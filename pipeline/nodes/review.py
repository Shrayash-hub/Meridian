"""
pipeline/nodes/review.py — Stage 5: Review Agents (parallel fan-out)

This file implements Stage 5 of the SDLC pipeline — three parallel review
agents running simultaneously via the LangGraph Send API fan-out. Each agent
reviews the accumulated code changes (state["code_changes"]) from a distinct
engineering dimension:
  - Security: vulnerabilities, authentication/authorization, data exposure
  - Performance: algorithmic complexity, query efficiency, resource leaks
  - Quality: error handling, readability, testability, design consistency

Results from all three reviewers accumulate into state["review_findings_raw"]
via the operator.add reducer defined on PipelineState.

Three separate node functions are defined (security_review_node,
performance_review_node, and quality_review_node) rather than a single generic
node because parallel execution in LangGraph requires separate named nodes in
the graph topology. Each function binds its own specialized system prompt.
"""

import operator
import time
import logging
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel as PydanticBaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.state import (
    CodeChange,
    RequirementsSpec,
    FindingItem,
    PipelineState,
)
from pipeline.models import review_model

logger = logging.getLogger(__name__)


class ReviewAgentState(TypedDict):
    """
    State slice passed to each review agent instance via Send API.
    reviewer_type determines which dimension of the code is reviewed.
    review_findings_raw uses operator.add so all three agents'
    findings accumulate into PipelineState["review_findings_raw"].
    """
    reviewer_type: str
    code_changes: list[CodeChange]
    requirements_spec: RequirementsSpec
    review_findings_raw: Annotated[list[FindingItem], operator.add]


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

SECURITY_REVIEW_PROMPT = """You are a senior security engineer reviewing code changes for security vulnerabilities.

Your job is to thoroughly inspect the provided code changes and identify genuine security concerns.

Focus areas to check explicitly:
- Injection vulnerabilities (SQL, command, LDAP, XPath, template)
- Authentication and authorization flaws
- Sensitive data exposure (secrets, PII, credentials or tokens hardcoded in code)
- Input validation gaps, boundary conditions, and missing sanitization
- Insecure dependencies or dangerous function usage
- OWASP Top 10 categories relevant to the modified code

Only report genuine security concerns with concrete risk, not general code style issues.
Be specific and actionable. Reference exact file paths and line numbers where possible.
If no security issues are found, return an empty list — do not invent issues just to fill the response.

Severity definitions:
- "blocking": must be fixed before merge — security vulnerability, data loss risk, or fundamental correctness issue
- "major": should be fixed — significant security gap or exposed attack surface that could cause issues in production
- "minor": worth fixing but not urgent — defense-in-depth improvement or low-impact hardening
- "suggestion": optional improvement — security best practice recommendation

Each finding item must use EXACTLY these field names:
- "reviewer": string — must be set to "security"
- "severity": string — one of "blocking", "major", "minor", "suggestion"
- "title": string — short descriptive title of the issue (under 10 words)
- "description": string — specific explanation of the problem, mechanism, and risk
- "file_path": string — the file where the issue exists
- "line_number": integer or null — integer if identifiable, null if not
- "suggestion": string — concrete, actionable fix recommendation

Respond with a valid JSON object containing a 'review_findings_raw' list. Each item must use exactly the field names listed above."""


PERFORMANCE_REVIEW_PROMPT = """You are a senior engineer reviewing code changes for performance issues and inefficiencies.

Your job is to inspect the provided code changes and identify genuine performance bottlenecks.

Focus areas to check explicitly:
- Algorithmic complexity (unnecessary O(n²) or worse where O(n) exists)
- N+1 query patterns in database access or repetitive external calls
- Unnecessary computation in hot paths, loops, or request handlers
- Missing caching where repeated identical expensive calls occur
- Memory leaks, unbounded data structure growth, or missing resource cleanup
- Synchronous blocking calls where async or background execution would be appropriate

Only report genuine performance concerns with measurable impact, not trivial micro-optimizations.
Be specific and actionable. Reference exact file paths and line numbers where possible.
If no performance issues are found, return an empty list — do not invent issues just to fill the response.

Severity definitions:
- "blocking": must be fixed before merge — severe denial-of-service risk, massive memory leak, or catastrophic algorithmic scaling
- "major": should be fixed — significant quality or performance problem that will cause latency or high resource usage in production
- "minor": worth fixing but not urgent — small inefficiency or sub-optimal data structure choice
- "suggestion": optional improvement — caching opportunity or non-critical optimization

Each finding item must use EXACTLY these field names:
- "reviewer": string — must be set to "performance"
- "severity": string — one of "blocking", "major", "minor", "suggestion"
- "title": string — short descriptive title of the issue (under 10 words)
- "description": string — specific explanation of the performance impact and scenario
- "file_path": string — the file where the issue exists
- "line_number": integer or null — integer if identifiable, null if not
- "suggestion": string — concrete, actionable fix recommendation

Respond with a valid JSON object containing a 'review_findings_raw' list. Each item must use exactly the field names listed above."""


QUALITY_REVIEW_PROMPT = """You are a senior engineer reviewing code changes for quality, maintainability, and correctness.

Your job is to inspect the provided code changes and identify quality, testing, and design problems.

Focus areas to check explicitly:
- Missing or inadequate error handling and recovery
- Insufficient test coverage for the changes made
- Unclear naming (variables, functions, classes) that obscures intent
- Missing or poor documentation and docstrings on public interfaces
- Code duplication that should be extracted or consolidated
- Violation of single responsibility principle or tight coupling
- Inconsistency with patterns described in requirements constraints

Focus on issues that will make the code harder to maintain, extend, or debug, not personal style preferences.
Be specific and actionable. Reference exact file paths and line numbers where possible.
If no quality issues are found, return an empty list — do not invent issues just to fill the response.

Severity definitions:
- "blocking": must be fixed before merge — fundamental correctness issue, broken contract, or regression of existing functionality
- "major": should be fixed — missing critical error handling, absent test coverage for core logic, or high maintenance hazard
- "minor": worth fixing but not urgent — naming clarity, missing docstring, or slight pattern deviation
- "suggestion": optional improvement — refactoring suggestion or idiom polish

Each finding item must use EXACTLY these field names:
- "reviewer": string — must be set to "quality"
- "severity": string — one of "blocking", "major", "minor", "suggestion"
- "title": string — short descriptive title of the issue (under 10 words)
- "description": string — specific explanation of the maintainability or correctness issue
- "file_path": string — the file where the issue exists
- "line_number": integer or null — integer if identifiable, null if not
- "suggestion": string — concrete, actionable fix recommendation

Respond with a valid JSON object containing a 'review_findings_raw' list. Each item must use exactly the field names listed above."""


# ---------------------------------------------------------------------------
# Shared Helper Function
# ---------------------------------------------------------------------------

# PONYTAIL: one helper is justified here — three identical formatting blocks across
# three node functions would be worse than one shared function.
def _build_code_context(code_changes: list[CodeChange]) -> str:
    """
    Builds the code section of the human message.
    Formats each CodeChange with file path, task ID, and complete content.
    """
    blocks = []
    for change in code_changes:
        block = (
            f"File: {change.file_path}\n"
            f"Task: {change.task_id}\n"
            f"-----------------------\n"
            f"{change.content}\n"
            f"-----------------------"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def security_review_node(state: ReviewAgentState) -> dict:
    """
    Stage 5: Security Review Agent.
    Reviews code_changes for security issues.
    Runs in parallel with other review agents via Send API.

    Reads:  state["code_changes"], state["requirements_spec"]
    Writes: state["review_findings_raw"] (appends findings list)
    """
    time.sleep(15)  # free-tier RPM guard — security reviewer fires first (no extra stagger)
    code_changes = state.get("code_changes") or []
    spec = state.get("requirements_spec")
    reviewer_type = state.get("reviewer_type", "security")

    if not code_changes:
        logger.warning(
            "Review agent called with no code changes",
            extra={"reviewer": reviewer_type}
        )
        return {"review_findings_raw": []}

    criteria_lines = [f"{i}. {c}" for i, c in enumerate(spec.acceptance_criteria, 1)] if spec and spec.acceptance_criteria else ["None specified."]
    criteria_text = "\n".join(criteria_lines)
    summary_text = spec.summary if spec else "None specified."

    human_message_content = (
        f"Reviewer Role: {reviewer_type}\n\n"
        f"Requirements Summary:\n"
        f"{summary_text}\n\n"
        f"Acceptance Criteria:\n"
        f"{criteria_text}\n\n"
        f"Code Changes to Review:\n"
        f"{_build_code_context(code_changes)}\n\n"
        f"Review the code changes above from a {reviewer_type} perspective.\n"
        f"Identify all {reviewer_type} issues following the severity definitions in your instructions."
    )

    class ReviewOutput(PydanticBaseModel):
        review_findings_raw: list[FindingItem]

    structured_model = review_model.with_structured_output(ReviewOutput)

    try:
        output = structured_model.invoke([
            SystemMessage(content=SECURITY_REVIEW_PROMPT),
            HumanMessage(content=human_message_content),
        ])
        findings = output.review_findings_raw

        logger.info(
            "Review complete",
            extra={
                "reviewer": reviewer_type,
                "findings_count": len(findings),
                "blocking": sum(1 for f in findings if f.severity == "blocking"),
            }
        )

        return {"review_findings_raw": findings}

    except Exception as e:
        _exc_type = type(e).__name__
        _exc_msg  = str(e)[:400]
        _http_body = ""
        _resp = getattr(e, "response", None)
        if _resp is not None:
            try:
                _http_body = f" | HTTP {_resp.status_code}: {_resp.text[:300]}"
            except Exception:
                pass
        logger.error(
            "Review agent failed",
            extra={
                "reviewer":  reviewer_type,
                "exc_type":  _exc_type,
                "exc_msg":   _exc_msg + _http_body,
            }
        )
        return {"review_findings_raw": []}


def performance_review_node(state: ReviewAgentState) -> dict:
    """
    Stage 5: Performance Review Agent.
    Reviews code_changes for performance issues.
    Runs in parallel with other review agents via Send API.

    Reads:  state["code_changes"], state["requirements_spec"]
    Writes: state["review_findings_raw"] (appends findings list)
    """
    time.sleep(15)  # free-tier RPM guard — performance reviewer stagger added before LLM call
    code_changes = state.get("code_changes") or []
    spec = state.get("requirements_spec")
    reviewer_type = state.get("reviewer_type", "performance")

    if not code_changes:
        logger.warning(
            "Review agent called with no code changes",
            extra={"reviewer": reviewer_type}
        )
        return {"review_findings_raw": []}

    criteria_lines = [f"{i}. {c}" for i, c in enumerate(spec.acceptance_criteria, 1)] if spec and spec.acceptance_criteria else ["None specified."]
    criteria_text = "\n".join(criteria_lines)
    summary_text = spec.summary if spec else "None specified."

    human_message_content = (
        f"Reviewer Role: {reviewer_type}\n\n"
        f"Requirements Summary:\n"
        f"{summary_text}\n\n"
        f"Acceptance Criteria:\n"
        f"{criteria_text}\n\n"
        f"Code Changes to Review:\n"
        f"{_build_code_context(code_changes)}\n\n"
        f"Review the code changes above from a {reviewer_type} perspective.\n"
        f"Identify all {reviewer_type} issues following the severity definitions in your instructions."
    )

    class ReviewOutput(PydanticBaseModel):
        review_findings_raw: list[FindingItem]

    time.sleep(5)  # stagger: fires 5s after security reviewer
    structured_model = review_model.with_structured_output(ReviewOutput)

    try:
        output = structured_model.invoke([
            SystemMessage(content=PERFORMANCE_REVIEW_PROMPT),
            HumanMessage(content=human_message_content),
        ])
        findings = output.review_findings_raw

        logger.info(
            "Review complete",
            extra={
                "reviewer": reviewer_type,
                "findings_count": len(findings),
                "blocking": sum(1 for f in findings if f.severity == "blocking"),
            }
        )

        return {"review_findings_raw": findings}

    except Exception as e:
        _exc_type = type(e).__name__
        _exc_msg  = str(e)[:400]
        _http_body = ""
        _resp = getattr(e, "response", None)
        if _resp is not None:
            try:
                _http_body = f" | HTTP {_resp.status_code}: {_resp.text[:300]}"
            except Exception:
                pass
        logger.error(
            "Review agent failed",
            extra={
                "reviewer":  reviewer_type,
                "exc_type":  _exc_type,
                "exc_msg":   _exc_msg + _http_body,
            }
        )
        return {"review_findings_raw": []}


def quality_review_node(state: ReviewAgentState) -> dict:
    """
    Stage 5: Quality Review Agent.
    Reviews code_changes for quality issues.
    Runs in parallel with other review agents via Send API.

    Reads:  state["code_changes"], state["requirements_spec"]
    Writes: state["review_findings_raw"] (appends findings list)
    """
    time.sleep(15)  # free-tier RPM guard — quality reviewer stagger added before LLM call
    code_changes = state.get("code_changes") or []
    spec = state.get("requirements_spec")
    reviewer_type = state.get("reviewer_type", "quality")

    if not code_changes:
        logger.warning(
            "Review agent called with no code changes",
            extra={"reviewer": reviewer_type}
        )
        return {"review_findings_raw": []}

    criteria_lines = [f"{i}. {c}" for i, c in enumerate(spec.acceptance_criteria, 1)] if spec and spec.acceptance_criteria else ["None specified."]
    criteria_text = "\n".join(criteria_lines)
    summary_text = spec.summary if spec else "None specified."

    human_message_content = (
        f"Reviewer Role: {reviewer_type}\n\n"
        f"Requirements Summary:\n"
        f"{summary_text}\n\n"
        f"Acceptance Criteria:\n"
        f"{criteria_text}\n\n"
        f"Code Changes to Review:\n"
        f"{_build_code_context(code_changes)}\n\n"
        f"Review the code changes above from a {reviewer_type} perspective.\n"
        f"Identify all {reviewer_type} issues following the severity definitions in your instructions."
    )

    class ReviewOutput(PydanticBaseModel):
        review_findings_raw: list[FindingItem]

    time.sleep(10)  # stagger: fires 10s after security reviewer
    structured_model = review_model.with_structured_output(ReviewOutput)

    try:
        output = structured_model.invoke([
            SystemMessage(content=QUALITY_REVIEW_PROMPT),
            HumanMessage(content=human_message_content),
        ])
        findings = output.review_findings_raw

        logger.info(
            "Review complete",
            extra={
                "reviewer": reviewer_type,
                "findings_count": len(findings),
                "blocking": sum(1 for f in findings if f.severity == "blocking"),
            }
        )

        return {"review_findings_raw": findings}

    except Exception as e:
        _exc_type = type(e).__name__
        _exc_msg  = str(e)[:400]
        _http_body = ""
        _resp = getattr(e, "response", None)
        if _resp is not None:
            try:
                _http_body = f" | HTTP {_resp.status_code}: {_resp.text[:300]}"
            except Exception:
                pass
        logger.error(
            "Review agent failed",
            extra={
                "reviewer":  reviewer_type,
                "exc_type":  _exc_type,
                "exc_msg":   _exc_msg + _http_body,
            }
        )
        return {"review_findings_raw": []}
