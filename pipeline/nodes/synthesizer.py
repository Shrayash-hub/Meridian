"""
pipeline/nodes/synthesizer.py — Stage 6: Review Synthesizer

This file implements Stage 6 of the SDLC pipeline — the aggregator node that
merges, deduplicates, and synthesizes findings from all three parallel review
agents (security, performance, quality) into a single, unified ReviewFindings
object with an overall recommendation ("approve", "request_changes", "reject").

Unlike the Stage 5 review worker nodes which receive scoped state slices, this
is an orchestrator/aggregator node that reads the full PipelineState.
"""

import logging
import time
from collections import defaultdict
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.state import (
    PipelineState,
    ReviewFindings,
    FindingItem,
)
from pipeline.models import flash_model

logger = logging.getLogger(__name__)


SYNTHESIZER_SYSTEM_PROMPT = """You are a senior engineering lead synthesizing code review findings from multiple specialist reviewers (security, performance, and quality) into a unified assessment.

Your job is to consolidate all individual reviewer findings into a structured ReviewFindings evaluation and determine a final PR recommendation.

Instructions for synthesis:

1. For each severity bucket, collect ALL findings of that severity from all reviewers.
   Do not drop or merge findings that address different issues, even if they touch the same file or function.

2. Resolve conflicts and redundancies between reviewers:
   - If two reviewers raise findings on the same underlying issue but disagree on severity, assign the higher severity.
   - If findings overlap on the exact same problem, keep the most detailed and actionable finding, and discard the duplicate.

3. Assign findings into the four severity buckets:
   - "blocking_issues": critical vulnerabilities, data loss risks, severe regressions, or fundamental correctness flaws.
   - "major_issues": significant quality, maintainability, or performance problems that will cause production issues.
   - "minor_issues": small inefficiencies, naming issues, missing docstrings, or non-urgent maintenance debt.
   - "suggestions": optional improvements, polish, or future cleanup ideas.

4. Make a final recommendation based on the consolidated findings:
   - "reject": if the overall architecture or approach is fundamentally flawed, unfixable, or would cause irreversible harm.
   - "request_changes": if there are any blocking issues, or 3 or more major issues.
   - "approve": if there are no blocking issues and fewer than 3 major issues (minor issues and suggestions do not block approval).

5. Provide a clear recommendation_reasoning:
   - Write one concise paragraph explaining the rationale for the recommendation.
   - Reference the most critical findings that drove the decision.
   - If requesting changes, clearly state what must be fixed for the next revision to be approvable.

The response must be a JSON object with EXACTLY these field names:
- "blocking_issues": list of FindingItem objects (severity == "blocking")
- "major_issues": list of FindingItem objects (severity == "major")
- "minor_issues": list of FindingItem objects (severity == "minor")
- "suggestions": list of FindingItem objects (severity == "suggestion")
- "recommendation": string — one of "approve", "request_changes", "reject"
- "recommendation_reasoning": string — one paragraph explaining the decision

Each FindingItem in the lists above must have:
- "reviewer": string ("security", "performance", or "quality")
- "severity": string ("blocking", "major", "minor", or "suggestion")
- "title": string
- "description": string
- "file_path": string
- "line_number": integer or null
- "suggestion": string

Respond with a valid JSON object that strictly follows the provided schema."""


def synthesizer_node(state: PipelineState) -> dict:
    time.sleep(15)
    """
    Stage 6: Review Synthesizer.

    Reads all findings accumulated from the three parallel review
    agents (review_findings_raw) and merges them into a unified
    ReviewFindings object. Resolves conflicts, deduplicates, and
    produces a single recommendation for the HITL gate.

    This is an aggregator/orchestrator node — it reads the full
    PipelineState, not a state slice.

    Reads:  state["review_findings_raw"], state["requirements_spec"]
    Writes: state["review_findings"], state["pipeline_status"]
    """
    raw_findings = state.get("review_findings_raw", [])
    spec = state.get("requirements_spec")

    # Guard: if no findings were produced by any reviewer, produce minimal approval
    if not raw_findings:
        logger.warning("Synthesizer called with no review findings")
        empty_findings = ReviewFindings(
            blocking_issues=[],
            major_issues=[],
            minor_issues=[],
            suggestions=[],
            recommendation="approve",
            recommendation_reasoning=(
                "No review findings were produced. "
                "All review agents returned empty results. "
                "Defaulting to approve — human should verify manually."
            ),
        )
        return {
            "review_findings": empty_findings,
            "pipeline_status": "review_complete",
        }

    # Group raw findings by reviewer for clear presentation to LLM
    by_reviewer = defaultdict(list)
    for f in raw_findings:
        by_reviewer[f.reviewer].append(f)

    findings_sections = []
    for reviewer_type, r_findings in by_reviewer.items():
        lines = [f"[{reviewer_type.upper()} REVIEWER — {len(r_findings)} findings]"]
        for f in r_findings:
            lines.append(
                f"  Severity: {f.severity}\n"
                f"  Title: {f.title}\n"
                f"  File: {f.file_path}\n"
                f"  Line: {f.line_number if f.line_number is not None else 'N/A'}\n"
                f"  Description: {f.description}\n"
                f"  Suggestion: {f.suggestion}\n"
            )
        findings_sections.append("\n".join(lines))

    findings_text = "\n\n".join(findings_sections)
    summary_text = spec.summary if spec else "Not available"

    human_message_content = (
        f"Total findings to synthesize: {len(raw_findings)}\n\n"
        f"Findings by reviewer:\n\n"
        f"{findings_text}\n\n"
        f"Requirements Summary (for context):\n"
        f"{summary_text}\n\n"
        f"Synthesize the above findings into a unified ReviewFindings."
    )

    structured_model = flash_model.with_structured_output(ReviewFindings, method="json_mode")

    try:
        findings = structured_model.invoke([
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=human_message_content),
        ])

        logger.info(
            "Review synthesis complete",
            extra={
                "recommendation": findings.recommendation,
                "blocking": len(findings.blocking_issues),
                "major": len(findings.major_issues),
                "minor": len(findings.minor_issues),
                "suggestions": len(findings.suggestions),
            }
        )

        return {
            "review_findings": findings,
            "pipeline_status": "review_complete",
        }

    except Exception as e:
        logger.error(
            "Synthesizer failed",
            extra={"error": str(e)}
        )
        return {
            "pipeline_status": f"synthesis_failed: {str(e)}",
        }
