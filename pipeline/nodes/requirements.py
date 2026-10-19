"""
pipeline/nodes/requirements.py — Stage 1: Requirements Agent

Analyzes raw GitHub issues received in the pipeline state and produces a
structured RequirementsSpec. This specification serves as the foundational data
contract for all downstream stages (Architecture planning, Coding, Testing,
and Reviews).
"""

import logging
import time
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.state import PipelineState, RequirementsSpec
from pipeline.models import flash_model
from pipeline.tools.code_search import search_codebase
from memory.store import get_recent_memories

logger = logging.getLogger(__name__)

# Module-level constant: Defining the system prompt at module scope decouples
# prompt maintenance and versioning from function execution logic, making it
# easily inspectable, reusable, and testable across test suites.
REQUIREMENTS_SYSTEM_PROMPT = """You are a senior software engineer analyzing a GitHub issue to produce a precise, actionable requirements specification.

Carefully extract and generate the following fields according to these strict behaviors:

1. summary: Restate the issue in one clear, concise paragraph from an engineering perspective — explaining what needs to be built or fixed. Do not simply copy the issue title.
2. acceptance_criteria: List specific, testable, and falsifiable conditions (aim for 3-6 criteria). For example, state "the endpoint returns a 400 status code for invalid email format" instead of vague phrases like "input is validated".
3. constraints: List explicit technical constraints only, such as existing patterns to follow, systems that must not break, or stated performance requirements. Return an empty list if none are specified.
4. files_likely_affected: List relative file paths likely relevant based on the issue description and context. Return an empty list if there is insufficient information.
5. ambiguities: List specific questions or blockers that must be answered before implementation can proceed. Only include genuine blockers, not minor uncertainties. Return an empty list if the issue is clear.
6. confidence: Set to "high" if the issue is clear and complete enough to implement without clarification; "medium" if there are minor uncertainties but implementation could proceed; "low" if critical information is missing.

Be precise, concise, and rigorous. Do not pad fields with vague content. An empty list is correct when there is nothing to put there.
Respond with a valid JSON object that strictly follows the provided schema."""


def requirements_node(state: PipelineState) -> dict:
    time.sleep(15)  # ponytail: free-tier 5 RPM limit; remove when on paid tier
    """
    Stage 1: Requirements Agent.

    Analyzes the GitHub issue and produces a structured RequirementsSpec.
    Uses Gemini 2.0 Flash with structured output to extract:
    - A clear summary of what needs to be built
    - Testable acceptance criteria
    - Technical constraints
    - Likely affected files
    - Ambiguities that need clarification
    - Confidence level in the spec's completeness

    Reads:  state["issue"]
    Writes: state["requirements_spec"], state["pipeline_status"]
    """
    issue = state["issue"]

    # Build the human message content step-by-step for clear parsing
    title = issue.get("title", "No title")
    number = issue.get("number", "Unknown")
    labels = ", ".join(issue.get("labels", [])) or "None"
    body = issue.get("body", "No description provided.")

    parts = [
        f"Issue Title: {title}",
        f"Issue Number: #{number}",
        f"Labels: {labels}",
        "",
        "Issue Description:",
        body,
    ]

    # Append any discussion comments present on the issue
    comments = issue.get("comments")
    if comments:
        parts.append("")
        parts.append("--- Comments ---")
        if isinstance(comments, list):
            for comment in comments:
                if isinstance(comment, dict):
                    author = comment.get("user") or comment.get("author", "Unknown")
                    comment_body = comment.get("body", "")
                    parts.append(f"Comment by {author}: {comment_body}")
                else:
                    parts.append(str(comment))
        else:
            parts.append(str(comments))

    human_message_content = "\n".join(parts)

    # Inject relevant existing code from the codebase index (RAG).
    # search_codebase returns "" when no index exists — graceful degradation.
    rag_context = search_codebase(query=title, k=3)
    if rag_context:
        human_message_content = human_message_content + "\n\n" + rag_context

    # Retrieve past memories relevant to similar issues.
    # Returns empty string if no memories exist — handled gracefully.
    past_memories = get_recent_memories(limit=3)
    if past_memories:
        human_message_content = (
            human_message_content
            + "\n\nPast Pipeline Runs (similar issues):\n"
            + past_memories
            + "\nUse past patterns to inform your analysis:\n"
            + "- Note recurring review findings to watch for\n"
            + "- Consider lessons learned from similar changes\n"
            + "- Past outcomes may indicate risk level"
        )

    messages = [
        SystemMessage(content=REQUIREMENTS_SYSTEM_PROMPT),
        HumanMessage(content=human_message_content),
    ]

    # Bind schema per-invocation; uses model configured in pipeline/models.py
    structured_model = flash_model.with_structured_output(RequirementsSpec, method="json_mode")

    try:
        spec = structured_model.invoke(messages)
        logger.info(
            "Requirements extracted",
            extra={
                "issue_number": issue.get("number"),
                "confidence": spec.confidence,
                "acceptance_criteria_count": len(spec.acceptance_criteria),
                "ambiguities_count": len(spec.ambiguities),
            },
        )
        return {
            "requirements_spec": spec,
            "pipeline_status": "requirements_complete",
        }

    except Exception as e:
        logger.error(
            "Requirements agent failed",
            extra={"issue_number": issue.get("number"), "error": str(e)},
        )
        return {
            "pipeline_status": f"requirements_failed: {str(e)}",
        }
