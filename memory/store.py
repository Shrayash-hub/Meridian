"""
memory/store.py — Long-Term Memory for the SDLC Pipeline

Provides a module-level InMemoryStore singleton, a PRMemory Pydantic schema,
and three functions the pipeline uses to interact with memory:

  - extract_memory()      — LLM call to distil a completed run into a PRMemory
  - save_memory()         — persists a PRMemory to the store under a timestamped key
  - get_recent_memories() — retrieves and formats recent memories for prompt injection

Namespace / Key pattern:
  All PR pattern memories are stored under the namespace ("sdlc_pipeline", "pr_patterns").
  Keys have the form "issue-{number}-{YYYYMMDD-HHMMSS}" so they sort chronologically
  and are traceable back to the originating issue.

Ponytail limitations:
  InMemoryStore is volatile — all memories are lost on process restart.
  get_recent_memories() fetches by recency (last N inserted items), not by semantic
  similarity to the current issue.

  Production upgrade path:
    Replace `store = InMemoryStore()` with `AsyncPostgresStore` + pgvector.
    That single line change enables persistence across restarts and semantic
    vector search over memories. No other code in this module needs to change.
"""

import logging
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.models import flash_model
from pipeline.state import (
    RequirementsSpec,
    CodeChange,
    TestReport,
    ReviewFindings,
)

logger = logging.getLogger(__name__)


# ponytail: InMemoryStore loses all data on process restart
# and does not support semantic search.
# Production upgrade: replace with AsyncPostgresStore + pgvector
# for persistence and semantic retrieval over memories.
# The rest of the system does not need to change — only this
# initialization line.
store = InMemoryStore()

# Memory namespace — all PR pattern memories live here.
MEMORY_NAMESPACE = ("sdlc_pipeline", "pr_patterns")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class PRMemory(BaseModel):
    """
    Structured memory extracted from a completed pipeline run.
    Stored in long-term memory for use in future runs.
    """

    issue_type: str = Field(
        description=(
            "Category of the issue in 2-4 words. "
            "Examples: 'input validation', 'authentication bug', "
            "'performance optimization', 'API endpoint addition'. "
            "Choose the most specific accurate category."
        )
    )
    files_changed: list[str] = Field(
        description=(
            "List of file paths that were modified in this PR. "
            "Copy exactly from the code changes."
        )
    )
    approach_summary: str = Field(
        description=(
            "2-3 sentence summary of the implementation approach "
            "taken. Focus on the pattern used, not the specific "
            "values. Future agents should be able to recognize "
            "when to apply this same pattern."
        )
    )
    review_patterns: list[str] = Field(
        description=(
            "List of review findings that came up, each as a "
            "short phrase. Example: 'missing input length validation', "
            "'HTTPException inside Pydantic validator'. "
            "These are patterns to watch for in similar issues."
        )
    )
    outcome: str = Field(
        description=(
            "One of: 'approved', 'rejected', 'changes_requested'. "
            "The human's final decision at the HITL gate."
        )
    )
    lessons_learned: str = Field(
        description=(
            "1-2 sentences describing what a future agent should "
            "do differently or the same when handling a similar "
            "issue. Be specific and actionable."
        )
    )


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a senior software engineer extracting reusable patterns from a completed code review pipeline run.

Your goal is to produce a compact, accurate memory record that future agents will use when handling similar issues. Be concise and specific — memories are read by AI agents, not humans. Brevity and accuracy matter more than completeness.

Focus on patterns that transfer to similar future issues, not the specifics of this exact issue. A future agent reading this memory should immediately understand:
- What type of issue this was
- Which files were typically affected
- What implementation approach worked
- What review findings to watch for
- What the human decided and why

Required output fields:
- issue_type: Category of the issue in 2-4 words
- files_changed: Exact file paths modified
- approach_summary: 2-3 sentences on the implementation pattern used
- review_patterns: Short phrases describing each review finding
- outcome: One of 'approved', 'rejected', 'changes_requested'
- lessons_learned: 1-2 actionable sentences for future agents

Respond with a valid JSON object that strictly follows the provided schema."""


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def extract_memory(
    issue: dict,
    code_changes: list[CodeChange],
    test_results: TestReport | None,
    review_findings: ReviewFindings | None,
    human_decision: str,
) -> PRMemory | None:
    """
    Calls the LLM to extract a PRMemory from a completed pipeline run.
    Returns None on failure — never raises.
    """
    # --- Build the files changed section ---
    files_section = "\n".join(
        f"- {change.file_path}" for change in code_changes
    ) or "None"

    # --- Build the implementation approach section ---
    approach_lines = []
    for change in code_changes:
        excerpt = change.explanation[:150]
        approach_lines.append(f"- {change.file_path}: {excerpt}")
    approach_section = "\n".join(approach_lines) or "Not available"

    # --- Build the test results section ---
    if test_results:
        test_section = (
            f"Status: {test_results.final_status}\n"
            f"Tests passed: {test_results.new_tests_passed}\n"
            f"Tests failed: {test_results.new_tests_failed}"
        )
    else:
        test_section = "Not available"

    # --- Build the review findings section ---
    if review_findings:
        review_lines = [
            f"Recommendation: {review_findings.recommendation}",
            f"Blocking: {len(review_findings.blocking_issues)}",
            f"Major: {len(review_findings.major_issues)}",
        ]
        for finding in review_findings.blocking_issues + review_findings.major_issues:
            review_lines.append(
                f"- [{finding.reviewer}] {finding.title}: "
                f"{finding.description[:100]}"
            )
        review_section = "\n".join(review_lines)
    else:
        review_section = "Not available"

    human_message_content = (
        f"Issue Title: {issue.get('title', 'Unknown')}\n"
        f"Issue Labels: {', '.join(issue.get('labels', []))}\n\n"
        f"Files Changed:\n{files_section}\n\n"
        f"Implementation Approach:\n{approach_section}\n\n"
        f"Test Results:\n{test_section}\n\n"
        f"Review Findings:\n{review_section}\n\n"
        f"Human Decision: {human_decision or 'Not recorded'}\n\n"
        f"Extract the reusable patterns from this pipeline run."
    )

    structured_model = flash_model.with_structured_output(PRMemory, method="json_mode")

    try:
        memory = structured_model.invoke([
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=human_message_content),
        ])
        logger.info(
            "Memory extracted",
            extra={
                "issue_type": memory.issue_type,
                "outcome": memory.outcome,
                "files": len(memory.files_changed),
            },
        )
        return memory
    except Exception as e:
        logger.error(
            "Memory extraction failed",
            extra={"error": str(e)},
        )
        return None


def save_memory(
    memory: PRMemory,
    issue_number: int | str,
) -> str:
    """
    Saves a PRMemory to the store under a timestamped key.
    Returns the key used to store it.
    """
    key = (
        f"issue-{issue_number}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )

    store.put(
        MEMORY_NAMESPACE,
        key,
        memory.model_dump(),
    )

    logger.info(
        "Memory saved",
        extra={"key": key, "namespace": str(MEMORY_NAMESPACE)},
    )
    return key


def get_recent_memories(limit: int = 5) -> str:
    """
    Retrieves recent memories from the store and formats them for injection
    into an agent's human message. Returns empty string if no memories exist.

    ponytail: retrieves by recency (all memories, take last N).
    Production upgrade: semantic search with pgvector — query by similarity
    to the current issue rather than recency.
    Upgrade requires AsyncPostgresStore with vector embeddings.
    """
    try:
        items = store.search(MEMORY_NAMESPACE)
        if not items:
            return ""

        # Take the most recent N items
        recent = items[-limit:] if len(items) > limit else items

        formatted = ["Relevant past pipeline runs (most recent first):\n"]
        for item in reversed(recent):
            value = item.value
            formatted.append(
                f"Issue type: {value.get('issue_type', 'unknown')}"
            )
            formatted.append(
                f"Files changed: {', '.join(value.get('files_changed', []))}"
            )
            formatted.append(
                f"Approach: {value.get('approach_summary', '')}"
            )
            patterns = value.get("review_patterns", [])
            if patterns:
                formatted.append(f"Watch for: {'; '.join(patterns[:3])}")
            formatted.append(
                f"Outcome: {value.get('outcome', 'unknown')}"
            )
            lessons = value.get("lessons_learned", "")
            if lessons:
                formatted.append(f"Lesson: {lessons}")
            formatted.append("")  # blank line between memories

        return "\n".join(formatted)

    except Exception as e:
        logger.warning(
            "Failed to retrieve memories",
            extra={"error": str(e)},
        )
        return ""
