"""
pipeline/nodes/hitl.py — Stage 7: Human-in-the-Loop Gate

This file implements Stage 7 of the SDLC pipeline — the Human-in-the-Loop gate.
It pauses graph execution using LangGraph's interrupt() primitive and serializes
the full execution state to the checkpointer. Execution remains paused until an
external engineer calls Command(resume={"decision": ..., "feedback": ...})
with the same thread_id.

This node surfaces a structured review summary to the engineer (including review
recommendation, reasoning, issue counts, changed files, and remaining revision
budget) and validates the human response. On "request_changes", it increments
revision_count to advance the revision loop towards max_revisions.
"""

import logging
from langgraph.types import interrupt
from pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def hitl_node(state: PipelineState) -> dict:
    """
    Stage 7: Human-in-the-Loop Gate.

    Pauses graph execution using interrupt() and surfaces the
    complete review package to the human engineer. Resumes when
    Command(resume={"decision": ..., "feedback": ...}) is received.

    Decision routing (handled by route_after_hitl in graph.py):
      "approve"         → PR Author (Stage 8)
      "request_changes" → Coding Agents (Stage 3, revision loop)
      "reject"          → END

    On "request_changes", increments revision_count so the
    revision loop terminates after max_revisions attempts.

    Reads:  state["review_findings"], state["requirements_spec"],
            state["implementation_plan"], state["test_results"],
            state["revision_count"], state["max_revisions"]
    Writes: state["human_decision"], state["human_feedback"],
            state["revision_count"] (incremented on request_changes),
            state["pipeline_status"]
    """
    # Step 1: Build the interrupt payload
    review_findings = state.get("review_findings")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)

    payload = {
        "stage": "human_review_required",
        "revision_count": revision_count,
        "max_revisions": max_revisions,
        "revisions_remaining": max_revisions - revision_count,
        "review_recommendation": (
            review_findings.recommendation
            if review_findings else "unknown"
        ),
        "review_reasoning": (
            review_findings.recommendation_reasoning
            if review_findings else "No review findings available."
        ),
        "blocking_issues_count": (
            len(review_findings.blocking_issues)
            if review_findings else 0
        ),
        "major_issues_count": (
            len(review_findings.major_issues)
            if review_findings else 0
        ),
        "requirements_summary": (
            state["requirements_spec"].summary
            if state.get("requirements_spec") else "Not available"
        ),
        "files_changed": [
            c.file_path for c in state.get("code_changes", [])
        ],
        "valid_decisions": (
            ["approve", "request_changes", "reject"]
            if revision_count < max_revisions
            else ["approve", "reject"]
        ),
    }

    # Step 2: Call interrupt()
    logger.info(
        "Pipeline paused for human review",
        extra={
            "revision_count": revision_count,
            "recommendation": payload["review_recommendation"],
        }
    )

    human_response = interrupt(payload)

    # Step 3: Read and validate the human response
    decision = human_response.get("decision", "reject")
    feedback = human_response.get("feedback", "")

    allowed = {"approve", "request_changes", "reject"}
    if decision not in allowed:
        logger.warning(
            "Invalid human decision received, defaulting to reject",
            extra={"received": decision}
        )
        decision = "reject"

    if decision == "request_changes" and revision_count >= max_revisions:
        logger.warning(
            "request_changes received but revision budget exhausted, "
            "treating as reject",
            extra={"revision_count": revision_count,
                   "max_revisions": max_revisions}
        )
        decision = "reject"

    # Step 4: Build the return dict
    result = {
        "human_decision": decision,
        "human_feedback": feedback,
        "pipeline_status": f"hitl_{decision}",
    }

    if decision == "request_changes":
        result["revision_count"] = revision_count + 1

    return result
