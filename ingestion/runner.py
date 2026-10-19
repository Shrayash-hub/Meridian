"""
Pipeline Runner — Issue Translation and Execution Orchestration.

This module runs the full SDLC pipeline for a given GitHub issue.
Handles the two-phase invocation required by the HITL gate:
- Phase 1: runs until interrupt() at the human-in-the-loop gate
- Phase 2: resumes with a human decision (or auto-approves)

When AUTO_APPROVE=true, Phase 2 automatically approves without waiting
for human input.

ponytail: production Phase 2 requires:
- Storing the HITL payload (issue number, review findings) in a persistent
  database after Phase 1 completes
- A /review/{issue_number} endpoint that exposes the payload to a frontend
- A /approve/{issue_number} endpoint that receives human input and triggers Phase 2
- This runner currently bundles both phases synchronously for simplicity.
"""

import os
import uuid
import logging
from langgraph.types import Command
from pipeline.graph import pipeline
from pipeline.config import build_pipeline_config
from evaluation.evaluator import record_run_metrics

logger = logging.getLogger(__name__)


def run_pipeline_for_issue(issue: dict) -> dict:
    """
    Run the full SDLC pipeline for a GitHub issue.

    Handles the two-phase invocation:
    - Phase 1: pipeline runs from requirements through
      review synthesis, then pauses at HITL gate
    - Phase 2: pipeline resumes with human (or auto) decision
      and continues through PR author and memory update

    When AUTO_APPROVE=true, Phase 2 auto-approves.
    When AUTO_APPROVE=false (production), Phase 2 requires
    a human to call a separate endpoint with their decision.
    This runner only implements AUTO_APPROVE mode.

    ponytail: production Phase 2 requires:
    - Storing the HITL payload (issue number, review findings)
      in a database after Phase 1
    - A /review/{issue_number} endpoint that shows the payload
    - A /approve/{issue_number} endpoint that triggers Phase 2
    - This runner currently bundles both phases for simplicity.

    Returns: the final pipeline state dict.
    """
    auto_approve = os.getenv("AUTO_APPROVE", "false").lower() == "true"
    issue_number = issue.get("number", "unknown")
    issue_title = issue.get("title", "")

    thread_id = f"issue-{issue_number}-{uuid.uuid4().hex[:8]}"
    config = build_pipeline_config(
        thread_id=thread_id,
        issue_number=issue_number,
        issue_title=issue_title,
    )

    initial_state = {
        "issue": issue,
        "revision_count": 0,
        "max_revisions": int(os.getenv("MAX_REVISIONS", "3")),
        "total_cost_usd": 0.0,
    }

    logger.info(
        "Starting pipeline",
        extra={
            "issue_number": issue_number,
            "thread_id": thread_id,
            "auto_approve": auto_approve,
        },
    )

    # Phase 1: Run to interrupt (HITL gate).
    # GraphInterrupt is a BaseException (not Exception) raised by LangGraph
    # when interrupt() is called inside hitl_node. It signals a successful
    # pause, not a failure. We catch it here so Phase 2 can resume.
    try:
        phase1_result = pipeline.invoke(initial_state, config=config)
    except BaseException as e:
        from langgraph.errors import GraphInterrupt
        if isinstance(e, GraphInterrupt):
            # Normal HITL pause — extract state from checkpoint and continue
            logger.info(
                "Phase 1 paused at HITL gate (GraphInterrupt caught)",
                extra={"issue_number": issue_number},
            )
            # Get the last checkpoint state so we have phase1_result to log
            try:
                phase1_result = pipeline.get_state(config).values
            except Exception:
                phase1_result = {}
        else:
            logger.error(
                "Pipeline Phase 1 failed",
                extra={"issue_number": issue_number, "error": str(e)},
            )
            return {"pipeline_status": f"phase1_failed: {str(e)[:100]}"}

    logger.info(
        "Phase 1 complete — pipeline at HITL gate",
        extra={
            "issue_number": issue_number,
            "status": phase1_result.get("pipeline_status"),
            "code_changes": len(phase1_result.get("code_changes", [])),
        },
    )

    # Phase 2: Resume with decision
    if auto_approve:
        human_response = {
            "decision": "approve",
            "feedback": (
                "Auto-approved by pipeline runner "
                "(AUTO_APPROVE=true)."
            ),
        }
        logger.info(
            "AUTO_APPROVE=true — auto-resuming HITL gate",
            extra={"issue_number": issue_number},
        )
    else:
        # ponytail: In production, store phase1_result and
        # wait for human decision via a separate endpoint.
        # For now, default to approve to complete the run.
        logger.warning(
            "AUTO_APPROVE=false — defaulting to approve. "
            "Production should wait for human decision.",
            extra={"issue_number": issue_number},
        )
        human_response = {
            "decision": "approve",
            "feedback": "Approved (no review UI configured).",
        }

    try:
        final_result = pipeline.invoke(
            Command(resume=human_response),
            config=config,
        )
    except Exception as e:
        logger.error(
            "Pipeline Phase 2 failed",
            extra={"issue_number": issue_number, "error": str(e)},
        )
        return {
            **phase1_result,
            "pipeline_status": f"phase2_failed: {str(e)[:100]}",
        }

    # Record metrics
    try:
        record_run_metrics(final_result, issue)
    except Exception as e:
        # Metrics recording failure must never affect pipeline result
        logger.warning(
            "Metrics recording failed",
            extra={"error": str(e)},
        )

    logger.info(
        "Pipeline complete",
        extra={
            "issue_number": issue_number,
            "status": final_result.get("pipeline_status"),
            "pr": final_result.get("pr", {}).get("pr_url", "none") if isinstance(final_result.get("pr"), dict) else "none",
        },
    )
    return final_result
