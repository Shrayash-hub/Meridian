"""
pipeline/nodes/memory_update.py — Post-Pipeline Memory Update Node

This node runs only on the success path, after the PR Author has opened a PR
(pipeline_status == "pr_opened"). It extracts reusable patterns from the
completed pipeline run using an LLM call (delegated to memory.store.extract_memory)
and saves the resulting PRMemory to the long-term store.

Memory update is best-effort. This node must never cause the pipeline to fail —
if extraction or storage fails for any reason, the node logs the error and returns
a status suffix, but the PR is already open and the pipeline result is preserved.

This node has no LLM call of its own. The LLM call is inside extract_memory().
"""

import time
import logging
from pipeline.state import PipelineState
from memory.store import extract_memory, save_memory

logger = logging.getLogger(__name__)


def memory_update_node(state: PipelineState) -> dict:
    """
    Post-pipeline: Memory Update Node.

    Runs after PR Author on the success path only.
    Extracts reusable patterns from the completed pipeline run and saves
    them to long-term memory (InMemoryStore).

    On failure, logs the error and returns without crashing — memory update
    failure must never affect the pipeline result.

    Reads:  state["issue"], state["code_changes"], state["test_results"],
            state["review_findings"], state["human_decision"]
    Writes: state["pipeline_status"] (appends memory status suffix)
    """
    time.sleep(15)  # ponytail: free-tier RPM guard; remove when on paid tier

    issue = state.get("issue", {})
    code_changes = state.get("code_changes", [])
    test_results = state.get("test_results")
    review_findings = state.get("review_findings")
    human_decision = state.get("human_decision", "")

    logger.info(
        "Running memory update",
        extra={"issue_number": issue.get("number")},
    )

    try:
        memory = extract_memory(
            issue=issue,
            code_changes=code_changes,
            test_results=test_results,
            review_findings=review_findings,
            human_decision=human_decision,
        )

        if memory is None:
            logger.warning("Memory extraction returned None, skipping save")
            return {"pipeline_status": "pr_opened_memory_skipped"}

        key = save_memory(memory, issue.get("number", "unknown"))
        logger.info(
            "Memory update complete",
            extra={"key": key},
        )
        return {"pipeline_status": "pr_opened_memory_saved"}

    except Exception as e:
        logger.error(
            "Memory update node failed",
            extra={"error": str(e)},
        )
        # Never crash the pipeline — memory update is best-effort
        return {"pipeline_status": "pr_opened_memory_failed"}
