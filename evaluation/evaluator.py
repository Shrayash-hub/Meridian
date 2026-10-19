"""
Evaluation Framework — Metrics Tracking and Aggregation.

This module tracks pipeline quality metrics across runs. Metrics are stored
in LangGraph's InMemoryStore under a dedicated namespace ("sdlc_pipeline", "run_metrics")
separate from PR memories.

Recorded metrics include:
- PR acceptance rate: ratio of runs reaching successful PR creation
- Test pass rate: ratio of testing runs where test suites passed
- Human approval rate: ratio of human review decisions that approved changes
- Review findings: frequency of blocking and major review issues
- Cost: estimated cumulative LLM token cost in USD

ponytail: production evaluation should use LangSmith's evaluation API with
golden test sets, human annotations, and LLM-as-a-judge scoring. This
in-memory summary is sufficient for development monitoring and local CI.
"""

import logging
from datetime import datetime, timezone
from memory.store import store

logger = logging.getLogger(__name__)
METRICS_NAMESPACE = ("sdlc_pipeline", "run_metrics")


def record_run_metrics(
    pipeline_result: dict,
    issue: dict,
) -> None:
    """
    Record metrics from a completed pipeline run.
    Stored in InMemoryStore for retrieval by get_metrics_summary.
    """
    issue_number = issue.get("number", "unknown")
    timestamp = datetime.now(timezone.utc).isoformat()

    status = pipeline_result.get("pipeline_status", "unknown")
    test_results = pipeline_result.get("test_results")
    review_findings = pipeline_result.get("review_findings")
    human_decision = pipeline_result.get("human_decision", "")
    code_changes = pipeline_result.get("code_changes", [])

    # Extract test metrics safely (supports Pydantic model or dict)
    test_passed = None
    test_retry_count = None
    if test_results is not None:
        if hasattr(test_results, "final_status"):
            test_passed = test_results.final_status == "passed"
            test_retry_count = getattr(test_results, "retry_count", None)
        elif isinstance(test_results, dict):
            test_passed = test_results.get("final_status") == "passed"
            test_retry_count = test_results.get("retry_count")

    # Extract review metrics safely (supports Pydantic model or dict)
    review_recommendation = None
    blocking_issues_count = 0
    major_issues_count = 0
    if review_findings is not None:
        if hasattr(review_findings, "recommendation"):
            review_recommendation = review_findings.recommendation
            blocking_issues_count = len(getattr(review_findings, "blocking_issues", []))
            major_issues_count = len(getattr(review_findings, "major_issues", []))
        elif isinstance(review_findings, dict):
            review_recommendation = review_findings.get("recommendation")
            blocking_issues_count = len(review_findings.get("blocking_issues", []))
            major_issues_count = len(review_findings.get("major_issues", []))

    metrics = {
        "issue_number": str(issue_number),
        "timestamp": timestamp,
        "pipeline_status": status,
        "pr_opened": status in (
            "pr_opened",
            "pr_opened_memory_saved",
            "pr_opened_memory_skipped",
            "pr_opened_memory_failed",
        ),
        "human_decision": human_decision,
        "code_changes_count": len(code_changes),
        "test_passed": test_passed,
        "test_retry_count": test_retry_count,
        "review_recommendation": review_recommendation,
        "blocking_issues_count": blocking_issues_count,
        "major_issues_count": major_issues_count,
        "total_cost_usd": pipeline_result.get("total_cost_usd", 0.0),
    }

    key = f"run-{issue_number}-{timestamp[:19].replace(':', '-')}"
    store.put(METRICS_NAMESPACE, key, metrics)

    logger.info(
        "Run metrics recorded",
        extra={
            "issue_number": issue_number,
            "pr_opened": metrics["pr_opened"],
            "human_decision": human_decision,
        },
    )


def get_metrics_summary() -> dict:
    """
    Compute aggregate metrics across all recorded runs.
    Returns a summary dict with key quality indicators.

    ponytail: production evaluation uses LangSmith's evaluation
    API with golden test sets, human annotations, and
    LLM-as-a-judge scoring. This in-memory summary is
    sufficient for development monitoring.
    """
    try:
        items = store.search(METRICS_NAMESPACE)
        if not items:
            return {"total_runs": 0, "message": "No runs recorded yet"}

        all_metrics = [item.value for item in items]
        total = len(all_metrics)

        pr_opened_count = sum(
            1 for m in all_metrics if m.get("pr_opened", False)
        )
        test_passed_count = sum(
            1 for m in all_metrics
            if m.get("test_passed") is True
        )
        test_ran_count = sum(
            1 for m in all_metrics
            if m.get("test_passed") is not None
        )
        approved_count = sum(
            1 for m in all_metrics
            if m.get("human_decision") == "approve"
        )
        decisions_count = sum(
            1 for m in all_metrics
            if m.get("human_decision")
        )
        blocking_issues = [
            m.get("blocking_issues_count", 0)
            for m in all_metrics
        ]
        costs = [
            m.get("total_cost_usd", 0.0)
            for m in all_metrics
            if m.get("total_cost_usd", 0.0) > 0
        ]

        return {
            "total_runs": total,
            "pr_acceptance_rate": (
                round(pr_opened_count / total, 3) if total else 0
            ),
            "test_pass_rate": (
                round(test_passed_count / test_ran_count, 3)
                if test_ran_count else None
            ),
            "human_approval_rate": (
                round(approved_count / decisions_count, 3)
                if decisions_count else None
            ),
            "avg_blocking_issues": (
                round(
                    sum(blocking_issues) / len(blocking_issues), 2
                )
                if blocking_issues else 0
            ),
            "avg_cost_usd": (
                round(sum(costs) / len(costs), 4)
                if costs else 0.0
            ),
            "runs_with_pr": pr_opened_count,
            "runs_without_pr": total - pr_opened_count,
        }

    except Exception as e:
        logger.error(
            "Metrics summary failed",
            extra={"error": str(e)}
        )
        return {"error": str(e)}
