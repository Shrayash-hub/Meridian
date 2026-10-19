"""
pipeline/config.py — Pipeline Invocation Config Helper

Provides build_pipeline_config(), the single place that constructs the
LangGraph config dict for every pipeline.invoke() call.

Centralising the config dict means:
- Every run carries consistent run_name, metadata, and tags in LangSmith.
- Adding a new metadata field (e.g. "triggered_by") requires one change here,
  not one change per call site.
- Validation scripts and the future FastAPI webhook layer all use the same
  structure.

LangSmith tracing is enabled by setting two environment variables in .env:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=<your key from smith.langchain.com>

No code changes are needed in node files or in pipeline/models.py — LangChain
detects these variables automatically and traces every LLM call.
See .env.example for full setup instructions.
"""

import os
from datetime import datetime, timezone


def build_pipeline_config(
    thread_id: str,
    issue_number: int | str = "unknown",
    issue_title: str = "",
    extra_metadata: dict = None,
) -> dict:
    """
    Build the LangGraph config dict for a pipeline invocation.

    Includes:
    - configurable.thread_id: required by MemorySaver checkpointer
    - run_name: human-readable label in LangSmith UI
    - metadata: structured fields for filtering and searching traces
    - tags: categorical labels for grouping traces in LangSmith

    Usage:
        config = build_pipeline_config(
            thread_id="issue-42-abc123",
            issue_number=42,
            issue_title="Add email validation",
        )
        pipeline.invoke(initial_state, config=config)

    Args:
        thread_id: Unique identifier for this pipeline run. Must be the same
            across all invoke() calls for the same run (required by the
            MemorySaver checkpointer for interrupt/resume support).
        issue_number: GitHub issue number. Included in run_name and tags so
            traces are searchable by issue in LangSmith.
        issue_title: GitHub issue title. Truncated to 80 chars in metadata
            to avoid token bloat in trace payloads.
        extra_metadata: Optional dict merged into the metadata field. Use for
            test scenario labels or any run-specific context.

    Returns:
        LangGraph config dict ready to pass as `config=` to pipeline.invoke().
    """
    pipeline_version = os.getenv("PIPELINE_VERSION", "1.0")
    environment = os.getenv("ENVIRONMENT", "development")
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    metadata = {
        "thread_id": thread_id,
        "issue_number": str(issue_number),
        "issue_title": issue_title[:80] if issue_title else "",
        "pipeline_version": pipeline_version,
        "environment": environment,
        "run_timestamp": run_timestamp,
    }

    if extra_metadata and isinstance(extra_metadata, dict):
        metadata.update(extra_metadata)

    return {
        "configurable": {
            "thread_id": thread_id,
        },
        "run_name": f"SDLC Pipeline - Issue #{issue_number}",
        "metadata": metadata,
        "tags": [
            "sdlc-pipeline",
            f"issue-{issue_number}",
            environment,
            f"v{pipeline_version}",
        ],
    }
