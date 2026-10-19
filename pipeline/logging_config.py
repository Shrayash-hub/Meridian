"""
pipeline/logging_config.py — Structured Logging Configuration

Provides configure_logging(), which sets up consistent log formatting for
the entire pipeline. Call it once at the start of any entry point:
- validation scripts (validate_*.py)
- the future FastAPI ingestion app (ingestion/webhook.py)

LangSmith and Python logging are complementary:
- LangSmith traces LLM calls, token usage, latencies, and graph edges.
- Python logging captures application-level events: node entry/exit, tool
  calls, error conditions, and business logic decisions.

Both are needed for full observability. LangSmith is enabled by environment
variables (see .env.example). Python logging is configured here.
"""

import os
import logging


def configure_logging() -> None:
    """
    Configure Python logging for the SDLC pipeline.

    Sets log level from the LOG_LEVEL environment variable (default: INFO).
    Uses a consistent format with timestamp, logger name, level, and message.
    Suppresses noisy third-party loggers to WARNING to keep output readable.

    Safe to call multiple times — logging.basicConfig() is a no-op if the
    root logger already has handlers, so this function is idempotent.

    Call once at the start of any entry point before importing pipeline modules.
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    format_string = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"

    logging.basicConfig(
        level=log_level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress third-party loggers that produce chatty output during pipeline runs.
    # These are informational for library developers but noise for application operators.
    for noisy_logger in [
        "httpx",
        "httpcore",
        "urllib3",
        "google.auth",
        "google.api_core",
        "grpc",
        "faiss",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug(
        "Logging configured",
        extra={"log_level": log_level_str},
    )
