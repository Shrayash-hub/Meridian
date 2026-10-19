"""
SDLC Pipeline — Application Entry Point.

Starts the FastAPI application with the GitHub webhook
endpoint. Run with: python main.py
Or with uvicorn directly: uvicorn main:app --reload
"""

from dotenv import load_dotenv
load_dotenv()

import os
import uvicorn
from fastapi import FastAPI
from pipeline.logging_config import configure_logging
from ingestion.webhook import router as webhook_router

configure_logging()

app = FastAPI(
    title="Autonomous SDLC Pipeline",
    description=(
        "Multi-agent LangGraph pipeline that processes GitHub "
        "issues and opens Pull Requests autonomously."
    ),
    version="1.0.0",
)

app.include_router(webhook_router)


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/metrics")
def get_metrics():
    """Return aggregate pipeline quality metrics."""
    from evaluation.evaluator import get_metrics_summary
    return get_metrics_summary()


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
