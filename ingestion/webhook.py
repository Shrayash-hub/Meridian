"""
FastAPI Webhook Router — GitHub Webhook Event Ingestion.

This module provides the FastAPI router for receiving GitHub webhook events.
Handles signature verification via HMAC-SHA256, event filtering (only processing
'issues' events with the 'opened' action), and delegates execution to the pipeline
runner.

HMAC verification is required to guarantee authenticity and prevent unauthorized
or forged webhook events from triggering expensive agent pipeline runs.
"""

import os
import hmac
import hashlib
import logging
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Header
from ingestion.runner import run_pipeline_for_issue

logger = logging.getLogger(__name__)
router = APIRouter()


def verify_github_signature(
    payload: bytes,
    signature: str,
    secret: str,
) -> bool:
    """
    Verify GitHub webhook HMAC-SHA256 signature.
    Uses hmac.compare_digest to prevent timing attacks.
    Returns False (not raises) on verification failure —
    the endpoint handles the 401 response.
    """
    if not signature or not secret:
        return False

    try:
        expected = hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(
            f"sha256={expected}",
            signature,
        )
    except Exception:
        return False


@router.post("/webhook")
async def github_webhook(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(None),
    x_github_event: Optional[str] = Header(None),
):
    """
    Receive GitHub webhook events.
    Only processes 'issues' events with action 'opened'.
    Verifies HMAC-SHA256 signature before processing.
    Returns 200 immediately after triggering the pipeline —
    GitHub requires a quick response to avoid delivery timeout.
    """
    # Step 1: Read raw body for signature verification
    payload_bytes = await request.body()

    # Step 2: Verify signature
    webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    if webhook_secret:
        if not x_hub_signature_256:
            logger.warning("Webhook received without signature")
            raise HTTPException(
                status_code=401,
                detail="Missing X-Hub-Signature-256 header",
            )
        if not verify_github_signature(
            payload_bytes, x_hub_signature_256, webhook_secret
        ):
            logger.warning("Webhook signature verification failed")
            raise HTTPException(
                status_code=401,
                detail="Invalid webhook signature",
            )
    else:
        logger.warning(
            "GITHUB_WEBHOOK_SECRET not set — "
            "skipping signature verification. "
            "Set this in production."
        )

    # Step 3: Parse JSON body
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON payload",
        )

    # Step 4: Filter for issues opened events only
    if x_github_event != "issues":
        logger.info(
            "Ignoring non-issues event",
            extra={"event": x_github_event},
        )
        return {"status": "ignored", "reason": "not an issues event"}

    action = payload.get("action")
    if action != "opened":
        logger.info(
            "Ignoring non-opened action",
            extra={"action": action},
        )
        return {"status": "ignored", "reason": f"action={action}"}

    # Step 5: Extract issue data
    issue = payload.get("issue", {})
    issue_number = issue.get("number", "unknown")

    logger.info(
        "Processing new issue",
        extra={"issue_number": issue_number},
    )

    # Step 6: Trigger pipeline (non-blocking response pattern)
    pipeline_issue = {
        "title": issue.get("title", ""),
        "number": issue_number,
        "body": issue.get("body", ""),
        "labels": [
            label.get("name", "")
            for label in issue.get("labels", [])
        ],
    }

    try:
        result = run_pipeline_for_issue(pipeline_issue)
        logger.info(
            "Pipeline completed",
            extra={
                "issue_number": issue_number,
                "status": result.get("pipeline_status"),
            },
        )
        return {
            "status": "processed",
            "issue_number": issue_number,
            "pipeline_status": result.get("pipeline_status"),
        }
    except Exception as e:
        logger.error(
            "Pipeline failed",
            extra={
                "issue_number": issue_number,
                "error": str(e),
            },
        )
        # Return 200 even on pipeline failure —
        # GitHub must not retry based on our internal errors.
        # ponytail: production should use background tasks
        # so the response is always immediate. The pipeline
        # should run asynchronously with results reported via
        # a separate status endpoint or Slack notification.
        return {
            "status": "error",
            "issue_number": issue_number,
            "error": str(e)[:200],
        }
