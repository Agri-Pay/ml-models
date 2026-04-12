"""Milestone verification endpoint — orchestrates multi-source verification."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from app.verification.orchestrator import verify_milestone as run_verification
from app.api.v1.oracle import release_payment, ReleaseRequest

logger = logging.getLogger(__name__)
router = APIRouter()


class VerifyMilestoneRequest(BaseModel):
    milestone_id: str


@router.post("/verify-milestone")
async def verify_milestone(request: VerifyMilestoneRequest):
    """Trigger multi-source verification for a milestone.

    Orchestrates satellite + drone + IoT analysis and writes
    the composite verdict to the database.

    If the verdict is PASS, automatically triggers the oracle
    to release the escrowed payment on-chain.
    """
    try:
        result = await run_verification(request.milestone_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}",
        )

    if result.get("verdict") == "ERROR":
        raise HTTPException(
            status_code=404,
            detail=result.get("error", "Unknown error"),
        )

    # ── Auto-trigger oracle release on PASS ──────────────────────────────────
    oracle_result = None
    if result.get("verdict") == "PASS":
        try:
            oracle_result = await release_payment(
                ReleaseRequest(milestone_id=request.milestone_id)
            )
            logger.info(
                "Oracle auto-release for %s: tx=%s",
                request.milestone_id,
                oracle_result.tx_hash,
            )
        except Exception as oracle_err:
            logger.error(
                "Oracle auto-release failed for %s: %s",
                request.milestone_id,
                oracle_err,
            )
            # Non-fatal — verification still succeeded

    return {
        "status": "verified",
        "verdict": result.get("verdict"),
        "overall_confidence": result.get("overall_confidence"),
        "recommendation": result.get("recommendation"),
        "report": result,
        "payment_release": {
            "triggered": oracle_result is not None,
            "tx_hash": oracle_result.tx_hash if oracle_result else None,
        },
    }
