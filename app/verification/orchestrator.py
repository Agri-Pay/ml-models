"""Milestone verification orchestrator.

Coordinates multi-source analysis (satellite + IoT + farmer input + drone)
and writes the composite verdict to the database.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from app.db.supabase_client import get_supabase
from app.verification.satellite_analyzer import analyze_satellite
from app.verification.drone_analyzer import analyze_drone
from app.verification.iot_analyzer import analyze_iot
from app.verification.farmer_input_analyzer import analyze_farmer_input
from app.verification.scorer import compute_verdict


async def verify_milestone(milestone_id: str) -> Dict[str, Any]:
    """Run full multi-source verification for a wheat milestone.

    Steps:
    1. Fetch milestone context (farm_id, crop_cycle_id, milestone_type)
    2. Resolve the completion date from growth_stage_inputs or actual_completion_date
    3. Run satellite, IoT, farmer-input, and drone analysis in parallel
    4. Compute weighted verdict
    5. Write results to DB (agro_augmentation + verifications + verification_evidence)
    6. Return verdict

    Returns:
        The complete verification result dict.
    """
    supabase = get_supabase()

    # ------------------------------------------------------------------
    # 1. Fetch milestone details
    # ------------------------------------------------------------------
    milestone_response = (
        supabase.table("cycle_milestones")
        .select(
            "id, crop_cycle_id, template_id, status, actual_completion_date, "
            "crop_cycles(id, farm_id, crop_id, crops(name)), "
            "milestone_templates(name, expected_ndvi_change, verification_type)"
        )
        .eq("id", milestone_id)
        .single()
        .execute()
    )

    if not milestone_response.data:
        return {
            "verdict": "ERROR",
            "error": f"Milestone {milestone_id} not found",
        }

    milestone    = milestone_response.data
    cycle        = milestone.get("crop_cycles") or {}
    farm_id      = cycle.get("farm_id")
    crop_cycle_id = cycle.get("id")
    crop_info    = cycle.get("crops") or {}
    crop_name    = crop_info.get("name", "").lower() if crop_info else None
    template     = milestone.get("milestone_templates") or {}

    # Normalise milestone_type to match keys in wheat_thresholds.py:
    # "Grain Filling and Ripening" → "grain_filling_and_ripening"
    milestone_type = template.get("name", "unknown").lower().replace(" ", "_")

    if not farm_id:
        return {
            "verdict": "ERROR",
            "error": "Could not determine farm_id for this milestone",
        }

    # ------------------------------------------------------------------
    # 2. Resolve completion date
    #    Priority: growth_stage_inputs.stage_date for the primary required
    #    stage → actual_completion_date on cycle_milestones → None (fallback
    #    to rolling window in each analyzer).
    # ------------------------------------------------------------------
    completion_date: Optional[str] = _resolve_completion_date(
        supabase, crop_cycle_id, milestone_type,
        milestone.get("actual_completion_date")
    )

    # ------------------------------------------------------------------
    # 3. Run all analyzers in parallel
    # ------------------------------------------------------------------
    satellite_task = analyze_satellite(
        farm_id=farm_id,
        milestone_type=milestone_type,
        completion_date=completion_date,
    )
    drone_task = analyze_drone(
        farm_id=farm_id,
        milestone_id=milestone_id,
        milestone_type=milestone_type,
        crop_name=crop_name,
    )
    iot_task = analyze_iot(
        farm_id=farm_id,
        milestone_type=milestone_type,
        completion_date=completion_date,
    )
    farmer_task = analyze_farmer_input(
        crop_cycle_id=crop_cycle_id,
        milestone_type=milestone_type,
    )

    satellite_result, drone_result, iot_result, farmer_result = await asyncio.gather(
        satellite_task, drone_task, iot_task, farmer_task
    )

    # ------------------------------------------------------------------
    # 4. Compute weighted verdict
    # ------------------------------------------------------------------
    verdict_result = compute_verdict(
        satellite_result=satellite_result,
        drone_result=drone_result,
        iot_result=iot_result,
        farmer_input_result=farmer_result,
    )
    verdict_result["processed_at"] = datetime.utcnow().isoformat()
    verdict_result["milestone_type"] = milestone_type
    verdict_result["completion_date_used"] = completion_date

    # ------------------------------------------------------------------
    # 5. Write to database
    # ------------------------------------------------------------------
    try:
        # Update milestone with full verification result
        supabase.table("cycle_milestones").update({
            "agro_augmentation": verdict_result,
            "status": "pending_verification",
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("id", milestone_id).execute()

        # Create verification audit record
        verification_insert = supabase.table("verifications").insert({
            "milestone_id": milestone_id,
            "source": "system",
            "evidence_data": verdict_result,
            "ai_confidence": verdict_result["overall_confidence"],
            "ai_analysis": {
                "satellite":    satellite_result,
                "drone":        drone_result,
                "iot":          iot_result,
                "farmer_input": farmer_result,
            },
            "status": _verdict_to_verification_status(verdict_result["verdict"]),
        }).execute()

        # Create verification_evidence rows for each source that had data
        verification_id = None
        if verification_insert.data:
            verification_id = verification_insert.data[0].get("id")

        if verification_id:
            evidence_rows = []
            for source_name, source_result in [
                ("satellite",    satellite_result),
                ("drone_output", drone_result),
                ("iot_reading",  iot_result),
                # farmer_result is stored in verifications.ai_analysis.farmer_input;
                # no 'farmer_input' evidence_type exists in the DB CHECK constraint.
            ]:
                if (source_result or {}).get("status") == "ANALYZED":
                    evidence_rows.append({
                        "verification_id":  verification_id,
                        "evidence_type":    source_name,
                        "evidence_source_id": milestone_id,
                        "captured_at":      datetime.utcnow().isoformat(),
                        "relevance_score":  source_result.get("confidence", 0),
                        "notes":            (source_result.get("interpretation") or "")[:500],
                    })
            if evidence_rows:
                supabase.table("verification_evidence").insert(evidence_rows).execute()

    except Exception as e:
        verdict_result["db_write_error"] = str(e)

    return verdict_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_completion_date(
    supabase,
    crop_cycle_id: Optional[str],
    milestone_type: str,
    actual_completion_date: Optional[str],
) -> Optional[str]:
    """Determine the best date to use as the center of the data window.

    1. Look up the primary growth stage date from growth_stage_inputs.
    2. Fall back to actual_completion_date on the cycle_milestones row.
    3. Return None if nothing is available (analyzers will use a rolling window).
    """
    if not crop_cycle_id:
        return _to_date_str(actual_completion_date)

    # Determine primary stage key from milestone type
    primary_stage_map = {
        "sowing":                     "sowing",
        "tillering":                  "tillering",
        "grain_filling_and_ripening": "grain_filling",
    }
    primary_stage = primary_stage_map.get(milestone_type)

    if primary_stage:
        try:
            resp = (
                supabase.table("growth_stage_inputs")
                .select("stage_date")
                .eq("crop_cycle_id", crop_cycle_id)
                .eq("stage_name", primary_stage)
                .order("updated_at", desc=True)
                .limit(1)
                .execute()
            )
            if resp.data and resp.data[0].get("stage_date"):
                return str(resp.data[0]["stage_date"])
        except Exception:
            pass   # Fall through to actual_completion_date

    return _to_date_str(actual_completion_date)


def _to_date_str(ts: Optional[str]) -> Optional[str]:
    """Convert a full ISO timestamp to a date string, or return None."""
    if not ts:
        return None
    return ts[:10]   # "2025-11-01T..." → "2025-11-01"


def _verdict_to_verification_status(verdict: str) -> str:
    """Map internal verdict string to the verifications.status enum value."""
    mapping = {
        "MILESTONE_COMPLETE":     "approved",
        "MANUAL_REVIEW_REQUIRED": "manual_review_required",
        "MILESTONE_FAILED":       "rejected",
        "INSUFFICIENT_EVIDENCE":  "manual_review_required",
    }
    return mapping.get(verdict, "processing")
