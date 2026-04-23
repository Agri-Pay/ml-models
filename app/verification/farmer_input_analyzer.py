"""Farmer input analyzer — verifies milestone completion via growth_stage_inputs.

Queries the `growth_stage_inputs` table (populated by the farmer in
CropStageSection.jsx) and scores confidence based on whether the required
growth stages have been logged with dates and expected agronomic activities.

This is the most direct evidence source: the farmer explicitly recording
that they performed each stage.
"""

from typing import Any, Dict, List, Optional

from app.db.supabase_client import get_supabase
from app.verification.wheat_thresholds import get_thresholds


# Human-readable labels for stage keys (for interpretation messages)
STAGE_LABELS: Dict[str, str] = {
    "sowing":        "Sowing",
    "emergence":     "Emergence",
    "tillering":     "Tillering",
    "jointing":      "Jointing",
    "booting":       "Booting",
    "heading":       "Heading",
    "anthesis":      "Anthesis",
    "grain_filling": "Grain Filling",
    "harvest":       "Harvest",
}


async def analyze_farmer_input(
    crop_cycle_id: str,
    milestone_type: str,
) -> Dict[str, Any]:
    """Score farmer-reported growth stage data for milestone verification.

    Checks whether the farmer has filled in the growth stages that are
    prerequisite to this milestone, and whether the dates and activities
    logged are agronomically plausible.

    Args:
        crop_cycle_id: UUID of the active crop cycle.
        milestone_type: Lowercased, underscored milestone name
                        (e.g. "sowing", "tillering", "grain_filling_and_ripening").

    Returns:
        dict with status, confidence, stages_found, missing_stages, activity_flags,
        das_reported, and interpretation.
    """
    try:
        supabase = get_supabase()
    except RuntimeError:
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": "Database not configured",
            "stages_found": [],
            "missing_stages": [],
        }

    thresholds = get_thresholds(milestone_type)
    if not thresholds:
        # Milestone type not in wheat thresholds — fall back gracefully
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": f"No farmer input thresholds defined for milestone type '{milestone_type}'",
            "stages_found": [],
            "missing_stages": [],
        }

    farmer_cfg = thresholds.get("farmer_input", {})
    required_stages: List[str] = farmer_cfg.get("required_stages", [])
    expected_das_min, expected_das_max = farmer_cfg.get("expected_das_range", (0, 200))
    positive_activities: List[str] = farmer_cfg.get("positive_activities", [])
    fertilizer_check: bool = farmer_cfg.get("fertilizer_check", False)

    if not required_stages:
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": "No required stages configured for this milestone",
            "stages_found": [],
            "missing_stages": [],
        }

    # ------------------------------------------------------------------
    # Query growth_stage_inputs for all required stages in this cycle
    # ------------------------------------------------------------------
    response = (
        supabase.table("growth_stage_inputs")
        .select(
            "stage_name, stage_date, das, irrigation, fertilizer_per_acre, "
            "herbicide, pesticide, fungicide, biologicals, notes"
        )
        .eq("crop_cycle_id", crop_cycle_id)
        .in_("stage_name", required_stages)
        .execute()
    )

    rows = response.data if response.data else []

    # Build a dict keyed by stage_name for quick lookup
    stage_map: Dict[str, Dict] = {r["stage_name"]: r for r in rows}

    stages_found = [s for s in required_stages if s in stage_map and stage_map[s].get("stage_date")]
    missing_stages = [s for s in required_stages if s not in stages_found]

    # No data at all — farmer hasn't logged anything relevant
    if not stages_found:
        missing_labels = [STAGE_LABELS.get(s, s) for s in missing_stages]
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": (
                f"Farmer has not filled in required growth stages: "
                f"{', '.join(missing_labels)}. Stage input log is empty for this milestone."
            ),
            "stages_found": stages_found,
            "missing_stages": missing_stages,
            "activity_flags": {},
            "das_reported": None,
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    score = 0.0

    # 1. Stage completeness (weight: 0.65 total)
    #    Each required stage with a date is worth an equal share of 0.65.
    stage_weight = 0.65 / len(required_stages)
    for stage in required_stages:
        if stage in stages_found:
            score += stage_weight

    # 2. Primary stage timing validation (weight: 0.15)
    #    The primary stage is the LAST in required_stages (the one directly
    #    named for this milestone, e.g. "tillering" for the Tillering milestone).
    das_reported: Optional[int] = None
    primary_stage = required_stages[-1]
    if primary_stage in stage_map:
        das_value = stage_map[primary_stage].get("das")
        if das_value is not None:
            das_reported = int(das_value)
            if expected_das_min <= das_reported <= expected_das_max:
                score += 0.15   # DAS is within the expected agronomic range
            else:
                # Partial credit if close but outside range
                overshoot = max(
                    das_reported - expected_das_max,
                    expected_das_min - das_reported,
                    0,
                )
                score += max(0.0, 0.08 * (1.0 - overshoot / 20.0))

    # 3. Agronomic activities (weight: 0.20 total)
    #    Presence of expected activities gives evidence the farmer actually
    #    performed the corresponding field operation.
    activity_flags: Dict[str, bool] = {}
    activity_score = 0.0

    if positive_activities:
        filled_activities = 0
        for activity in positive_activities:
            # Check ANY of the found stage rows for this activity flag
            flag = False
            for stage_row in stage_map.values():
                val = stage_row.get(activity)
                if isinstance(val, bool) and val:
                    flag = True
                    break
            activity_flags[activity] = flag
            if flag:
                filled_activities += 1
        # Give proportional credit for activities found
        if positive_activities:
            activity_score += 0.15 * (filled_activities / len(positive_activities))

    if fertilizer_check:
        has_fertilizer = any(
            (row.get("fertilizer_per_acre") or 0) > 0
            for row in stage_map.values()
        )
        activity_flags["fertilizer"] = has_fertilizer
        if has_fertilizer:
            activity_score += 0.05

    score += activity_score

    # Clamp to [0.0, 1.0]
    confidence = round(min(1.0, max(0.0, score)), 3)

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    found_labels = [STAGE_LABELS.get(s, s) for s in stages_found]
    missing_labels = [STAGE_LABELS.get(s, s) for s in missing_stages]
    activity_summary = ", ".join(
        f"{k}: {'✓' if v else '✗'}" for k, v in activity_flags.items()
    )

    if confidence >= 0.80:
        interpretation = (
            f"Farmer input strongly supports milestone. "
            f"Stages logged: {', '.join(found_labels)}. "
        )
    elif confidence >= 0.50:
        interpretation = (
            f"Moderate farmer input evidence. "
            f"Stages logged: {', '.join(found_labels)}. "
        )
    else:
        interpretation = (
            f"Incomplete farmer input. "
        )

    if missing_labels:
        interpretation += f"Missing stages: {', '.join(missing_labels)}. "
    if activity_summary:
        interpretation += f"Activities: {activity_summary}."
    if das_reported is not None:
        interpretation += f" DAS reported: {das_reported} (expected {expected_das_min}–{expected_das_max})."

    return {
        "status": "ANALYZED",
        "confidence": confidence,
        "interpretation": interpretation,
        "stages_found": stages_found,
        "missing_stages": missing_stages,
        "stages_required": required_stages,
        "activity_flags": activity_flags,
        "das_reported": das_reported,
        "expected_das_range": [expected_das_min, expected_das_max],
    }
