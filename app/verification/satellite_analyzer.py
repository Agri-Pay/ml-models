"""Satellite data analyzer — wheat-specific range-based verification.

Queries `field_metrics` for NDVI, SAVI, LAI, and NDMI time-series data,
then checks whether each index falls within the agronomic range expected
for the specific wheat milestone being verified (Punjab Pakistan, Rabi season).

Replaces the previous generic NDVI-delta-only approach.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from app.db.supabase_client import get_supabase
from app.verification.wheat_thresholds import get_thresholds, score_range


async def analyze_satellite(
    farm_id: str,
    milestone_type: str,
    completion_date: Optional[str] = None,
    expected_ndvi_change: Optional[float] = None,   # legacy param, ignored for wheat
    lookback_days: int = 21,
) -> Dict[str, Any]:
    """Analyze satellite vegetation index data for wheat milestone verification.

    Args:
        farm_id:         Farm UUID.
        milestone_type:  Lowercase underscored milestone name
                         ("sowing", "tillering", "grain_filling_and_ripening").
        completion_date: ISO date string (YYYY-MM-DD) of when the farmer
                         reported this stage. Used to center the data window.
                         Falls back to `lookback_days` from now if None.
        expected_ndvi_change: Ignored (legacy). Thresholds come from
                              wheat_thresholds.py instead.
        lookback_days:  Fallback window size when no completion_date is given.

    Returns:
        dict with status, confidence, per-index values, sub_scores,
        interpretation, and raw data_points count.
    """
    try:
        supabase = get_supabase()
    except RuntimeError:
        return _no_data("Database not configured", 0)

    # ------------------------------------------------------------------
    # Determine query time window
    # ------------------------------------------------------------------
    thresholds = get_thresholds(milestone_type)
    time_window = thresholds.get("time_window", {"before_days": 7, "after_days": 14})

    if completion_date:
        try:
            base_date = datetime.fromisoformat(completion_date)
        except ValueError:
            base_date = datetime.utcnow()
        window_start = (base_date - timedelta(days=time_window["before_days"])).date().isoformat()
        window_end   = (base_date + timedelta(days=time_window["after_days"])).date().isoformat()
    else:
        # No stage date — fall back to a rolling window ending now
        now = datetime.utcnow()
        window_end   = now.date().isoformat()
        window_start = (now - timedelta(days=lookback_days)).date().isoformat()

    # ------------------------------------------------------------------
    # Query field_metrics
    # ------------------------------------------------------------------
    response = (
        supabase.table("field_metrics")
        .select("date, ndvi_mean, savi_mean, lai_mean, moisture_mean, cloud_coverage_pct")
        .eq("farm_id", farm_id)
        .gte("date", window_start)
        .lte("date", window_end)
        .order("date")
        .execute()
    )

    records = response.data if response.data else []
    if len(records) < 2:
        return _no_data(
            f"Insufficient satellite data: only {len(records)} record(s) "
            f"between {window_start} and {window_end}",
            len(records),
        )

    # Filter by cloud coverage (prefer cloud-free observations)
    clean = [r for r in records if (r.get("cloud_coverage_pct") or 100) <= 30]
    if len(clean) < 2:
        clean = records   # fall back to all records if too few cloud-free ones

    # Derive representative values
    # "current" = mean of the 3 most recent clean records (noise reduction)
    # "baseline" = mean of the first 2 records (before the milestone event)
    recent = clean[-3:] if len(clean) >= 3 else clean[-1:]
    first  = clean[:2]

    def _mean(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    ndvi_current  = _mean(recent, "ndvi_mean")
    ndvi_baseline = _mean(first,  "ndvi_mean")
    savi_current  = _mean(recent, "savi_mean")
    lai_current   = _mean(recent, "lai_mean")
    ndmi_current  = _mean(recent, "moisture_mean")

    if ndvi_current is None:
        return _no_data("NDVI values are missing from field_metrics records", len(clean))

    ndvi_change = (
        (ndvi_current - ndvi_baseline) if ndvi_baseline is not None else None
    )

    # ------------------------------------------------------------------
    # Milestone-specific scoring
    # ------------------------------------------------------------------
    sat_cfg     = thresholds.get("satellite", {})
    sub_weights = sat_cfg.get("sub_weights", {"ndvi": 1.0})

    sub_scores: Dict[str, Optional[float]] = {}

    if milestone_type == "sowing":
        sub_scores = _score_sowing(ndvi_current, savi_current, ndvi_change, sat_cfg)

    elif milestone_type == "tillering":
        sub_scores = _score_tillering(
            ndvi_current, savi_current, lai_current, ndvi_change,
            ndvi_baseline, sat_cfg
        )

    elif milestone_type == "grain_filling_and_ripening":
        sub_scores = _score_grain_filling(
            ndvi_current, savi_current, lai_current, ndmi_current,
            ndvi_change, sat_cfg
        )

    else:
        # Unknown milestone type — return low-confidence generic result
        sub_scores = {"ndvi": score_range(
            ndvi_current,
            (0.20, 0.80),
            (0.10, 0.90),
        )}

    # Compute weighted overall confidence from sub-scores
    total_weight = sum(sub_weights.get(k, 0) for k in sub_scores if sub_scores[k] is not None)
    if total_weight == 0:
        confidence = 0.0
    else:
        confidence = sum(
            (sub_scores[k] or 0) * sub_weights.get(k, 0)
            for k in sub_scores
            if sub_scores[k] is not None
        ) / total_weight

    confidence = round(min(1.0, max(0.0, confidence)), 3)

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    interpretation = _build_interpretation(
        milestone_type, confidence, ndvi_current, ndvi_change, sub_scores
    )

    return {
        "status": "ANALYZED",
        "confidence": confidence,
        "interpretation": interpretation,
        "milestone_type": milestone_type,
        "ndvi_current": round(ndvi_current, 4),
        "ndvi_baseline": round(ndvi_baseline, 4) if ndvi_baseline is not None else None,
        "ndvi_change": round(ndvi_change, 4) if ndvi_change is not None else None,
        "savi_current": round(savi_current, 4) if savi_current is not None else None,
        "lai_current":  round(lai_current,  4) if lai_current  is not None else None,
        "ndmi_current": round(ndmi_current,  4) if ndmi_current is not None else None,
        "sub_scores": {k: round(v, 3) if v is not None else None for k, v in sub_scores.items()},
        "data_points": len(clean),
        "date_range": {
            "start": clean[0]["date"],
            "end":   clean[-1]["date"],
        },
        "query_window": {"start": window_start, "end": window_end},
    }


# ---------------------------------------------------------------------------
# Per-milestone scorer functions
# ---------------------------------------------------------------------------

def _score_sowing(
    ndvi: float,
    savi: Optional[float],
    ndvi_change: Optional[float],
    cfg: Dict,
) -> Dict[str, float]:
    """Score satellite indices for the Sowing milestone.

    At sowing the field is bare soil / germinating. NDVI must be LOW.
    If NDVI is high the field has residual vegetation and the milestone
    evidence is weak.
    """
    scores: Dict[str, Optional[float]] = {}

    # NDVI absolute: bare soil range
    ndvi_opt = cfg.get("ndvi_optimal",    (0.06, 0.20))
    ndvi_acc = cfg.get("ndvi_acceptable", (0.04, 0.28))
    scores["ndvi"] = score_range(ndvi, ndvi_opt, ndvi_acc)

    # SAVI (if available)
    if savi is not None:
        savi_opt = cfg.get("savi_optimal",    (0.04, 0.16))
        savi_acc = cfg.get("savi_acceptable", (0.02, 0.22))
        scores["savi"] = score_range(savi, savi_opt, savi_acc)

    # NDVI trend: should not be rising steeply
    if ndvi_change is not None:
        max_rise = cfg.get("ndvi_trend_max_rise", 0.08)
        if ndvi_change <= max_rise:
            # Flat or slight rise — good
            scores["trend"] = 1.0 - max(0.0, ndvi_change / max(max_rise, 0.001)) * 0.4
        else:
            # Rising more than expected — suggests existing vegetation
            overshoot = ndvi_change - max_rise
            scores["trend"] = max(0.0, 0.6 - overshoot * 5)

    return {k: v for k, v in scores.items() if v is not None}


def _score_tillering(
    ndvi: float,
    savi: Optional[float],
    lai: Optional[float],
    ndvi_change: Optional[float],
    ndvi_baseline: Optional[float],
    cfg: Dict,
) -> Dict[str, float]:
    """Score satellite indices for the Tillering milestone.

    NDVI should be in the active-canopy range (0.35–0.62) and should have
    risen meaningfully from the post-sowing bare-soil baseline.
    """
    scores: Dict[str, Optional[float]] = {}

    ndvi_opt = cfg.get("ndvi_optimal",    (0.35, 0.62))
    ndvi_acc = cfg.get("ndvi_acceptable", (0.26, 0.70))
    scores["ndvi"] = score_range(ndvi, ndvi_opt, ndvi_acc)

    if savi is not None:
        savi_opt = cfg.get("savi_optimal",    (0.26, 0.50))
        savi_acc = cfg.get("savi_acceptable", (0.18, 0.58))
        scores["savi"] = score_range(savi, savi_opt, savi_acc)

    if lai is not None:
        lai_opt = cfg.get("lai_optimal",    (0.8, 3.0))
        lai_acc = cfg.get("lai_acceptable", (0.4, 3.8))
        scores["lai"] = score_range(lai, lai_opt, lai_acc)

    # NDVI rise from sowing baseline
    min_rise = cfg.get("ndvi_min_rise", 0.10)
    if ndvi_change is not None:
        if ndvi_change >= min_rise:
            scores["ndvi_rise"] = min(1.0, ndvi_change / (min_rise * 2))
        else:
            scores["ndvi_rise"] = max(0.0, ndvi_change / min_rise)

    return {k: v for k, v in scores.items() if v is not None}


def _score_grain_filling(
    ndvi: float,
    savi: Optional[float],
    lai: Optional[float],
    ndmi: Optional[float],
    ndvi_change: Optional[float],
    cfg: Dict,
) -> Dict[str, float]:
    """Score satellite indices for the Grain Filling and Ripening milestone.

    NDVI spans a wide range (peak fill → senescence) so both ends are valid.
    The key indicator is that NDVI is neither rising steeply (still vegetative
    growth) nor already at bare-soil levels (harvest already happened).
    NDMI declining toward zero confirms active desiccation during ripening.
    """
    scores: Dict[str, Optional[float]] = {}

    ndvi_opt = cfg.get("ndvi_optimal",    (0.40, 0.82))
    ndvi_acc = cfg.get("ndvi_acceptable", (0.22, 0.88))
    scores["ndvi"] = score_range(ndvi, ndvi_opt, ndvi_acc)

    if savi is not None:
        savi_opt = cfg.get("savi_optimal",    (0.32, 0.72))
        savi_acc = cfg.get("savi_acceptable", (0.18, 0.80))
        scores["savi"] = score_range(savi, savi_opt, savi_acc)

    if lai is not None:
        lai_opt = cfg.get("lai_optimal",    (2.0, 7.0))
        lai_acc = cfg.get("lai_acceptable", (1.0, 8.0))
        scores["lai"] = score_range(lai, lai_opt, lai_acc)

    if ndmi is not None:
        ndmi_opt = cfg.get("ndmi_optimal",    (-0.05, 0.50))
        ndmi_acc = cfg.get("ndmi_acceptable", (-0.20, 0.58))
        scores["ndmi"] = score_range(ndmi, ndmi_opt, ndmi_acc)

    # NDVI trend: should be flat or declining (senescence)
    if ndvi_change is not None:
        max_rise = cfg.get("ndvi_trend_max_rise", 0.10)
        min_trend = cfg.get("ndvi_trend_min", -0.50)
        if ndvi_change <= max_rise:
            if ndvi_change >= min_trend:
                # Normal declining or flat NDVI — best evidence
                scores["trend"] = 1.0 if ndvi_change <= 0 else 0.8
            else:
                # Collapsed too fast — possible crop loss or data error
                scores["trend"] = 0.3
        else:
            # NDVI still rising — probably not at grain fill yet
            overshoot = ndvi_change - max_rise
            scores["trend"] = max(0.0, 0.5 - overshoot * 4)

    return {k: v for k, v in scores.items() if v is not None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_data(reason: str, data_points: int) -> Dict[str, Any]:
    return {
        "status": "NO_DATA",
        "confidence": 0.0,
        "interpretation": reason,
        "data_points": data_points,
    }


def _build_interpretation(
    milestone_type: str,
    confidence: float,
    ndvi: float,
    ndvi_change: Optional[float],
    sub_scores: Dict[str, Optional[float]],
) -> str:
    change_str = f", change {ndvi_change:+.3f}" if ndvi_change is not None else ""
    ndvi_str = f"NDVI={ndvi:.3f}{change_str}"

    if confidence >= 0.80:
        verdict = "strongly supports"
    elif confidence >= 0.55:
        verdict = "moderately supports"
    elif confidence > 0.25:
        verdict = "weakly supports"
    else:
        verdict = "does not support"

    base = f"Satellite data {verdict} {milestone_type} milestone ({ndvi_str})."

    # Note any sub-scores that dragged confidence down
    low = [k for k, v in sub_scores.items() if v is not None and v < 0.45]
    if low:
        base += f" Low sub-scores: {', '.join(low)}."

    return base
