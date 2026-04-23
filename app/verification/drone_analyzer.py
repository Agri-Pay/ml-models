"""Drone imagery analyzer — spectral band statistics for wheat verification.

For wheat crops (no plant-counting model available), verification uses the
statistical summary stored in `drone_imagery_layers.statistics` (populated at
upload time by TiTiler). This avoids running any ML model while still extracting
a meaningful greenness signal from the drone orthomosaic.

No plant counting is performed. The previous rice_plant_counter_v1 dependency
is removed entirely for wheat.
"""

from typing import Any, Dict, Optional

from app.db.supabase_client import get_supabase
from app.verification.wheat_thresholds import get_thresholds, score_range


# Layer type preference order: NDVI layer is most informative, fall back to
# multispectral, then RGB.
_LAYER_PREFERENCE = ["ndvi", "multispectral", "rgb"]


async def analyze_drone(
    farm_id: str,
    milestone_id: str,
    milestone_type: str,
    crop_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze drone imagery for wheat milestone verification.

    Uses pre-computed band statistics stored in drone_imagery_layers.statistics
    (a JSONB column populated when each GeoTIFF is uploaded). No ML model is run.

    Args:
        farm_id:        Farm UUID.
        milestone_id:   Milestone UUID (kept for API compatibility).
        milestone_type: Lowercase underscored milestone name.
        crop_name:      Crop name (kept for API compatibility, not used here).

    Returns:
        dict with status, confidence, ndvi_mean (if available), flight_date,
        layer_type, and interpretation.
    """
    try:
        supabase = get_supabase()
    except RuntimeError:
        return _no_data("Database not configured")

    # ------------------------------------------------------------------
    # Find the most recent drone flight for this farm
    # ------------------------------------------------------------------
    flights_resp = (
        supabase.table("drone_flights")
        .select("id, flight_date")
        .eq("farm_id", farm_id)
        .order("flight_date", desc=True)
        .limit(1)
        .execute()
    )

    if not flights_resp.data:
        return _no_data("No drone flights found for this farm")

    flight = flights_resp.data[0]
    flight_id   = flight["id"]
    flight_date = flight["flight_date"]

    # ------------------------------------------------------------------
    # Find the best available imagery layer for this flight
    # Priority: ndvi → multispectral → rgb
    # ------------------------------------------------------------------
    layers_resp = (
        supabase.table("drone_imagery_layers")
        .select("id, layer_type, statistics, filename")
        .eq("flight_id", flight_id)
        .execute()
    )

    layers = layers_resp.data if layers_resp.data else []
    if not layers:
        return _no_data(
            f"No imagery layers found for the latest drone flight ({flight_date})"
        )

    # Build a dict: layer_type → layer record
    layer_by_type: Dict[str, Dict] = {}
    for layer in layers:
        lt = (layer.get("layer_type") or "").lower()
        if lt not in layer_by_type:
            layer_by_type[lt] = layer

    # Pick the best available layer type
    chosen_layer = None
    chosen_type  = None
    for preferred in _LAYER_PREFERENCE:
        if preferred in layer_by_type:
            chosen_layer = layer_by_type[preferred]
            chosen_type  = preferred
            break

    if chosen_layer is None:
        # Fall back to any layer that has statistics
        for layer in layers:
            if layer.get("statistics"):
                chosen_layer = layer
                chosen_type  = (layer.get("layer_type") or "unknown").lower()
                break

    if chosen_layer is None:
        return _no_data(
            f"Drone imagery layers have no statistics JSONB for flight {flight_date}"
        )

    # ------------------------------------------------------------------
    # Extract NDVI (or greenness proxy) from statistics
    # ------------------------------------------------------------------
    statistics = chosen_layer.get("statistics") or {}

    ndvi_mean = _extract_ndvi_mean(statistics, chosen_type)

    if ndvi_mean is None:
        return _no_data(
            f"Could not extract NDVI from {chosen_type} layer statistics "
            f"(flight {flight_date}). Statistics field: {list(statistics.keys())}"
        )

    # ------------------------------------------------------------------
    # Score against wheat milestone thresholds
    # ------------------------------------------------------------------
    thresholds  = get_thresholds(milestone_type)
    drone_cfg   = thresholds.get("drone", {})

    ndvi_opt = tuple(drone_cfg.get("ndvi_layer_optimal",    (0.10, 0.80)))
    ndvi_acc = tuple(drone_cfg.get("ndvi_layer_acceptable", (0.02, 0.90)))

    confidence = round(score_range(ndvi_mean, ndvi_opt, ndvi_acc), 3)

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    if confidence >= 0.80:
        verdict = "strongly supports"
    elif confidence >= 0.55:
        verdict = "moderately supports"
    elif confidence > 0.25:
        verdict = "weakly supports"
    else:
        verdict = "does not support"

    interpretation = (
        f"Drone {chosen_type} layer {verdict} {milestone_type} milestone. "
        f"NDVI mean={ndvi_mean:.3f} "
        f"(expected {ndvi_opt[0]:.2f}–{ndvi_opt[1]:.2f}). "
        f"Flight date: {flight_date}."
    )

    return {
        "status": "ANALYZED",
        "confidence": confidence,
        "interpretation": interpretation,
        "ndvi_mean": round(ndvi_mean, 4),
        "layer_type": chosen_type,
        "flight_date": flight_date,
        "filename": chosen_layer.get("filename", ""),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_ndvi_mean(statistics: Dict, layer_type: str) -> Optional[float]:
    """Extract a NDVI mean value from the statistics JSONB.

    The structure of statistics depends on how TiTiler populated it at upload:
    - Single-band layers: {"mean": 0.45, "min": ..., "max": ..., "std": ...}
    - Multi-band layers:  {"1": {"mean": ...}, "2": {"mean": ...}, ...}
                          or {"R": {"mean": ...}, "G": {"mean": ...}, ...}
    - NDVI layer with a named band: {"ndvi": {"mean": ...}}
    """
    if not statistics:
        return None

    # Direct mean at top level (single-band NDVI layer)
    if "mean" in statistics:
        val = statistics["mean"]
        if val is not None:
            return float(val)

    # Named key "ndvi" (some TiTiler outputs)
    if "ndvi" in statistics:
        sub = statistics["ndvi"]
        if isinstance(sub, dict) and sub.get("mean") is not None:
            return float(sub["mean"])
        if isinstance(sub, (int, float)):
            return float(sub)

    # Multi-band: if it's an NDVI layer, band 1 is the NDVI channel
    if layer_type in ("ndvi", "multispectral"):
        for key in ("1", "band1", "Band1", "b1"):
            if key in statistics:
                sub = statistics[key]
                if isinstance(sub, dict) and sub.get("mean") is not None:
                    return float(sub["mean"])

    # RGB layer fallback: compute crude NDVI proxy from red/green channels
    # (R reflectance high → low NDVI, G high → healthy canopy indicator)
    # This is a rough proxy only — used when no better layer exists.
    if layer_type == "rgb":
        r_mean = _band_mean(statistics, ("R", "red",   "1", "Band1"))
        g_mean = _band_mean(statistics, ("G", "green", "2", "Band2"))
        if r_mean is not None and g_mean is not None and (r_mean + g_mean) > 0:
            # Simple green ratio index as NDVI proxy
            return float((g_mean - r_mean) / (g_mean + r_mean))

    return None


def _band_mean(statistics: Dict, keys) -> Optional[float]:
    """Try multiple band name variants to find a mean value."""
    for key in keys:
        sub = statistics.get(key)
        if isinstance(sub, dict) and sub.get("mean") is not None:
            return float(sub["mean"])
        if isinstance(sub, (int, float)):
            return float(sub)
    return None


def _no_data(reason: str) -> Dict[str, Any]:
    return {
        "status": "NO_DATA",
        "confidence": 0.0,
        "interpretation": reason,
    }
