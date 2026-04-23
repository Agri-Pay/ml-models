"""IoT sensor analyzer — per-sensor-type wheat milestone verification.

Queries `iot_readings` grouped by sensor type and checks whether each sensor's
readings fall within the agronomically expected range for the milestone being
verified (Punjab Pakistan, Rabi season).

Key fix over the previous version: readings from all sensor types are NO LONGER
pooled into a single mixed array. Soil temperature and soil moisture — which
have completely different units and scales — are now analyzed separately.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from app.db.supabase_client import get_supabase
from app.verification.wheat_thresholds import get_thresholds, score_range


async def analyze_iot(
    farm_id: str,
    milestone_type: str,
    completion_date: Optional[str] = None,
    lookback_days: int = 21,
) -> Dict[str, Any]:
    """Analyze IoT sensor data for wheat milestone verification.

    Args:
        farm_id:         Farm UUID.
        milestone_type:  Lowercase underscored milestone name.
        completion_date: ISO date string used to center the data window.
        lookback_days:   Fallback window when completion_date is None.

    Returns:
        dict with status, confidence, per-sensor-type scores and means,
        readings_per_sensor, and interpretation.
    """
    try:
        supabase = get_supabase()
    except RuntimeError:
        return _no_data("Database not configured", {})

    # ------------------------------------------------------------------
    # Time window
    # ------------------------------------------------------------------
    thresholds = get_thresholds(milestone_type)
    time_window = thresholds.get("time_window", {"before_days": 7, "after_days": 14})

    if completion_date:
        try:
            base = datetime.fromisoformat(completion_date)
        except ValueError:
            base = datetime.utcnow()
        window_start = (base - timedelta(days=time_window["before_days"])).isoformat()
        window_end   = (base + timedelta(days=time_window["after_days"])).isoformat()
    else:
        now = datetime.utcnow()
        window_end   = now.isoformat()
        window_start = (now - timedelta(days=lookback_days)).isoformat()

    # ------------------------------------------------------------------
    # Fetch active devices WITH their sensor type name
    # Join to iot_sensor_types via sensor_type_id
    # ------------------------------------------------------------------
    devices_resp = (
        supabase.table("iot_devices")
        .select("id, sensor_type_id, iot_sensor_types(name, unit)")
        .eq("farm_id", farm_id)
        .eq("is_active", True)
        .execute()
    )

    devices = devices_resp.data if devices_resp.data else []
    if not devices:
        return _no_data("No active IoT devices found for this farm", {})

    # Build map: device_id → sensor_type_name
    device_sensor_type: Dict[str, str] = {}
    for device in devices:
        sensor_type_info = device.get("iot_sensor_types") or {}
        sensor_name = sensor_type_info.get("name") if isinstance(sensor_type_info, dict) else None
        if sensor_name:
            device_sensor_type[device["id"]] = sensor_name

    # ------------------------------------------------------------------
    # Fetch readings for all devices in the time window
    # ------------------------------------------------------------------
    all_readings: List[Dict] = []
    for device in devices:
        resp = (
            supabase.table("iot_readings")
            .select("device_id, captured_at, value, quality_flag")
            .eq("device_id", device["id"])
            .gte("captured_at", window_start)
            .lte("captured_at", window_end)
            .eq("quality_flag", "normal")
            .order("captured_at")
            .execute()
        )
        if resp.data:
            all_readings.extend(resp.data)

    if not all_readings:
        return _no_data(
            f"No IoT readings in the window {window_start[:10]} to {window_end[:10]}",
            {}
        )

    # ------------------------------------------------------------------
    # Group readings by sensor type
    # ------------------------------------------------------------------
    sensor_values: Dict[str, List[float]] = defaultdict(list)
    for reading in all_readings:
        device_id = reading.get("device_id")
        sensor_type = device_sensor_type.get(device_id)
        if sensor_type and reading.get("value") is not None:
            sensor_values[sensor_type].append(float(reading["value"]))

    # ------------------------------------------------------------------
    # Score each sensor type against wheat thresholds
    # ------------------------------------------------------------------
    iot_cfg = thresholds.get("iot", {})
    sensor_scores: Dict[str, float] = {}
    sensor_means: Dict[str, float] = {}
    readings_per_sensor: Dict[str, int] = {}
    skipped_sensors: List[str] = []

    for sensor_type, values in sensor_values.items():
        readings_per_sensor[sensor_type] = len(values)

        if len(values) < 3:
            # Too few readings for reliable statistics
            skipped_sensors.append(sensor_type)
            continue

        mean_val = float(np.mean(values))
        sensor_means[sensor_type] = round(mean_val, 3)

        # Check if we have threshold config for this sensor type
        if sensor_type in iot_cfg:
            sensor_thresholds = iot_cfg[sensor_type]
            optimal    = sensor_thresholds.get("optimal")
            acceptable = sensor_thresholds.get("acceptable")

            if optimal and acceptable:
                sensor_scores[sensor_type] = round(
                    score_range(mean_val, tuple(optimal), tuple(acceptable)), 3
                )
        # Sensors not in the milestone's iot config are ignored (no penalty)

    if not sensor_scores:
        reason = (
            "No scoreable IoT sensor types found for this milestone "
            f"(devices report: {list(sensor_values.keys()) or 'none'})"
        )
        return _no_data(reason, readings_per_sensor)

    # ------------------------------------------------------------------
    # Aggregate: equal-weight average of available scored sensor types
    # ------------------------------------------------------------------
    confidence = round(
        sum(sensor_scores.values()) / len(sensor_scores), 3
    )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    interpretation = _build_interpretation(
        milestone_type, confidence, sensor_scores, sensor_means,
        skipped_sensors, iot_cfg
    )

    return {
        "status": "ANALYZED",
        "confidence": confidence,
        "interpretation": interpretation,
        "sensor_scores": sensor_scores,
        "sensor_means":  sensor_means,
        "readings_per_sensor": readings_per_sensor,
        "devices_queried": len(devices),
        "skipped_sensors": skipped_sensors,
        "query_window": {"start": window_start[:10], "end": window_end[:10]},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_data(reason: str, readings_per_sensor: Dict) -> Dict[str, Any]:
    return {
        "status": "NO_DATA",
        "confidence": 0.0,
        "interpretation": reason,
        "sensor_scores": {},
        "sensor_means":  {},
        "readings_per_sensor": readings_per_sensor,
        "devices_queried": 0,
    }


def _build_interpretation(
    milestone_type: str,
    confidence: float,
    sensor_scores: Dict[str, float],
    sensor_means: Dict[str, float],
    skipped_sensors: List[str],
    iot_cfg: Dict,
) -> str:
    if confidence >= 0.80:
        verdict = "strongly supports"
    elif confidence >= 0.55:
        moderately = "moderately supports"
        verdict = moderately
    elif confidence > 0.25:
        verdict = "weakly supports"
    else:
        verdict = "does not support"

    parts = [f"IoT data {verdict} the {milestone_type} milestone."]

    for sensor_type, score in sensor_scores.items():
        mean = sensor_means.get(sensor_type)
        cfg  = iot_cfg.get(sensor_type, {})
        opt  = cfg.get("optimal")
        unit = ""
        if sensor_type == "soil_temperature":
            unit = "°C"
        elif sensor_type == "soil_moisture":
            unit = "%"

        mean_str  = f"{mean:.1f}{unit}" if mean is not None else "n/a"
        range_str = f"{opt[0]}-{opt[1]}{unit}" if opt else "?"
        badge     = "✓" if score >= 0.6 else "✗"
        parts.append(f"{sensor_type}: {mean_str} (optimal {range_str}) {badge}")

    if skipped_sensors:
        parts.append(f"Skipped (too few readings): {', '.join(skipped_sensors)}.")

    return " ".join(parts)
