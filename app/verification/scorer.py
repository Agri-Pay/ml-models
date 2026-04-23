"""Verification scorer — weighted combination of multi-source analysis.

Supports 4 evidence sources for wheat:
  satellite (0.40) + iot (0.25) + farmer_input (0.20) + drone (0.15)

When a source has no data its weight is redistributed proportionally to
the remaining available sources, ensuring the overall confidence still
reflects a 0.0–1.0 scale regardless of data availability.
"""

from typing import Any, Dict, Optional


# Default source weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "satellite":    0.40,
    "iot":          0.25,
    "farmer_input": 0.20,
    "drone":        0.15,
}

# Verdict thresholds
COMPLETE_THRESHOLD = 0.75   # → MILESTONE_COMPLETE
REVIEW_THRESHOLD   = 0.40   # → MANUAL_REVIEW_REQUIRED
# below REVIEW_THRESHOLD    → MILESTONE_FAILED


def compute_verdict(
    satellite_result:    Dict[str, Any],
    drone_result:        Dict[str, Any],
    iot_result:          Dict[str, Any],
    farmer_input_result: Optional[Dict[str, Any]] = None,
    weights:             Dict[str, float] = None,
) -> Dict[str, Any]:
    """Compute a weighted verification verdict from multi-source analysis.

    Dynamically adjusts weights when data sources are missing or have
    zero confidence (i.e. status != "ANALYZED").

    Args:
        satellite_result:    Result from analyze_satellite().
        drone_result:        Result from analyze_drone().
        iot_result:          Result from analyze_iot().
        farmer_input_result: Result from analyze_farmer_input() (optional,
                             for backwards compatibility with old callers).
        weights:             Override DEFAULT_WEIGHTS (optional).

    Returns:
        dict with verdict, overall_confidence, recommendation, per-source
        details, and weights_used.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Build the sources dict; farmer_input is included only if provided
    sources: Dict[str, Dict[str, Any]] = {
        "satellite":    satellite_result,
        "drone":        drone_result,
        "iot":          iot_result,
    }
    if farmer_input_result is not None:
        sources["farmer_input"] = farmer_input_result

    # Ensure all sources that are present have a weight
    for name in sources:
        if name not in weights:
            weights[name] = 0.0

    # ------------------------------------------------------------------
    # Classify sources as "available" (have real data) or "unavailable"
    # ------------------------------------------------------------------
    available:   Dict[str, Dict] = {}
    unavailable: Dict[str, Dict] = {}

    for name, result in sources.items():
        if result and result.get("status") == "ANALYZED" and result.get("confidence", 0) > 0:
            available[name] = result
        else:
            unavailable[name] = result

    if not available:
        return {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "overall_confidence": 0.0,
            "recommendation": (
                "Unable to verify milestone — no data sources available. "
                "Ensure satellite monitoring is active and the farmer has "
                "filled in their growth stage input log."
            ),
            "sources_available": 0,
            "sources_total": len(sources),
            "satellite_analysis":    satellite_result,
            "drone_analysis":        drone_result,
            "iot_analysis":          iot_result,
            "farmer_input_analysis": farmer_input_result,
        }

    # ------------------------------------------------------------------
    # Redistribute weights from unavailable sources proportionally
    # ------------------------------------------------------------------
    total_available_weight = sum(weights.get(name, 0) for name in available)
    if total_available_weight <= 0:
        total_available_weight = 1.0

    adjusted_weights: Dict[str, float] = {
        name: weights.get(name, 0) / total_available_weight
        for name in available
    }

    # ------------------------------------------------------------------
    # Weighted average confidence
    # ------------------------------------------------------------------
    overall_confidence = sum(
        available[name]["confidence"] * adjusted_weights[name]
        for name in available
    )

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    if overall_confidence >= COMPLETE_THRESHOLD:
        verdict = "MILESTONE_COMPLETE"
    elif overall_confidence >= REVIEW_THRESHOLD:
        verdict = "MANUAL_REVIEW_REQUIRED"
    else:
        verdict = "MILESTONE_FAILED"

    recommendation = _generate_recommendation(
        verdict, overall_confidence, available, unavailable
    )

    return {
        "verdict":            verdict,
        "overall_confidence": round(overall_confidence, 3),
        "recommendation":     recommendation,
        "sources_available":  len(available),
        "sources_total":      len(sources),
        "weights_used":       {k: round(v, 3) for k, v in adjusted_weights.items()},
        "satellite_analysis":    satellite_result,
        "drone_analysis":        drone_result,
        "iot_analysis":          iot_result,
        "farmer_input_analysis": farmer_input_result,
    }


def _generate_recommendation(
    verdict:     str,
    confidence:  float,
    available:   Dict,
    unavailable: Dict,
) -> str:
    parts = []

    if verdict == "MILESTONE_COMPLETE":
        sources_str = ", ".join(available.keys())
        parts.append(
            f"All available indicators ({sources_str}) confirm milestone completion "
            f"with {confidence:.0%} confidence. Recommend approving payment."
        )
    elif verdict == "MANUAL_REVIEW_REQUIRED":
        parts.append(
            f"Evidence is inconclusive ({confidence:.0%} confidence). "
            "Manual review recommended before approving payment."
        )
        for name, result in available.items():
            if result["confidence"] < 0.5:
                parts.append(
                    f" - {name}: low confidence ({result['confidence']:.0%}) — "
                    f"{result.get('interpretation', '')}"
                )
    else:
        parts.append(
            f"Verification failed ({confidence:.0%} confidence). "
            "Do not approve payment without additional evidence."
        )

    if unavailable:
        missing = ", ".join(unavailable.keys())
        parts.append(f"Missing data from: {missing}.")

    return " ".join(parts)
