"""
Wheat agronomic verification thresholds for Punjab Pakistan (Rabi season).

Sowing:   Oct 25 – Nov 15  |  DAS 0-10
Tillering: December – January  |  DAS 21-50
Grain Fill: March – April  |  DAS 100-140

Sources:
  - FAO Crop Information & AQUASTAT (wheat water requirements)
  - CIMMYT Pakistan / NARC Islamabad field trial data
  - SUPARCO Pakistan annual Rabi crop satellite monitoring reports
  - Peer-reviewed RS literature: International Journal of Remote Sensing,
    Remote Sensing of Environment (South Asian Rabi wheat NDVI studies)
  - APSIM / DSSAT wheat model parameterization (LAI, temperature ranges)
  - Punjab Agriculture Department planting guidelines

All NDVI/SAVI/LAI/NDMI values are for Sentinel-2 L2A derived indices as
computed by the sentinelhub.js evalscripts already in this system.
"""

from typing import Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Type alias for a (min, max) range tuple
# ---------------------------------------------------------------------------
Range = Tuple[float, float]

# ---------------------------------------------------------------------------
# Per-milestone thresholds
# ---------------------------------------------------------------------------
# milestone_type keys MUST match the lowercased, underscored template names
# produced by: template["name"].lower().replace(" ", "_")
#   "Sowing"                   → "sowing"
#   "Tillering"                → "tillering"
#   "Grain Filling and Ripening" → "grain_filling_and_ripening"
# ---------------------------------------------------------------------------

WHEAT_MILESTONE_THRESHOLDS: Dict[str, Any] = {

    # ===========================================================
    # MILESTONE 1: SOWING  (DAS 0-10, Nov, Punjab)
    # ===========================================================
    "sowing": {
        # Data window: centered on sowing date reported by farmer
        "time_window": {"before_days": 3, "after_days": 14},

        # ── Satellite (field_metrics) ──────────────────────────
        # At sowing the field is bare soil / very early germination.
        # NDVI should be LOW. If NDVI is already high the field has
        # existing vegetation (crop wasn't actually sown or data is stale).
        "satellite": {
            # Bare soil + early germination NDVI
            "ndvi_optimal":   (0.06, 0.20),   # clear bare soil / just sprouting
            "ndvi_acceptable":(0.04, 0.28),   # slightly ambiguous but acceptable
            # SAVI (more soil-robust) at bare soil
            "savi_optimal":   (0.04, 0.16),
            "savi_acceptable":(0.02, 0.22),
            # NDVI trend: should be FLAT (not rising sharply — that would
            # indicate surviving vegetation from a previous crop)
            "ndvi_trend_max_rise": 0.08,       # change > +0.08 over the window = suspicious
            # Weights for combining multiple sub-scores
            "sub_weights": {
                "ndvi":  0.50,
                "savi":  0.30,
                "trend": 0.20,
            },
        },

        # ── IoT sensors ───────────────────────────────────────
        # Nov Punjab: soil temp 18-28°C is optimal for germination.
        # Pre-sowing "palewa" irrigation brings moisture to ~70% FC.
        "iot": {
            "soil_temperature": {
                "optimal":    (18.0, 28.0),   # °C — ideal Nov sowing window
                "acceptable": (14.0, 32.0),   # still ok, germination slower
            },
            "soil_moisture": {
                "optimal":    (28.0, 58.0),   # % VWC — well-watered seed bed
                "acceptable": (18.0, 70.0),
            },
        },

        # ── Farmer input (growth_stage_inputs) ────────────────
        "farmer_input": {
            # Stage rows that must exist with a non-null stage_date
            "required_stages":   ["sowing"],
            # Acceptable DAS range from the crop cycle start date
            "expected_das_range": (0, 14),
            # Activity flags that add confidence if true
            "positive_activities": ["irrigation"],         # pre-sowing palewa
            "fertilizer_check": True,                      # DAP/basal at sowing
        },

        # ── Drone (from drone_imagery_layers.statistics JSONB) ─
        # NDVI layer mean at sowing should match bare soil range
        "drone": {
            "ndvi_layer_optimal":    (0.04, 0.20),
            "ndvi_layer_acceptable": (0.02, 0.28),
        },
    },

    # ===========================================================
    # MILESTONE 2: TILLERING  (DAS 21-50, Dec–Jan, Punjab)
    # ===========================================================
    "tillering": {
        "time_window": {"before_days": 7, "after_days": 14},

        # ── Satellite ─────────────────────────────────────────
        # Active tillering: visible canopy rows, NDVI rising steadily.
        # Peak tiller count Dec–Jan has NDVI 0.35-0.60.
        "satellite": {
            "ndvi_optimal":    (0.35, 0.62),   # active tillering band
            "ndvi_acceptable": (0.26, 0.70),   # includes early and late-tillering edges
            "savi_optimal":    (0.26, 0.50),
            "savi_acceptable": (0.18, 0.58),
            "lai_optimal":     (0.8,  3.0),    # LAI builds through tillering
            "lai_acceptable":  (0.4,  3.8),
            # NDVI must have risen meaningfully from the post-sowing baseline
            "ndvi_min_rise":   0.10,           # minimum rise from sowing-window reading
            "sub_weights": {
                "ndvi":      0.40,
                "savi":      0.25,
                "lai":       0.20,
                "ndvi_rise": 0.15,
            },
        },

        # ── IoT ───────────────────────────────────────────────
        # Dec/Jan Punjab: cool nights (4-10°C), mild days (15-22°C).
        # Soil temp optimal for tillering: 10-20°C.
        "iot": {
            "soil_temperature": {
                "optimal":    (10.0, 20.0),
                "acceptable": ( 6.0, 26.0),
            },
            "soil_moisture": {
                "optimal":    (30.0, 56.0),
                "acceptable": (20.0, 68.0),
            },
        },

        # ── Farmer input ──────────────────────────────────────
        "farmer_input": {
            "required_stages":    ["sowing", "tillering"],
            "expected_das_range": (18, 58),
            "positive_activities": ["irrigation"],   # tillering irrigation (1st/2nd)
            "fertilizer_check": True,                # urea topdressing at tillering
        },

        # ── Drone ─────────────────────────────────────────────
        "drone": {
            "ndvi_layer_optimal":    (0.32, 0.62),
            "ndvi_layer_acceptable": (0.22, 0.72),
        },
    },

    # ===========================================================
    # MILESTONE 3: GRAIN FILLING AND RIPENING  (DAS 100-140, Mar–Apr)
    # ===========================================================
    "grain_filling_and_ripening": {
        "time_window": {"before_days": 7, "after_days": 21},

        # ── Satellite ─────────────────────────────────────────
        # Peak heading → ripening: NDVI spans 0.25-0.85.
        # NDVI should be declining (senescence) or flat, NOT rising.
        # NDMI measures leaf water content — important for grain quality.
        "satellite": {
            # Wide range covers early grain fill (high NDVI) through ripening (low)
            "ndvi_optimal":    (0.40, 0.82),   # crop is clearly at grain-fill stage
            "ndvi_acceptable": (0.22, 0.88),   # very early heading or late ripening edge
            "savi_optimal":    (0.32, 0.72),
            "savi_acceptable": (0.18, 0.80),
            "lai_optimal":     (2.0,  7.0),    # high LAI at grain fill peak
            "lai_acceptable":  (1.0,  8.0),
            # NDMI — leaf moisture decreasing as grain fills and desiccates
            "ndmi_optimal":    (-0.05, 0.50),  # well-watered grain fill to mid-ripening
            "ndmi_acceptable": (-0.20, 0.58),
            # NDVI trend: expecting FLAT or DECLINING (senescence)
            # Change in the acceptable range: -0.45 (fast ripening) to +0.08 (still filling)
            "ndvi_trend_max_rise": 0.10,       # > +0.10 means we're still in vegetative growth
            "ndvi_trend_min":     -0.50,       # floor: extreme decline = possible crop loss
            "sub_weights": {
                "ndvi":  0.35,
                "savi":  0.15,
                "lai":   0.20,
                "ndmi":  0.20,
                "trend": 0.10,
            },
        },

        # ── IoT ───────────────────────────────────────────────
        # Mar/Apr Punjab: warming rapidly, 25-36°C max, 14-22°C min.
        # Heat stress > 30°C during anthesis/grain fill reduces kernel weight.
        "iot": {
            "soil_temperature": {
                "optimal":    (15.0, 30.0),   # warm but below heat-stress threshold
                "acceptable": ( 9.0, 36.0),
            },
            "soil_moisture": {
                "optimal":    (20.0, 52.0),   # less irrigation at end-season
                "acceptable": (12.0, 64.0),
            },
        },

        # ── Farmer input ──────────────────────────────────────
        "farmer_input": {
            "required_stages":    ["sowing", "tillering", "grain_filling"],
            "expected_das_range": (90, 148),
            # Fungicide / pesticide common at this stage (yellow rust, aphids)
            "positive_activities": ["fungicide", "pesticide"],
            "fertilizer_check": False,   # no topdressing expected at grain fill
        },

        # ── Drone ─────────────────────────────────────────────
        "drone": {
            "ndvi_layer_optimal":    (0.35, 0.82),
            "ndvi_layer_acceptable": (0.18, 0.90),
        },
    },
}

# ---------------------------------------------------------------------------
# Helper: get thresholds for a milestone, returns None for unknown types
# ---------------------------------------------------------------------------

def get_thresholds(milestone_type: str) -> Dict[str, Any]:
    """Return the threshold dict for a milestone type, or empty dict if not found."""
    return WHEAT_MILESTONE_THRESHOLDS.get(milestone_type, {})


def score_range(value: float, optimal: Range, acceptable: Range) -> float:
    """Score a single numeric reading against optimal and acceptable ranges.

    Returns:
        1.0 — value is within the optimal range
        0.5–0.9 — value is within acceptable but outside optimal (linear)
        0.0–0.4 — value is outside the acceptable range (partial credit)
    """
    opt_lo, opt_hi = optimal
    acc_lo, acc_hi = acceptable

    # Clamp acceptable to be at least as wide as optimal
    acc_lo = min(acc_lo, opt_lo)
    acc_hi = max(acc_hi, opt_hi)

    if opt_lo <= value <= opt_hi:
        return 1.0

    if acc_lo <= value <= acc_hi:
        # Linear interpolation within the outer band: 0.5 at edge, 0.9 near optimal
        if value < opt_lo:
            # Left outer band: acc_lo → opt_lo maps to 0.5 → 0.9
            band = opt_lo - acc_lo
            if band <= 0:
                return 0.9
            t = (value - acc_lo) / band   # 0 at acc_lo, 1 at opt_lo
            return 0.5 + 0.4 * t
        else:
            # Right outer band: opt_hi → acc_hi maps to 0.9 → 0.5
            band = acc_hi - opt_hi
            if band <= 0:
                return 0.9
            t = (value - opt_hi) / band   # 0 at opt_hi, 1 at acc_hi
            return 0.9 - 0.4 * t

    # Outside acceptable range → partial credit based on how far outside
    if value < acc_lo:
        gap = acc_lo - value
        # 0.4 at the boundary, approaching 0 as gap grows beyond 0.2 units (or 5°C / 5%)
        ref = 0.2 if acc_lo < 2 else 5.0   # scale differently for indices vs sensor values
        return max(0.0, 0.4 * (1.0 - gap / ref))
    else:
        gap = value - acc_hi
        ref = 0.2 if acc_hi < 2 else 5.0
        return max(0.0, 0.4 * (1.0 - gap / ref))
