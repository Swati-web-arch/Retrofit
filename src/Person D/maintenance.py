"""
Maintenance Axis
================
Person D — Retrofit Recommendation Engine

Scores expected maintenance-burden reduction for a given retrofit option,
1-5. Mirrors Person C's comfort_score/sustainability_score pattern for
consistency across the team's scoring functions:

    - If the building dict already carries a `maintenance_reduction_score`
      (e.g. we're scoring one of EESL's own 16 buildings), use it directly.
    - Otherwise fall back to the EESL group-average maintenance score for
      buildings that implemented that retrofit measure.

Shared interface:
    maintenance_score(building: dict, retrofit_option: str) -> float (1-5)
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EESL_FILE = PROJECT_ROOT / "data" / "processed" / "eesl_commercial_retrofits_clean.csv"

RETROFIT_COLUMNS = {
    "Smart_Controls": "retrofit_smart_controls",
    "AHU_VFD": "retrofit_ahu_vfd",
    "DCV": "retrofit_dcv",
    "Chiller_Optimization": "retrofit_chiller_opt",
    "Zoning_Optimization": "retrofit_zoning_opt",
}

if not EESL_FILE.exists():
    raise FileNotFoundError(
        f"maintenance.py expected the EESL dataset at {EESL_FILE} but it "
        f"isn't there. This file is loaded at import time (module-level), "
        f"so anything that imports maintenance.py -- including combiner.py "
        f"-- will fail immediately with this same error until the path is "
        f"fixed or the CSV is restored."
    )
_eesl = pd.read_csv(EESL_FILE)

# NOTE on the group-average baseline:
# The EESL dataset's only maintenance column is `maintenance_reduction_score`,
# which is the label itself (range 3.9-4.5, std=0.15 across 16 rows). There
# are no EESL building-level predictive features (equipment age, service
# frequency, condition rating, historical downtime, etc.) that could drive a
# per-building regression the way energy_scoring uses EUI/area/floors, so the
# group average by retrofit measure remains the baseline for the fallback
# case (no direct EESL label for this exact option).
#
# On top of that baseline, though, the app itself already collects several
# maintenance-relevant signals that were previously ignored entirely by this
# module: `building_age` (older equipment needs more upkeep -- a retrofit
# has more maintenance burden to remove), `hvac_type` (mechanically simple/
# manually-serviced baseline equipment has more truck-rolls to eliminate),
# and `hvac_distribution` (many independent decentralized units mean many
# separate maintenance touchpoints vs. one centralized plant). Previously
# none of these moved this score at all -- see the adjustment helpers below,
# which mirror the same contextual-nudge pattern energy_scoring.py already
# uses. This was the actual cause of the maintenance axis "barely changing":
# with only the flat 3.9-4.5 group averages and no adjustments, every
# building scored the same ~4 regardless of any input.
#
# Rounding: previously returned round(float, 2) (e.g. 4.19), inconsistent
# with Energy/Cost Benefit's whole-number 1-5 scale. Now rounds (half-up) to
# an int, same as every other axis.
_maintenance_means = {}
for _retrofit, _column in RETROFIT_COLUMNS.items():
    _subset = _eesl[_eesl[_column] == 1]
    _maintenance_means[_retrofit] = (
        float(_subset["maintenance_reduction_score"].mean())
        if len(_subset) else float(_eesl["maintenance_reduction_score"].mean())
    )


def _age_adjustment(building):
    """Older baseline equipment carries more maintenance burden for a
    retrofit to remove. `building_age` is collected by the app's Building
    Characteristics form but was previously unused by every scoring axis,
    including this one."""
    age = building.get("building_age")
    if age is None:
        return 0.0
    try:
        age = float(age)
    except (TypeError, ValueError):
        return 0.0
    if age < 10:
        return -0.5
    elif age < 20:
        return 0.0
    elif age < 30:
        return 0.6
    return 1.2


def _distribution_group(building):
    """Classify `hvac_distribution` into 'centralized' / 'localized', or
    None if unspecified. Mirrors Person C's comfort_sustainability.py
    helper of the same name."""
    dist = str(building.get("hvac_distribution", "")).lower()
    if not dist or dist == "nan":
        return None
    if any(w in dist for w in ("decentral", "local", "split", "vrf", "window")):
        return "localized"
    if "central" in dist:
        return "centralized"
    return None


def _hvac_type_adjustment(building, retrofit_option):
    """Bump maintenance-reduction potential up or down depending on how
    well `retrofit_option` matches the described baseline HVAC equipment
    (`hvac_type`). Extends to maintenance the same keyword-matching pattern
    energy_scoring.py already uses for Chiller_Optimization/AHU_VFD --
    previously maintenance_score() ignored `hvac_type` entirely."""
    hvac_desc = str(building.get("hvac_type", "")).lower()
    adjustment = 0.0

    if retrofit_option == "Chiller_Optimization":
        if any(w in hvac_desc for w in ("constant", "reciprocating", "old", "without vfd", "no vfd", "screw")):
            adjustment += 0.7  # old, mechanically-serviced chillers need frequent manual upkeep
        elif any(w in hvac_desc for w in ("split", "window", "dx", "vrf")):
            adjustment -= 0.6  # chiller optimization doesn't apply to non-central-plant baselines

    elif retrofit_option == "AHU_VFD":
        if any(w in hvac_desc for w in ("cav", "fixed speed", "fixed")):
            adjustment += 0.5
        elif any(w in hvac_desc for w in ("vfd", "variable")):
            adjustment -= 0.5

    elif retrofit_option == "Zoning_Optimization":
        if building.get("poor_zoning", 0) >= 3:
            adjustment += 0.4

    elif retrofit_option == "DCV":
        if building.get("ventilation_imbalance", 0) >= 3:
            adjustment += 0.4

    elif retrofit_option == "Smart_Controls":
        if building.get("economizer_fault", 0) >= 3 or any(w in hvac_desc for w in ("manual", "pneumatic")):
            adjustment += 0.6  # automating manual/pneumatic controls removes a lot of routine truck-rolls

    return adjustment


def _distribution_adjustment(building, retrofit_option):
    """Bump maintenance-reduction potential based on HVAC distribution.
    Many independent decentralized units (VRF/split/window) mean many
    separate maintenance touchpoints; a centralized plant concentrates
    maintenance into fewer, larger pieces of equipment. `hvac_distribution`
    was previously collected by the app but read by no scoring axis at all.
    """
    group = _distribution_group(building)
    if group is None:
        return 0.0
    if retrofit_option in ("Chiller_Optimization", "AHU_VFD"):
        # These target central-plant equipment specifically.
        return 0.4 if group == "centralized" else -0.5
    if retrofit_option == "Smart_Controls":
        # BMS-style centralized monitoring saves the most truck-rolls when
        # there are many independent zone units to coordinate.
        return 0.5 if group == "localized" else -0.2
    return 0.0


def _round_score(value):
    """Round-half-up to a whole number -- see the module-level rounding
    note above."""
    return int(np.floor(float(value) + 0.5))


def maintenance_score(building: dict, retrofit_option: str) -> int:
    """Return a 1-5 maintenance-reduction score (whole number) for a
    building + retrofit option.

    Uses the building's own `maintenance_reduction_score` ONLY when this
    specific retrofit_option was actually part of what that building
    implemented (via its `retrofit_measures_implemented` list or the
    matching per-measure binary column) -- that label reflects the outcome
    of whatever combo of measures was implemented together, so it isn't a
    valid stand-in for a *different*, hypothetical option. Otherwise falls
    back to the EESL group average for buildings that implemented that
    measure, nudged by building age, baseline HVAC type, and HVAC
    distribution (see the adjustment helpers above) so the score actually
    responds to the inputs the app collects.
    """
    if retrofit_option not in RETROFIT_COLUMNS:
        raise ValueError(f"Unknown retrofit option: {retrofit_option}")

    value = building.get("maintenance_reduction_score")
    option_was_implemented = bool(building.get(RETROFIT_COLUMNS[retrofit_option]))
    if not option_was_implemented and building.get("retrofit_measures_implemented"):
        implemented = str(building["retrofit_measures_implemented"]).split(";")
        option_was_implemented = retrofit_option in implemented

    if value is not None and option_was_implemented:
        base = float(np.clip(value, 1, 5))
    else:
        base = _maintenance_means[retrofit_option]
        base += _age_adjustment(building)
        base += _hvac_type_adjustment(building, retrofit_option)
        base += _distribution_adjustment(building, retrofit_option)
        base = float(np.clip(base, 1, 5))

    return _round_score(base)


if __name__ == "__main__":
    print("Person D — Maintenance Axis")
    print("----------------------------")
    print("EESL group-average maintenance scores by measure:")
    for option in RETROFIT_COLUMNS:
        print(f"  {option}: {_maintenance_means[option]:.2f}")

    test_building = {}  # no direct label -> falls back to group averages
    print("\nScoring a building with no known label:")
    for option in RETROFIT_COLUMNS:
        print(f"  {option}: {maintenance_score(test_building, option)}")
