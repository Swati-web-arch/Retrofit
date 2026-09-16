import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# The model file should be placed beside this script.
MODEL_FILE = BASE_DIR / "comfort_tsv_model.joblib"

# If the CSVs are kept in the same folder, these names work directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

EESL_FILE = PROJECT_ROOT / "data" / "processed" / "eesl_commercial_retrofits_clean.csv"


FEATURES = [
    "indoor_temp_c",
    "indoor_rh_pct",
    "indoor_air_velocity_ms",
]

RETROFIT_COLUMNS = {
    "Smart_Controls": "retrofit_smart_controls",
    "AHU_VFD": "retrofit_ahu_vfd",
    "DCV": "retrofit_dcv",
    "Chiller_Optimization": "retrofit_chiller_opt",
    "Zoning_Optimization": "retrofit_zoning_opt",
}

eesl = pd.read_csv(EESL_FILE)

saved = joblib.load(MODEL_FILE)
tsv_model = saved["model"]

co2 = eesl["avoided_co2_tons_yr"].dropna()

CO2_BINS = [
    -np.inf,
    co2.quantile(0.20),
    co2.quantile(0.40),
    co2.quantile(0.60),
    co2.quantile(0.80),
    np.inf,
]

MEAN_EESL_AREA_M2 = float(eesl["gross_floor_area_m2"].mean())

comfort_means = {}
co2_means = {}

for retrofit, column in RETROFIT_COLUMNS.items():
    subset = eesl[eesl[column] == 1]

    comfort_means[retrofit] = (
        float(subset["comfort_impact_score"].mean())
        if len(subset) else 3.0
    )

    co2_means[retrofit] = (
        float(subset["avoided_co2_tons_yr"].mean())
        if len(subset) else float(co2.mean())
    )


def _score_from_quintile(value):
    return int(np.digitize(value, CO2_BINS[1:-1], right=True) + 1)


def _option_was_implemented(building, retrofit_option):
    """True if this specific retrofit_option is part of what `building`
    actually implemented -- via its per-measure binary column or its
    `retrofit_measures_implemented` list. Mirrors maintenance_score()'s
    check in src/Person D/maintenance.py."""
    implemented = bool(building.get(RETROFIT_COLUMNS[retrofit_option]))
    if not implemented and building.get("retrofit_measures_implemented"):
        parsed = str(building["retrofit_measures_implemented"]).split(";")
        implemented = retrofit_option in parsed
    return implemented


# ---------------------------------------------------------------------------
# Contextual adjustments (building age / baseline HVAC type / HVAC
# distribution)
#
# Previously, whenever `building` wasn't itself one of the 16 EESL rows
# (i.e. every building entered through the Streamlit app), comfort_score()
# and sustainability_score() fell straight through to a flat per-retrofit
# EESL group average. That average only varies by `retrofit_option` -- it
# never reads `building` -- so these two axes barely moved no matter what
# the user changed (building type, floor area, HVAC type, HVAC
# distribution, building age, etc.).
#
# This adds a small, bounded, documented rule-based adjustment layer on top
# of that EESL fallback baseline, in the same "base value + contextual
# nudge, clipped to the valid range" style Person B's energy_scoring.py
# already uses for its own contextual adjustments. It only fires in the
# fallback branch -- a real EESL row's own directly-labeled outcome is left
# untouched.
# ---------------------------------------------------------------------------

def _age_adjustment(building):
    """Older baseline equipment has more comfort headroom for a retrofit to
    recover (standard facilities-engineering assumption: more worn/drifted
    equipment -> bigger perceptible improvement). `building_age` is
    collected by the app's Building Characteristics form but was previously
    unused by every scoring axis, including this one."""
    age = building.get("building_age")
    if age is None:
        return 0.0
    try:
        age = float(age)
    except (TypeError, ValueError):
        return 0.0
    if age < 10:
        return -0.4
    elif age < 20:
        return 0.0
    elif age < 30:
        return 0.5
    return 1.0


def _distribution_group(building):
    """Classify `hvac_distribution` into 'centralized' / 'localized', or
    None if unspecified. Handles both the app's exact dropdown labels
    ('Centralized (central plant...)' / 'Localized (Split, Window, or
    VRF...)') and freeform equivalents."""
    dist = str(building.get("hvac_distribution", "")).lower()
    if not dist or dist == "nan":
        return None
    if any(w in dist for w in ("decentral", "local", "split", "vrf", "window")):
        return "localized"
    if "central" in dist:
        return "centralized"
    return None


def _hvac_type_adjustment(building, retrofit_option):
    """Bump comfort improvement up or down depending on how well
    `retrofit_option` matches the described baseline HVAC equipment
    (`hvac_type`). Extends to comfort the same keyword-matching pattern
    energy_scoring.py already uses for Chiller_Optimization/AHU_VFD --
    previously comfort_score() ignored `hvac_type` entirely."""
    hvac_desc = str(building.get("hvac_type", "")).lower()
    adjustment = 0.0

    if retrofit_option == "Chiller_Optimization":
        if any(w in hvac_desc for w in ("constant", "reciprocating", "old", "without vfd", "no vfd", "screw")):
            adjustment += 0.6
        elif any(w in hvac_desc for w in ("split", "window", "dx", "vrf")):
            adjustment -= 0.6

    elif retrofit_option == "AHU_VFD":
        if any(w in hvac_desc for w in ("cav", "fixed speed", "fixed")):
            adjustment += 0.5
        elif any(w in hvac_desc for w in ("vfd", "variable")):
            adjustment -= 0.5

    elif retrofit_option == "Zoning_Optimization":
        if building.get("poor_zoning", 0) >= 3 or any(w in hvac_desc for w in ("split", "window")):
            adjustment += 0.5

    elif retrofit_option == "DCV":
        if building.get("ventilation_imbalance", 0) >= 3:
            adjustment += 0.4

    elif retrofit_option == "Smart_Controls":
        if building.get("economizer_fault", 0) >= 3 or any(w in hvac_desc for w in ("manual", "pneumatic")):
            adjustment += 0.5

    return adjustment


def _distribution_adjustment(building, retrofit_option):
    """Bump comfort improvement based on HVAC distribution. Centralized
    plants suffer more from zone-to-zone imbalance, so zoning/ventilation
    fixes recover more comfort there; decentralized/localized (VRF/split)
    estates are made up of many independently-controlled units, so
    coordinated smart controls help more. `hvac_distribution` was
    previously collected by the app but read by no scoring axis at all."""
    group = _distribution_group(building)
    if group is None:
        return 0.0
    if retrofit_option in ("Zoning_Optimization", "DCV"):
        return 0.4 if group == "centralized" else -0.3
    if retrofit_option == "Smart_Controls":
        return 0.4 if group == "localized" else -0.2
    return 0.0


def _round_score(value):
    """Round-half-up to a whole number. Energy and Cost Benefit are
    already discrete 1-5 ints (threshold-based bins); comfort previously
    returned round(x, 2) decimals (e.g. 3.45), which is why the ranked
    table showed decimals for Comfort/Maintenance next to whole numbers for
    the other three axes."""
    return int(np.floor(float(value) + 0.5))


def comfort_score(building, retrofit_option):
    """Return a 1-5 comfort score (whole number) for the selected retrofit
    option.

    Uses the building's own `comfort_impact_score` ONLY when this specific
    retrofit_option was actually part of what that building implemented --
    that label reflects the outcome of whatever combo of measures was
    implemented together, so it isn't a valid stand-in for a *different*,
    hypothetical option. Otherwise falls back to the EESL group average for
    buildings that implemented that measure, nudged by building age,
    baseline HVAC type, and HVAC distribution (see the adjustment helpers
    above) so the score actually responds to the inputs the app collects.
    """
    if retrofit_option not in RETROFIT_COLUMNS:
        raise ValueError(f"Unknown retrofit option: {retrofit_option}")

    value = building.get("comfort_impact_score")
    if value is not None and _option_was_implemented(building, retrofit_option):
        base = float(np.clip(value, 1, 5))
    else:
        base = comfort_means[retrofit_option]
        base += _age_adjustment(building)
        base += _hvac_type_adjustment(building, retrofit_option)
        base += _distribution_adjustment(building, retrofit_option)
        base = float(np.clip(base, 1, 5))

    return _round_score(base)


def _age_co2_multiplier(building):
    """Older baseline equipment runs less efficiently, so replacing/
    optimizing it avoids more CO2 per year. Same age bands as
    _age_adjustment(), expressed as a multiplier since avoided CO2 is a
    continuous physical quantity rather than a 1-5 ordinal rating."""
    age = building.get("building_age")
    if age is None:
        return 1.0
    try:
        age = float(age)
    except (TypeError, ValueError):
        return 1.0
    if age < 10:
        return 0.9
    elif age < 20:
        return 1.0
    elif age < 30:
        return 1.15
    return 1.3


def _hvac_type_co2_multiplier(building, retrofit_option):
    """Same baseline-HVAC-type signal as _hvac_type_adjustment(), expressed
    as a multiplier on avoided CO2."""
    hvac_desc = str(building.get("hvac_type", "")).lower()
    if retrofit_option == "Chiller_Optimization":
        if any(w in hvac_desc for w in ("constant", "reciprocating", "old", "without vfd", "no vfd", "screw")):
            return 1.2
        if any(w in hvac_desc for w in ("split", "window", "dx", "vrf")):
            return 0.85
    elif retrofit_option == "AHU_VFD":
        if any(w in hvac_desc for w in ("cav", "fixed speed", "fixed")):
            return 1.15
        if any(w in hvac_desc for w in ("vfd", "variable")):
            return 0.9
    return 1.0


def _distribution_co2_multiplier(building):
    """From data/processed/buildheat_clean.csv (16 EU building-retrofit
    case studies): grouping `pre_hvac_type` into Centralized/Decentralized
    and comparing (pre_total_co2 - post_total_co2) / pre_total_co2 shows
    decentralized baselines averaging ~78.6% CO2 reduction after retrofit
    vs. ~61.8% for centralized baselines (~1.27x). Applied here toned down,
    since BuildHeat (EU residential heating retrofits) is a cross-domain
    directional proxy rather than a same-domain regression input, and as a
    building-level multiplier rather than a per-measure one, since the
    underlying pattern is about overall system distribution, not a single
    catalog measure."""
    group = _distribution_group(building)
    if group == "localized":
        return 1.15
    if group == "centralized":
        return 0.92
    return 1.0


def sustainability_score(building, retrofit_option):
    """Return a 1-5 sustainability score for the selected retrofit option.

    Same direct-label pattern as comfort_score(): uses the building's own
    `avoided_co2_tons_yr` only when this option was actually what that
    building implemented, otherwise falls back to the EESL group-average
    avoided-CO2 for that measure -- scaled by this building's own floor
    area relative to the EESL buildings' average floor area (avoided CO2
    is a size-dependent absolute quantity, so a building's own
    `gross_floor_area_m2` should move it, and previously never did), then
    nudged by building age, baseline HVAC type, and HVAC distribution.
    """
    if retrofit_option not in RETROFIT_COLUMNS:
        raise ValueError(f"Unknown retrofit option: {retrofit_option}")

    value = building.get("avoided_co2_tons_yr")
    if value is not None and _option_was_implemented(building, retrofit_option):
        co2_value = float(value)
    else:
        co2_value = co2_means[retrofit_option]

        area = (
            building.get("gross_floor_area_m2")
            or building.get("floor_area")
            or building.get("area_tot_m2")
        )
        if area:
            try:
                co2_value *= float(area) / MEAN_EESL_AREA_M2
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        co2_value *= _age_co2_multiplier(building)
        co2_value *= _hvac_type_co2_multiplier(building, retrofit_option)
        co2_value *= _distribution_co2_multiplier(building)

    return _score_from_quintile(co2_value)


def predict_tsv(building):
    """Predict TSV on the approximate -3 to +3 scale."""
    missing = [f for f in FEATURES if f not in building]
    if missing:
        raise ValueError(f"Missing thermal inputs: {missing}")

    row = pd.DataFrame(
        [[building[f] for f in FEATURES]],
        columns=FEATURES
    )

    return float(tsv_model.predict(row)[0])


def comfort_support(building):
    """Return supporting thermal-neutrality information."""
    tsv = predict_tsv(building)
    distance = abs(tsv)

    if distance <= 0.5:
        interpretation = "near neutral"
    elif distance <= 1.0:
        interpretation = "slightly away from neutral"
    elif distance <= 2.0:
        interpretation = "noticeably away from neutral"
    else:
        interpretation = "far from neutral"

    return {
        "predicted_tsv": round(tsv, 3),
        "distance_from_neutral": round(distance, 3),
        "interpretation": interpretation,
    }
if __name__ == "__main__":

    print("\nPerson C — Comfort + Sustainability")
    print("-----------------------------------")

    indoor_temp = float(input("Enter indoor temperature (°C): "))
    indoor_rh = float(input("Enter indoor RH (%): "))
    air_velocity = float(input("Enter indoor air velocity (m/s): "))

    test_building = {
        "indoor_temp_c": indoor_temp,
        "indoor_rh_pct": indoor_rh,
        "indoor_air_velocity_ms": air_velocity
    }

    print("\nComfort Support:")
    print(comfort_support(test_building))

    print("\nRetrofit Scores:")

    for option in RETROFIT_COLUMNS:
        comfort = comfort_score(test_building, option)
        sustainability = sustainability_score(test_building, option)

        print(
            f"{option} | "
            f"Comfort: {comfort} | "
            f"Sustainability: {sustainability}"
        )