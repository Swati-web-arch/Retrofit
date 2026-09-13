"""
Unit and Integration Tests for Person D — Maintenance Axis + Combiner
======================================================================
Run with: pytest "tests/Person D/test_combiner_maintenance.py"
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure Person D's modules can be imported (mirrors the exact pattern used
# in tests/Person B/test_energy_cost.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSON_D_DIR = PROJECT_ROOT / "src" / "Person D"
if str(PERSON_D_DIR) not in sys.path:
    sys.path.insert(0, str(PERSON_D_DIR))

from maintenance import maintenance_score, RETROFIT_COLUMNS, _maintenance_means  # noqa: E402
from combiner import (  # noqa: E402
    recommend_retrofits,
    score_option,
    filter_catalog,
    RETROFIT_CATALOG,
    DEFAULT_WEIGHTS,
)

EESL_FILE = PROJECT_ROOT / "data" / "processed" / "eesl_commercial_retrofits_clean.csv"


# =====================================================================
# 1. maintenance_score()
# =====================================================================

def test_maintenance_unknown_option_raises():
    with pytest.raises(ValueError):
        maintenance_score({}, "Not_A_Real_Option")


def test_maintenance_fallback_when_no_label():
    """A building with no maintenance_reduction_score falls back to the
    EESL group average for that measure, still within the valid 1-5 range."""
    building = {}
    for option in RETROFIT_COLUMNS:
        score = maintenance_score(building, option)
        assert 1 <= score <= 5
        assert score == round(float(_maintenance_means[option]), 2)


def test_maintenance_uses_direct_label_when_option_was_implemented():
    """If the building's own label corresponds to a retrofit option it
    actually implemented, that direct label should be used instead of the
    group average."""
    building = {
        "maintenance_reduction_score": 4.5,
        "retrofit_ahu_vfd": 1,
    }
    assert maintenance_score(building, "AHU_VFD") == 4.5


def test_maintenance_ignores_direct_label_for_a_different_option():
    """The building's own label reflects whatever bundle of measures it
    actually implemented -- it must NOT be reused for a option that
    building never implemented (this was the original bug)."""
    building = {
        "maintenance_reduction_score": 4.5,
        "retrofit_ahu_vfd": 1,
        # Chiller_Optimization was NOT implemented by this building
    }
    score = maintenance_score(building, "Chiller_Optimization")
    assert score == round(float(_maintenance_means["Chiller_Optimization"]), 2)
    assert score != 4.5


def test_maintenance_reads_measures_implemented_list():
    """retrofit_measures_implemented (semicolon-separated) should also
    count as evidence the option was implemented, not just the binary
    per-measure column."""
    building = {
        "maintenance_reduction_score": 4.2,
        "retrofit_measures_implemented": "DCV;Smart_Controls",
    }
    assert maintenance_score(building, "DCV") == 4.2
    assert maintenance_score(building, "AHU_VFD") != 4.2


# =====================================================================
# 2. combiner.filter_catalog()
# =====================================================================

def test_filter_catalog_always_includes_ungated_measures():
    """AHU_VFD and Chiller_Optimization aren't gated by any of Person A's
    four flags, so they should always be candidates."""
    result = filter_catalog({})
    assert "AHU_VFD" in result
    assert "Chiller_Optimization" in result


def test_filter_catalog_gates_in_measures_above_threshold():
    flags = {"poor_zoning": 3, "ventilation_imbalance": 0, "economizer_fault": 0, "sensor_mismatch": 0}
    result = filter_catalog(flags)
    assert "Zoning_Optimization" in result
    assert "DCV" not in result  # ventilation_imbalance below threshold


def test_filter_catalog_returns_subset_of_full_catalog():
    flags = {"poor_zoning": 5, "ventilation_imbalance": 5, "economizer_fault": 5, "sensor_mismatch": 5}
    result = filter_catalog(flags)
    assert set(result) <= set(RETROFIT_CATALOG)
    assert set(result) == set(RETROFIT_CATALOG)  # everything gated in at max severity


# =====================================================================
# 3. combiner.score_option() / recommend_retrofits() -- integration
# =====================================================================

def test_score_option_returns_all_five_axes():
    building = {"gross_floor_area_m2": 15000, "baseline_eui_kwh_per_m2": 180}
    row = score_option(building, "Smart_Controls", DEFAULT_WEIGHTS)
    for key in ("Energy", "Comfort", "Cost Benefit", "Sustainability", "Maintenance", "Final Score"):
        assert key in row
    for key in ("Energy", "Comfort", "Cost Benefit", "Sustainability", "Maintenance"):
        assert 1 <= row[key] <= 5


def test_recommend_retrofits_on_real_eesl_building():
    """End-to-end smoke test: run the full pipeline on an actual EESL row
    and check the ranked output is well-formed (this is the same building
    combiner.py's own __main__ block scores)."""
    eesl = pd.read_csv(EESL_FILE)
    sample = eesl.iloc[0].to_dict()

    result = recommend_retrofits(building_features=sample)

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert list(result["Final Score"]) == sorted(result["Final Score"], reverse=True)
    assert set(result["Retrofit Option"]) <= set(RETROFIT_CATALOG)


def test_recommend_retrofits_respects_custom_weights():
    """Weighting Energy at 100% should reproduce the same ranking as
    sorting by the Energy column alone."""
    eesl = pd.read_csv(EESL_FILE)
    sample = eesl.iloc[0].to_dict()
    energy_only_weights = {
        "energy": 1.0, "comfort": 0.0, "cost_benefit": 0.0,
        "sustainability": 0.0, "maintenance": 0.0,
    }
    result = recommend_retrofits(building_features=sample, weights=energy_only_weights)
    assert list(result["Final Score"]) == list(result["Energy"])


def test_recommend_retrofits_filters_by_inefficiency_flags():
    eesl = pd.read_csv(EESL_FILE)
    sample = eesl.iloc[0].to_dict()
    no_faults = {"poor_zoning": 0, "ventilation_imbalance": 0, "economizer_fault": 0, "sensor_mismatch": 0}
    result = recommend_retrofits(building_features=sample, inefficiency_flags=no_faults)
    # Only the two ungated measures should be scored when nothing is flagged
    assert set(result["Retrofit Option"]) == {"AHU_VFD", "Chiller_Optimization"}
