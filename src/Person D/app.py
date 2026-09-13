"""
Interface
=========
Person D — Retrofit Recommendation Engine

Streamlit app: enter building characteristics, review real-time detected
conditions from sensor telemetry, adjust the 5 axis weights, and see the
ranked retrofit table.

Axis weights are NOT auto-normalized. Each slider can be individually
locked (frozen at its current value) while you adjust the others, but the
five raw values must sum to exactly 1.00 before recommendations will run --
see the sidebar for the live sum and error state.

Run with:
    streamlit run "src/Person D/app.py"
"""

import io
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from combiner import (
    DEFAULT_WEIGHTS,
    PROJECT_ROOT,
    classify_eui,
    detect_conditions,
    recommend_retrofits,
)

st.set_page_config(page_title="Retrofit Recommendation Engine", layout="wide")
st.title("Retrofit Recommendation Engine")
st.caption(
    "Enter building characteristics, review real-time detected conditions from sensor telemetry, "
    "adjust axis weights, and get a ranked, scored table of retrofit options."
)

# ---------------------------------------------------------------------------
# Sidebar: axis weights.
#
# Each axis has a "Lock" toggle next to its slider. Locking disables that
# slider (it can't be dragged) but keeps whatever value it already has --
# useful for pinning a weight you're happy with while you experiment with
# the rest. Weights are used exactly as entered: there is no behind-the-
# scenes normalization anymore, so the five sliders must add up to exactly
# 1.00 or "Get Recommendations" stays disabled and a red error shows below.
# ---------------------------------------------------------------------------
st.sidebar.header("Axis Weights")
st.sidebar.caption(
    "Lock an axis to freeze its value while you adjust the others. "
    "The five weights must sum to exactly 1.00 to run."
)

AXES = [
    ("energy", "Energy"),
    ("comfort", "Comfort"),
    ("cost_benefit", "Cost Benefit"),
    ("sustainability", "Sustainability"),
    ("maintenance", "Maintenance"),
]

weight_values = {}
for _key, _label in AXES:
    _locked = st.sidebar.toggle(f"Lock {_label}", key=f"lock_{_key}", value=False)
    weight_values[_key] = st.sidebar.slider(
        _label,
        0.0,
        1.0,
        DEFAULT_WEIGHTS[_key],
        0.05,
        key=f"weight_{_key}",
        disabled=_locked,
    )

raw_total = sum(weight_values.values())
# Round before comparing -- slider steps of 0.05 can accumulate tiny
# floating-point noise (e.g. 0.05 * 3 == 0.15000000000000002).
weights_valid = round(raw_total, 2) == 1.00

st.sidebar.markdown(f"**Sum of weights: {raw_total:.2f}** (must equal 1.00)")
if weights_valid:
    st.sidebar.success("Weights sum to 1.00 — ready to run.")
else:
    st.sidebar.error(
        f"Weights sum to {raw_total:.2f}, not 1.00. Adjust the sliders "
        f"above (or unlock a locked one) until they add up to exactly "
        f"1.00. 'Get Recommendations' is disabled until then."
    )

weights = weight_values

# ---------------------------------------------------------------------------
# Building Characteristics & Dropdown Options
# Sourced from project datasets (BDG2/EESL) to reliably match model categories.
# ---------------------------------------------------------------------------

BUILDING_TYPE_OPTIONS = [
    ("Office", "Office"),
    ("Education", "Education"),
    ("Healthcare", "Healthcare"),
    ("Retail", "Retail"),
    ("Food Sales and Service", "Food sales and service"),
    ("Residential or Lodging", "Lodging/residential"),
    ("Entertainment or Public Assembly", "Entertainment/public assembly"),
    ("Public Services", "Public services"),
    ("Warehouse or Storage", "Warehouse/storage"),
    ("Manufacturing or Industrial", "Manufacturing/industrial"),
    ("Technology or Science", "Technology/science"),
    ("Religious Worship", "Religious worship"),
    ("Utility", "Utility"),
    ("Other", None),
]

HVAC_TYPE_OPTIONS = [
    "Constant-speed Chillers + CAV AHUs",
    "Water-cooled Screw Chillers + CAV AHUs",
    "Reciprocating Chillers + Fixed Speed AHUs",
    "Old Centrifugal Chillers + Primary-Secondary Pumping",
    "Centrifugal Chillers with VFD + Variable Primary Pumping",
    "Water-cooled Chillers without VFD",
    "Mixed Window and Split Units + Constant Volume Ducting",
    "Direct Expansion (DX) Central Units + Split ACs",
    "VRF System with Individual Zone Control",
    "Other",
]

HVAC_DISTRIBUTION_OPTIONS = [
    "Centralized (central plant serving the whole building)",
    "Localized (Split, Window, or VRF units per zone)",
]

# Standard severity scale from Person A's inefficiency_detection module
SEVERITY_LEGEND = {
    0: "No evidence",
    1: "Very weak",
    2: "Mild",
    3: "Moderate",
    4: "Strong",
    5: "Severe",
}

# ---------------------------------------------------------------------------
# Cached Telemetry Loading & Detection
# Cached to avoid re-reading and re-scoring 4-9MB CSVs on each widget interaction.
# ---------------------------------------------------------------------------

@st.cache_data
def load_and_evaluate_sample_telemetry(file_path_str: str) -> Dict[str, Any]:
    df = pd.read_csv(file_path_str)
    return detect_conditions(df)


@st.cache_data
def evaluate_uploaded_telemetry(file_bytes: bytes) -> Dict[str, Any]:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return detect_conditions(df)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Building Characteristics")

    _bt_label = st.selectbox(
        "Building Type",
        [label for label, _ in BUILDING_TYPE_OPTIONS],
        index=0,
    )
    _bt_value = dict(BUILDING_TYPE_OPTIONS)[_bt_label]
    if _bt_value is None:
        building_type = st.text_input("Specify building type", "")
        st.caption(
            "A custom building type that doesn't match a known category "
            "will be benchmarked against Office EUI ranges by default."
        )
    else:
        building_type = _bt_value

    floor_area = st.number_input("Gross Floor Area (m²)", min_value=100, value=15000, step=500)

    # Note: building_age is collected for building audit records but currently unused by scoring models.
    building_age = st.number_input("Building Age (years)", min_value=0, value=15)
    st.caption("Contextual metric (recorded for audit history; not consumed by scoring models).")

    _hvac_label = st.selectbox("Baseline HVAC Type", HVAC_TYPE_OPTIONS, index=0)
    if _hvac_label == "Other":
        baseline_hvac_type = st.text_input("Specify baseline HVAC type", "")
    else:
        baseline_hvac_type = _hvac_label

    # Note: hvac_distribution is collected for layout metadata but currently unused by scoring models.
    hvac_distribution = st.selectbox("HVAC Distribution", HVAC_DISTRIBUTION_OPTIONS)
    st.caption("Contextual layout metadata (not consumed by scoring models).")

    n_floors = st.number_input("Number of Floors", min_value=1, value=4)
    fan_type = st.selectbox("Fan Type", ["Constant Speed", "Variable Speed (VFD)"])

    # -----------------------------------------------------------------------
    # Problem 2a: Real Energy & Cost Inputs
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("##### Energy Consumption & EUI Benchmarking")
    energy_input_method = st.radio(
        "Energy Metric Input Method",
        ["Baseline EUI (kWh/m²/yr)", "Annual Energy Consumption (kWh/yr)"],
        horizontal=True,
    )

    building_features = {
        "building_type": building_type,
        "gross_floor_area_m2": floor_area,
        "building_age": building_age,
        "hvac_type": baseline_hvac_type,
        "hvac_distribution": hvac_distribution,
        "n_floors": n_floors,
        "fan_type": fan_type,
    }

    if energy_input_method == "Baseline EUI (kWh/m²/yr)":
        eui_input = st.number_input(
            "Baseline EUI (kWh/m²/yr)",
            min_value=1.0,
            max_value=2500.0,
            value=150.0,
            step=10.0,
            help="Annual energy consumed per unit area. Average commercial office benchmark is ~150 kWh/m².",
        )
        if eui_input > 0:
            building_features["eui"] = float(eui_input)
    else:
        default_annual = float(floor_area * 150)
        energy_input = st.number_input(
            "Annual Energy Consumption (kWh/yr)",
            min_value=100.0,
            value=default_annual,
            step=10000.0,
            help="Total annual electricity consumption in kWh.",
        )
        if energy_input > 0:
            building_features["annual_energy"] = float(energy_input)

    # Predicted EUI Benchmark via Person B's classify_eui()
    try:
        eui_result = classify_eui(building_features)
        b_class = eui_result["benchmark_class"]
        pct = int(round(eui_result["percentile"] * 100))
        suffix = "th" if 11 <= pct % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(pct % 10, "th")
        b_type = eui_result["building_type"]
        eui_val_disp = round(eui_result["eui"], 1)
        st.info(
            f"**Predicted EUI Benchmark:** **{b_class}** ({pct}{suffix} percentile vs. peer {b_type} buildings) — "
            f"Calculated EUI: {eui_val_disp} kWh/m²/yr"
        )
    except Exception as exc:
        st.caption(f"EUI classification unavailable: {exc}")

    # Optional Tariff & CAPEX Overrides (Checkbox-gated)
    st.markdown("---")
    st.markdown("##### Financial Overrides (Optional)")
    override_tariff = st.checkbox("Override default electricity tariff (Default: 9.00 INR/kWh)", value=False)
    if override_tariff:
        tariff_val = st.number_input(
            "Electricity Tariff (INR/kWh)",
            min_value=0.1,
            value=9.0,
            step=0.5,
            help="Custom electricity unit rate in INR/kWh.",
        )
        if tariff_val > 0:
            building_features["electricity_price"] = float(tariff_val)

    override_capex = st.checkbox("Specify custom CAPEX / Investment Budget (INR)", value=False)
    if override_capex:
        capex_val = st.number_input(
            "CAPEX Budget (INR)",
            min_value=1000.0,
            value=float(floor_area * 700),
            step=50000.0,
            help="Custom total investment budget. If omitted, uses EESL empirical benchmark rates per m².",
        )
        if capex_val > 0:
            building_features["capex"] = float(capex_val)


with col2:
    st.subheader("Detected Conditions")
    st.caption(
        "Evaluated from raw sensor telemetry by Person A's physics-based "
        "inefficiency detection layer (0=no evidence ... 5=severe)."
    )

    SAMPLE_DATASETS = {
        "Sample: Office HVAC testbed (bldg59)": PROJECT_ROOT / "data" / "processed" / "bldg59_master_hourly_clean.csv",
        "Sample: Multi-zone lab testbed": PROJECT_ROOT / "data" / "processed" / "TestBedClean.csv",
        "Upload sensor CSV": None,
    }

    telemetry_source = st.radio(
        "Select Telemetry Source",
        list(SAMPLE_DATASETS.keys()),
        index=0,
    )

    detection_data = None
    if telemetry_source == "Upload sensor CSV":
        uploaded_file = st.file_uploader(
            "Upload Sensor Telemetry CSV",
            type=["csv"],
            help="Upload raw sensor log (must include temperature, damper, or zone readings).",
        )
        if uploaded_file is not None:
            try:
                detection_data = evaluate_uploaded_telemetry(uploaded_file.getvalue())
            except Exception as exc:
                st.error(f"Failed to evaluate uploaded telemetry: {exc}")
        else:
            st.info("Upload a sensor CSV file to compute building inefficiencies.")
    else:
        sample_path = SAMPLE_DATASETS[telemetry_source]
        if sample_path.exists():
            detection_data = load_and_evaluate_sample_telemetry(str(sample_path))
        else:
            st.error(f"Sample telemetry file not found at: {sample_path}")

    if detection_data is not None:
        scores = detection_data.get("scores", {})
        details = detection_data.get("details", {})

        st.markdown("#### Model-Predicted Severity Scores")
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            pz = scores.get("poor_zoning", 0)
            st.metric("Poor Zoning", f"{pz} / 5", SEVERITY_LEGEND.get(pz, ""))
            ef = scores.get("economizer_fault", 0)
            st.metric("Economizer Fault", f"{ef} / 5", SEVERITY_LEGEND.get(ef, ""))
        with mcol2:
            vi = scores.get("ventilation_imbalance", 0)
            st.metric("Ventilation Imbalance", f"{vi} / 5", SEVERITY_LEGEND.get(vi, ""))
            sm = scores.get("sensor_mismatch", 0)
            st.metric("Sensor Mismatch", f"{sm} / 5", SEVERITY_LEGEND.get(sm, ""))

        with st.expander("Diagnostic Telemetry Evidence & Detector Details", expanded=False):
            # Zoning
            z_det = details.get("zoning", {})
            if "median_zone_spread_c" in z_det:
                st.markdown(
                    f"• **Poor Zoning ({SEVERITY_LEGEND.get(pz, '')}):** "
                    f"Median inter-zone spread is {z_det.get('median_zone_spread_c')}°C "
                    f"(P90: {z_det.get('p90_zone_spread_c')}°C, Mean: {z_det.get('mean_zone_spread_c')}°C). "
                    f"{round(z_det.get('fraction_hours_spread_above_3c', 0) * 100, 1)}% of hours have >3°C spread across "
                    f"{z_det.get('num_zones_evaluated')} zones. VAV CV: {z_det.get('vav_energy_cv')}."
                )
            elif "mean_temp_drift_from_setpoint_c" in z_det:
                st.markdown(
                    f"• **Poor Zoning ({SEVERITY_LEGEND.get(pz, '')}):** "
                    f"Mean drift from comfort setpoint: {z_det.get('mean_temp_drift_from_setpoint_c')}°C. "
                    f"*{z_det.get('note', '')}*"
                )
            else:
                st.markdown(f"• **Poor Zoning ({SEVERITY_LEGEND.get(pz, '')}):** {z_det}")

            # Ventilation
            v_det = details.get("ventilation", {})
            st.markdown(
                f"• **Ventilation Imbalance ({SEVERITY_LEGEND.get(vi, '')}):** "
                f"Violation rate: {v_det.get('violation_rate_pct', 0)}% ({v_det.get('total_violation_hours', 0)} total hours). "
                f"Over-ventilation during extreme weather: {v_det.get('over_ventilation_hours', 0)} hrs; "
                f"Under-ventilation during occupancy: {v_det.get('under_ventilation_hours', 0)} hrs. "
                f"Damper standard deviation: {v_det.get('damper_std', 0)}%."
            )

            # Economizer
            e_det = details.get("economizer", {})
            st.markdown(
                f"• **Economizer Fault ({SEVERITY_LEGEND.get(ef, '')}):** "
                f"Fault rate: {e_det.get('fault_rate_pct', 0)}% ({e_det.get('total_fault_hours', 0)} of {e_det.get('eligible_hours', 0)} eligible hours). "
                f"Stuck open: {e_det.get('stuck_open_hours', 0)} hrs; "
                f"Stuck closed: {e_det.get('stuck_closed_hours', 0)} hrs. "
                f"Max consecutive duration: {e_det.get('max_consecutive_fault_hours', 0)} hrs."
            )

            # Sensor Mismatch
            s_det = details.get("sensor_mismatch", {})
            st.markdown(
                f"• **Sensor Mismatch ({SEVERITY_LEGEND.get(sm, '')}):** "
                f"Mismatch rate: {s_det.get('mismatch_rate_pct', 0)}% ({s_det.get('mismatch_hours', 0)} hours with |dev| > 15°F). "
                f"Mean absolute deviation: {s_det.get('mean_abs_deviation_f', 0)}°F, "
                f"Max deviation: {s_det.get('max_abs_deviation_f', 0)}°F."
            )

        inefficiency_flags = scores
    else:
        inefficiency_flags = None


# ---------------------------------------------------------------------------
# Recommendation Action
# ---------------------------------------------------------------------------
st.markdown("---")
if st.button("Get Recommendations", type="primary", disabled=not weights_valid):
    result = recommend_retrofits(
        building_features=building_features,
        inefficiency_flags=inefficiency_flags,
        weights=weights,
    )
    st.subheader("Ranked Retrofit Recommendations")
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.caption(
        f"{len(result)} of 5 catalog options shown "
        f"(filtered by detected conditions above)."
    )

if not weights_valid:
    st.info("Fix the axis weights in the sidebar (they must sum to 1.00) to enable recommendations.")
