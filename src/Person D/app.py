"""
Interface
=========
Person D — Retrofit Recommendation Engine

Streamlit app: enter building characteristics (including zoning & occupancy),
review real-time detected conditions from sensor telemetry, adjust the 5 axis
weights, and view an executive dashboard with clean 1-5 scale grades (A-F),
detailed scoring explanations, visual charts, and a complete financial
feasibility table with dark green to dark red color mapping.

Run with:
    streamlit run "src/Person D/app.py"
"""

import io
from pathlib import Path
from typing import Any, Dict, Optional
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from combiner import (
    DEFAULT_WEIGHTS,
    PROJECT_ROOT,
    RETROFIT_CATALOG,
    classify_eui,
    detect_conditions,
    recommend_retrofits,
)


def estimate_declared_zoning_severity(building: Dict[str, Any]) -> int:
    """Estimate poor_zoning severity (0-5) from the user's DECLARED zoning setup.
    Rule of thumb: roughly 1 zone per 300-500 m2 is reasonable practice;
    fewer than that, or a single/uniform zone on a large building, is a poor-zoning signal.
    """
    n_zones = int(building.get("n_thermal_zones") or building.get("n_zones") or 1)
    zoning_type = str(building.get("zoning_type", "")).lower()
    area = float(building.get("gross_floor_area_m2", 0) or 0)

    if "single" in zoning_type or "uniform" in zoning_type:
        if area > 1000:
            return 5
        elif area > 300:
            return 3
        return 1

    if n_zones <= 0 or area <= 0:
        return 0

    m2_per_zone = area / max(n_zones, 1)
    if m2_per_zone > 1000:
        return 4
    elif m2_per_zone > 600:
        return 3
    elif m2_per_zone > 300:
        return 1
    return 0


st.set_page_config(
    page_title="RetrofitIQ — Building Recommendation Engine",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation & Input Persistence State
if "page_view" not in st.session_state:
    st.session_state["page_view"] = "input"

if "user_inputs" not in st.session_state:
    st.session_state["user_inputs"] = {}

# ---------------------------------------------------------------------------
# Sidebar: Axis Weights
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Axis Weights")
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
weights_valid = round(raw_total, 2) == 1.00

st.sidebar.markdown(f"**Sum of weights: `{raw_total:.2f}`** (Target: `1.00`)")
if weights_valid:
    st.sidebar.success("Weights sum to 1.00 — ready to run.")
else:
    st.sidebar.error(f"Weights sum to {raw_total:.2f}. Adjust sliders until they equal 1.00.")

weights = weight_values

# Sidebar: Direct Grading System Overview
with st.sidebar.expander("Grading Scale (1 to 5)", expanded=True):
    st.markdown(
        """
        **Final Score (Scale 1–5):**
        - **Grade A** (≥ 4.0): Highly Recommended
        - **Grade B** (3.0 – 3.99): Recommended
        - **Grade C** (2.0 – 2.99): Consider
        - **Grade D** (1.0 – 1.99): Low Priority
        - **Grade F** (< 1.0): Not Recommended
        """
    )

# ---------------------------------------------------------------------------
# Options & Constants
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

ZONING_TYPE_OPTIONS = [
    "Multiple Zones",
    "Single Zone",
    "Floor-wise",
    "Occupancy-based",
    "Other",
]

SEVERITY_LEGEND = {
    0: "No evidence",
    1: "Very weak",
    2: "Mild",
    3: "Moderate",
    4: "Strong",
    5: "Severe",
}

# ---------------------------------------------------------------------------
# Cached Telemetry Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_evaluate_sample_telemetry(file_path_str: str) -> Dict[str, Any]:
    df = pd.read_csv(file_path_str)
    return detect_conditions(df)


@st.cache_data
def evaluate_uploaded_telemetry(file_bytes: bytes) -> Dict[str, Any]:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return detect_conditions(df)


# ===========================================================================
# VIEW 1: INPUT & CONFIGURATION PAGE
# ===========================================================================
if st.session_state["page_view"] == "input":
    st.title("🏢 RetrofitIQ — Recommendation Engine")
    st.markdown(
        "Enter building characteristics, configure **zoning and occupancy**, review real-time "
        "sensor telemetry, and generate an executive retrofit plan."
    )

    saved = st.session_state.get("user_inputs", {})

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.subheader("1. Building Characteristics & Systems")

        bt_labels = [label for label, _ in BUILDING_TYPE_OPTIONS]
        default_bt_label = saved.get("building_type_label", bt_labels[0])
        bt_idx = bt_labels.index(default_bt_label) if default_bt_label in bt_labels else 0

        _bt_label = st.selectbox("Building Type", bt_labels, index=bt_idx)
        _bt_value = dict(BUILDING_TYPE_OPTIONS)[_bt_label]
        if _bt_value is None:
            building_type = st.text_input("Specify building type", saved.get("custom_building_type", ""))
            st.caption("Custom types are benchmarked against Office EUI distributions.")
        else:
            building_type = _bt_value

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            floor_area = st.number_input(
                "Gross Floor Area (m²)",
                min_value=100,
                value=int(saved.get("gross_floor_area_m2", 15000)),
                step=500,
            )
            n_floors = st.number_input(
                "Number of Floors",
                min_value=1,
                value=int(saved.get("n_floors", 4)),
            )
        with b_c2:
            building_age = st.number_input(
                "Building Age (years)",
                min_value=0,
                value=int(saved.get("building_age", 15)),
            )
            fan_options = ["Constant Speed", "Variable Speed (VFD)"]
            fan_idx = fan_options.index(saved.get("fan_type", "Constant Speed")) if saved.get("fan_type") in fan_options else 0
            fan_type = st.selectbox("Fan Type", fan_options, index=fan_idx)

        hvac_idx = HVAC_TYPE_OPTIONS.index(saved.get("hvac_type", HVAC_TYPE_OPTIONS[0])) if saved.get("hvac_type") in HVAC_TYPE_OPTIONS else 0
        _hvac_label = st.selectbox("Baseline HVAC System", HVAC_TYPE_OPTIONS, index=hvac_idx)
        if _hvac_label == "Other":
            baseline_hvac_type = st.text_input("Specify baseline HVAC type", saved.get("custom_hvac", "Custom HVAC"))
        else:
            baseline_hvac_type = _hvac_label

        dist_idx = HVAC_DISTRIBUTION_OPTIONS.index(saved.get("hvac_distribution", HVAC_DISTRIBUTION_OPTIONS[0])) if saved.get("hvac_distribution") in HVAC_DISTRIBUTION_OPTIONS else 0
        hvac_distribution = st.selectbox("HVAC Distribution Architecture", HVAC_DISTRIBUTION_OPTIONS, index=dist_idx)

        # -------------------------------------------------------------------
        # Zoning & Occupancy Inputs
        # -------------------------------------------------------------------
        st.markdown("---")
        st.subheader("2. Zoning & Occupancy Configuration")
        st.caption("Define spatial control granularity and occupant density variation.")

        zcol1, zcol2 = st.columns(2)
        with zcol1:
            n_zones = st.number_input(
                "Number of Thermal Zones",
                min_value=1,
                max_value=300,
                value=int(saved.get("n_zones", 4)),
                step=1,
                help="Total count of separately controlled temperature/airflow zones.",
            )
            z_idx = ZONING_TYPE_OPTIONS.index(saved.get("zoning_type", "Multiple Zones")) if saved.get("zoning_type") in ZONING_TYPE_OPTIONS else 0
            zoning_type = st.selectbox(
                "Zoning Type",
                ZONING_TYPE_OPTIONS,
                index=z_idx,
                help="Spatial division approach: Single, Multiple, Floor-wise, Occupancy-based, or Other.",
            )
            if zoning_type == "Other":
                custom_zoning = st.text_input("Specify Zoning Type", saved.get("custom_zoning", "Custom Zone Division"))
            else:
                custom_zoning = zoning_type

        with zcol2:
            occ_options = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
            occ_idx = occ_options.index(saved.get("occupancy_level", "Medium")) if saved.get("occupancy_level") in occ_options else 2
            occupancy_level = st.select_slider(
                "Occupancy Level & Dynamic Variation",
                options=occ_options,
                value=occ_options[occ_idx],
                help="Occupancy density. Higher levels increase value of Demand-Controlled Ventilation (DCV).",
            )
            st.info(
                f"**Zoning Setup:** {n_zones} zones ({custom_zoning}) with **{occupancy_level}** internal load variation."
            )

        # -------------------------------------------------------------------
        # Energy Baseline & Financial Parameters
        # -------------------------------------------------------------------
        st.markdown("---")
        st.subheader("3. Energy Baseline & Financial Parameters")

        energy_methods = ["Baseline EUI (kWh/m²/yr)", "Annual Energy Consumption (kWh/yr)"]
        em_idx = energy_methods.index(saved.get("energy_input_method", energy_methods[0])) if saved.get("energy_input_method") in energy_methods else 0
        energy_input_method = st.radio(
            "Energy Metric Input Method",
            energy_methods,
            index=em_idx,
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
            "n_zones": n_zones,
            "n_thermal_zones": n_zones,
            "zoning_type": custom_zoning,
            "occupancy_level": occupancy_level,
        }

        if energy_input_method == "Baseline EUI (kWh/m²/yr)":
            eui_input = st.number_input(
                "Baseline EUI (kWh/m²/yr)",
                min_value=1.0,
                max_value=2500.0,
                value=float(saved.get("eui", 150.0)),
                step=10.0,
                help="Energy Use Intensity in kWh/m²/year. Commercial office average is ~150.",
            )
            building_features["eui"] = float(eui_input)
        else:
            default_annual = float(saved.get("annual_energy", floor_area * 150.0))
            energy_input = st.number_input(
                "Annual Energy Consumption (kWh/yr)",
                min_value=100.0,
                value=default_annual,
                step=10000.0,
                help="Total annual electricity consumed in kWh.",
            )
            building_features["annual_energy"] = float(energy_input)

        # EUI Benchmarking feedback
        try:
            eui_result = classify_eui(building_features)
            b_class = eui_result["benchmark_class"]
            pct = int(round(eui_result["percentile"] * 100))
            suffix = "th" if 11 <= pct % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(pct % 10, "th")
            st.info(
                f"**Benchmarked EUI:** **{b_class}** ({pct}{suffix} percentile vs. peer {eui_result['building_type']} buildings) — "
                f"Calculated: **{round(eui_result['eui'], 1)} kWh/m²/yr**"
            )
        except Exception as exc:
            st.caption(f"EUI classification unavailable: {exc}")

        # Tariff and Available Budget Inputs
        f_c1, f_c2 = st.columns(2)
        with f_c1:
            tariff_val = st.number_input(
                "Electricity Tariff (INR/kWh)",
                min_value=0.1,
                value=float(saved.get("electricity_price", 9.0)),
                step=0.5,
                help="Commercial utility tariff (default 9.00 INR/kWh from EESL dataset).",
            )
            building_features["electricity_price"] = float(tariff_val)

        with f_c2:
            default_budget = float(saved.get("available_budget", 15000000.0))
            available_budget = st.number_input(
                "Available Investment Budget (INR)",
                min_value=0.0,
                value=default_budget,
                step=500000.0,
                help="Total capital funds available to upgrade the building. Used to determine financial feasibility.",
            )
            building_features["available_budget"] = float(available_budget)
            building_features["budget"] = float(available_budget)

    with col2:
        st.subheader("4. Telemetry Diagnostics & Detected Conditions")
        st.caption("Operational fault detection evaluated from sensor logs or declared building setup.")

        SAMPLE_DATASETS = {
            "Sample: Office HVAC testbed (bldg59)": PROJECT_ROOT / "data" / "processed" / "bldg59_master_hourly_clean.csv",
            "Sample: Multi-zone lab testbed": PROJECT_ROOT / "data" / "processed" / "TestBedClean.csv",
            "Upload sensor CSV": None,
        }

        saved_source = saved.get("telemetry_source", list(SAMPLE_DATASETS.keys())[0])
        src_idx = list(SAMPLE_DATASETS.keys()).index(saved_source) if saved_source in SAMPLE_DATASETS else 0

        telemetry_source = st.radio(
            "Select Telemetry Source",
            list(SAMPLE_DATASETS.keys()),
            index=src_idx,
        )

        detection_data = None
        if telemetry_source == "Upload sensor CSV":
            uploaded_file = st.file_uploader(
                "Upload Sensor Telemetry CSV",
                type=["csv"],
                help="Must contain temperature, damper, or zone readings.",
            )
            if uploaded_file is not None:
                try:
                    detection_data = evaluate_uploaded_telemetry(uploaded_file.getvalue())
                except Exception as exc:
                    st.error(f"Failed to evaluate uploaded telemetry: {exc}")
        else:
            sample_path = SAMPLE_DATASETS[telemetry_source]
            if sample_path.exists():
                detection_data = load_and_evaluate_sample_telemetry(str(sample_path))
            else:
                st.error(f"Sample file not found at: {sample_path}")

        # Calculate zoning severity from declared setup
        declared_pz = estimate_declared_zoning_severity(building_features)

        if detection_data is not None:
            scores = detection_data.get("scores", {})
            details = detection_data.get("details", {})

            # Take the max of telemetry zoning and declared zoning setup
            effective_pz = max(scores.get("poor_zoning", 0), declared_pz)

            st.markdown("##### Operational Severity Scores (0=No evidence ... 5=Severe)")
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric("Poor Zoning", f"{effective_pz} / 5", SEVERITY_LEGEND.get(effective_pz, ""))
                ef = scores.get("economizer_fault", 0)
                st.metric("Economizer Fault", f"{ef} / 5", SEVERITY_LEGEND.get(ef, ""))
            with mcol2:
                vi = scores.get("ventilation_imbalance", 0)
                st.metric("Ventilation Imbalance", f"{vi} / 5", SEVERITY_LEGEND.get(vi, ""))
                sm = scores.get("sensor_mismatch", 0)
                st.metric("Sensor Mismatch", f"{sm} / 5", SEVERITY_LEGEND.get(sm, ""))

            with st.expander("Diagnostic Telemetry Details", expanded=False):
                z_det = details.get("zoning", {})
                if "median_zone_spread_c" in z_det:
                    st.markdown(
                        f"• **Poor Zoning ({SEVERITY_LEGEND.get(effective_pz, '')}):** Median spread: {z_det.get('median_zone_spread_c')}°C "
                        f"(P90: {z_det.get('p90_zone_spread_c')}°C). Declared setup evaluation: {declared_pz}/5."
                    )
                else:
                    st.markdown(f"• **Poor Zoning:** Declared zoning setup severity = {declared_pz}/5 ({custom_zoning}, {n_zones} zones).")

                v_det = details.get("ventilation", {})
                st.markdown(f"• **Ventilation:** Violation rate: {v_det.get('violation_rate_pct', 0)}% ({v_det.get('total_violation_hours', 0)} hrs).")

                e_det = details.get("economizer", {})
                st.markdown(f"• **Economizer:** Fault rate: {e_det.get('fault_rate_pct', 0)}% ({e_det.get('total_fault_hours', 0)} hrs).")

                s_det = details.get("sensor_mismatch", {})
                st.markdown(f"• **Sensor Mismatch:** Mismatch rate: {s_det.get('mismatch_rate_pct', 0)}%.")

            inefficiency_flags = {
                **scores,
                "poor_zoning": effective_pz,
            }
        else:
            inefficiency_flags = {
                "poor_zoning": declared_pz,
                "ventilation_imbalance": 0,
                "economizer_fault": 0,
                "sensor_mismatch": 0,
            }
            st.info(f"Declared zoning setup reflects an initial severity of **{declared_pz} / 5** ({SEVERITY_LEGEND.get(declared_pz, '')}).")

    # -----------------------------------------------------------------------
    # Get Recommendations Button
    # -----------------------------------------------------------------------
    st.markdown("---")
    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        get_recs = st.button("🚀 Get Recommendations", type="primary", disabled=not weights_valid, use_container_width=True)

    if get_recs:
        # Save all current inputs to session_state so they are preserved
        st.session_state["user_inputs"] = {
            "building_type_label": _bt_label,
            "custom_building_type": building_type if _bt_value is None else "",
            "gross_floor_area_m2": floor_area,
            "building_age": building_age,
            "hvac_type": baseline_hvac_type,
            "custom_hvac": baseline_hvac_type if _hvac_label == "Other" else "",
            "hvac_distribution": hvac_distribution,
            "n_floors": n_floors,
            "fan_type": fan_type,
            "n_zones": n_zones,
            "zoning_type": zoning_type,
            "custom_zoning": custom_zoning if zoning_type == "Other" else "",
            "occupancy_level": occupancy_level,
            "energy_input_method": energy_input_method,
            "eui": float(building_features.get("eui", 150.0)),
            "annual_energy": float(building_features.get("annual_energy", floor_area * 150.0)),
            "electricity_price": tariff_val,
            "available_budget": available_budget,
            "telemetry_source": telemetry_source,
        }

        results_df = recommend_retrofits(
            building_features=building_features,
            inefficiency_flags=inefficiency_flags,
            weights=weights,
            show_all=True,
            budget_inr=available_budget,
        )
        st.session_state["results_df"] = results_df
        st.session_state["building_features"] = building_features
        st.session_state["inefficiency_flags"] = inefficiency_flags
        st.session_state["weights"] = weights
        st.session_state["budget_inr"] = available_budget
        st.session_state["page_view"] = "dashboard"
        st.rerun()

    if not weights_valid:
        st.warning("Adjust sidebar weights to sum to exactly 1.00 to generate recommendations.")


# ===========================================================================
# VIEW 2: DEDICATED DASHBOARD & RESULTS PAGE
# ===========================================================================
elif st.session_state["page_view"] == "dashboard":
    results_df: pd.DataFrame = st.session_state.get("results_df")
    b_features: Dict[str, Any] = st.session_state.get("building_features", {})
    budget: float = float(st.session_state.get("budget_inr", 0.0))

    # Top action bar with back button that preserves inputs
    top_nav_col1, top_nav_col2 = st.columns([3, 7])
    with top_nav_col1:
        if st.button("← Back to Building Inputs", type="secondary", use_container_width=True):
            st.session_state["page_view"] = "input"
            st.rerun()

    st.title("Retrofit Recommendation & Executive Dashboard")

    # Building Context Chips
    z_label = f"{b_features.get('n_zones', 4)} Zones ({b_features.get('zoning_type', 'Multi-zone')})"
    st.caption(
        f"**Facility:** {b_features.get('building_type', 'Office')} | **Area:** {b_features.get('gross_floor_area_m2', 0):,.0f} m² | "
        f"**Floors:** {b_features.get('n_floors', 1)} | **Zoning:** {z_label} | "
        f"**Occupancy:** {b_features.get('occupancy_level', 'Medium')} | "
        f"**Available Budget:** ₹{budget:,.0f} | **Tariff:** ₹{b_features.get('electricity_price', 9.0):.2f}/kWh"
    )

    if results_df is not None and len(results_df) > 0:
        top_option = results_df.iloc[0]

        # -------------------------------------------------------------------
        # Executive KPI Cards Row
        # -------------------------------------------------------------------
        st.markdown("---")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.metric(
                "Top Recommendation",
                f"{top_option['Retrofit Option']}",
                f"{top_option['Recommendation']}",
            )
        with kpi2:
            st.metric(
                "Grade & Final Score",
                f"{top_option['Grade']}",
                f"{top_option['Final Score']:.2f} / 5.00",
            )
        with kpi3:
            st.metric(
                "Peak Annual Savings",
                f"₹{top_option['Annual Savings (INR)']:,.0f}",
                f"{top_option['Savings %']:.1f}% Energy Reduction",
            )
        with kpi4:
            st.metric(
                "Fastest Payback",
                f"{results_df['Payback (Years)'].min():.2f} yrs",
                "Full Capex Recovery",
            )
        with kpi5:
            surplus_or_short = budget - top_option["Upgrade Cost (INR)"]
            if surplus_or_short >= 0:
                st.metric("Budget Feasibility", "Feasible", f"Surplus: ₹{surplus_or_short:,.0f}")
            else:
                st.metric("Budget Feasibility", "Budget Deficit", f"Shortfall: ₹{abs(surplus_or_short):,.0f}")

        # -------------------------------------------------------------------
        # Top Recommended Spotlight Hero Card
        # -------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Top Recommended Retrofit Spotlight")

        hero_col1, hero_col2 = st.columns([1.2, 0.8])
        with hero_col1:
            st.success(
                f"### {top_option['Recommendation']} — **{top_option['Retrofit Option']}** ({top_option['Grade']})\n\n"
                f"{top_option['Explanation']['verdict_explanation']}\n\n"
                f"• **Technical Performance:** Energy: **{top_option['Energy']}/5** | Comfort: **{top_option['Comfort']}/5** | "
                f"Cost-Benefit: **{top_option['Cost Benefit']}/5** | Sustainability: **{top_option['Sustainability']}/5** | "
                f"Maintenance: **{top_option['Maintenance']}/5**\n\n"
                f"• **Final Score:** **{top_option['Final Score']:.3f} / 5.000**"
            )
        with hero_col2:
            st.info(
                f"#### Financial & Investment Metrics\n\n"
                f"- **Upgrade Cost Required:** **₹{top_option['Upgrade Cost (INR)']:,.0f}**\n"
                f"- **Available Budget:** **₹{budget:,.0f}**\n"
                f"- **Annual Utility Savings:** **₹{top_option['Annual Savings (INR)']:,.0f} / year**\n"
                f"- **Payback Period:** **{top_option['Payback (Years)']:.2f} years**\n"
                f"- **Budget Feasibility:** {top_option['Budget Feasibility']}"
            )

        # -------------------------------------------------------------------
        # Visual Dashboards: Multi-Axis & Financials
        # -------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Multi-Option Performance & Financial Dashboards")
        st.caption("Direct visual comparison across all 5 catalog candidates.")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### Multi-Axis Performance Scores (Scale 1–5)")
            melted_scores = []
            for _, r in results_df.iterrows():
                for axis in ["Energy", "Comfort", "Cost Benefit", "Sustainability", "Maintenance"]:
                    melted_scores.append({
                        "Retrofit": r["Retrofit Option"],
                        "Axis": axis,
                        "Score": r[axis],
                    })
            scores_df = pd.DataFrame(melted_scores)
            axis_chart = (
                alt.Chart(scores_df)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Axis:N", title="Evaluation Axis", sort=None),
                    y=alt.Y("Score:Q", title="Score (1 to 5)", scale=alt.Scale(domain=[0, 5])),
                    color=alt.Color("Retrofit:N", title="Retrofit Option", scale=alt.Scale(scheme="category10")),
                    xOffset="Retrofit:N",
                    tooltip=["Retrofit", "Axis", "Score"],
                )
                .properties(height=320)
            )
            st.altair_chart(axis_chart, use_container_width=True)

        with chart_col2:
            st.markdown("##### Financial Comparison: Upgrade Cost vs. Annual Savings (INR)")
            fin_melted = []
            for _, r in results_df.iterrows():
                fin_melted.append({
                    "Retrofit": r["Retrofit Option"],
                    "Metric": "Annual Savings (INR)",
                    "Amount": r["Annual Savings (INR)"],
                })
                fin_melted.append({
                    "Retrofit": r["Retrofit Option"],
                    "Metric": "Upgrade Cost (INR)",
                    "Amount": r["Upgrade Cost (INR)"],
                })
            fin_melted_df = pd.DataFrame(fin_melted)

            fin_chart = (
                alt.Chart(fin_melted_df)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Retrofit:N", title="Retrofit Option", sort=None),
                    y=alt.Y("Amount:Q", title="Amount (INR)"),
                    color=alt.Color("Metric:N", title="Financial Metric", scale=alt.Scale(scheme="set2")),
                    xOffset="Metric:N",
                    tooltip=["Retrofit", "Metric", alt.Tooltip("Amount:Q", format=",.0f")],
                )
                .properties(height=320)
            )
            st.altair_chart(fin_chart, use_container_width=True)

        # -------------------------------------------------------------------
        # Scoring Model Explanations (Detailed Why?)
        # -------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Scoring Model Explanations — Why Did Each Option Receive Its Score?")
        st.caption(
            "Granular engineering justifications across Energy, Comfort, Cost-Benefit, Sustainability, "
            "Maintenance, and the final weighted arithmetic formula."
        )

        for _, row in results_df.iterrows():
            opt = row["Retrofit Option"]
            exp = row["Explanation"]
            grade = row["Grade"]
            fscore = row["Final Score"]
            tier = row["Recommendation"]

            with st.expander(f"**{opt}** — Score: `{fscore:.3f}/5` | {grade} — *{tier}*", expanded=(opt == top_option["Retrofit Option"])):
                e_tab, c_tab, cb_tab, s_tab, m_tab, f_tab = st.tabs([
                    f"Energy ({row['Energy']}/5)",
                    f"Comfort ({row['Comfort']}/5)",
                    f"Cost Benefit ({row['Cost Benefit']}/5)",
                    f"Sustainability ({row['Sustainability']}/5)",
                    f"Maintenance ({row['Maintenance']}/5)",
                    "Math & Formula",
                ])

                with e_tab:
                    st.markdown(f"**Energy Rationale:**\n\n{exp['energy_explanation']}")
                    st.markdown(f"• **Predicted Energy Savings:** `{row['Savings %']:.1f}%`\n• **Annual Electricity Saved:** `{round(row['Savings %']*0.01 * (b_features.get('eui', 150)*b_features.get('gross_floor_area_m2', 15000))):,.0f} kWh/year`")

                with c_tab:
                    st.markdown(f"**Comfort Rationale:**\n\n{exp['comfort_explanation']}")

                with cb_tab:
                    st.markdown(f"**Cost Benefit Rationale:**\n\n{exp['cost_benefit_explanation']}")
                    st.markdown(
                        f"• **Annual Savings:** `₹{row['Annual Savings (INR)']:,.0f}`\n"
                        f"• **Upgrade Cost Required:** `₹{row['Upgrade Cost (INR)']:,.0f}`\n"
                        f"• **Payback Period:** `{row['Payback (Years)']:.2f} years`"
                    )

                with s_tab:
                    st.markdown(f"**Sustainability Rationale:**\n\n{exp['sustainability_explanation']}")

                with m_tab:
                    st.markdown(f"**Maintenance Rationale:**\n\n{exp['maintenance_explanation']}")

                with f_tab:
                    st.markdown("**Weighted Final Score Arithmetic:**")
                    st.code(exp["formula_explanation"], language="text")
                    st.markdown(f"**Score-to-Grade Mapping:** Final Score `{fscore:.3f}` maps directly to **{grade}** ({tier}).")

        # -------------------------------------------------------------------
        # Comprehensive Recommendation & Financial Table
        # Color mapped from Dark Green to Neutral to Dark Red (NO EMOJIS)
        # -------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Comprehensive Retrofit Ranking & Financial Feasibility Table")
        st.caption(
            "Showing **all 5 catalog options** styled from dark green (best) to neutral to dark red. "
            "All upgrade costs and feasibility depend directly on capital requirements."
        )

        display_df = results_df[[
            "Retrofit Option",
            "Recommendation",
            "Grade",
            "Final Score",
            "Energy",
            "Comfort",
            "Cost Benefit",
            "Sustainability",
            "Maintenance",
            "Savings %",
            "Annual Savings (INR)",
            "Upgrade Cost (INR)",
            "Budget Balance (INR)",
            "Budget Feasibility",
            "Payback (Years)",
        ]].copy()

        # Sort: feasible items first (by Final Score desc), deficit items last (by Final Score desc)
        display_df["_is_deficit"] = display_df["Budget Feasibility"].str.contains("Deficit", case=False, na=False).astype(int)
        display_df = display_df.sort_values(
            ["_is_deficit", "Final Score"], ascending=[True, False]
        ).reset_index(drop=True)
        display_df = display_df.drop(columns=["_is_deficit"])

        # Format columns for display
        formatted_df = display_df.copy()
        formatted_df["Annual Savings (INR)"] = formatted_df["Annual Savings (INR)"].apply(lambda x: f"₹{x:,.0f}")
        formatted_df["Upgrade Cost (INR)"] = formatted_df["Upgrade Cost (INR)"].apply(lambda x: f"₹{x:,.0f}")
        formatted_df["Budget Balance (INR)"] = formatted_df["Budget Balance (INR)"].apply(lambda x: f"+₹{x:,.0f}" if x >= 0 else f"-₹{abs(x):,.0f}")
        formatted_df["Savings %"] = formatted_df["Savings %"].apply(lambda x: f"{x:.1f}%")
        formatted_df["Payback (Years)"] = formatted_df["Payback (Years)"].apply(lambda x: f"{x:.2f} yrs")
        formatted_df["Final Score"] = formatted_df["Final Score"].apply(lambda x: f"{x:.3f}")

        # Color-styling helpers: TEXT color only (Dark green -> neutral -> dark red)
        def color_recommendation(val):
            v = str(val).strip()
            if v == "Highly Recommended":
                return "color: #1b5e20; font-weight: bold;"
            elif v == "Recommended":
                return "color: #2e7d32; font-weight: bold;"
            elif v == "Consider":
                return "color: #f57f17; font-weight: bold;"
            elif v == "Low Priority":
                return "color: #e65100; font-weight: bold;"
            elif v == "Not Recommended":
                return "color: #b71c1c; font-weight: bold;"
            return ""

        def color_grade(val):
            v = str(val).strip()
            if v == "Grade A":
                return "color: #1b5e20; font-weight: bold;"
            elif v == "Grade B":
                return "color: #2e7d32; font-weight: bold;"
            elif v == "Grade C":
                return "color: #f57f17; font-weight: bold;"
            elif v == "Grade D":
                return "color: #e65100; font-weight: bold;"
            elif v == "Grade F":
                return "color: #b71c1c; font-weight: bold;"
            return ""

        def color_score(val):
            try:
                s = float(val)
                if s >= 4.0:
                    return "color: #1b5e20; font-weight: bold;"
                elif s >= 3.0:
                    return "color: #2e7d32; font-weight: bold;"
                elif s >= 2.0:
                    return "color: #f57f17; font-weight: bold;"
                elif s >= 1.0:
                    return "color: #e65100; font-weight: bold;"
                else:
                    return "color: #b71c1c; font-weight: bold;"
            except (ValueError, TypeError):
                return ""

        def color_feasibility(val):
            v = str(val)
            if "Feasible" in v or "High ROI" in v:
                return "color: #1b5e20; font-weight: bold;"
            elif "Deficit" in v or "Shortfall" in v:
                return "color: #b71c1c; font-weight: bold;"
            return ""

        styled_df = (
            formatted_df.style
            .map(color_recommendation, subset=["Recommendation"])
            .map(color_grade, subset=["Grade"])
            .map(color_score, subset=["Final Score", "Energy", "Comfort", "Cost Benefit", "Sustainability", "Maintenance"])
            .map(color_feasibility, subset=["Budget Feasibility"])
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
        )

        # Download CSV report
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Recommendations Report (CSV)",
            data=csv_data,
            file_name="Retrofit_Recommendations_Report.csv",
            mime="text/csv",
        )

        # -------------------------------------------------------------------
        # Navigation Options: Edit Inputs vs Configure Another Building
        # -------------------------------------------------------------------
        st.markdown("---")
        b_col1, b_col2, b_col3 = st.columns([2, 2.5, 4])
        with b_col1:
            if st.button("✏️ Edit Current Inputs", type="primary", use_container_width=True):
                # Keeps st.session_state["user_inputs"] exactly as typed!
                st.session_state["page_view"] = "input"
                st.rerun()
        with b_col2:
            if st.button("🆕 Configure Another Building", type="secondary", use_container_width=True):
                # Clears user_inputs so fresh default values appear!
                st.session_state["user_inputs"] = {}
                st.session_state["results_df"] = None
                st.session_state["page_view"] = "input"
                st.rerun()
    else:
        st.warning("No recommendations calculated. Click 'Back to Building Inputs' to configure.")

