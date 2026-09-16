"""
Interface
=========
Person D — Retrofit Recommendation Engine

Streamlit app: enter building characteristics, assess operational HVAC conditions,
adjust the 5 axis weights, and view an executive dashboard with clean 1-5 scale grades (A-F),
detailed scoring explanations, visual charts, and a complete financial
feasibility table with dark green to dark red color mapping.

Run with:
    streamlit run "src/Person D/app.py"
"""

from pathlib import Path
from typing import Any, Dict, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from llm_explainer import ask_retrofit_ai, build_retrofit_context

from prediction.monitoring_engine import MonitoringEngine
from prediction.realtime_simulator import RealtimeSimulator

from combiner import (
    DEFAULT_WEIGHTS,
    classify_eui,
    recommend_retrofits,
    generate_retrofit_packages,
)



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

if "show_individual_analysis" not in st.session_state:
    st.session_state["show_individual_analysis"] = False

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
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Monitoring")
if st.sidebar.button("Open Real-Time Energy Monitoring", use_container_width=True):
    st.session_state["page_view"] = "monitoring"
    st.rerun()

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


SEVERITY_LEGEND = {
    0: "No evidence",
    1: "Very weak",
    2: "Mild",
    3: "Moderate",
    4: "Strong",
    5: "Severe",
}


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
        # Energy Baseline & Financial Parameters
        # -------------------------------------------------------------------
        st.markdown("---")
        st.subheader("2. Energy Baseline & Financial Parameters")

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
        }

        if energy_input_method == "Baseline EUI (kWh/m²/yr)":
            eui_input = st.number_input(
                "Baseline EUI (kWh/m²/yr)",
                min_value=1.0,
                max_value=2500.0,
                value=float(saved.get("eui") if saved.get("eui") is not None else 150.0),
                step=10.0,
                help="Energy Use Intensity in kWh/m²/year. Commercial office average is ~150.",
            )
            building_features["eui"] = float(eui_input)
            building_features["energy_input_method"] = "eui"
        else:
            default_annual = float(saved.get("annual_energy") if saved.get("annual_energy") is not None else floor_area * 150.0)
            energy_input = st.number_input(
                "Annual Energy Consumption (kWh/yr)",
                min_value=100.0,
                value=default_annual,
                step=10000.0,
                help="Total annual electricity consumed in kWh.",
            )
            building_features["annual_energy"] = float(energy_input)
            building_features["energy_input_method"] = "annual_energy"

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
        st.subheader("3. Operational Condition Assessment")
        st.caption(
            "Rate the current severity of each HVAC operating condition from 0 (no evidence) to 5 (severe). "
            "These assessments directly inform retrofit relevance and scoring."
        )

        def condition_slider(label: str, key: str, help_text: str) -> int:
            return int(
                st.slider(
                    label,
                    min_value=0,
                    max_value=5,
                    value=int(saved.get(key, 0)),
                    step=1,
                    help=help_text,
                )
            )

        operational_scores = {}

        operational_scores["poor_zoning"] = condition_slider(
            "Poor Thermal Zoning",
            "poor_zoning",
            "Rate how strongly different rooms or zones experience uneven heating/cooling. "
            "0 = no issue, 5 = severe and persistent zone imbalance.",
        )
        operational_scores["ventilation_imbalance"] = condition_slider(
            "Ventilation Imbalance",
            "ventilation_imbalance",
            "Rate how strongly the building appears to have too much or too little outdoor-air ventilation. "
            "0 = no issue, 5 = severe imbalance.",
        )
        operational_scores["economizer_fault"] = condition_slider(
            "Economizer Fault",
            "economizer_fault",
            "Rate the likelihood/severity of a problem with the economizer or free-cooling controls. "
            "0 = no evidence, 5 = severe fault.",
        )
        operational_scores["sensor_mismatch"] = condition_slider(
            "Sensor / Measurement Mismatch",
            "sensor_mismatch",
            "Rate the severity of inconsistent, drifting, or unreliable HVAC sensor readings. "
            "0 = no evidence, 5 = severe mismatch.",
        )

        st.markdown("##### Current Operational Severity")
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            pz = operational_scores["poor_zoning"]
            st.metric("Poor Zoning", f"{pz} / 5", SEVERITY_LEGEND.get(pz, ""))
            ef = operational_scores["economizer_fault"]
            st.metric("Economizer Fault", f"{ef} / 5", SEVERITY_LEGEND.get(ef, ""))
        with mcol2:
            vi = operational_scores["ventilation_imbalance"]
            st.metric("Ventilation Imbalance", f"{vi} / 5", SEVERITY_LEGEND.get(vi, ""))
            sm = operational_scores["sensor_mismatch"]
            st.metric("Sensor Mismatch", f"{sm} / 5", SEVERITY_LEGEND.get(sm, ""))

        inefficiency_flags = operational_scores

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
            "energy_input_method": energy_input_method,
            "eui": float(building_features.get("eui")) if "eui" in building_features else None,
            "annual_energy": float(building_features.get("annual_energy")) if "annual_energy" in building_features else None,
            "electricity_price": tariff_val,
            "available_budget": available_budget,
            **operational_scores,
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
# VIEW 2: REAL-TIME ENERGY MONITORING PAGE
# ===========================================================================
elif st.session_state["page_view"] == "monitoring":
    st.title("📡 RetrofitIQ — Real-Time Energy Monitoring")
    st.caption(
        "Telemetry Replay Prototype — the cleaned historical building dataset is replayed as "
        "incoming readings. This demonstrates the monitoring pipeline; it is not a live sensor feed."
    )

    # Keep the trained monitoring engine inside this user's Streamlit session.
    if "monitoring_engine" not in st.session_state:
        with st.spinner("Loading and training the energy monitoring model..."):
            engine = MonitoringEngine()
            engine.load()
            engine.train()
        st.session_state["monitoring_engine"] = engine
        st.session_state["monitoring_history"] = []
        st.session_state["monitoring_simulator"] = RealtimeSimulator(
            data=engine.data, engine=engine
        )

    engine = st.session_state["monitoring_engine"]
    simulator = st.session_state["monitoring_simulator"]

    nav1, nav2, nav3 = st.columns([1.5, 1.5, 5])
    with nav1:
        if st.button("← Recommendation", use_container_width=True):
            st.session_state["page_view"] = "input"
            st.rerun()
    with nav2:
        if st.button("🔄 Reset Replay", use_container_width=True):
            simulator.reset()
            st.session_state["monitoring_history"] = []
            st.rerun()

    st.markdown("---")
    st.subheader("Model Performance")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("MAE", f"{engine.predictor.metrics.get('mae', 0):.2f} kW")
    with m2:
        st.metric("R²", f"{engine.predictor.metrics.get('r2', 0):.3f}")
    with m3:
        st.metric("Tariff", f"₹{engine.predictor.tariff:.2f}/kWh")

    st.info(
        "The monitoring model estimates expected power from operating conditions. "
        "Actual power is then compared with expected power to estimate deviation and financial impact."
    )

    st.subheader("Telemetry Replay")
    progress = simulator.get_progress()
    st.progress(min(progress, 1.0))
    st.caption(f"Replay progress: {simulator.current_index:,} / {len(simulator.data):,} readings")

    r1, r2 = st.columns([1.5, 5])
    with r1:
        next_reading = st.button(
            "▶ Next Reading",
            type="primary",
            disabled=not simulator.has_next(),
            use_container_width=True,
        )
    with r2:
        if simulator.has_next():
            preview_row = simulator.data.iloc[simulator.current_index]
            preview_time = preview_row.get("date", "")
            st.caption(f"Next telemetry timestamp: **{preview_time}**")
        else:
            st.caption("Replay complete. Press Reset Replay to start again.")

    if next_reading:
        result = simulator.next_reading()
        if result is not None:
            st.session_state["monitoring_history"].append(result)

    history = st.session_state.get("monitoring_history", [])
    if history:
        current = history[-1]

        st.markdown("---")
        st.subheader("Current Energy Condition")

        # Defensive handling in case an older replay result contains NO_DATA.
        actual_power = current.get("actual_power_kw")
        expected_power = current.get("expected_power_kw")
        deviation = current.get("power_deviation_percent")
        expected_cost = current.get("expected_cost_inr")
        actual_cost = current.get("actual_cost_inr")
        excess = current.get("excess_cost_inr")

        if current.get("status") == "NO_DATA" or actual_power is None:
            st.warning("This telemetry reading does not contain a valid power measurement.")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Actual Power", "N/A")
            with k2:
                st.metric("Expected Power", "N/A")
            with k3:
                st.metric("Deviation", "N/A")
            with k4:
                st.metric("Status", current.get("status", "NO_DATA"))
        else:
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Actual Power", f"{float(actual_power):.2f} kW")
            with k2:
                st.metric("Expected Power", f"{float(expected_power):.2f} kW")
            with k3:
                st.metric("Deviation", f"{float(deviation):+.2f}%")
            with k4:
                st.metric("Status", current.get("status", "NORMAL"))

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Expected Hourly Cost", f"₹{float(expected_cost):,.2f}")
            with c2:
                st.metric("Actual Hourly Cost", f"₹{float(actual_cost):,.2f}")
            with c3:
                label = "Excess Cost" if float(excess) >= 0 else "Saved Cost"
                st.metric(
                    label,
                    f"₹{abs(float(excess)):,.2f}",
                    delta=f"₹{float(excess):+,.2f}",
                )

        alert = current["alert"]
        severity = alert.get("severity", "NORMAL")
        if severity == "CRITICAL":
            st.error(f"**{alert['title']}**\n\n{alert['message']}\n\n**Action:** {alert['action']}")
        elif severity == "WARNING":
            st.warning(f"**{alert['title']}**\n\n{alert['message']}\n\n**Action:** {alert['action']}")
        elif severity == "WATCH":
            st.warning(f"**{alert['title']}**\n\n{alert['message']}\n\n**Action:** {alert['action']}")
        elif severity == "INFO":
            st.info(f"**{alert['title']}**\n\n{alert['message']}\n\n**Action:** {alert['action']}")
        elif severity == "NO_DATA":
            st.error(f"**{alert['title']}**\n\n{alert['message']}\n\n**Action:** {alert['action']}")
        else:
            st.success(f"**{alert['title']}**\n\n{alert['message']}\n\n**Action:** {alert['action']}")

        if current.get("persistent"):
            st.warning(
                f"Persistent deviation detected: {current['consecutive_abnormal']} consecutive abnormal readings."
            )

        st.markdown("---")
        st.subheader("Actual vs Expected Energy Trend")
        chart_rows = []
        for item in history:
            if item["status"] == "NO_DATA":
                continue
            chart_rows.extend([
                {"Timestamp": item["timestamp"], "Metric": "Actual Power", "Power (kW)": item["actual_power_kw"]},
                {"Timestamp": item["timestamp"], "Metric": "Expected Power", "Power (kW)": item["expected_power_kw"]},
            ])
        chart_df = pd.DataFrame(chart_rows)
        if not chart_df.empty:
            chart = (
                alt.Chart(chart_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Timestamp:T", title="Time"),
                    y=alt.Y("Power (kW):Q", title="Power (kW)"),
                    color=alt.Color("Metric:N", title="Metric"),
                    tooltip=[
                        alt.Tooltip("Timestamp:T", title="Time"),
                        alt.Tooltip("Metric:N", title="Metric"),
                        alt.Tooltip("Power (kW):Q", title="Power", format=".2f"),
                    ],
                )
                .properties(height=400)
            )
            st.altair_chart(chart, use_container_width=True)

        st.subheader("Monitoring History")
        history_df = pd.DataFrame(history)
        history_display = history_df[[
            "timestamp", "actual_power_kw", "expected_power_kw",
            "power_deviation_percent", "status", "excess_cost_inr",
            "persistent",
        ]].copy()
        history_display.columns = [
            "Timestamp", "Actual kW", "Expected kW", "Deviation %",
            "Status", "Excess / Saved Cost (INR)", "Persistent",
        ]
        st.dataframe(history_display, use_container_width=True, hide_index=True)
    else:
        st.info("Press **Next Reading** to begin the telemetry replay.")


# ===========================================================================
# VIEW 3: DEDICATED DASHBOARD & RESULTS PAGE
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
    st.caption(
        f"**Facility:** {b_features.get('building_type', 'Office')} | **Area:** {b_features.get('gross_floor_area_m2', 0):,.0f} m² | "
        f"**Floors:** {b_features.get('n_floors', 1)} | "
        f"**Available Budget:** ₹{budget:,.0f} | **Tariff:** ₹{b_features.get('electricity_price', 9.0):.2f}/kWh"
    )

    if results_df is not None and len(results_df) > 0:
        top_option = results_df.iloc[0]

        # -------------------------------------------------------------------
        # Budget-Constrained Retrofit Packages
        # -------------------------------------------------------------------
        package_df = generate_retrofit_packages(
            results_df=results_df,
            building_features=b_features,
            budget_inr=budget,
            minimum_score=3.0,
            include_single_options=False,
        )
        st.session_state["package_df"] = package_df

        st.markdown("---")
        st.subheader("Retrofit Packages Within Your Budget")
        st.caption(
            "Click a package row to view it in detail. Packages contain only Grade A/B retrofits "
            "and remain within the available CAPEX."
        )

        if not package_df.empty:
            package_display = package_df[[
                "Package", "Package Grade", "Package Score", "Package CAPEX (INR)",
                "Annual Savings (INR)", "Payback (Years)",
            ]].copy()
            package_display["Package Score"] = package_display["Package Score"].map(lambda x: f"{x:.2f}")
            package_display["Package CAPEX (INR)"] = package_display["Package CAPEX (INR)"].map(lambda x: f"₹{x:,.0f}")
            package_display["Annual Savings (INR)"] = package_display["Annual Savings (INR)"].map(lambda x: f"₹{x:,.0f}/yr")
            package_display["Payback (Years)"] = package_display["Payback (Years)"].map(lambda x: f"{x:.2f} yrs" if pd.notna(x) else "—")

            st.markdown("**Available packages — click a row to elaborate:**")
            package_event = st.dataframe(
                package_display,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                key="package_table",
            )
            selected_rows = package_event.selection.rows if hasattr(package_event, "selection") else []
            selected_idx = int(selected_rows[0]) if selected_rows else 0
            pkg = package_df.iloc[selected_idx]

            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("Package Grade", pkg["Package Grade"], f"Score {pkg['Package Score']:.2f} / 5")
            with p2:
                st.metric("Package CAPEX", f"₹{pkg['Package CAPEX (INR)']:,.0f}", f"₹{pkg['Budget Remaining (INR)']:,.0f} remaining")
            with p3:
                st.metric("Annual Savings", f"₹{pkg['Annual Savings (INR)']:,.0f}", f"{pkg['Combined Savings %']:.1f}% energy")
            with p4:
                st.metric("Package Payback", f"{pkg['Payback (Years)']:.2f} yrs")

            st.success(
                f"### {pkg['Package']}\n\n"
                f"**Grade {pkg['Package Grade'][-1]} — {pkg['Package Score']:.2f}/5**  ·  "
                f"**₹{pkg['Package CAPEX (INR)']:,.0f}** of **₹{budget:,.0f}** available CAPEX"
            )

            condition_labels = {
                "poor_zoning": "thermal zoning",
                "ventilation_imbalance": "ventilation balance",
                "economizer_fault": "economizer performance",
                "sensor_mismatch": "sensor/measurement consistency",
            }
            active_conditions = [
                label for key, label in condition_labels.items()
                if float(b_features.get(key, 0) or 0) >= 2
            ]
            if active_conditions:
                if len(active_conditions) == 1:
                    condition_text = active_conditions[0]
                elif len(active_conditions) == 2:
                    condition_text = f"{active_conditions[0]} and {active_conditions[1]}"
                else:
                    condition_text = ", ".join(active_conditions[:-1]) + f", and {active_conditions[-1]}"
                context_sentence = f"Your operational assessment indicates issues around **{condition_text}**."
            else:
                context_sentence = "Your operational assessment does not show a dominant issue, so this package is driven mainly by overall energy and financial performance."

            retrofit_names = {
                "Smart_Controls": "Smart Controls",
                "AHU_VFD": "AHU VFD",
                "DCV": "Demand-Controlled Ventilation",
                "Chiller_Optimization": "Chiller Optimization",
                "Zoning_Optimization": "Zoning Optimization",
            }
            readable = [retrofit_names.get(r, r) for r in pkg["Retrofits"]]
            if len(readable) == 1:
                measure_text = readable[0]
            elif len(readable) == 2:
                measure_text = f"{readable[0]} and {readable[1]}"
            else:
                measure_text = ", ".join(readable[:-1]) + f", and {readable[-1]}"

            st.markdown("### Why this package?")
            st.markdown(
                f"{context_sentence} The **{measure_text}** combination was selected because "
                f"every measure is rated Grade A or B and the complete package remains within your "
                f"available CAPEX. Together, the package is projected to save approximately "
                f"**{pkg['Combined Savings %']:.1f}% of annual energy**, worth about "
                f"**₹{pkg['Annual Savings (INR)']:,.0f} per year**, with an estimated payback of "
                f"**{pkg['Payback (Years)']:.2f} years**."
            )
            st.caption("Select another package in the table above to update the details and explanation.")

            # -------------------------------------------------------------------
            # RetrofitIQ AI — grounded explanation layer
            # -------------------------------------------------------------------
            st.markdown("### 🤖 Ask RetrofitIQ AI")
            st.caption(
                "Ask questions about the current building, selected package, scores, "
                "costs, savings, diagnostics, or trade-offs. The LLM explains the "
                "calculated RetrofitIQ results; it does not change them."
            )

            ai_question = st.text_input(
                "Your question",
                placeholder="Why was this package selected?",
                key="retrofit_ai_question",
            )
            ai_col1, ai_col2 = st.columns([1, 4])
            with ai_col1:
                ask_ai = st.button(
                    "Ask AI",
                    type="primary",
                    use_container_width=True,
                    key="ask_retrofit_ai",
                )

            if ask_ai:
                if not ai_question.strip():
                    st.warning("Enter a question first.")
                else:
                    selected_package_context = pkg.to_dict()
                    context = build_retrofit_context(
                        building_features=b_features,
                        inefficiency_flags=st.session_state.get("inefficiency_flags", {}),
                        weights=st.session_state.get("weights", {}),
                        package=selected_package_context,
                        results_df=results_df,
                    )
                    with st.spinner("RetrofitIQ AI is explaining the current results..."):
                        try:
                            answer = ask_retrofit_ai(ai_question, context)
                            st.session_state["retrofit_ai_answer"] = answer
                        except Exception as exc:
                            st.error(f"Could not contact RetrofitIQ AI: {exc}")

            if st.session_state.get("retrofit_ai_answer"):
                st.markdown("**RetrofitIQ AI response**")
                st.info(st.session_state["retrofit_ai_answer"])
                st.caption(
                    "Grounding rule: RetrofitIQ AI explains the existing engineering/ML "
                    "results and cannot modify the calculated score or recommendation."
                )
        else:
            st.info(
                "No combination of Grade A/B retrofits currently fits within the available CAPEX. "
                "Try increasing the budget."
            )

        # -------------------------------------------------------------------
        # Package-Level Decision Charts
        # -------------------------------------------------------------------
        if not package_df.empty:
            st.markdown("---")
            st.subheader("Package-Level Decision Comparison")
            st.caption(
                "These charts compare complete retrofit packages only. Package rank follows the "
                "RetrofitIQ package score, not the score of any individual measure."
            )

            package_chart_df = package_df.copy().reset_index(drop=True)
            package_chart_df["Rank"] = [f"Rank {i + 1}" for i in range(len(package_chart_df))]
            package_chart_df["Package Label"] = package_chart_df.apply(
                lambda r: f"Rank {r['Rank'].split()[-1]} — {r['Package']}", axis=1
            )

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("##### Retrofit Package Ranking")
                ranking_chart = (
                    alt.Chart(package_chart_df)
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                    .encode(
                        x=alt.X("Package Score:Q", title="Package Score (1–5)", scale=alt.Scale(domain=[0, 5])),
                        y=alt.Y("Package Label:N", title="Package", sort=None),
                        color=alt.Color("Package Label:N", title="Package"),
                        tooltip=[
                            alt.Tooltip("Rank:N", title="Rank"),
                            alt.Tooltip("Package:N", title="Package"),
                            alt.Tooltip("Package Grade:N", title="Grade"),
                            alt.Tooltip("Package Score:Q", title="Score", format=".3f"),
                            alt.Tooltip("Combined Savings %:Q", title="Savings", format=".1f"),
                            alt.Tooltip("Package CAPEX (INR):Q", title="CAPEX", format=",.0f"),
                            alt.Tooltip("Annual Savings (INR):Q", title="Annual savings", format=",.0f"),
                            alt.Tooltip("Payback (Years):Q", title="Payback", format=".2f"),
                        ],
                    )
                    .properties(height=max(300, 55 * len(package_chart_df)))
                )
                st.altair_chart(ranking_chart, use_container_width=True)

            with chart_col2:
                st.markdown("##### CAPEX vs Annual Savings by Package Rank")
                financial_rows = []
                for _, row in package_chart_df.iterrows():
                    financial_rows.extend([
                        {
                            "Rank": row["Rank"],
                            "Package": row["Package"],
                            "Metric": "CAPEX",
                            "Amount": row["Package CAPEX (INR)"],
                            "Package Score": row["Package Score"],
                            "Package Grade": row["Package Grade"],
                            "Combined Savings %": row["Combined Savings %"],
                            "Payback (Years)": row["Payback (Years)"],
                        },
                        {
                            "Rank": row["Rank"],
                            "Package": row["Package"],
                            "Metric": "Annual Savings",
                            "Amount": row["Annual Savings (INR)"],
                            "Package Score": row["Package Score"],
                            "Package Grade": row["Package Grade"],
                            "Combined Savings %": row["Combined Savings %"],
                            "Payback (Years)": row["Payback (Years)"],
                        },
                    ])
                financial_df = pd.DataFrame(financial_rows)
                financial_chart = (
                    alt.Chart(financial_df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("Rank:N", title="Package Rank", sort=list(package_chart_df["Rank"])),
                        xOffset=alt.XOffset("Metric:N"),
                        y=alt.Y("Amount:Q", title="Amount (INR)"),
                        color=alt.Color("Metric:N", title="Financial Metric"),
                        tooltip=[
                            alt.Tooltip("Rank:N", title="Rank"),
                            alt.Tooltip("Package:N", title="Package"),
                            alt.Tooltip("Metric:N", title="Metric"),
                            alt.Tooltip("Amount:Q", title="Amount", format=",.0f"),
                            alt.Tooltip("Package Score:Q", title="Package score", format=".3f"),
                            alt.Tooltip("Package Grade:N", title="Grade"),
                            alt.Tooltip("Combined Savings %:Q", title="Savings", format=".1f"),
                            alt.Tooltip("Payback (Years):Q", title="Payback", format=".2f"),
                        ],
                    )
                    .properties(height=max(300, 55 * len(package_chart_df)))
                )
                st.altair_chart(financial_chart, use_container_width=True)


        # -------------------------------------------------------------------
        # Expandable Individual Retrofit Analysis
        # -------------------------------------------------------------------
        st.markdown("---")
        heading_col, button_col = st.columns([4.5, 1.5])
        with heading_col:
            st.subheader("Individual Retrofit Analysis")
            st.caption(
                "Expand this section to view the top recommendation spotlight, explanations, "
                "visual comparisons, and the complete financial feasibility analysis."
            )
        with button_col:
            st.write("")
            button_label = (
                "🔼 Collapse Analysis"
                if st.session_state["show_individual_analysis"]
                else "🔽 Expand Analysis"
            )
            if st.button(button_label, use_container_width=True, key="toggle_individual_analysis"):
                st.session_state["show_individual_analysis"] = not st.session_state["show_individual_analysis"]
                st.rerun()

        if st.session_state["show_individual_analysis"]:
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

            # Build the same styled ranking table used by the dashboard, while
            # keeping the underlying results_df untouched for all core logic.
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

            # Keep the same ordering convention: feasible options first, then
            # budget-deficit options, with higher final score first.
            display_df["_is_deficit"] = display_df["Budget Feasibility"].str.contains(
                "Deficit", case=False, na=False
            ).astype(int)
            display_df = display_df.sort_values(
                ["_is_deficit", "Final Score"], ascending=[True, False]
            ).reset_index(drop=True)
            display_df = display_df.drop(columns=["_is_deficit"])

            formatted_df = display_df.copy()
            formatted_df["Annual Savings (INR)"] = formatted_df["Annual Savings (INR)"].apply(lambda x: f"₹{x:,.0f}")
            formatted_df["Upgrade Cost (INR)"] = formatted_df["Upgrade Cost (INR)"].apply(lambda x: f"₹{x:,.0f}")
            formatted_df["Budget Balance (INR)"] = formatted_df["Budget Balance (INR)"].apply(
                lambda x: f"+₹{x:,.0f}" if x >= 0 else f"-₹{abs(x):,.0f}"
            )
            formatted_df["Savings %"] = formatted_df["Savings %"].apply(lambda x: f"{x:.1f}%")
            formatted_df["Payback (Years)"] = formatted_df["Payback (Years)"].apply(lambda x: f"{x:.2f} yrs")
            formatted_df["Final Score"] = formatted_df["Final Score"].apply(lambda x: f"{x:.3f}")

            def color_recommendation(val):
                v = str(val).strip()
                if v == "Highly Recommended":
                    return "color: #1b5e20; font-weight: bold;"
                if v == "Recommended":
                    return "color: #2e7d32; font-weight: bold;"
                if v == "Consider":
                    return "color: #f57f17; font-weight: bold;"
                if v == "Low Priority":
                    return "color: #e65100; font-weight: bold;"
                if v == "Not Recommended":
                    return "color: #b71c1c; font-weight: bold;"
                return ""

            def color_grade(val):
                v = str(val).strip()
                if v == "Grade A":
                    return "color: #1b5e20; font-weight: bold;"
                if v == "Grade B":
                    return "color: #2e7d32; font-weight: bold;"
                if v == "Grade C":
                    return "color: #f57f17; font-weight: bold;"
                if v == "Grade D":
                    return "color: #e65100; font-weight: bold;"
                if v == "Grade F":
                    return "color: #b71c1c; font-weight: bold;"
                return ""

            def color_score(val):
                try:
                    score = float(val)
                    if score >= 4.0:
                        return "color: #1b5e20; font-weight: bold;"
                    if score >= 3.0:
                        return "color: #2e7d32; font-weight: bold;"
                    if score >= 2.0:
                        return "color: #f57f17; font-weight: bold;"
                    if score >= 1.0:
                        return "color: #e65100; font-weight: bold;"
                    return "color: #b71c1c; font-weight: bold;"
                except (ValueError, TypeError):
                    return ""

            def color_feasibility(val):
                v = str(val)
                if "Feasible" in v or "High ROI" in v:
                    return "color: #1b5e20; font-weight: bold;"
                if "Deficit" in v or "Shortfall" in v:
                    return "color: #b71c1c; font-weight: bold;"
                return ""

            styled_df = (
                formatted_df.style
                .map(color_recommendation, subset=["Recommendation"])
                .map(color_grade, subset=["Grade"])
                .map(
                    color_score,
                    subset=["Final Score", "Energy", "Comfort", "Cost Benefit", "Sustainability", "Maintenance"],
                )
                .map(color_feasibility, subset=["Budget Feasibility"])
            )

            # Clickable individual retrofit table with detail shown below.
            st.markdown("**Individual retrofit options — click a row to view its analysis:**")
            individual_event = st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                key="individual_retrofit_table",
            )
            selected_rows = individual_event.selection.rows if hasattr(individual_event, "selection") else []
            selected_display_idx = int(selected_rows[0]) if selected_rows else 0
            selected_option = display_df.iloc[selected_display_idx]["Retrofit Option"]
            selected_row = results_df[results_df["Retrofit Option"] == selected_option].iloc[0]

            # -------------------------------------------------------------------
            # Selected Retrofit Analysis — appears directly below the table
            # -------------------------------------------------------------------
            st.markdown("### Selected Retrofit Analysis")
            st.caption("Select another row above to update the analysis and score explanation.")

            opt = selected_row["Retrofit Option"]
            exp = selected_row["Explanation"]
            grade = selected_row["Grade"]
            fscore = selected_row["Final Score"]
            tier = selected_row["Recommendation"]

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("Final Score", f"{fscore:.3f} / 5")
            with d2:
                st.metric("Grade", grade, tier)
            with d3:
                st.metric("Annual Savings", f"₹{selected_row['Annual Savings (INR)']:,.0f}")
            with d4:
                st.metric("Payback", f"{selected_row['Payback (Years)']:.2f} yrs")

            st.info(
                f"### {opt} — {tier}\n\n"
                f"{exp['verdict_explanation']}"
            )

            e_tab, c_tab, cb_tab, s_tab, m_tab, f_tab = st.tabs([
                f"Energy ({selected_row['Energy']}/5)",
                f"Comfort ({selected_row['Comfort']}/5)",
                f"Cost Benefit ({selected_row['Cost Benefit']}/5)",
                f"Sustainability ({selected_row['Sustainability']}/5)",
                f"Maintenance ({selected_row['Maintenance']}/5)",
                "Math & Formula",
            ])

            with e_tab:
                st.markdown(f"**Energy Rationale:**\n\n{exp['energy_explanation']}")
                annual_saved_kwh = selected_row.get("Annual Energy Saved (kWh)")
                if annual_saved_kwh is None:
                    annual_saved_kwh = selected_row["Savings %"] * 0.01 * (
                        b_features.get("eui", b_features.get("baseline_eui", 150.0))
                        * b_features.get("gross_floor_area_m2", b_features.get("floor_area", 15000.0))
                    )
                st.markdown(
                    f"• **Predicted Energy Savings:** `{selected_row['Savings %']:.1f}%`\n"
                    f"• **Annual Electricity Saved:** `{round(float(annual_saved_kwh)):,.0f} kWh/year`"
                )

            with c_tab:
                st.markdown(f"**Comfort Rationale:**\n\n{exp['comfort_explanation']}")

            with cb_tab:
                st.markdown(f"**Cost Benefit Rationale:**\n\n{exp['cost_benefit_explanation']}")
                st.markdown(
                    f"• **Annual Savings:** `₹{selected_row['Annual Savings (INR)']:,.0f}`\n"
                    f"• **Upgrade Cost Required:** `₹{selected_row['Upgrade Cost (INR)']:,.0f}`\n"
                    f"• **Payback Period:** `{selected_row['Payback (Years)']:.2f} years`"
                )

            with s_tab:
                st.markdown(f"**Sustainability Rationale:**\n\n{exp['sustainability_explanation']}")

            with m_tab:
                st.markdown(f"**Maintenance Rationale:**\n\n{exp['maintenance_explanation']}")

            with f_tab:
                st.markdown("**Weighted Final Score Arithmetic:**")
                st.code(exp["formula_explanation"], language="text")
                st.markdown(
                    f"**Score-to-Grade Mapping:** Final Score `{fscore:.3f}` maps directly to **{grade}** ({tier})."
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
        with b_col3:
            if st.button("📡 Energy Monitoring", type="secondary", use_container_width=True):
                st.session_state["page_view"] = "monitoring"
                st.rerun()
    else:
        st.warning("No recommendations calculated. Click 'Back to Building Inputs' to configure.")

