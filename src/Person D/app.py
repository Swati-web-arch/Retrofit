"""
Interface
=========
Person D — Retrofit Recommendation Engine

Streamlit app: enter building characteristics, adjust the 5 axis weights,
and see the ranked retrofit table.

Axis weights are NOT auto-normalized. Each slider can be individually
locked (frozen at its current value) while you adjust the others, but the
five raw values must sum to exactly 1.00 before recommendations will run --
see the sidebar for the live sum and error state.

Run with:
    streamlit run app.py
"""

import streamlit as st
from combiner import recommend_retrofits, DEFAULT_WEIGHTS

st.set_page_config(page_title="Retrofit Recommendation Engine", layout="wide")
st.title("Retrofit Recommendation Engine")
st.caption(
    "Enter building characteristics, adjust axis weights, and get a ranked, "
    "scored table of retrofit options."
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

# Used as-is -- no silent renormalization. If this isn't exactly 1.00,
# the button below is disabled, so recommend_retrofits() is never called
# with invalid weights.
weights = weight_values

# ---------------------------------------------------------------------------
# Building Characteristics: dropdown options sourced from the project's
# datasets rather than free text, so entries reliably match what the
# scoring functions recognize (see src/Person B/eui_benchmark.py's
# TYPOLOGY_MAP and src/Person B/energy_scoring.py's hvac_desc matching).
# ---------------------------------------------------------------------------

# (display label, internal value sent to building_features). Internal
# values are exact BDG2/EESL-derived typology strings that
# eui_benchmark.py's TYPOLOGY_MAP already recognizes case-insensitively --
# the slash-free labels are display-only. "Other" lets the user type a
# value that doesn't fit any bucket (falls back to the "Office" benchmark
# per TYPOLOGY_MAP's default -- flagged to the user in the UI below).
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

# Representative baseline HVAC descriptions drawn from the real
# baseline_hvac_type values in data/processed/eesl_commercial_retrofits_clean.csv.
# The wording matters: energy_scoring.py's estimate_energy_savings() scans
# this string for words like "constant", "reciprocating", "old", "screw",
# "no vfd" (centralized, inefficient -> bigger Chiller_Optimization
# savings) vs "split", "window", "dx" (localized/decentralized -> smaller
# central-chiller savings), so these options were chosen to trigger that
# logic the same way a real EESL row would.
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
    building_age = st.number_input("Building Age (years)", min_value=0, value=15)

    _hvac_label = st.selectbox("Baseline HVAC Type", HVAC_TYPE_OPTIONS, index=0)
    if _hvac_label == "Other":
        baseline_hvac_type = st.text_input("Specify baseline HVAC type", "")
    else:
        baseline_hvac_type = _hvac_label

    hvac_distribution = st.selectbox("HVAC Distribution", HVAC_DISTRIBUTION_OPTIONS)

    eui_level = st.selectbox("Energy Use Intensity", ["High", "Typical", "Low"])
    n_floors = st.number_input("Number of Floors", min_value=1, value=4)
    fan_type = st.selectbox("Fan Type", ["Constant Speed", "Variable Speed (VFD)"])
    zoning_condition = st.selectbox("Zoning Condition", ["Poor", "Good"])
    ventilation_condition = st.selectbox("Ventilation Condition", ["Excess Outdoor Air", "Normal", "Insufficient"])
    controls = st.selectbox("Controls", ["Conventional", "Automated"])

with col2:
    st.subheader("Detected Conditions")
    st.caption(
        "From Person A's inefficiency detector (0=no evidence, 5=severe). "
        "Enter manually for now, or wire up telemetry_data once available."
    )
    poor_zoning = st.slider("Poor Zoning severity", 0, 5, 0)
    ventilation_imbalance = st.slider("Ventilation Imbalance severity", 0, 5, 0)
    economizer_fault = st.slider("Economizer Fault severity", 0, 5, 0)
    sensor_mismatch = st.slider("Sensor Mismatch severity", 0, 5, 0)

inefficiency_flags = {
    "poor_zoning": poor_zoning,
    "ventilation_imbalance": ventilation_imbalance,
    "economizer_fault": economizer_fault,
    "sensor_mismatch": sensor_mismatch,
}

building_features = {
    "building_type": building_type,
    "gross_floor_area_m2": floor_area,
    "building_age": building_age,
    "hvac_type": baseline_hvac_type,      # renamed key to match what energy_scoring.py expects
    "hvac_distribution": hvac_distribution,
    "eui_level": eui_level,
    "n_floors": n_floors,
    "fan_type": fan_type,
    "zoning_condition": zoning_condition,
    "ventilation_condition": ventilation_condition,
    "controls": controls,
}

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
