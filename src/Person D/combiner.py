"""
Combiner
========
Person D — Retrofit Recommendation Engine

Integration point: takes Person A's inefficiency flags, filters the retrofit
catalog to relevant options, runs Persons B/C/D's per-axis scoring functions
on each remaining option, applies the weighted Final Score formula, and
returns a ranked table.

NOTE on the two "building" inputs (documented for the technical report's
assumptions section):
  - Person A's detect_inefficiencies() operates on raw sensor TELEMETRY
    (bldg59 / testbedclean style: rtu_oa_damper_avg, zone temps, hvac_kw,
    etc. -- a DataFrame/Series/dict of hourly readings, or a dataset id).
  - Persons B/C/D's scoring functions operate on a per-building FEATURE
    dict (EESL-style: comfort_impact_score, avoided_co2_tons_yr,
    maintenance_reduction_score, energy_savings_pct, simple_payback_years,
    indoor_temp_c, etc.).
  These are two different views of the same physical building. The combiner
  accepts them separately -- `telemetry_data` for A, `building_features`
  for B/C/D -- rather than forcing one shared schema. If `telemetry_data`
  isn't available (or the caller already has A's flags), pass
  `inefficiency_flags` directly instead and no telemetry is needed.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# ===========================================================================
# 1. ENVIRONMENT & PATH SETUP (MUST REMAIN AT VERY TOP OF FILE)
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add teammate packages with relative/plain imports to sys.path immediately,
# ensuring any Person B/D module imports succeed regardless of import position.
_PERSON_B_DIR = PROJECT_ROOT / "src" / "Person B"
if str(_PERSON_B_DIR) not in sys.path:
    sys.path.insert(0, str(_PERSON_B_DIR))

_PERSON_D_DIR = PROJECT_ROOT / "src" / "Person D"
if str(_PERSON_D_DIR) not in sys.path:
    sys.path.insert(0, str(_PERSON_D_DIR))


def _load_module(name: str, path: Path):
    """Load a module from a file path (needed because teammates' folder
    names contain spaces, e.g. 'Person A', 'person c/comfort and
    sustainability', so they can't be imported as normal packages).

    Raises a clear, actionable RuntimeError instead of letting a bare
    FileNotFoundError/ImportError surface from deep inside importlib's
    internals.
    """
    if not path.exists():
        raise RuntimeError(
            f"combiner.py could not load '{name}': expected a file at "
            f"{path}, but nothing exists there. If a teammate's folder was "
            f"renamed or moved, update this path in combiner.py to match."
        )
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not build an import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        raise RuntimeError(
            f"combiner.py failed to load '{name}' from {path}: {exc}"
        ) from exc


# ===========================================================================
# 2. CROSS-PERSON SCORING & UTILITY IMPORTS
# ===========================================================================
# Person A & Person C (folders contain spaces -> loaded via _load_module)
_person_a = _load_module(
    "person_a", PROJECT_ROOT / "src" / "Person A" / "inefficiency_detection.py"
)
_person_c = _load_module(
    "person_c",
    PROJECT_ROOT / "src" / "person c" / "comfort and sustainability" / "comfort_sustainability.py",
)
_person_d_maintenance = _load_module(
    "person_d_maintenance", PROJECT_ROOT / "src" / "Person D" / "maintenance.py"
)

# Person B (plain imports resolved via _PERSON_B_DIR in sys.path above)
from energy_scoring import energy_score, estimate_energy_savings  # noqa: E402
from cost_benefit import analyze_cost_benefit, cost_benefit_score  # noqa: E402
from eui_benchmark import classify_eui  # noqa: E402
import numpy as np

# Export canonical scoring and detection functions
detect_inefficiencies = _person_a.detect_inefficiencies
comfort_score = _person_c.comfort_score
sustainability_score = _person_c.sustainability_score
maintenance_score = _person_d_maintenance.maintenance_score


def detect_conditions(telemetry_df: Any) -> Dict[str, Any]:
    """Wrapper so app.py doesn't need its own import gymnastics.
    Calls Person A's detect_inefficiencies with return_details=True.
    """
    return detect_inefficiencies(telemetry_df, return_details=True)


RETROFIT_CATALOG = [
    "Smart_Controls",
    "AHU_VFD",
    "DCV",
    "Chiller_Optimization",
    "Zoning_Optimization",
]

DEFAULT_WEIGHTS = {
    "energy": 0.25,
    "comfort": 0.20,
    "cost_benefit": 0.25,
    "sustainability": 0.20,
    "maintenance": 0.10,
}

# Which of Person A's four inefficiency flags gates which catalog measure.
FLAG_TO_MEASURE = {
    "poor_zoning": "Zoning_Optimization",
    "ventilation_imbalance": "DCV",
    "economizer_fault": "Smart_Controls",
    "sensor_mismatch": "Smart_Controls",
}
ALWAYS_CANDIDATE = {"AHU_VFD", "Chiller_Optimization"}
FLAG_THRESHOLD = 2  # matches Person A's own "flags" threshold (score >= 2)


def filter_catalog(inefficiency_flags: Dict[str, int]) -> list:
    """Return the subset of the retrofit catalog that's relevant given
    Person A's severity scores, e.g. don't recommend DCV if ventilation
    is already fine."""
    gated_in = set()
    for flag, measure in FLAG_TO_MEASURE.items():
        if inefficiency_flags.get(flag, 0) >= FLAG_THRESHOLD:
            gated_in.add(measure)
    relevant = gated_in | ALWAYS_CANDIDATE
    return [m for m in RETROFIT_CATALOG if m in relevant]


def get_recommendation_and_grade(final_score: float) -> tuple[str, str]:
    """
    Direct grading scale derived purely from Final Score (1-5 scale):
    - 4.0+: Highly Recommended | Grade A
    - 3.0 - 3.99: Recommended | Grade B
    - 2.0 - 2.99: Consider | Grade C
    - 1.0 - 1.99: Low Priority | Grade D
    - < 1.0: Not Recommended | Grade F
    """
    if final_score >= 4.0:
        return "Highly Recommended", "Grade A"
    elif final_score >= 3.0:
        return "Recommended", "Grade B"
    elif final_score >= 2.0:
        return "Consider", "Grade C"
    elif final_score >= 1.0:
        return "Low Priority", "Grade D"
    else:
        return "Not Recommended", "Grade F"


def explain_retrofit_score(
    building_features: Dict[str, Any],
    option: str,
    scores: Dict[str, int],
    financials: Dict[str, Any],
    weights: Dict[str, float],
    grade: str,
    tier: str,
    energy_est: Dict[str, Any],
) -> Dict[str, str]:
    """
    Generate comprehensive, human-readable explanations detailing exactly why
    this retrofit option received its scores on each of the 5 axes and overall.
    """
    e_score = scores.get("Energy", 3)
    c_score = scores.get("Comfort", 3)
    cb_score = scores.get("Cost Benefit", 3)
    s_score = scores.get("Sustainability", 3)
    m_score = scores.get("Maintenance", 3)
    final_score = scores.get("Final Score", 3.0)

    area = float(building_features.get("gross_floor_area_m2") or building_features.get("floor_area") or 15000.0)
    savings_pct = energy_est.get("predicted_savings_pct", 30.0)
    saved_kwh = energy_est.get("annual_energy_saved_kwh", 0.0)
    est_basis = energy_est.get("estimation_basis", "Ridge regression on EESL baseline")
    capex = financials.get("capex_inr", 0.0)
    savings_inr = financials.get("annual_cost_savings_inr", 0.0)
    payback = financials.get("payback_years", 3.0)
    tariff = financials.get("electricity_tariff_inr_kwh", 9.0)

    # 1. Energy
    if option == "AHU_VFD":
        fan_t = building_features.get("fan_type", "Constant Speed")
        energy_why = (
            f"Rated {e_score}/5 based on {savings_pct}% predicted energy savings ({saved_kwh:,.0f} kWh/yr). "
            f"Baseline fan is '{fan_t}'. Adding Variable Frequency Drives (VFD) to AHU supply fans enables speed reduction "
            f"matching cooling load. Under fan affinity laws (P ∝ RPM³), a 20% drop in speed yields ~50% fan motor power savings."
        )
    elif option == "Chiller_Optimization":
        hvac_t = building_features.get("hvac_type", "Central Chiller")
        energy_why = (
            f"Rated {e_score}/5 based on {savings_pct}% predicted energy savings ({saved_kwh:,.0f} kWh/yr). "
            f"Baseline plant is '{hvac_t}'. Chiller optimization (condenser water reset, staging control, and variable flow) "
            f"elevates chiller COP by 0.5–1.2 W/W across part-load hours, eliminating compressor over-cycling."
        )
    elif option == "DCV":
        energy_why = (
            f"Rated {e_score}/5 based on {savings_pct}% predicted energy savings ({saved_kwh:,.0f} kWh/yr). "
            f"Demand-Controlled Ventilation throttles outside air intake based on real-time occupancy/air-quality signals, "
            f"cutting unnecessary conditioned ventilation air when spaces are unoccupied."
        )
    elif option == "Smart_Controls":
        age = building_features.get("building_age", 15)
        energy_why = (
            f"Rated {e_score}/5 based on {savings_pct}% predicted energy savings ({saved_kwh:,.0f} kWh/yr). "
            f"Baseline system age is {age} years. Smart IoT controls introduce automated optimal start/stop, night setbacks, "
            f"and economizer logic, preventing off-hour cooling and drift from setpoints. Estimation basis: {est_basis}."
        )
    elif option == "Zoning_Optimization":
        energy_why = (
            f"Rated {e_score}/5 based on {savings_pct}% predicted energy savings ({saved_kwh:,.0f} kWh/yr). "
            f"Telemetry indicates thermal imbalance across zones. Zoning optimization balances air distribution via motorized VAV "
            f"dampers and smart zone thermostats, reducing perimeter-to-core temperature fighting."
        )
    else:
        energy_why = f"Rated {e_score}/5 based on {savings_pct}% predicted energy savings ({saved_kwh:,.0f} kWh/yr). Basis: {est_basis}."

    # 2. Comfort
    if option == "AHU_VFD":
        comfort_why = f"Rated {c_score}/5. Modulating airflow delivers smoother room circulation, avoiding harsh cold air drafts and cycling noise caused by on/off constant-speed fans."
    elif option == "Chiller_Optimization":
        comfort_why = f"Rated {c_score}/5. Precision chilled water temperature control maintains steady supply air dewpoints and eliminates indoor humidity spikes."
    elif option == "DCV":
        comfort_why = f"Rated {c_score}/5. Actively maintains indoor CO2 below 800-1000 ppm during peak meetings while preventing over-ventilation chills in quiet areas."
    elif option == "Smart_Controls":
        comfort_why = f"Rated {c_score}/5. Adaptive algorithms pre-cool spaces before arrival and maintain tighter ±0.5°C comfort bands against ambient weather fluctuations."
    else:
        comfort_why = f"Rated {c_score}/5. Balances inter-zone temperature spreads across floors and rooms, eliminating local hotspots and cold draft complaints."

    # 3. Cost Benefit
    cost_why = (
        f"Rated {cb_score}/5. Generates ₹{savings_inr:,.0f}/year in utility savings at ₹{tariff:.2f}/kWh. "
        f"Against an estimated upgrade cost of ₹{capex:,.0f}, full capital investment is recovered in {payback:.2f} years. "
        f"Ranked favorably relative to peer options for commercial payback speed."
    )

    # 4. Sustainability
    co2_est = round(saved_kwh * 0.82 / 1000.0, 1)
    sust_why = (
        f"Rated {s_score}/5. Displaces ~{co2_est:,.1f} metric tons of CO2 equivalent emissions each year from the power grid. "
        f"Avoided carbon intensity scales directly with facility size ({area:,.0f} m²), placing it in quintile {s_score}."
    )

    # 5. Maintenance
    if option == "AHU_VFD":
        maint_why = f"Rated {m_score}/5. Soft-start capability reduces torque shock on motor bearings and belts, extending fan motor operational lifespan and minimizing belt replacements."
    elif option == "Chiller_Optimization":
        maint_why = f"Rated {m_score}/5. Eliminates compressor short-cycling and optimizes motor run-hours, decreasing mechanical wear on impellers and refrigerant seals."
    elif option == "DCV":
        maint_why = f"Rated {m_score}/5. Reduces continuous airflow burden on intake filters and air-handling units during low occupancy, extending filter replacement intervals."
    elif option == "Smart_Controls":
        maint_why = f"Rated {m_score}/5. Early fault detection and sensor diagnostic alerts prevent minor drifts from escalating into costly equipment failures."
    else:
        maint_why = f"Rated {m_score}/5. Balancing dampers reduces static pressure build-up and duct vibration, lowering long-term maintenance calls."

    # Formula breakdown
    formula_why = (
        f"Final Score: {final_score:.3f} / 5.00\n"
        f"= ({weights['energy']:.2f} × Energy {e_score}) + "
        f"({weights['comfort']:.2f} × Comfort {c_score}) + "
        f"({weights['cost_benefit']:.2f} × Cost-Benefit {cb_score}) + "
        f"({weights['sustainability']:.2f} × Sustainability {s_score}) + "
        f"({weights['maintenance']:.2f} × Maintenance {m_score})"
    )

    verdict_why = (
        f"{tier} — {grade} (Final Score: {final_score:.2f} / 5.00). "
        f"Combines {savings_pct}% energy savings, ₹{savings_inr:,.0f}/yr financial savings, "
        f"and an investment payback of {payback:.2f} years."
    )

    return {
        "energy_explanation": energy_why,
        "comfort_explanation": comfort_why,
        "cost_benefit_explanation": cost_why,
        "sustainability_explanation": sust_why,
        "maintenance_explanation": maint_why,
        "formula_explanation": formula_why,
        "verdict_explanation": verdict_why,
    }


def score_option(
    building_features: Dict[str, Any],
    option: str,
    weights: Dict[str, float],
    budget_inr: Optional[float] = None,
) -> Dict[str, Any]:
    energy = energy_score(building_features, option)
    comfort = comfort_score(building_features, option)
    sustainability = sustainability_score(building_features, option)
    maintenance = maintenance_score(building_features, option)

    financials = analyze_cost_benefit(building_features, option)
    cost_benefit = financials["cost_benefit_score"]

    energy_est = estimate_energy_savings(building_features, option)
    savings_pct = energy_est["predicted_savings_pct"]

    final_score = (
        weights["energy"] * energy
        + weights["comfort"] * comfort
        + weights["cost_benefit"] * cost_benefit
        + weights["sustainability"] * sustainability
        + weights["maintenance"] * maintenance
    )
    final_score_rounded = round(final_score, 3)

    tier, grade = get_recommendation_and_grade(final_score_rounded)
    payback = financials["payback_years"]
    upgrade_cost = financials["capex_inr"]

    # Budget & financial feasibility check
    budget = building_features.get("available_budget") or building_features.get("budget") or budget_inr

    if budget is not None and float(budget) > 0:
        budget_val = float(budget)
        balance = budget_val - upgrade_cost
        if balance >= 0:
            fin_feasibility = f"Feasible (Within Budget, Surplus: ₹{balance:,.0f})"
            capital_shortfall = 0.0
        else:
            shortfall = abs(balance)
            fin_feasibility = f"Budget Deficit (Shortfall: ₹{shortfall:,.0f})"
            capital_shortfall = shortfall
    else:
        capital_shortfall = upgrade_cost
        balance = -upgrade_cost
        if payback <= 3.2:
            fin_feasibility = "High ROI (Payback < 3.2 yrs)"
        elif payback <= 5.0:
            fin_feasibility = "Feasible (Payback 3-5 yrs)"
        else:
            fin_feasibility = "Marginal (Payback > 5 yrs)"

    scores_dict = {
        "Energy": energy,
        "Comfort": comfort,
        "Cost Benefit": cost_benefit,
        "Sustainability": sustainability,
        "Maintenance": maintenance,
        "Final Score": final_score_rounded,
    }

    explanations = explain_retrofit_score(
        building_features=building_features,
        option=option,
        scores=scores_dict,
        financials=financials,
        weights=weights,
        grade=grade,
        tier=tier,
        energy_est=energy_est,
    )

    return {
        "Retrofit Option": option,
        "Recommendation": tier,
        "Grade": grade,
        "Final Score": final_score_rounded,
        "Energy": energy,
        "Comfort": comfort,
        "Cost Benefit": cost_benefit,
        "Sustainability": sustainability,
        "Maintenance": maintenance,
        "Savings %": savings_pct,
        "Annual Energy Saved (kWh)": energy_est.get("annual_energy_saved_kwh", 0.0),
        "Annual Savings (INR)": financials["annual_cost_savings_inr"],
        "Upgrade Cost (INR)": upgrade_cost,
        "Estimated CAPEX (INR)": upgrade_cost,
        "Budget Balance (INR)": balance,
        "Amount Required (INR)": capital_shortfall,
        "Capital Shortfall (INR)": capital_shortfall,
        "Budget Feasibility": fin_feasibility,
        "Financial Sustainability": fin_feasibility,
        "Payback (Years)": payback,
        "Baseline Annual Cost (INR)": financials["baseline_annual_opcost_inr"],
        "Post-Retrofit Annual Cost (INR)": financials["post_annual_opcost_inr"],
        "Explanation": explanations,
    }


def recommend_retrofits(
    building_features: Dict[str, Any],
    telemetry_data: Optional[Any] = None,
    inefficiency_flags: Optional[Dict[str, int]] = None,
    weights: Optional[Dict[str, float]] = None,
    show_all: bool = True,
    budget_inr: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run the full pipeline: score options on all 5 axes, compute financial viability,
    grade, and explanations.

    Parameters
    ----------
    building_features : dict
        Per-building features/labels consumed by the B/C/D axis functions.
    telemetry_data : DataFrame/Series/dict/str, optional
        Raw sensor telemetry passed to Person A's detect_inefficiencies().
    inefficiency_flags : dict, optional
        Precomputed inefficiency scores (0-5).
    weights : dict, optional
        Overrides for axis weights.
    show_all : bool, default True
        If True, displays all 5 catalog options at all times regardless of score.
        If False, filters by Person A's inefficiency gating.
    budget_inr : float, optional
        Available CAPEX budget to evaluate financial sustainability.

    Returns
    -------
    pandas.DataFrame, sorted descending by Final Score.
    """
    weights = weights or DEFAULT_WEIGHTS

    if inefficiency_flags is None and telemetry_data is not None:
        inefficiency_flags = detect_inefficiencies(telemetry_data)

    if show_all:
        candidates = RETROFIT_CATALOG
    else:
        candidates = (
            filter_catalog(inefficiency_flags) if inefficiency_flags is not None else RETROFIT_CATALOG
        )

    # Merge A's flags into the feature dict
    scoring_features = {**building_features, **(inefficiency_flags or {})}

    rows = [
        score_option(scoring_features, option, weights, budget_inr=budget_inr)
        for option in candidates
    ]
    df = pd.DataFrame(rows).sort_values("Final Score", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    eesl = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "eesl_commercial_retrofits_clean.csv")
    sample = eesl.iloc[0].to_dict()

    print(f"Building: {sample.get('building_name')} ({sample.get('location')})")
    print(f"Actually implemented: {sample.get('retrofit_measures_implemented')}\n")

    result = recommend_retrofits(building_features=sample)
    print(result[[
        "Retrofit Option", "Recommendation", "Grade", "Final Score",
        "Energy", "Comfort", "Cost Benefit", "Sustainability", "Maintenance",
        "Annual Savings (INR)", "Estimated CAPEX (INR)", "Financial Sustainability", "Payback (Years)"
    ]].to_string(index=False))

