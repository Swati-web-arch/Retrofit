"""
Cost Benefit Scoring & Financial Analysis Module
================================================
Person B — Retrofit Recommendation Engine

This module computes the financial attractiveness and simple payback period
for retrofit options, outputting a Cost Benefit Score on a 1–5 scale.

Core Principle:
- Higher financial savings + shorter payback + reasonable CAPEX -> higher Cost Benefit score (4–5).
- Low financial savings + high CAPEX + long payback -> lower Cost Benefit score (1–2).

Standard Commercial Tariff:
- Directly derived from EESL commercial retrofit dataset: 9.00 INR / kWh
  (annual_cost_savings_inr / energy_savings_kwh = 9.00 across all 16 projects).
- Never uses arbitrary or invented tariffs without documented dataset derivation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from .energy_scoring import (
        RETROFIT_CATALOG,
        estimate_energy_savings,
        normalize_retrofit_name,
    )
except (ImportError, ValueError):
    from energy_scoring import (
        RETROFIT_CATALOG,
        estimate_energy_savings,
        normalize_retrofit_name,
    )

# Empirical electricity tariff directly from EESL dataset (INR/kWh)
EESL_DEFAULT_ELECTRICITY_TARIFF_INR = 9.00

# Measure-level CAPEX benchmarks derived from EESL package projects.
# EESL records package CAPEX for several measures implemented together, so
# we first allocate each package cost equally across the measures present in
# that project, convert to INR/m², and then use the median rate per measure.
# This avoids incorrectly charging the full package CAPEX to every single
# retrofit option.
EESL_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "eesl_commercial_retrofits_clean.csv"

MEASURE_FLAG_COLUMNS = {
    "Smart_Controls": "retrofit_smart_controls",
    "AHU_VFD": "retrofit_ahu_vfd",
    "DCV": "retrofit_dcv",
    "Chiller_Optimization": "retrofit_chiller_opt",
    "Zoning_Optimization": "retrofit_zoning_opt",
}

DEFAULT_CAPEX_PER_M2_INR = {
    "Smart_Controls": 350.0,
    "AHU_VFD": 333.0,
    "DCV": 303.0,
    "Chiller_Optimization": 380.0,
    "Zoning_Optimization": 308.0,
}

CAPEX_RATE_RANGES_INR_PER_M2 = {
    "Smart_Controls": (305.2, 410.0),
    "AHU_VFD": (302.8, 410.4),
    "DCV": (287.2, 309.6),
    "Chiller_Optimization": (315.8, 411.4),
    "Zoning_Optimization": (289.6, 332.7),
}


def _load_eesl_capex_rates() -> Dict[str, float]:
    """Estimate measure-level CAPEX rates from EESL package projects.

    Each EESL project may contain multiple retrofit measures but reports one
    project-level CAPEX. We allocate that project CAPEX equally across the
    measures implemented in the project, normalize by floor area, and use the
    median rate per measure. This is a transparent proxy, not a quotation.
    """
    if not EESL_FILE.exists():
        return DEFAULT_CAPEX_PER_M2_INR.copy()

    try:
        df = pd.read_csv(EESL_FILE)
        rates: Dict[str, list] = {k: [] for k in MEASURE_FLAG_COLUMNS}
        for _, row in df.iterrows():
            area = pd.to_numeric(row.get("gross_floor_area_m2"), errors="coerce")
            capex = pd.to_numeric(row.get("project_capex_inr"), errors="coerce")
            if pd.isna(area) or pd.isna(capex) or area <= 0 or capex <= 0:
                continue
            measures = [
                key for key, col in MEASURE_FLAG_COLUMNS.items()
                if pd.to_numeric(row.get(col), errors="coerce") == 1
            ]
            if not measures:
                continue
            allocated_rate = float(capex) / float(area) / len(measures)
            for measure in measures:
                rates[measure].append(allocated_rate)

        result = DEFAULT_CAPEX_PER_M2_INR.copy()
        for measure, values in rates.items():
            if values:
                result[measure] = round(float(np.median(values)), 2)
        return result
    except Exception:
        return DEFAULT_CAPEX_PER_M2_INR.copy()


CAPEX_PER_M2_INR = _load_eesl_capex_rates()


def analyze_cost_benefit(
    building: Dict[str, Any],
    retrofit_option: str,
    electricity_price_inr: Optional[float] = None,
    capex_inr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive financial return metrics for a retrofit option.

    Parameters
    ----------
    building : dict
        Building features including floor area and energy telemetry.
    retrofit_option : str
        Target retrofit measure.
    electricity_price_inr : float, optional
        Custom power price in INR/kWh. If omitted, uses EESL dataset rate (9.00 INR/kWh).
    capex_inr : float, optional
        Custom initial investment in INR. If omitted, estimated from building floor area.

    Returns
    -------
    dict
        {
            'retrofit_option': str,
            'annual_energy_saved_kwh': float,
            'electricity_tariff_inr_kwh': float,
            'annual_cost_savings_inr': float,
            'capex_inr': float,
            'payback_years': float,
            'cost_benefit_score': int (1-5),
            'currency': 'INR',
            'financial_summary': str
        }
    """
    canon_retrofit = normalize_retrofit_name(retrofit_option)
    energy_est = estimate_energy_savings(building, canon_retrofit)
    saved_kwh = energy_est["annual_energy_saved_kwh"]

    tariff = (
        electricity_price_inr
        or building.get("electricity_price")
        or building.get("energy_price_inr")
        or EESL_DEFAULT_ELECTRICITY_TARIFF_INR
    )
    tariff = float(tariff)

    annual_cost_savings = round(saved_kwh * tariff, 2)
    baseline_annual_opcost_inr = round(energy_est["baseline_annual_kwh"] * tariff, 2)
    post_annual_opcost_inr = round(energy_est["post_annual_kwh"] * tariff, 2)

    area = float(
        building.get("floor_area")
        or building.get("gross_floor_area_m2")
        or building.get("area_tot_m2")
        or 10000.0
    )

    if capex_inr is not None:
        investment = float(capex_inr)
        capex_source = "user_provided"
    elif building.get("custom_retrofit_capex"):
        investment = float(building.get("custom_retrofit_capex"))
        capex_source = "custom_override"
    else:
        # `project_capex_inr` in EESL is package-level CAPEX for all measures
        # implemented together. It must not be reused as the cost of one
        # individual retrofit option.
        unit_rate = CAPEX_PER_M2_INR.get(canon_retrofit, 350.0)
        investment = round(unit_rate * area, 2)
        capex_source = "eesl_measure_level_median_benchmark"

    unit_rate_used = investment / area if area > 0 else 0.0
    low_rate, high_rate = CAPEX_RATE_RANGES_INR_PER_M2.get(
        canon_retrofit, (unit_rate_used, unit_rate_used)
    )
    estimated_cost_low = round(low_rate * area, 2)
    estimated_cost_high = round(high_rate * area, 2)

    if annual_cost_savings > 0:
        payback = round(investment / annual_cost_savings, 2)
    else:
        payback = 99.9

    # Absolute financial attractiveness: unlike the old relative-ranking
    # method, the same retrofit gets the same score whenever its payback is
    # the same. This is easier to explain and makes tariff/cost assumptions
    # traceable.
    if payback <= 2.0:
        score = 5
    elif payback <= 4.0:
        score = 4
    elif payback <= 6.0:
        score = 3
    elif payback <= 10.0:
        score = 2
    else:
        score = 1

    summary = (
        f"Retrofit {canon_retrofit} yields annual savings of INR {annual_cost_savings:,.0f} "
        f"against an estimated CAPEX of INR {investment:,.0f}, reaching payback in {payback:.2f} years "
        f"(estimated measure-level benchmark range: INR {estimated_cost_low:,.0f}–{estimated_cost_high:,.0f}; "
        f"CAPEX source: {capex_source})."
    )

    return {
    "retrofit_option": canon_retrofit,
    "annual_energy_saved_kwh": saved_kwh,
    "electricity_tariff_inr_kwh": tariff,
    "baseline_annual_opcost_inr": baseline_annual_opcost_inr,   # new
    "post_annual_opcost_inr": post_annual_opcost_inr,           # new
    "annual_cost_savings_inr": annual_cost_savings,
    "capex_inr": investment,
    "capex_low_inr": estimated_cost_low,
    "capex_high_inr": estimated_cost_high,
    "capex_unit_rate_inr_m2": round(unit_rate_used, 2),
    "capex_source": capex_source,
    "payback_years": payback,
    "cost_benefit_score": score,
    "currency": "INR",
    "financial_summary": summary,
}


def cost_benefit_score(
    building: Dict[str, Any],
    retrofit_option: str,
    electricity_price_inr: Optional[float] = None,
    capex_inr: Optional[float] = None,
) -> int:
    """Calculate an absolute Cost Benefit Score (1–5) from payback.

    The score no longer depends on which other retrofit options happen to be
    in the current catalog. This makes the financial meaning stable and
    directly traceable to the estimated investment and annual savings.
    """
    result = analyze_cost_benefit(
        building,
        retrofit_option,
        electricity_price_inr=electricity_price_inr,
        capex_inr=capex_inr,
    )
    return int(result["cost_benefit_score"])


def score_all_cost_benefit_retrofits(
    building: Dict[str, Any],
    electricity_price_inr: Optional[float] = None,
) -> Dict[str, int]:
    """Calculate Cost Benefit Scores (1–5) across all catalog retrofits."""
    return {
        retrofit: cost_benefit_score(
            building, retrofit, electricity_price_inr=electricity_price_inr
        )
        for retrofit in RETROFIT_CATALOG
    }
