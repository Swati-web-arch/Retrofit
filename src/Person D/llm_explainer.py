"""RetrofitIQ LLM explanation layer using Hugging Face Inference Providers.

The LLM is deliberately kept separate from the engineering/scoring engine.
It receives already-calculated RetrofitIQ results and explains them in plain
language. It does not calculate or modify scores, savings, CAPEX, or rankings.
"""

import json
import os
from typing import Any, Dict, Optional


def _get_hf_token() -> Optional[str]:
    """Read the Hugging Face token from Streamlit secrets or the environment."""
    try:
        import streamlit as st
        try:
            token = st.secrets.get("HF_TOKEN")
            if token:
                return str(token)
        except Exception:
            pass
    except Exception:
        pass
    return os.getenv("HF_TOKEN")


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def build_retrofit_context(
    building_features: Dict[str, Any],
    inefficiency_flags: Dict[str, Any],
    weights: Dict[str, float],
    package: Optional[Dict[str, Any]] = None,
    results_df=None,
    selected_retrofit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a compact, factual context object for the LLM."""
    context: Dict[str, Any] = {
        "building": _json_safe(building_features),
        "operational_assessment": _json_safe(inefficiency_flags),
        "axis_weights": _json_safe(weights),
        "important_rule": (
            "The supplied scores, savings, CAPEX, payback, package ranking and grades "
            "are authoritative RetrofitIQ calculations. Do not recalculate or change them."
        ),
    }

    if package is not None:
        context["selected_package"] = _json_safe(package)

    if selected_retrofit is not None:
        context["selected_retrofit"] = _json_safe(selected_retrofit)

    if results_df is not None:
        cols = [
            "Retrofit Option", "Recommendation", "Grade", "Final Score", "Energy",
            "Comfort", "Cost Benefit", "Sustainability", "Maintenance", "Savings %",
            "Annual Savings (INR)", "Upgrade Cost (INR)", "Payback (Years)",
            "Budget Feasibility",
        ]
        available = [c for c in cols if c in results_df.columns]
        context["retrofit_options"] = _json_safe(
            results_df[available].to_dict(orient="records")
        )

    return context


SYSTEM_PROMPT = """
You are RetrofitIQ AI, an explanation assistant inside a building-retrofit
engineering decision-support application.

Your job is to explain the calculations and recommendations already produced by
RetrofitIQ. You are NOT the decision engine.

Rules:
1. Never invent a value, dataset result, model accuracy, savings percentage,
   CAPEX, payback, grade, ranking, diagnostic severity, or retrofit outcome.
2. Treat supplied RetrofitIQ numbers as authoritative. Do not recompute or
   alter them.
3. Do not claim that climate normalization was used if it is not present in the
   supplied context. The current project may have explored weather-based
   normalization but it should not be described as part of the final result
   unless the context explicitly says so.
4. Explain the existing ML as a hybrid system: Ridge Regression estimates
   package-level energy savings; other components use deterministic engineering
   rules and scoring. Do not describe the LLM as the model that predicts energy
   savings.
5. Be explicit when a value is an engineering assumption rather than a learned
   quantity. In particular, individual retrofit savings may be decomposed from
   package savings using predefined allocation factors.
6. If information is missing, say that it is not available in the supplied
   RetrofitIQ context.
7. Keep answers concise, practical, and understandable to a building manager,
   engineer, or judge. Use bullets when helpful.
8. Do not provide financial, safety, or engineering certainty beyond the data.
9. When asked "why", cite the supplied numbers and relationships rather than
   inventing causal evidence.
""".strip()


def ask_retrofit_ai(
    question: str,
    context: Dict[str, Any],
    model: Optional[str] = None,
) -> str:
    """Ask a Hugging Face Inference Provider for a grounded explanation."""
    question = (question or "").strip()
    if not question:
        return "Please enter a question."

    hf_token = _get_hf_token()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not configured. Add it to Streamlit secrets or the environment."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI Python package is not installed. Run `pip install openai`."
        ) from exc

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )
    chosen_model = model or os.getenv(
        "RETROFITIQ_LLM_MODEL",
        "openai/gpt-oss-120b:fastest",
    )

    payload = json.dumps(context, indent=2, ensure_ascii=False, default=str)
    user_message = (
        "Here is the current RetrofitIQ context in JSON. Use only this data "
        "for project-specific facts.\n\n"
        f"{payload}\n\n"
        f"User question:\n{question}"
    )

    response = client.chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=700,
    )

    text = response.choices[0].message.content if response.choices else None
    if text:
        return text.strip()

    return "The LLM returned no explanation text."
