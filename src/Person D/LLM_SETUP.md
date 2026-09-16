# RetrofitIQ LLM integration

## Files
- `app.py` — dashboard with the new **Ask RetrofitIQ AI** panel.
- `llm_explainer.py` — OpenAI Responses API integration and grounding rules.
- `requirements_llm.txt` — adds the `openai` Python package.

## Install
```bash
pip install -r requirements_llm.txt
```

## API key
Set `OPENAI_API_KEY` as an environment variable, or add it to Streamlit secrets:

```toml
OPENAI_API_KEY = "your-key-here"
```

For Streamlit Cloud, put the key under **Settings → Secrets**. Do not hard-code the key in `app.py` or commit it to Git.

## Optional model override
The default is `gpt-5.6-luna`. To override it:

```bash
set RETROFITIQ_LLM_MODEL=gpt-5.6-luna
```

or set `RETROFITIQ_LLM_MODEL` in the deployment environment.

## What the LLM does
The LLM receives the already-calculated building inputs, diagnostics, axis weights, retrofit options, and selected package. It explains those results in natural language.

It does **not** calculate or modify:
- Ridge savings predictions
- CAPEX
- payback
- five-axis scores
- grades
- package ranking
- budget eligibility

This keeps the engineering/ML decision engine deterministic while using the LLM as an explanation and Q&A layer.
