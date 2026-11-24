"""The AI assistant. It only ever sees numbers the app computed, so it explains rather than invents.

Set the key in .streamlit/secrets.toml (see secrets.toml.example) or as the OPENAI_API_KEY environment variable.
"""
import json
import os

import streamlit as st

DEFAULT_MODEL = "gpt-4o-mini"  # override with OPENAI_MODEL (use whatever model the class key allows)

SYSTEM = """You are the assistant inside Glassbox, a machine-learning toolbox for business users with little
technical background. Explain clearly and briefly, in plain English, like a good analytics consultant.
Rules:
- Use ONLY the numbers in the CONTEXT block. If something is not in the context, say the user has not run that step yet.
- Never invent metrics, column names or results.
- When you recommend something, say which page of the app to do it on (Data check, Prep, Model lab, Explain, Decide).
- Keep answers under 150 words unless asked for more."""


def _secret(name):
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets.toml yet
        return None


def api_key():
    return _secret("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")


def model_name():
    return _secret("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def available() -> bool:
    return bool(api_key())


def context() -> dict:
    """Compact, JSON-safe summary of everything the user has done so far."""
    s = st.session_state
    ctx = {}
    if s.get("df") is not None:
        ctx["dataset"] = {"name": s.get("dataset_name"), "rows": int(len(s.df)), "columns": list(map(str, s.df.columns))}
    if s.get("target"):
        ctx["target"] = {"column": s.target, "positive_value": str(s.positive)}
    if s.get("report"):
        r = s.report
        ctx["data_check"] = {
            "positive_rate": round(r["positive_rate"], 3),
            "suspected_leakage": r["leakage"],
            "issues": [f"{sev}: {col}: {msg}" for sev, col, msg in r["issues"][:12]],
        }
    if s.get("prep"):
        ctx["prep"] = {"dropped": s.prep["drop"], "treated_as_category": s.prep["as_category"]}
    if s.get("lab"):
        lab = s.lab
        ctx["models"] = {
            name: {k: round(float(r[k]), 3) for k in ("ROC AUC", "Accuracy", "Precision", "Recall", "F1")}
            | {"tuned": r["param"], "best_value": str(r["best"])}
            for name, r in lab["results"].items()
        }
        ctx["best_model"] = lab["best"]
        if lab.get("importance") is not None:
            imp = lab["importance"].head(10)
            ctx["top_features_auc_drop"] = {f: round(float(v), 4) for f, v in zip(imp["feature"], imp["AUC drop"])}
    if s.get("decision"):
        ctx["decision"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.decision.items()}
    return ctx


def ask(messages: list[dict], extra: str = "") -> str:
    """Send the conversation plus the current app context to the model."""
    from openai import OpenAI  # imported lazily so the app runs without the package/key

    client = OpenAI(api_key=api_key())
    ctx = json.dumps(context(), default=str)
    full = [{"role": "system", "content": SYSTEM + f"\n\nCONTEXT:\n{ctx}" + (f"\n\n{extra}" if extra else "")}] + messages
    resp = client.chat.completions.create(model=model_name(), messages=full, temperature=0.3)
    return resp.choices[0].message.content


def explain_button(topic: str, key: str):
    """'Explain this in plain English' button for any page. Hidden until an API key is configured."""
    if not available():
        return
    if st.button("✨ Explain this in plain English", key=key):
        with st.spinner("Asking the assistant…"):
            try:
                st.info(ask([{"role": "user", "content": f"Explain the {topic} results to me."}]))
            except Exception as e:  # network/key problems shouldn't crash the page
                st.warning(f"The assistant is unavailable: {e}")
