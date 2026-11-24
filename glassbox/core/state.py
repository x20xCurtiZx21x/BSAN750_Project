"""Shared session state: one place that knows what the user has done so far."""
import streamlit as st

DEFAULTS = {
    "df": None,             # the raw dataset (pandas DataFrame)
    "dataset_name": None,
    "target": None,         # name of the column to predict
    "positive": None,       # which value of the target counts as "yes" (e.g. 1 = cancelled)
    "report": None,         # data health report (core.data.health_report)
    "prep": None,           # preprocessing choices (core.prep)
    "lab": None,            # trained models + test split (core.models)
    "decision": None,       # chosen costs + threshold (core.decide)
    "chat": [],             # assistant conversation
}

# downstream state that becomes stale when an earlier step changes
DOWNSTREAM = {
    "df": ["target", "positive", "report", "prep", "lab", "decision", "chat"],
    "target": ["report", "prep", "lab", "decision"],
    "prep": ["lab", "decision"],
    "lab": ["decision"],
}


def init():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v if not isinstance(v, list) else list(v)


def get(key):
    return st.session_state.get(key)


def set(key, value):  # noqa: A001 - mirrors st.session_state semantics
    """Set a value and clear everything that depended on the old one."""
    st.session_state[key] = value
    for k in DOWNSTREAM.get(key, []):
        st.session_state[k] = DEFAULTS[k] if not isinstance(DEFAULTS[k], list) else []


def progress():
    """(label, done) for each workflow step, used by the sidebar checklist."""
    s = st.session_state
    return [
        ("Load data", s.get("df") is not None),
        ("Data check", s.get("report") is not None),
        ("Prep", s.get("prep") is not None),
        ("Model lab", s.get("lab") is not None),
        ("Explain", s.get("lab") is not None),
        ("Decide", s.get("decision") is not None),
    ]
