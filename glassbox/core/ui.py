"""Small UI helpers so every page looks and behaves the same."""
from __future__ import annotations
import inspect

import streamlit as st


def _has_param(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


# Streamlit renamed use_container_width -> width="stretch"; support both (Python 3.9 tops out at Streamlit 1.50)
_ALTAIR_WIDTH = _has_param(st.altair_chart, "width")


def chart(c):
    """Full-width Altair chart on any Streamlit version."""
    if _ALTAIR_WIDTH:
        st.altair_chart(c, width="stretch")
    else:
        st.altair_chart(c, use_container_width=True)


def table(data, **kwargs):
    """Full-width dataframe on any Streamlit version."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except Exception:
        st.dataframe(data, use_container_width=True, **kwargs)


def header(step: int | None, title: str, subtitle: str):
    if step:
        st.caption(f"STEP {step} OF 5")
    st.title(title)
    st.markdown(subtitle)


def explainer(markdown: str, title: str = "What's happening here?"):
    with st.expander(f"💡 {title}"):
        st.markdown(markdown)


def require(condition: bool, message: str, page: str | None = None):
    """Stop the page with a friendly pointer to the step that's missing."""
    if not condition:
        st.info(message)
        if page:
            st.page_link(page, label="Go there →")
        st.stop()


def next_step(page: str, label: str):
    st.divider()
    st.page_link(page, label=f"Next: {label} →", icon="➡️")


def todo(items: list[str]):
    """Visible checklist for teammates while the app is being built. Remove before the demo."""
    with st.expander("🛠 Team to-do for this page"):
        for i in items:
            st.markdown(f"- [ ] {i}")
