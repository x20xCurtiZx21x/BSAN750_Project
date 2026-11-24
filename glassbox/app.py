"""Glassbox: an ML toolbox where nothing is a black box.

Run with:  streamlit run app.py
"""
import streamlit as st

from core import llm, state

st.set_page_config(page_title="Glassbox · ML toolbox", page_icon="🔍", layout="wide")
state.init()

pages = {
    "": [st.Page("steps/home.py", title="Start", icon="🏠", default=True)],
    "Workflow": [
        st.Page("steps/data_check.py", title="1 · Data check", icon="🩺"),
        st.Page("steps/prep.py", title="2 · Prep", icon="🧹"),
        st.Page("steps/model_lab.py", title="3 · Model lab", icon="🧪"),
        st.Page("steps/explain.py", title="4 · Explain", icon="🔍"),
        st.Page("steps/decide.py", title="5 · Decide", icon="💰"),
    ],
    "Help": [st.Page("steps/assistant.py", title="AI assistant", icon="✨")],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🔍 Glassbox")
    st.caption("From spreadsheet to decision, with every step explained.")
    sidebar_status = st.empty()


def draw_status():
    with sidebar_status.container():
        if state.get("dataset_name"):
            st.markdown(f"**Data:** {state.get('dataset_name')}")
        st.markdown("**Progress**")
        for label, done in state.progress():
            st.markdown(f"{'✅' if done else '⬜️'} {label}")
        st.caption("AI assistant: " + ("connected" if llm.available() else "add an API key to enable"))


try:
    nav.run()
finally:  # drawn after the page runs, so the checklist reflects what the page just did
    draw_status()
