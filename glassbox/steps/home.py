import streamlit as st

from core import data, state, ui

ui.header(None, "🔍 Glassbox", "**An ML toolbox where nothing is a black box.** Bring a spreadsheet and a yes/no "
          "question. Glassbox guides you from messy data to a decision you can defend, explaining every step "
          "using *your* numbers.")

c1, c2, c3, c4, c5 = st.columns(5)
for col, (icon, name, text) in zip(
    (c1, c2, c3, c4, c5),
    [("🩺", "Data check", "Health report and leakage detector"),
     ("🧹", "Prep", "Cleaning, with a reason for each step"),
     ("🧪", "Model lab", "Four models, tuned and compared"),
     ("🔍", "Explain", "Why the model said what it said"),
     ("💰", "Decide", "The cutoff that makes the most money")],
):
    with col:
        st.markdown(f"#### {icon} {name}")
        st.caption(text)

st.divider()
st.subheader("1. Choose your data")
left, right = st.columns(2)

with left:
    st.markdown(f"**Demo: {data.DEMO['name']}**")
    st.caption(f"119,390 real hotel bookings. *{data.DEMO['question']}*")
    if st.button("Use the demo dataset", type="primary"):
        state.set("df", data.load_demo())
        st.session_state.dataset_name = data.DEMO["name"]
        st.session_state.target = data.DEMO["target"]
        st.session_state.positive = data.DEMO["positive"]
        st.switch_page("steps/data_check.py")

with right:
    st.markdown("**Or upload your own CSV**")
    st.caption("One row per case (customer, booking, loan…) and one column with the yes/no outcome you want to predict.")
    up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if up is not None and st.button("Use this file"):
        state.set("df", data.load_upload(up))
        st.session_state.dataset_name = up.name
        st.switch_page("steps/data_check.py")

if state.get("df") is not None:
    st.success(f"Loaded **{state.get('dataset_name')}**: {len(state.get('df')):,} rows × {state.get('df').shape[1]} columns.")
    st.page_link("steps/data_check.py", label="Continue to Data check →", icon="➡️")
