import streamlit as st

from core import prep, state, ui

ui.header(2, "🧹 Prep", "Decide which columns the model may use and how to turn them into numbers. "
          "Glassbox suggests defaults, and you stay in charge.")
ui.require(state.get("report") is not None, "Run the Data check first.", "steps/data_check.py")
df, target, report = state.get("df"), state.get("target"), state.get("report")

sug = prep.suggest(df, target, report)
cfg = state.get("prep") or sug
features = [c for c in df.columns if c != target]

st.subheader("Columns to leave out")
drop = st.multiselect("Dropped columns", features, default=[c for c in cfg["drop"] if c in features])
for c in drop:
    if c in sug["drop_reasons"]:
        st.caption(f"`{c}`: {sug['drop_reasons'][c]}")

st.subheader("Numbers that are really labels")
numeric_left = [c for c in features if c not in drop and prep.is_numeric(df[c])]
as_cat = st.multiselect("Treat these as categories", numeric_left,
                        default=[c for c in cfg["as_category"] if c in numeric_left],
                        help="ID-like codes (agent #9 is not 'more' than agent #3) should be categories, not quantities.")

st.subheader("Encoding")
max_cat = st.slider("Keep the most common categories per column (the rest become 'other')", 5, 50, cfg["max_categories"])

ui.explainer(
    "- **Missing numbers** are filled with the column's median, which is robust to outliers.\n"
    "- **Missing categories** get their own label, `(missing)`, because *missing* can itself be informative.\n"
    "- **Numbers are standardised** (mean 0, spread 1) so models that use distances or weights treat them fairly.\n"
    "- **Categories are one-hot encoded**: each common value becomes its own yes/no column.\n"
    "- All of this is learned on the training data only, then applied to the test data, so nothing leaks."
)

new_cfg = {"drop": drop, "drop_reasons": sug["drop_reasons"], "as_category": as_cat, "max_categories": max_cat}
_, num, cat = prep.split_columns(df, target, new_cfg)
st.markdown(f"**Result:** {len(num)} numeric and {len(cat)} categorical columns go into the model.")

if st.button("Save prep choices", type="primary"):
    state.set("prep", new_cfg)
    st.success("Saved. Models will be trained with these choices.")
elif state.get("prep") is not None and new_cfg != state.get("prep"):
    st.warning("You changed something. Save to apply it (this resets trained models).")

ui.todo([
    "Option: log-transform skewed numbers",
    "Option: class weighting for imbalanced targets",
    "Feature engineering: total nights = weekend + week nights",
])
ui.next_step("steps/model_lab.py", "Model lab")
