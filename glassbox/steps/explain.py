import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import explain, llm, state, ui

ui.header(4, "🔍 Explain", "Which inputs matter overall, and **why did the model score this one case the way it did?**")
ui.require(state.get("lab") is not None, "Train at least one model in the Model lab first.", "steps/model_lab.py")
lab, target, positive = state.get("lab"), state.get("target"), state.get("positive")
res = lab["results"]
name = st.selectbox("Model", list(res), index=list(res).index(lab["best"]))
model = res[name]["model"]
X_test, y_test = lab["X_test"], lab["y_test"]

# ---------------------------------------------------------------- global importance
st.subheader("What matters overall")
key = f"importance::{name}"
if key not in lab:
    with st.spinner("Shuffling each column to see how much the model relies on it…"):
        lab[key] = explain.global_importance(model, X_test, y_test)
imp = lab[key]
if name == lab["best"]:
    lab["importance"] = imp  # shared with the AI assistant
top = imp.head(12)
ui.chart(alt.Chart(top).mark_bar().encode(
    x=alt.X("AUC drop:Q", title="Drop in AUC when this column is scrambled"),
    y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260))))
st.markdown(f"Scrambling **`{top.iloc[0]['feature']}`** hurts the most (AUC −{top.iloc[0]['AUC drop']:.3f}), so the model "
            f"leans on it heavily. " + (f"`{top.iloc[1]['feature']}` and `{top.iloc[2]['feature']}` come next." if len(top) > 2 else ""))
ui.explainer("**Permutation importance**: shuffle one column so it no longer lines up with the outcome, then re-score. "
             "A big drop means the model depends on that column. It works for any model. Important ≠ causal: "
             "a column can matter because it is a *signal of* something else.")

# ---------------------------------------------------------------- walkthrough
st.subheader("Walk me through one prediction")
proba = res[name]["proba"]
order = np.argsort(proba)
pick_mode = st.radio("Pick a case", ["Borderline", "Highest risk", "Lowest risk", "Random"], horizontal=True)
if pick_mode == "Highest risk":
    i = order[-1]
elif pick_mode == "Lowest risk":
    i = order[0]
elif pick_mode == "Borderline":
    i = int(np.argmin(np.abs(proba - 0.5)))
else:
    i = int(st.session_state.setdefault("random_case", np.random.default_rng().integers(len(proba))))
    if st.button("Another random case"):
        st.session_state.random_case = int(np.random.default_rng().integers(len(proba)))
        st.rerun()
row = X_test.iloc[[i]]
actual = int(y_test.iloc[i])

ref = explain.typical_values(lab["X_train"])
contrib = explain.contributions(model, row, ref)
whatif = explain.what_if(model, row, lab["X_train"], list(contrib["feature"].head(6)))
label = f"{target} = {positive}"

left, right = st.columns([3, 2])
with left:
    for line in explain.walkthrough(model, row, ref, lab["base_rate"], contrib, whatif, positive_label=label):
        st.markdown(line)
    st.caption(f"What actually happened: {label} was **{'true' if actual else 'false'}**.")
with right:
    show = contrib.head(8).copy()
    show["value"] = show["value"].map(explain.fmt_value)
    show["typical"] = show["typical"].map(explain.fmt_value)
    show["label"] = show["feature"] + " = " + show["value"]
    show["odds"] = np.exp(show["effect"])
    ui.chart(alt.Chart(show).mark_bar().encode(
        x=alt.X("effect:Q", title="← toward no · push · toward yes → (log-odds)"),
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.condition("datum.effect > 0", alt.value("#e4572e"), alt.value("#2e86ab")),
        tooltip=["feature", "value", "typical", alt.Tooltip("odds:Q", format=".2f", title="odds multiplier")]))
with st.expander("This case's raw values"):
    ui.table(row.T.rename(columns={row.index[0]: "value"}).astype(str))

llm.explain_button("feature importance and the selected case", key="explain_case")
ui.todo([
    "Partial dependence plot: how the probability changes across one feature",
    "Let the user edit values (what-if sliders) and re-score live",
    "SHAP values as an advanced option",
])
ui.next_step("steps/decide.py", "Decide")
