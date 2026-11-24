import altair as alt
import pandas as pd
import streamlit as st

from core import llm, models, state, ui

ui.header(3, "🧪 Model lab", "Train several kinds of model, tune each one fairly, and compare them on data they have never seen.")
ui.require(state.get("prep") is not None, "Save your Prep choices first.", "steps/prep.py")
df, target, positive, cfg = state.get("df"), state.get("target"), state.get("positive"), state.get("prep")

with st.form("train"):
    names = st.multiselect("Models", list(models.ZOO), default=list(models.ZOO))
    c1, c2, c3 = st.columns(3)
    sample = c1.select_slider("Rows to use", [5000, 10000, 20000, 50000, len(df)],
                              value=min(20000, len(df)), format_func=lambda v: f"{v:,}" if v < len(df) else f"all {v:,}")
    test_size = c2.slider("Held-out test share", 0.1, 0.4, 0.25, 0.05)
    tune = c3.checkbox("Tune with 3-fold cross-validation", value=True)
    go = st.form_submit_button("Train models", type="primary")

with st.expander("💡 What do these models do?"):
    for name, (_, param, grid, text) in models.ZOO.items():
        st.markdown(f"**{name}**: {text} *(tuning `{param}` over {grid})*")

if go and names:
    bar = st.progress(0.0, "Starting…")
    lab = models.train(df, target, positive, cfg, names, sample=sample, test_size=test_size, tune=tune,
                       progress=lambda f, t: bar.progress(f, t))
    state.set("lab", lab)
    bar.empty()

lab = state.get("lab")
if lab is None:
    st.stop()

res = lab["results"]
best = lab["best"]
table = pd.DataFrame({n: {k: r[k] for k in ("ROC AUC", "Accuracy", "Precision", "Recall", "F1")} | {"seconds": r["seconds"]}
                      for n, r in res.items()}).T.sort_values("ROC AUC", ascending=False)
st.subheader("Results on the held-out test set")
ui.table(table.style.format("{:.3f}").format("{:.1f}", subset=["seconds"]).highlight_max(subset=["ROC AUC", "F1"], color="#1f6f5c"))
r = res[best]
st.markdown(
    f"**{best}** ranks cases best (AUC {r['ROC AUC']:.3f}): pick a random “{positive}” and a random other case, "
    f"and it scores the “{positive}” one higher {r['ROC AUC']:.0%} of the time. At the default 50% cutoff it catches "
    f"**{r['Recall']:.0%}** of the real “{positive}” cases, and **{r['Precision']:.0%}** of its alarms are correct. "
    f"For comparison, always guessing the most common answer would be right {max(lab['base_rate'], 1 - lab['base_rate']):.0%} of the time."
)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**ROC curves**: higher and further left is better")
    roc = pd.concat([pd.DataFrame({"False positive rate": v["roc"][0], "True positive rate": v["roc"][1], "model": n})
                     for n, v in res.items()])
    ui.chart(alt.Chart(roc).mark_line().encode(x="False positive rate", y="True positive rate", color="model"))
with c2:
    pick = st.selectbox("Tuning curve for", list(res), index=list(res).index(best))
    rr = res[pick]
    if rr["curve"] is not None:
        long = rr["curve"].melt("value", var_name="set", value_name="AUC")
        ui.chart(alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("value:N", title=rr["param"], sort=None), y=alt.Y("AUC:Q", scale=alt.Scale(zero=False)), color="set"))
    st.markdown(models.tuning_story(pick, rr))

llm.explain_button("model comparison", key="explain_models")
ui.todo([
    "Confusion matrix for the selected model",
    "Tune two hyperparameters (grid or randomised search)",
    "Regression models when the target is numeric",
])
ui.next_step("steps/explain.py", "Explain")
