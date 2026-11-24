import altair as alt
import streamlit as st

from core import data, llm, state, ui

ui.header(1, "🩺 Data check", "Before any modelling: what are we predicting, and can we trust this data?")
ui.require(state.get("df") is not None, "Load a dataset on the Start page first.", "steps/home.py")
df = state.get("df")

# ---------------------------------------------------------------- target
st.subheader("What do you want to predict?")
binary_cols = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
options = binary_cols + [c for c in df.columns if c not in binary_cols]
current = state.get("target")
c1, c2 = st.columns(2)
target = c1.selectbox("Outcome column", options, index=options.index(current) if current in options else 0,
                      help="Glassbox currently handles yes/no outcomes (binary classification).")
values = df[target].dropna().unique().tolist()
if len(values) > 10:
    st.warning("This column has many values. Pick a yes/no column (or create one) for now.")
pos_default = state.get("positive") if state.get("positive") in values else values[-1]
positive = c2.selectbox("Which value counts as “yes”?", values, index=values.index(pos_default))

if target != current or positive != state.get("positive"):
    state.set("target", target)
    st.session_state.positive = positive

if state.get("report") is None:
    with st.spinner("Checking every column…"):
        st.session_state.report = data.health_report(df, target, positive)
report = state.get("report")

# ---------------------------------------------------------------- overview
m = st.columns(5)
m[0].metric("Rows", f"{report['rows']:,}")
m[1].metric("Columns", report["cols"])
m[2].metric("Missing cells", f"{report['missing_cells_pct']}%")
m[3].metric("Duplicate rows", f"{report['duplicates']:,}")
m[4].metric(f"Rate of “{positive}”", f"{report['positive_rate']:.1%}")

ui.explainer(
    f"The model will learn to predict whether **{target} = {positive}**. "
    f"In this data that happens in **{report['positive_rate']:.1%}** of rows. That is the number to beat: "
    "a model that just guessed the most common answer every time would already be right "
    f"{max(report['positive_rate'], 1 - report['positive_rate']):.1%} of the time."
)

# ---------------------------------------------------------------- issues
st.subheader("Health report")
icons = {"high": "🔴", "medium": "🟠", "low": "🟡"}
if report["leakage"]:
    st.error("**Too good to be true:** " + ", ".join(f"`{c}`" for c in report["leakage"]) +
             " leak the outcome: they appear to be recorded *after* it happens, so they would not be available "
             "when making a real prediction. Prep drops them by default.")
shown = [i for i in report["issues"] if i[0] != "low"]
for sev, col, msg in shown[:15]:
    st.markdown(f"{icons[sev]} **`{col}`**: {msg}")
lows = [i for i in report["issues"] if i[0] == "low"]
if lows:
    st.caption(f"Plus {len(lows)} minor note(s): " + ", ".join(f"`{c}`" for _, c, _ in lows) + " have a few missing values.")
if not report["issues"]:
    st.success("No problems found.")

with st.expander("Column profile"):
    ui.table(report["profile"], hide_index=True)

# ---------------------------------------------------------------- quick EDA
st.subheader("Explore: what goes with the outcome?")
single = {k: v for k, v in report["single_auc"].items() if k not in report["leakage"]}
top = list(single)[:8]
st.caption("Columns ranked by how well each one *alone* separates the outcomes (AUC: 0.5 = no signal, 1.0 = perfect). "
           "Strongest non-leaky ones: " + ", ".join(f"`{c}` ({single[c]:.2f})" for c in top[:4]) + ".")
col = st.selectbox("Column to explore", [c for c in df.columns if c != target], index=[c for c in df.columns if c != target].index(top[0]))
by = data.rate_by(df, col, target, positive)
chart = alt.Chart(by).mark_bar().encode(
    x=alt.X("rate:Q", title=f"Share with {target} = {positive}", axis=alt.Axis(format="%")),
    y=alt.Y("group:N", sort="-x", title=col),
    tooltip=["group", alt.Tooltip("rate:Q", format=".1%"), "rows"],
)
rule = alt.Chart(by.assign(avg=report["positive_rate"])).mark_rule(color="red", strokeDash=[4, 4]).encode(x="avg:Q")
ui.chart(chart + rule)
hi, lo = by.iloc[0], by.iloc[-1]
st.markdown(f"For `{col}`, the rate ranges from **{lo['rate']:.0%}** ({lo['group']}) to **{hi['rate']:.0%}** "
            f"({hi['group']}), against an overall {report['positive_rate']:.0%} (red line).")

llm.explain_button("data check", key="explain_data")
ui.todo([
    "Support numeric (regression) targets",
    "Correlation heat map for numeric columns",
    "Date columns: detect and split into month / weekday",
])
ui.next_step("steps/prep.py", "Prep")
