import altair as alt
import pandas as pd
import streamlit as st

from core import data, decide, llm, state, ui

ui.header(5, "💰 Decide", "A probability is not a decision. Put a price on each kind of mistake, and Glassbox finds the cutoff "
          "that is worth the most.")
ui.require(state.get("lab") is not None, "Train models in the Model lab first.", "steps/model_lab.py")
lab, target, positive = state.get("lab"), state.get("target"), state.get("positive")
demo = state.get("dataset_name") == data.DEMO["name"]


def money(v: float) -> str:
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"

name = st.selectbox("Model", list(lab["results"]), index=list(lab["results"]).index(lab["best"]))
proba = lab["results"][name]["proba"]
y = lab["y_test"].to_numpy()

st.subheader("What is each outcome worth? (per case)")
c1, c2, c3 = st.columns(3)
if demo:
    st.caption("Hotel framing: if we predict a cancellation, we **act** (ask for a deposit, or overbook that room).")
    gain_tp = c1.number_input("✅ Caught a real cancellation: value gained ($)", 0.0, value=60.0, step=5.0,
                              help="E.g. the room is resold or the deposit is kept.")
    cost_fp = c2.number_input("⚠️ False alarm: cost ($)", 0.0, value=25.0, step=5.0,
                              help="E.g. an annoyed guest, a discount to keep them, or an overbooking walk.")
    cost_fn = c3.number_input("❌ Missed cancellation: cost ($)", 0.0, value=90.0, step=5.0,
                              help="E.g. a room that stays empty for the night.")
else:
    gain_tp = c1.number_input(f"✅ Correctly flagged “{positive}”: value ($)", 0.0, value=50.0, step=5.0)
    cost_fp = c2.number_input("⚠️ False alarm: cost ($)", 0.0, value=20.0, step=5.0)
    cost_fn = c3.number_input(f"❌ Missed “{positive}”: cost ($)", 0.0, value=50.0, step=5.0)

tab = decide.threshold_table(y, proba, gain_tp, cost_fp, cost_fn)
base = decide.baselines(y, gain_tp, cost_fp, cost_fn)
best = tab.loc[tab["value"].idxmax()]
at50 = tab.loc[(tab["threshold"] - 0.5).abs().idxmin()]

m = st.columns(4)
m[0].metric("Best cutoff", f"{best['threshold']:.0%}")
m[1].metric("Value per 1,000 cases", money(best['value per 1,000']),
            f"{best['value per 1,000'] - at50['value per 1,000']:+,.0f} vs. 50% cutoff")
m[2].metric("Cases flagged", f"{best['flagged'] / len(y):.0%}")
m[3].metric("Act on nobody", money(base['act on nobody']))

chart = alt.Chart(tab).mark_line().encode(
    x=alt.X("threshold:Q", title="Cutoff: act when probability ≥", axis=alt.Axis(format="%")),
    y=alt.Y("value per 1,000:Q", title="Value per 1,000 cases ($)"),
    tooltip=[alt.Tooltip("threshold:Q", format=".0%"), alt.Tooltip("value per 1,000:Q", format="$,.0f"), "flagged"])
point = alt.Chart(pd.DataFrame([best])).mark_point(size=120, filled=True, color="#e4572e").encode(x="threshold:Q", y="value per 1,000:Q")
ui.chart(chart + point)

caught = best["TP"] / max(1, best["TP"] + best["FN"])
st.markdown(
    f"**Recommendation:** act on cases the {name.lower()} scores at **{best['threshold']:.0%} or higher**. "
    f"On the test set that flags {int(best['flagged']):,} of {len(y):,} cases, catches **{caught:.0%}** of the real "
    f"“{positive}” cases, and raises {int(best['FP']):,} false alarms. Expected value is **{money(best['value per 1,000'])} per 1,000 cases**, "
    f"compared with {money(base['act on nobody'])} if you do nothing and {money(base['act on everybody'])} if you act on everyone."
)
ui.explainer("The model's default 50% cutoff treats both kinds of mistake as equally bad. In business they rarely are. "
             "When a miss costs more than a false alarm, it pays to act earlier (a lower cutoff), and vice versa. "
             "These dollar values are assumptions: change them and watch the best cutoff move.")

st.session_state.decision = {"model": name, "threshold": float(best["threshold"]), "value_per_1000": float(best["value per 1,000"]),
                             "gain_tp": gain_tp, "cost_fp": cost_fp, "cost_fn": cost_fn,
                             "recall": float(caught), "flagged_share": float(best["flagged"] / len(y))}

flagged = lab["X_test"].assign(probability=proba).query("probability >= @best['threshold']").sort_values("probability", ascending=False)
st.download_button(f"Download the {len(flagged):,} flagged cases (CSV)", flagged.to_csv(index=True), "glassbox_action_list.csv", "text/csv")

llm.explain_button("decision and recommended cutoff", key="explain_decision")
ui.todo([
    "One-page decision memo (PDF) written by the AI assistant",
    "Sensitivity table: best cutoff under low / mid / high cost assumptions",
    "Score a new upload (e.g. next month's bookings) with the saved model",
])
