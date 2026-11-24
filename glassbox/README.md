# 🔍 Glassbox: an ML toolbox where nothing is a black box

**From spreadsheet to decision, with every step explained using your own data.**

Glassbox is a Streamlit app that guides business users (no ML background needed) through a full predictive-modelling workflow, from data check to dollar decision. It explains each result in plain English, computed from the user's actual numbers. BSAN 750 final group project, Fall 2026.

## Run it locally (Mac)

```bash
cd "path/to/glassbox"
python3 -m venv .venv               # one time
source .venv/bin/activate
pip install -r requirements.txt     # one time
streamlit run app.py
```

The app opens at http://localhost:8501. Press `Ctrl+C` in the terminal to stop it. Next time, you only need the `source` and `streamlit run` lines.

**AI assistant (optional):** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste in the class OpenAI key. Everything else works without it. `secrets.toml` is git-ignored, so the key never reaches GitHub.

## The workflow

| Step | Page | What it does |
|---|---|---|
| 0 | **Start** | Load the demo dataset or upload a CSV |
| 1 | **Data check** | Pick the yes/no outcome; health report (missing values, duplicates, imbalance); **leakage detector** flags columns that are "too good to be true"; quick exploration chart |
| 2 | **Prep** | Suggested column drops with reasons, ID codes treated as categories, one-hot encoding; all fitted on training data only |
| 3 | **Model lab** | Logistic regression, random forest, gradient boosting and a neural net, each tuned by 3-fold cross-validation; ROC curves; **tuning curves with a plain-English overfitting check** |
| 4 | **Explain** | Permutation importance, plus **"Walk me through one prediction"**: why this case got its score (odds multipliers per feature) and the single change that would flip it |
| 5 | **Decide** | Enter the dollar value of hits, false alarms and misses; Glassbox finds the profit-maximising cutoff and exports the action list |
| – | **AI assistant** | OpenAI chat that only sees the numbers the app computed (shown on the page), so it explains instead of inventing |

## Demo: hotel cancellations

`data/hotel_bookings.csv.gz` has 119,390 real bookings at two Portuguese hotels (Antonio, Almeida & Nunes, 2019). The question: *which bookings will cancel, and should the hotel ask for a deposit or overbook?*

Demo storyline (about 8 minutes):
1. **Data check:** the leakage detector catches `reservation_status` (it literally says "Canceled") and `reservation_status_date`.
2. **Model lab:** gradient boosting wins (AUC ≈ 0.95 on 20k rows). The tuning curve shows the best depth is the largest one tried, so we discuss what to try next.
3. **Explain:** non-refundable deposits and long lead times push cancellation risk up; special requests and parking requests pull it down.
4. **Decide:** with $60 gained per caught cancellation, $25 per false alarm and $90 per missed one, the best cutoff is about 16%, not 50%. That's worth about $4,700 more per 1,000 bookings than the default.

## Project structure

```
app.py               navigation + sidebar progress
steps/               one file per page (each teammate can own one on their own branch)
core/                the logic, independent of the UI
  data.py            loading, health report, leakage detector
  prep.py            preprocessing choices and sklearn ColumnTransformer
  models.py          model zoo, cross-validated tuning, evaluation, tuning explanations
  explain.py         permutation importance, per-row walkthrough, what-if
  decide.py          cost-based threshold optimisation
  llm.py             OpenAI assistant (context = computed results only)
  state.py           shared session state; changing a step resets later steps
  ui.py              shared page helpers
data/                demo dataset
```

Each page ends with a **🛠 Team to-do** list of next features. Remove these before the demo.

## Suggested split for the team

Each person creates a branch and owns a page plus its `core/` module: Data check + Prep · Model lab · Explain · Decide + AI assistant · demo storyline, README and deployment (Streamlit Community Cloud).
