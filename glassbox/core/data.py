"""Loading data and the plain-English data health check, including a leakage detector."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEMO = {
    "name": "Hotel bookings (Portugal, 2015–2017)",
    "file": DATA_DIR / "hotel_bookings.csv.gz",
    "target": "is_canceled",
    "positive": 1,
    "question": "Which bookings are likely to be cancelled, and should the hotel ask for a deposit or overbook?",
}

LEAKAGE_AUC = 0.95   # a single column this predictive is "too good to be true"
ID_UNIQUE_RATIO = 0.9


@st.cache_data(show_spinner="Loading the demo dataset…")
def load_demo() -> pd.DataFrame:
    return pd.read_csv(DEMO["file"])


def load_upload(file) -> pd.DataFrame:
    return pd.read_csv(file)


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def binary_target(df: pd.DataFrame, target: str, positive) -> pd.Series:
    return (df[target] == positive).astype(int)


def single_feature_auc(x: pd.Series, y: pd.Series) -> float:
    """How well one column on its own separates the classes (0.5 = useless, 1.0 = perfect)."""
    if x.nunique(dropna=True) <= 1:
        return 0.5
    if is_numeric(x):
        score = x.fillna(x.median())
    else:  # categorical: score each category by its positive rate
        cat = x.astype("string").fillna("(missing)")
        score = cat.map(y.groupby(cat).mean())
    try:
        auc = roc_auc_score(y, score)
    except ValueError:
        return 0.5
    return float(max(auc, 1 - auc))


def health_report(df: pd.DataFrame, target: str, positive, sample: int = 20000, seed: int = 0) -> dict:
    """Everything the Data check page shows, computed once and stored in session state."""
    y_full = binary_target(df, target, positive)
    d = df.sample(min(sample, len(df)), random_state=seed) if len(df) > sample else df
    y = binary_target(d, target, positive)
    features = [c for c in df.columns if c != target]

    profile = []
    for c in features:
        s = df[c]
        profile.append({
            "column": c,
            "type": "number" if is_numeric(s) else "category",
            "missing %": round(100 * s.isna().mean(), 1),
            "unique values": int(s.nunique(dropna=True)),
            "example": str(s.dropna().iloc[0]) if s.notna().any() else "",
        })
    profile = pd.DataFrame(profile)

    issues = []  # (severity, column, message)
    auc = {c: single_feature_auc(d[c], y) for c in features}
    for c in features:
        s = df[c]
        miss = s.isna().mean()
        if auc[c] >= LEAKAGE_AUC:
            issues.append(("high", c, f"Predicts the target almost perfectly on its own (AUC {auc[c]:.2f}). "
                                      "It probably contains information you would only know *after* the outcome "
                                      "(data leakage). Drop it, or the model will look great and fail in real use."))
        if miss > 0.6:
            issues.append(("medium", c, f"{miss:.0%} missing. Mostly empty columns rarely help."))
        elif miss > 0:
            issues.append(("low", c, f"{miss:.1%} missing. These will be filled in during Prep."))
        if s.nunique(dropna=True) <= 1:
            issues.append(("medium", c, "Has only one value, so it cannot help predict anything."))
        elif not is_numeric(s) and s.nunique() / max(1, s.notna().sum()) > ID_UNIQUE_RATIO:
            issues.append(("medium", c, "Almost every row is different. Looks like an ID or free text."))

    # columns derived from a leaky column (e.g. reservation_status -> reservation_status_date) leak too
    leaky = [c for c in features if auc[c] >= LEAKAGE_AUC]
    for c in features:
        if c not in leaky and any(c != l and c.lower().startswith(l.lower()) for l in leaky):
            src = next(l for l in leaky if c.lower().startswith(l.lower()))
            leaky.append(c)
            issues.append(("high", c, f"Looks derived from `{src}`, which leaks the outcome, so it probably leaks too "
                                      f"(on its own AUC {auc[c]:.2f})."))

    dup = int(df.duplicated().sum())
    if dup / len(df) > 0.01:
        issues.append(("medium", "(all columns)", f"{dup:,} rows ({dup / len(df):.0%}) are exact copies of another row. "
                       "They may be genuine (e.g. identical group bookings), but copies that land in both the training "
                       "and test data make the test score look better than it really is."))

    pos_rate = float(y_full.mean())
    if pos_rate < 0.1 or pos_rate > 0.9:
        issues.append(("medium", target, f"Imbalanced target: only {min(pos_rate, 1 - pos_rate):.1%} of rows are in the "
                                         "rarer class. Accuracy will be misleading, so look at recall and AUC instead."))

    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda t: order[t[0]])
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "duplicates": int(df.duplicated().sum()),
        "missing_cells_pct": round(100 * df.isna().mean().mean(), 2),
        "positive_rate": pos_rate,
        "profile": profile,
        "issues": issues,
        "single_auc": dict(sorted(auc.items(), key=lambda kv: -kv[1])),
        "leakage": leaky,
    }


def rate_by(df: pd.DataFrame, col: str, target: str, positive, bins: int = 8, top: int = 12) -> pd.DataFrame:
    """Positive rate by category (or by bin for numbers), for the quick-EDA chart."""
    y = binary_target(df, target, positive)
    s = df[col]
    if is_numeric(s) and s.nunique() > top:
        groups = pd.qcut(s, q=bins, duplicates="drop").astype(str)
    else:
        groups = s.astype("string").fillna("(missing)")
        keep = groups.value_counts().index[:top]
        groups = groups.where(groups.isin(keep), "(other)")
    out = pd.DataFrame({"group": groups, "y": y}).groupby("group", observed=True)["y"].agg(["mean", "size"])
    out = out.rename(columns={"mean": "rate", "size": "rows"}).reset_index()
    return out.sort_values("rate", ascending=False)
