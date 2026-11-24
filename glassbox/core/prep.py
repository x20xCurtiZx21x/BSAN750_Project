"""Turning raw columns into model-ready numbers, with suggestions the user can accept or change."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.data import is_numeric

# numeric columns whose numbers are really labels (agent #9 is not "more" than agent #3)
CODE_HINTS = ("id", "agent", "company", "code", "zip", "postal")


def suggest(df: pd.DataFrame, target: str, report: dict) -> dict:
    """Default Prep choices, each with a reason shown to the user."""
    drop, reasons = [], {}
    for sev, col, msg in report["issues"]:
        if col == target or col in drop or col not in df.columns:
            continue
        if sev == "high" or "only one value" in msg or "ID or free text" in msg or "missing. Mostly" in msg:
            drop.append(col)
            reasons[col] = msg
    as_category = [
        c for c in df.columns
        if c != target and c not in drop and is_numeric(df[c])
        and any(h in c.lower() for h in CODE_HINTS)
    ]
    return {"drop": drop, "drop_reasons": reasons, "as_category": as_category, "max_categories": 15}


def split_columns(df: pd.DataFrame, target: str, cfg: dict):
    feats = [c for c in df.columns if c != target and c not in cfg["drop"]]
    num = [c for c in feats if is_numeric(df[c]) and c not in cfg["as_category"]]
    cat = [c for c in feats if c not in num]
    return feats, num, cat


def to_frame(df: pd.DataFrame, target: str, cfg: dict) -> pd.DataFrame:
    """Raw feature frame with code-like numbers cast to text so they are treated as categories."""
    feats, num, cat = split_columns(df, target, cfg)
    X = df[feats].copy()
    for c in cat:  # missing becomes its own category: "no agent" can itself be informative
        col = X[c]
        if is_numeric(col) and (col.dropna() % 1 == 0).all():
            col = col.astype("Int64")  # agent 19.0 -> "19"
        X[c] = col.astype("string").fillna("(missing)").astype(object)
    return X


def build_preprocessor(df: pd.DataFrame, target: str, cfg: dict) -> ColumnTransformer:
    """Numbers: fill gaps with the median, then standardise. Categories: one-hot encode the most common values."""
    _, num, cat = split_columns(df, target, cfg)
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=cfg["max_categories"],
                                 sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, num), ("cat", categorical, cat)])
