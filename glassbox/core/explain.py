"""Why did the model say that? Global importance and a per-row walkthrough computed from the row's own values."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def global_importance(model, X_test, y_test, n=2000, seed=0) -> pd.DataFrame:
    """Shuffle one column at a time and measure how much the AUC drops (model-agnostic)."""
    Xs = X_test.sample(min(n, len(X_test)), random_state=seed)
    ys = y_test.loc[Xs.index]
    r = permutation_importance(model, Xs, ys, scoring="roc_auc", n_repeats=3, random_state=seed, n_jobs=1)
    out = pd.DataFrame({"feature": Xs.columns, "AUC drop": r.importances_mean})
    return out.sort_values("AUC drop", ascending=False).reset_index(drop=True)


def typical_values(X: pd.DataFrame) -> pd.Series:
    """The 'average booking' used as a reference: median for numbers, most common value for categories."""
    vals = {}
    for c in X.columns:
        s = X[c]
        vals[c] = s.median() if pd.api.types.is_numeric_dtype(s) else s.mode(dropna=True).iloc[0]
    return pd.Series(vals, dtype=object)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def pct(p: float) -> str:
    """Readable probability: never shows 100% or 0% for a model output."""
    if p > 0.999:
        return ">99.9%"
    if p < 0.001:
        return "<0.1%"
    return f"{p:.1%}" if (p > 0.99 or p < 0.01) else f"{p:.0%}"


def contributions(model, row: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """For each feature: how much this row's value pushes the prediction compared with the typical value.
    Measured in log-odds, so it still works when the probability is close to 0% or 100%.
    Positive = pushes toward "yes". exp(effect) is the odds multiplier."""
    p_row = float(model.predict_proba(row)[:, 1][0])
    probes = pd.concat([row] * len(row.columns), ignore_index=True)
    for i, c in enumerate(row.columns):
        probes.loc[i, c] = reference[c]
    p_probe = model.predict_proba(probes)[:, 1]
    out = pd.DataFrame({
        "feature": row.columns,
        "value": [row.iloc[0][c] for c in row.columns],
        "typical": [reference[c] for c in row.columns],
        "effect": logit(p_row) - logit(p_probe),
    })
    out = out[out["value"].astype(str) != out["typical"].astype(str)]
    return out.reindex(out["effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def what_if(model, row: pd.DataFrame, X_ref: pd.DataFrame, features, threshold=0.5, per_feature=6):
    """Try changing one feature at a time to common alternative values; return the single change that
    moves the prediction most toward the other side of the threshold."""
    p_row = float(model.predict_proba(row)[:, 1][0])
    want_down = p_row >= threshold
    probes, meta = [], []
    for c in features:
        s = X_ref[c]
        if pd.api.types.is_numeric_dtype(s):
            cands = np.unique(np.round(s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy(), 2))
        else:
            cands = s.value_counts().index[:per_feature].to_numpy()
        for v in cands:
            if str(v) == str(row.iloc[0][c]):
                continue
            r = row.copy()
            r.loc[r.index[0], c] = v
            probes.append(r)
            meta.append((c, v))
    if not probes:
        return None
    p = model.predict_proba(pd.concat(probes, ignore_index=True))[:, 1]
    lp = logit(p)
    i = int(np.argmin(lp) if want_down else np.argmax(lp))
    return {"feature": meta[i][0], "from": row.iloc[0][meta[i][0]], "to": meta[i][1],
            "p_before": p_row, "p_after": float(p[i]), "flips": (p[i] < threshold) == want_down}


def fmt_value(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)


def walkthrough(model, row, reference, base_rate, contrib: pd.DataFrame, whatif, positive_label="positive",
                threshold=0.5) -> list[str]:
    """The step-by-step story for one row. Every number comes from the model and this row."""
    p = float(model.predict_proba(row)[:, 1][0])
    steps = [
        f"**Starting point.** Across the training data, {base_rate:.0%} of rows are {positive_label}. "
        f"That is the baseline before looking at this row.",
        f"**This row.** The model gives it a **{pct(p)}** chance of being {positive_label}, "
        f"{'above' if p >= threshold else 'below'} the {threshold:.0%} decision line.",
    ]
    ups = contrib[contrib["effect"] > 0.05].head(3)      # > ~5% change in odds
    downs = contrib[contrib["effect"] < -0.05].head(3)
    if len(ups):
        items = "; ".join(f"`{r.feature}` = {fmt_value(r.value)} (typical: {fmt_value(r.typical)}) "
                          f"**×{np.exp(r.effect):.1f} odds**" for r in ups.itertuples())
        steps.append(f"**Pushing it up:** {items}.")
    if len(downs):
        items = "; ".join(f"`{r.feature}` = {fmt_value(r.value)} (typical: {fmt_value(r.typical)}) "
                          f"**÷{np.exp(-r.effect):.1f} odds**" for r in downs.itertuples())
        steps.append(f"**Pulling it down:** {items}.")
    if len(ups) or len(downs):
        steps.append("_“×3 odds” means: compared with a typical value, this value makes “yes” three times as likely "
                     "relative to “no”._")
    if not len(ups) and not len(downs):
        steps.append("This row looks very typical: no single feature moves the prediction much.")
    if whatif:
        change = (f"changing `{whatif['feature']}` from {fmt_value(whatif['from'])} to {fmt_value(whatif['to'])} "
                  f"would move it from {pct(whatif['p_before'])} to **{pct(whatif['p_after'])}**")
        steps.append(f"**What would change the answer?** The single biggest lever: {change}"
                     + (", enough to flip the decision." if whatif["flips"] else ", though not enough to flip the decision on its own."))
    steps.append("_Effects are measured by swapping one value for the typical one and re-scoring, "
                 "so they show this model's behaviour, not proof of cause and effect._")
    return steps
