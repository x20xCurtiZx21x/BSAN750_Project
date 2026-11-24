"""Turn probabilities into a business decision: pick the cutoff that maximises expected value."""
import numpy as np
import pandas as pd


def threshold_table(y_true, proba, gain_tp, cost_fp, cost_fn, gain_tn=0.0) -> pd.DataFrame:
    """Expected value of acting on every row scored at or above each threshold."""
    y = np.asarray(y_true)
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        act = proba >= t
        tp = int(np.sum(act & (y == 1))); fp = int(np.sum(act & (y == 0)))
        fn = int(np.sum(~act & (y == 1))); tn = int(np.sum(~act & (y == 0)))
        value = tp * gain_tp - fp * cost_fp - fn * cost_fn + tn * gain_tn
        rows.append({"threshold": t, "flagged": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "value": value, "value per 1,000": 1000 * value / len(y)})
    return pd.DataFrame(rows)


def baselines(y_true, gain_tp, cost_fp, cost_fn, gain_tn=0.0) -> dict:
    """What happens if you act on nobody, or on everybody."""
    y = np.asarray(y_true)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    nobody = -pos * cost_fn + neg * gain_tn
    everybody = pos * gain_tp - neg * cost_fp
    return {"act on nobody": 1000 * nobody / len(y), "act on everybody": 1000 * everybody / len(y)}
