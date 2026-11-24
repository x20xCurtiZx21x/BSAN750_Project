"""Model zoo, cross-validated tuning, and evaluation."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from core.data import binary_target
from core.prep import build_preprocessor, to_frame

# name -> (factory, tuned hyperparameter, values to try, what it means in plain English)
ZOO = {
    "Logistic regression": (
        lambda: LogisticRegression(max_iter=2000),
        "C", [0.01, 0.1, 1, 10],
        "Adds up weighted features into a score. Easy to explain, and a strong baseline. "
        "C controls how freely weights can grow: small C = simpler model.",
    ),
    "Random forest": (
        lambda: RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=0),
        "max_depth", [4, 8, 12, 20],
        "Hundreds of decision trees vote. Handles interactions automatically. "
        "max_depth limits how many questions each tree may ask: deeper trees can memorise.",
    ),
    "Gradient boosting": (
        lambda: HistGradientBoostingClassifier(random_state=0),
        "max_depth", [2, 4, 6, 10],
        "Builds trees one after another, each fixing the last one's mistakes. Often the most accurate on tables. "
        "max_depth controls how complex each tree is.",
    ),
    "Neural network": (
        lambda: MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=0),
        "alpha", [1e-4, 1e-3, 1e-2, 1e-1],
        "Layers of neurons learn their own feature combinations. Flexible but harder to explain. "
        "alpha penalises large weights: bigger alpha = smoother, simpler model.",
    ),
}


def make_split(df, target, positive, cfg, sample, test_size, seed=0):
    d = df.sample(min(sample, len(df)), random_state=seed) if len(df) > sample else df
    X = to_frame(d, target, cfg)
    y = binary_target(d, target, positive)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def evaluate(model, X_test, y_test, threshold=0.5) -> dict:
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, p)
    return {
        "ROC AUC": roc_auc_score(y_test, p),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "proba": p,
        "roc": (fpr, tpr),
    }


def train(df, target, positive, cfg, names, sample=20000, test_size=0.25, tune=True, cv=3, progress=None) -> dict:
    """Train each chosen model, optionally tuning one hyperparameter with k-fold cross-validation."""
    X_train, X_test, y_train, y_test = make_split(df, target, positive, cfg, sample, test_size)
    results = {}
    for i, name in enumerate(names):
        if progress:
            progress(i / len(names), f"Training {name}…")
        factory, param, grid, _ = ZOO[name]
        pipe = Pipeline([("prep", build_preprocessor(df, target, cfg)), ("model", factory())])
        t0 = time.time()
        curve = None
        if tune:
            search = GridSearchCV(pipe, {f"model__{param}": grid}, cv=cv, scoring="roc_auc",
                                  return_train_score=True, n_jobs=1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best = search.best_params_[f"model__{param}"]
            cvr = search.cv_results_
            curve = pd.DataFrame({
                "value": [str(v) for v in grid],
                "train AUC": cvr["mean_train_score"],
                "validation AUC": cvr["mean_test_score"],
            })
        else:
            model = pipe.fit(X_train, y_train)
            best = model.named_steps["model"].get_params()[param]
        results[name] = {
            "model": model,
            "param": param,
            "best": best,
            "curve": curve,
            "seconds": time.time() - t0,
            **evaluate(model, X_test, y_test),
        }
    if progress:
        progress(1.0, "Done")
    best_name = max(results, key=lambda n: results[n]["ROC AUC"])
    base_rate = float(y_train.mean())
    return {"results": results, "best": best_name, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test, "base_rate": base_rate,
            "settings": {"sample": sample, "test_size": test_size, "tune": tune, "cv": cv}}


def tuning_story(name: str, r: dict) -> str:
    """One or two sentences explaining the tuning curve, computed from the actual scores."""
    c = r["curve"]
    if c is None:
        return f"Trained with default settings ({r['param']} = {r['best']})."
    i = int(np.argmax(c["validation AUC"]))
    gap = float(c["train AUC"].iloc[i] - c["validation AUC"].iloc[i])
    text = (f"Tried {r['param']} = {', '.join(c['value'])}. The best validation AUC "
            f"({c['validation AUC'].iloc[i]:.3f}) came at {r['param']} = {c['value'].iloc[i]}. ")
    if gap > 0.05:
        text += (f"Training AUC is {gap:.2f} higher than validation, so the model is partly **memorising** the "
                 "training data (overfitting). A simpler setting or more data would help.")
    elif gap < 0.01:
        text += "Training and validation scores are almost equal, so there is no sign of overfitting."
    else:
        text += f"The small train–validation gap ({gap:.3f}) is normal."
    if 0 < i == len(c) - 1:
        text += f" The best value is the **largest one tried**, so even larger {r['param']} values might score higher."
    elif i == 0 and len(c) > 1:
        text += f" The best value is the **smallest one tried**, so even smaller {r['param']} values might score higher."
    return text
