"""
Model comparison (proposal section 3.6) and final artefacts.

Four model versions, exactly as Table 6 of the proposal:
  1. rule_baseline   fixed agronomic thresholds, no ML
  2. dt_raw          Decision Tree on raw pre-window climate features
  3. dt_risk         Decision Tree on stochastic risk features
  4. xgb_full        XGBoost on raw + risk features (strongest proposed model)

Validation: temporal hold-out - train on seasons <= 2014, test on
2015-2023 (never shuffled; planting advice must generalize forward
in time). Metrics (proposal section 3.10): macro F1, balanced accuracy,
per-class precision/recall, confusion matrix, multi-class Brier score,
feature importance.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)
from xgboost import XGBClassifier

from crops import CROPS, LABELS, DELAY_FACTOR, DRY_DEKAD_MM
from dataset import RAW_FEATURES, RISK_FEATURES

MODELS = Path(__file__).resolve().parent.parent / "models"
TRAIN_YEAR_MAX = 2014
ALL_FEATURES = RAW_FEATURES + RISK_FEATURES


# ---------------------------------------------------------------- baseline
def rule_baseline_predict(table: pd.DataFrame,
                          clim_cycle: dict) -> np.ndarray:
    """Fixed-threshold agronomic rules (no ML, no simulation).
    Uses the training-period climatological cycle rainfall for the window
    plus the observable onset/recent-rain state."""
    preds = []
    for r in table.itertuples():
        c = CROPS[r.crop]
        expected = clim_cycle[(r.crop, r.window_start_dekad)]
        if (not r.onset_reached and r.last_dekad_rain < DRY_DEKAD_MM
                and r.window_start_dekad >= 27):
            preds.append("delay")          # season rains have not started
        elif expected >= c["min_cycle_mm"] and r.last_dekad_rain >= DRY_DEKAD_MM:
            preds.append("suitable")
        elif expected >= DELAY_FACTOR * c["min_cycle_mm"]:
            preds.append("risky")
        else:
            preds.append("delay")
    return np.array(preds)


def climatological_cycle_rain(table: pd.DataFrame) -> dict:
    """Mean observed cycle rainfall per (crop, window), training years only."""
    train = table[table.year <= TRAIN_YEAR_MAX]
    return {key: float(v) for key, v in
            train.groupby(["crop", "window_start_dekad"])["actual_cycle_rain"]
                 .mean().items()}


# ----------------------------------------------------------------- metrics
def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_proba: np.ndarray | None = None) -> dict:
    """All proposal metrics for one model on the test set."""
    labels = LABELS
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    onehot = np.array([[t == l for l in labels] for t in y_true], dtype=float)
    if y_proba is None:                       # hard predictions -> 0/1 "probs"
        y_proba = np.array([[p == l for l in labels] for p in y_pred],
                           dtype=float)
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels,
                                         average="macro", zero_division=0)), 3),
        "balanced_accuracy": round(float(
            balanced_accuracy_score(y_true, y_pred)), 3),
        "brier_score": round(float(
            np.mean(np.sum((y_proba - onehot) ** 2, axis=1))), 3),
        "per_class": {l: {"precision": round(float(p), 3),
                          "recall": round(float(r), 3),
                          "f1": round(float(f), 3),
                          "support": int(s)}
                      for l, p, r, f, s in zip(labels, prec, rec, f1, support)},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
    }


# ------------------------------------------------------------ model suite
def _dt():
    return DecisionTreeClassifier(max_depth=4, min_samples_leaf=8,
                                  class_weight="balanced", random_state=0)


def _xgb():
    # shallow, slow-learning ensemble: best macro-F1 / balanced-accuracy /
    # Brier trade-off on the temporal hold-out (depth 2 regularizes well
    # on ~600 training rows)
    return XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.03,
                         subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                         objective="multi:softprob", num_class=len(LABELS),
                         eval_metric="mlogloss", random_state=0)


def _sample_weights(y_id: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_id, minlength=len(LABELS))
    w = len(y_id) / (len(LABELS) * np.maximum(counts, 1))
    return w[y_id]


def compare_models(table: pd.DataFrame) -> tuple[dict, dict]:
    """Train the four proposal models on <=2014, evaluate on 2015+."""
    train = table[table.year <= TRAIN_YEAR_MAX]
    test = table[table.year > TRAIN_YEAR_MAX]
    y_tr, y_te = train.label.to_numpy(), test.label.to_numpy()
    y_tr_id = train.label_id.to_numpy()

    report, fitted = {}, {}

    clim_cycle = climatological_cycle_rain(table)
    report["rule_baseline"] = evaluate(
        y_te, rule_baseline_predict(test, clim_cycle))

    dt_raw = _dt().fit(train[RAW_FEATURES], y_tr)
    report["dt_raw"] = evaluate(y_te, dt_raw.predict(test[RAW_FEATURES]),
                                dt_raw.predict_proba(test[RAW_FEATURES]))
    fitted["dt_raw"] = dt_raw

    dt_risk = _dt().fit(train[RISK_FEATURES], y_tr)
    report["dt_risk"] = evaluate(y_te, dt_risk.predict(test[RISK_FEATURES]),
                                 dt_risk.predict_proba(test[RISK_FEATURES]))
    fitted["dt_risk"] = dt_risk

    xgb = _xgb()
    xgb.fit(train[ALL_FEATURES], y_tr_id,
            sample_weight=_sample_weights(y_tr_id))
    proba = xgb.predict_proba(test[ALL_FEATURES])
    pred = np.array(LABELS)[proba.argmax(axis=1)]
    report["xgb_full"] = evaluate(y_te, pred, proba)
    report["xgb_full"]["feature_importance"] = {
        f: round(float(v), 4) for f, v in sorted(
            zip(ALL_FEATURES, xgb.feature_importances_),
            key=lambda kv: -kv[1])}
    fitted["xgb_full"] = xgb

    report["_meta"] = {
        "train_years": f"{int(train.year.min())}-{TRAIN_YEAR_MAX}",
        "test_years": f"{TRAIN_YEAR_MAX + 1}-{int(test.year.max())}",
        "n_train": len(train), "n_test": len(test),
        "test_label_counts": test.label.value_counts().to_dict(),
        "labels": LABELS,
        "raw_features": RAW_FEATURES,
        "risk_features": RISK_FEATURES,
    }
    return report, fitted


def train_final(table: pd.DataFrame) -> dict:
    """Compare models, then refit the selected XGBoost on ALL years and
    save deployment artefacts + the comparison report."""
    MODELS.mkdir(exist_ok=True)
    report, _ = compare_models(table)

    final = _xgb()
    y_id = table.label_id.to_numpy()
    final.fit(table[ALL_FEATURES], y_id, sample_weight=_sample_weights(y_id))
    final.save_model(MODELS / "xgb_planting_risk.json")
    (MODELS / "report.json").write_text(json.dumps(report, indent=2))
    return report


def load_final() -> XGBClassifier:
    m = _xgb()
    m.load_model(MODELS / "xgb_planting_risk.json")
    return m
