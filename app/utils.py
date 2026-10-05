"""Helpers for the Streamlit demo: load the model, build a patient row, predict.

The saved model is a full scikit-learn pipeline (preprocessing + classifier),
so the app only has to hand it one row of raw patient values with the same
column names as the dataset. Missing lab values are passed as NaN and handled
exactly as in training (median imputation + a "was missing" indicator).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data import split  # noqa: E402

APP_INPUTS = ROOT / "app" / "app_inputs.json"

# Human-readable labels and units for the inputs shown in the app.
LABELS = {
    "age": "Age (years)", "stage": "Cancer stage", "ecog_status": "ECOG performance status",
    "baseline_gph": "Baseline GPH score (physical)", "baseline_gmh": "Baseline GMH score (mental)",
    "num_comorbidities": "Number of comorbidities", "depression_history": "History of depression",
    "albumin_g_dl": "Albumin (g/dL)", "hemoglobin_g_dl": "Hemoglobin (g/dL)",
    "weight_loss_pct_6m": "Weight loss in last 6 months (%)", "cancer_type": "Cancer type",
    "chemo_regimen": "Chemotherapy regimen",
}
INTEGER_INPUTS = {"age", "stage", "ecog_status", "num_comorbidities", "depression_history"}
BINARY_INPUTS = {"depression_history"}
LAB_INPUTS = {"albumin_g_dl", "hemoglobin_g_dl"}  # can be "not measured"


def label(col):
    return LABELS.get(col, col.replace("_", " "))


def load_bundle(path=None):
    path = Path(path) if path else config.MODELS_DIR / "final_model.joblib"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"{path.relative_to(ROOT)} is missing. Train it first:  python -m src.train")
    return joblib.load(path)


def load_app_inputs():
    with open(APP_INPUTS) as f:
        return json.load(f)


def training_defaults():
    """Typical patient: median of every numeric input, most common category.

    Taken from the TRAINING split only. Inputs not shown in the app keep these
    values, so the prediction reflects 'an otherwise typical patient'.
    """
    X_train, *_ = split()
    defaults = {c: float(X_train[c].median()) for c in config.NUMERIC_COLS}
    defaults.update({c: X_train[c].mode().iloc[0] for c in config.CATEGORICAL_COLS})
    for c in INTEGER_INPUTS:
        defaults[c] = int(round(defaults[c]))
    return defaults


def make_row(values: dict, feature_cols) -> pd.DataFrame:
    row = {c: values.get(c, np.nan) for c in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)


def predict_risk(bundle, values: dict) -> float:
    X = make_row(values, bundle["feature_cols"])
    return float(bundle["model"].predict_proba(X)[0, 1])


def sweep(bundle, values: dict, feature: str, grid) -> pd.DataFrame:
    """Risk as one input moves across its range, everything else held fixed."""
    rows = []
    for v in grid:
        changed = dict(values, **{feature: v})
        rows.append(make_row(changed, bundle["feature_cols"]))
    X = pd.concat(rows, ignore_index=True)
    return pd.DataFrame({"value": list(grid), "risk": bundle["model"].predict_proba(X)[:, 1]})


def risk_band(risk, threshold):
    """Three bands around the model's decision threshold."""
    if risk >= threshold:
        return "High", "Predicted to decline"
    if risk >= threshold * 0.6:
        return "Moderate", "Below the alert threshold, worth monitoring"
    return "Low", "Not predicted to decline"
