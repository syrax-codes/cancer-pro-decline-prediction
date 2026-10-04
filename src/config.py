"""Shared paths, column lists and constants for the modelling pipeline.

These mirror exactly what notebooks/02_preprocessing.ipynb (Person 1) used,
so the models are trained and tested on the same patients.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "synthetic_chemo_pro_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Supported prediction targets (the paper's two outcomes).
TARGETS = {
    "drop_outcome": "GPH drops by 10+ points on chemotherapy",
    "threshold_outcome": "On-chemo GPH score is 40 or below",
}
DEFAULT_TARGET = "drop_outcome"

# Columns that must never be used as inputs:
# patient_id is an identifier; on_chemo_gph is measured AFTER chemo starts and
# directly defines both labels; the outcomes are the labels themselves.
LEAKAGE_COLS = ["patient_id", "on_chemo_gph", "drop_outcome", "threshold_outcome"]

CATEGORICAL_COLS = [
    "sex", "race", "insurance", "marital_status", "smoking_status",
    "cancer_type", "chemo_regimen", "treatment_intent",
]

NUMERIC_COLS = [
    "age", "bmi", "stage", "ecog_status", "prior_surgery", "prior_radiation",
    "months_since_diagnosis", "num_comorbidities", "diabetes", "hypertension",
    "copd", "heart_disease", "depression_history", "med_analgesic", "med_opioid",
    "med_antidepressant", "med_antiemetic", "med_steroid", "num_med_classes",
    "ed_visits_6m", "hospitalizations_6m", "psychiatry_visits_6m", "heart_rate",
    "systolic_bp", "diastolic_bp", "temperature_c", "resp_rate", "spo2",
    "weight_loss_pct_6m", "hemoglobin_g_dl", "wbc_k_ul", "platelets_k_ul",
    "albumin_g_dl", "creatinine_mg_dl", "alt_u_l", "bilirubin_mg_dl",
    "sodium_mmol_l", "potassium_mmol_l", "glucose_mg_dl", "crp_mg_l", "ldh_u_l",
    "psa_ng_ml", "baseline_gph", "baseline_gmh",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Feature groups for the ablation study (mirrors the paper's Table 2).
FEATURE_GROUPS = {
    "Demographics": ["age", "sex", "race", "insurance", "marital_status", "bmi", "smoking_status"],
    "Cancer & treatment": ["cancer_type", "stage", "chemo_regimen", "treatment_intent",
                           "ecog_status", "prior_surgery", "prior_radiation", "months_since_diagnosis"],
    "Diagnoses": ["num_comorbidities", "diabetes", "hypertension", "copd", "heart_disease",
                  "depression_history"],
    "Medications": ["med_analgesic", "med_opioid", "med_antidepressant", "med_antiemetic",
                    "med_steroid", "num_med_classes"],
    "Utilization": ["ed_visits_6m", "hospitalizations_6m", "psychiatry_visits_6m"],
    "Vitals": ["heart_rate", "systolic_bp", "diastolic_bp", "temperature_c", "resp_rate",
               "spo2", "weight_loss_pct_6m"],
    "Labs": ["hemoglobin_g_dl", "wbc_k_ul", "platelets_k_ul", "albumin_g_dl", "creatinine_mg_dl",
             "alt_u_l", "bilirubin_mg_dl", "sodium_mmol_l", "potassium_mmol_l", "glucose_mg_dl",
             "crp_mg_l", "ldh_u_l", "psa_ng_ml"],
    "Baseline PROMIS scores": ["baseline_gph", "baseline_gmh"],
}
