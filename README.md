# Predicting Cancer PRO Physical Health Decline on Chemotherapy

An applied Machine Learning project predicting clinically meaningful decline in Patient-Reported Outcome (PROMIS Global Physical Health, GPH) for cancer patients receiving chemotherapy, based on the research framework by Ostberg & Peterson (Stanford CS229).

---

## Project Overview

- **Objective:** Predict which cancer patients will experience a $\ge 10$-point decline in self-reported physical health during chemotherapy.
- **Dataset:** 1,800 synthetic patient records across 56 clinical, demographic, laboratory, and patient-reported outcome variables (`data/raw/synthetic_chemo_pro_data.csv`).
- **Primary Target:** `drop_outcome` (binary classification):
  $$\text{drop\_outcome} = \begin{cases} 1 & \text{if } (on\_chemo\_gph - baseline\_gph) \le -10 \\ 0 & \text{otherwise} \end{cases}$$
- **Data Leakage Safeguard:** `on_chemo_gph` is collected during chemotherapy and directly defines the labels; it is strictly excluded from all feature sets alongside `threshold_outcome` and `patient_id`.

---

## Repository Structure

```text
├── data/
│   ├── raw/                 # Raw dataset (synthetic_chemo_pro_data.csv)
│   └── processed/           # Leakage-free train/test splits (X_train, X_test, y_train, y_test)
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis (8 sections, 10 figures)
│   ├── 02_preprocessing.ipynb # Preprocessing pipeline & data verification checks
│   ├── 03_model_training.ipynb# Model training and hyperparameter tuning
│   └── 04_evaluation.ipynb  # Evaluation metrics (AUROC, PR-AUC, calibration)
├── models/
│   ├── preprocessor.joblib  # Fitted scikit-learn ColumnTransformer
│   └── final_model.joblib   # Best performing model artifact
├── reports/
│   ├── figures/             # Exported visualizations from EDA
│   ├── data_dictionary.csv  # 56-column clinical metadata dictionary
│   └── notes.md             # Project research notes
├── app/
│   └── app_inputs.json      # Streamlit demo configuration (12 key inputs, schema, test cases)
├── requirements.txt         # Project dependencies
└── README.md
```

---

## How to Run

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute notebooks in sequential order:**
   - **`notebooks/01_eda.ipynb`**: Explores missingness, target class balance, clinical subgroup drop rates, feature correlations, and baseline vs. on-chemo distributions.
   - **`notebooks/02_preprocessing.ipynb`**: Executes stratified 80/20 train/test split, builds `ColumnTransformer` (median imputation + missing indicators + standard scaling + one-hot encoding), verifies zero NaNs and label integrity, and saves processed splits.
   - **`notebooks/03_model_training.ipynb`**: Trains baseline and advanced classifiers on processed data.
   - **`notebooks/04_evaluation.ipynb`**: Evaluates model performance on the held-out test set.

---

## Handover to Modeling

The preprocessing phase is complete. The modeling workflow (`03_model_training.ipynb`) should directly ingest the prepared artifacts:

- **Target Variable:** `drop_outcome` (Class 0: 79.58% no drop, Class 1: 20.42% clinically meaningful decline).
- **Data Location:**
  - `data/processed/X_train.csv` (1,440 rows $\times$ 89 features)
  - `data/processed/X_test.csv` (360 rows $\times$ 89 features)
  - `data/processed/y_train.csv` (1,440 rows)
  - `data/processed/y_test.csv` (360 rows)
- **Pre-computed Transformations:**
  - Continuous and integer features are centered and scaled (`StandardScaler`).
  - Missing laboratory values have been imputed using training set medians.
  - 13 binary missingness indicator features (`num__missingindicator_*`) are included to preserve clinical ordering signals.
  - Categorical variables are one-hot encoded (`cat__*`).
  - No missing values (`NaN`) exist in train or test matrices.
- **Fitted Pipeline:** Saved at `models/preprocessor.joblib` for transforming new raw patient data in production or the demo app.
- **Global Reproducibility Seed:** `random_state=42`.
- **Recommended Evaluation Metrics:** Given the ~20.4% class prevalence, prioritize **AUROC**, **Precision-Recall AUC (PR-AUC)**, and **Recall / F1-score** over raw accuracy.
