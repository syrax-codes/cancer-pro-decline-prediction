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
│   ├── raw/                     # Raw dataset (synthetic_chemo_pro_data.csv)
│   └── processed/               # Leakage-free train/test splits (X_train, X_test, y_train, y_test)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis (8 sections, 10 figures)
│   ├── 02_preprocessing.ipynb   # Preprocessing pipeline & data verification checks
│   ├── 03_model_training.ipynb  # Model training and hyperparameter tuning
│   └── 04_evaluation.ipynb      # Test-set evaluation, ROC, confusion matrix, ablation
├── src/
│   ├── config.py                # Paths, column lists, leakage columns, seed
│   ├── data.py                  # Train/test split + preprocessing (same as 02_preprocessing)
│   ├── models.py                # The five models and their hyperparameter grids
│   ├── train.py                 # CV grid search, voting ensemble, final-model selection
│   └── evaluate.py              # Test metrics, figures, feature importance, ablation
├── models/
│   ├── preprocessor.joblib      # Fitted ColumnTransformer from preprocessing (scikit-learn 1.7)
│   └── final_model.joblib       # Final model: full pipeline (preprocessing + classifier) + threshold
├── reports/
│   ├── figures/                 # EDA figures (01-10) and model figures (11-14)
│   ├── results/                 # Model comparison tables, ablation, feature importance
│   ├── data_dictionary.csv      # 56-column clinical metadata dictionary
│   └── notes.md                 # Project research notes
├── app/
│   ├── app.py                   # Streamlit demo with What-If Risk Simulator
│   ├── utils.py                 # Model loading, input row building, risk sweeps
│   └── app_inputs.json          # Demo configuration (12 key inputs, schema, example patients)
├── requirements.txt
└── README.md
```

---

## How to Run

1. **Clone the repository and install dependencies** (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```

2. **Data side - execute notebooks in order:**
   - **`notebooks/01_eda.ipynb`**: Explores missingness, target class balance, clinical subgroup drop rates, feature correlations, and baseline vs. on-chemo distributions.
   - **`notebooks/02_preprocessing.ipynb`**: Executes stratified 80/20 train/test split, builds `ColumnTransformer` (median imputation + missing indicators + standard scaling + one-hot encoding), verifies zero NaNs and label integrity, and saves processed splits.

3. **Modelling side - from the repo root:**
   ```bash
   python -m src.train       # ~2 min: tunes 5 models + voting ensemble, saves models/final_model.joblib
   python -m src.evaluate    # ~30 s: test metrics, figures 11-14, tables in reports/results/
   ```
   Or run `notebooks/03_model_training.ipynb` then `notebooks/04_evaluation.ipynb`, which call the same code.
   Add `--target threshold_outcome` to either command for the paper's second outcome (on-chemo GPH <= 40).

4. **Live demo:**
   ```bash
   streamlit run app/app.py
   ```

---

## Modelling Approach

- **Models:** Logistic Regression baseline (Ridge / Elastic Net / LASSO via `l1_ratio`), KNN, SVM (Platt-calibrated), Random Forest, Gradient Boosting, and a soft-voting ensemble averaging the probabilities of the 3 best models.
- **Tuning:** grid search with 5-fold stratified cross-validation on the training set, scored by AUROC (the paper's primary metric).
- **No leakage in CV:** preprocessing lives inside each model's pipeline, so imputation medians and scaling are refit on the training folds only. The rebuilt preprocessing reproduces `data/processed/X_test.csv` exactly.
- **Final model choice:** highest cross-validated AUROC; the test set is used only once, for the final report.
- **Class imbalance (20% positive):** probabilities are left unweighted (so they read as real risks) and the decision threshold is the one that maximises F1 on out-of-fold training predictions.

## Results (test set, 360 patients, `drop_outcome`)

| Model | CV AUROC | Test AUROC | PR-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (final)** | **0.753** | **0.764** | **0.513** | 0.725 | 0.398 | 0.699 | **0.507** |
| KNN | 0.719 | 0.705 | 0.406 | 0.664 | 0.338 | 0.685 | 0.452 |
| SVM | 0.733 | 0.772 | 0.488 | 0.719 | 0.387 | 0.658 | 0.487 |
| Random Forest | 0.741 | 0.750 | 0.457 | 0.714 | 0.375 | 0.616 | 0.466 |
| Gradient Boosting | 0.738 | 0.739 | 0.438 | 0.622 | 0.309 | 0.699 | 0.429 |
| Voting Ensemble | 0.752 | 0.761 | 0.485 | 0.683 | 0.361 | 0.726 | 0.482 |

Final model test AUROC 0.764 (bootstrap 95% CI 0.695-0.823), close to the paper's best of 0.771 on real Stanford data. The LASSO keeps 35 of 89 features; the strongest are cancer stage, hemoglobin, albumin, baseline GMH/GPH and ECOG status. Second outcome (`threshold_outcome`): Logistic Regression test AUROC 0.913.

## Demo: What-If Risk Simulator

The Streamlit app takes the 12 key pre-treatment inputs (the other 40 stay at a typical or loaded patient's values), shows the predicted risk and risk band, and lets you change one or two inputs to see the risk move, with a curve of risk across the full range of the first input. Example patients from `app/app_inputs.json` can be loaded in one click. A second tab shows all model results and figures.

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
