"""Model definitions and the hyperparameter grids searched for each one.

Every model is wrapped in a Pipeline(preprocess -> classifier), so the grid
search refits imputation and scaling inside each cross-validation fold.
Grids are deliberately small: enough to tune the settings that matter most,
fast enough to rerun before the demo (a few minutes on a laptop).
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from . import config
from .data import build_preprocessor

RS = config.RANDOM_STATE


def model_specs():
    """name -> (estimator, param_grid). Grid keys use the 'clf__' prefix."""
    return {
        # Baseline. l1_ratio=0 is Ridge (L2), 1 is LASSO (L1), in-between is Elastic Net,
        # so one grid covers all three logistic models from the paper.
        "Logistic Regression": (
            LogisticRegression(solver="saga", max_iter=5000, random_state=RS),
            {"clf__C": np.logspace(-3, 2, 6), "clf__l1_ratio": [0.0, 0.5, 1.0]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 11, 21, 41, 71, 101, 151, 201], "clf__weights": ["uniform", "distance"]},
        ),
        "SVM": (
            # SVC outputs margins, not probabilities; Platt scaling (sigmoid) turns
            # them into probabilities so AUROC/voting work.
            CalibratedClassifierCV(SVC(random_state=RS),
                                   method="sigmoid", cv=3, ensemble=False),
            {"clf__estimator__C": [0.01, 0.1, 1, 10],
             "clf__estimator__kernel": ["linear", "rbf"]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=RS),
            {"clf__min_samples_leaf": [1, 5, 20], "clf__max_features": ["sqrt", 0.3]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RS, subsample=0.8),
            {"clf__learning_rate": [0.03, 0.1], "clf__n_estimators": [100, 300],
             "clf__max_depth": [2, 3]},
        ),
    }


def make_pipeline(estimator, numeric_cols=None, categorical_cols=None) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
        ("clf", estimator),
    ])
