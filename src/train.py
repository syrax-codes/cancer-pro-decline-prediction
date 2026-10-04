"""Train and tune all models, build the voting ensemble, pick the final model.

Usage (from the repo root):
    python -m src.train                       # drop_outcome (main task)
    python -m src.train --target threshold_outcome

Model selection uses ONLY the training set (5-fold cross-validated AUROC);
the test set is kept untouched until src.evaluate.
"""
import argparse
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from . import config
from .data import split
from .models import make_pipeline, model_specs


def best_f1_threshold(y_true, proba):
    """Decision threshold that maximises F1 on out-of-fold training predictions.

    With only ~20% positives a 0.5 cut-off is arbitrary, so we choose it from
    training data and then apply the same threshold to the test set.
    """
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return round(float(grid[int(np.argmax(scores))]), 2)


def model_path(target, name):
    slug = name.lower().replace(" ", "_")
    return config.MODELS_DIR / target / f"{slug}.joblib"


def main(target=config.DEFAULT_TARGET, n_vote=3):
    X_train, X_test, y_train, y_test = split(target)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    (config.MODELS_DIR / target).mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Target: {target} | train {X_train.shape}, positives {y_train.mean():.1%}")
    results, fitted = {}, {}

    # Steps 1-2: baseline first (dict order), then the other four models.
    for name, (estimator, grid) in model_specs().items():
        t0 = time.time()
        search = GridSearchCV(make_pipeline(estimator), grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        results[name] = {
            "cv_auroc": float(search.best_score_),
            "cv_auroc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "best_params": {k.replace("clf__", "").replace("estimator__", ""): (v.item() if hasattr(v, "item") else v)
                            for k, v in search.best_params_.items()},
            "threshold": best_f1_threshold(y_train, oof),
        }
        fitted[name] = best
        joblib.dump(best, model_path(target, name))
        print(f"  {name:20s} CV AUROC {search.best_score_:.4f}  "
              f"{results[name]['best_params']}  ({time.time() - t0:.0f}s)")

    # Step 5: soft-voting ensemble = average predicted probability of the
    # n_vote best individual models (ranked by CV AUROC, never by test score).
    ranked = sorted(results, key=lambda n: results[n]["cv_auroc"], reverse=True)
    members = ranked[:n_vote]
    voting = VotingClassifier([(n, fitted[n]) for n in members], voting="soft", n_jobs=-1)
    oof = cross_val_predict(voting, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    fold_aucs = [roc_auc_score(y_train.iloc[va], oof[va]) for _, va in cv.split(X_train, y_train)]
    voting.fit(X_train, y_train)
    results["Voting Ensemble"] = {
        "cv_auroc": float(np.mean(fold_aucs)),
        "cv_auroc_std": float(np.std(fold_aucs)),
        "best_params": {"members": members},
        "threshold": best_f1_threshold(y_train, oof),
    }
    fitted["Voting Ensemble"] = voting
    joblib.dump(voting, model_path(target, "Voting Ensemble"))
    print(f"  {'Voting Ensemble':20s} CV AUROC {np.mean(fold_aucs):.4f}  members={members}")

    # Final model = highest CV AUROC. Ties within 0.002 go to the simpler model,
    # since a difference that small is noise.
    order = ["Logistic Regression", "KNN", "SVM", "Random Forest", "Gradient Boosting", "Voting Ensemble"]
    top = max(r["cv_auroc"] for r in results.values())
    final = next(n for n in order if results[n]["cv_auroc"] >= top - 0.002)
    print(f"Final model: {final}")

    bundle = {
        "model": fitted[final],
        "model_name": final,
        "target": target,
        "target_description": config.TARGETS[target],
        "threshold": results[final]["threshold"],
        "cv_auroc": results[final]["cv_auroc"],
        "feature_cols": config.FEATURE_COLS,
        "numeric_cols": config.NUMERIC_COLS,
        "categorical_cols": config.CATEGORICAL_COLS,
    }
    joblib.dump(bundle, config.MODELS_DIR / ("final_model.joblib" if target == config.DEFAULT_TARGET
                                             else f"final_model_{target}.joblib"))
    with open(config.RESULTS_DIR / f"train_results_{target}.json", "w") as f:
        json.dump({"target": target, "final_model": final, "models": results}, f, indent=2)
    return results, final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=config.DEFAULT_TARGET, choices=list(config.TARGETS))
    main(ap.parse_args().target)
