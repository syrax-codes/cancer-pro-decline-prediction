"""Evaluate every trained model on the held-out test set.

Usage (from the repo root, after src.train):
    python -m src.evaluate
    python -m src.evaluate --target threshold_outcome

Outputs
  reports/results/model_comparison_<target>.csv / .md   all models, all metrics
  reports/results/ablation_<target>.csv                 feature-group ablation
  reports/results/feature_importance_<target>.csv       permutation importance
  reports/figures/11_roc_curves_<target>.png            ROC curves, all models
  reports/figures/12_confusion_matrix_<target>.png      final model
  reports/figures/13_feature_importance_<target>.png    final model
  reports/figures/14_ablation_<target>.png              final model type
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)

from . import config
from .data import split
from .models import make_pipeline
from .train import model_path

MODEL_ORDER = ["Logistic Regression", "KNN", "SVM", "Random Forest",
               "Gradient Boosting", "Voting Ensemble"]
# Categorical palette slots 1-6 (fixed order, one colour per model everywhere).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlecolor": INK, "figure.dpi": 150, "savefig.bbox": "tight",
})


def metrics_at(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "AUROC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
    }


def bootstrap_auc_ci(y, proba, n=2000, seed=config.RANDOM_STATE):
    """95% CI for AUROC by resampling test patients (as the paper did)."""
    rng = np.random.default_rng(seed)
    y, proba = np.asarray(y), np.asarray(proba)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], proba[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def plot_roc(y, probas, final, target, path):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK_2, alpha=0.6, label="Chance (0.50)")
    for name, color in zip(MODEL_ORDER, COLORS):
        fpr, tpr, _ = roc_curve(y, probas[name])
        is_final = name == final
        ax.plot(fpr, tpr, color=color, lw=2.6 if is_final else 1.6, alpha=1 if is_final else 0.85,
                label=f"{name}{' (final)' if is_final else ''}  {roc_auc_score(y, probas[name]):.3f}",
                zorder=3 if is_final else 2)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate",
           title=f"ROC curves on the test set - {config.TARGETS[target]}")
    ax.grid(color=GRID, lw=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, title="Model  AUROC",
              title_fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(y, pred, name, threshold, path):
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.6
            ax.text(j, i, f"{cm[i, j]}\n{labels[i][j]}", ha="center", va="center",
                    color="white" if dark else INK, fontsize=10)
    ax.set_xticks([0, 1], ["No decline", "Decline"])
    ax.set_yticks([0, 1], ["No decline", "Decline"])
    ax.set(xlabel="Predicted", ylabel="Actual",
           title=f"{name} - confusion matrix\n(threshold {threshold:.2f}, n={len(y)})")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def plot_barh(labels, values, xlabel, title, path, errors=None):
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(labels) + 1.2))
    ypos = np.arange(len(labels))[::-1]
    ax.barh(ypos, values, color=COLORS[0], height=0.62, xerr=errors,
            error_kw={"ecolor": INK_2, "lw": 1})
    ax.set_yticks(ypos, labels)
    ends = values + (errors if errors is not None else 0)
    for y_, v, e in zip(ypos, values, ends):
        ax.text(e, y_, f"  {v:.3f}", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_xlim(0, max(ends) * 1.18)
    ax.set(xlabel=xlabel, title=title)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def plot_ablation(abl, full_auc, final, path):
    """Dot + 95% CI per feature set; the dashed line is the full-feature model."""
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(abl) + 1.3))
    ypos = np.arange(len(abl))[::-1]
    ax.axvline(full_auc, color=INK_2, ls="--", lw=1, zorder=1)
    ax.hlines(ypos, abl["CI low"], abl["CI high"], color=COLORS[0], lw=2, alpha=0.45, zorder=2)
    ax.scatter(abl["AUROC"], ypos, s=46, color=COLORS[0], edgecolor="white", lw=1.5, zorder=3)
    for y_, v, hi in zip(ypos, abl["AUROC"], abl["CI high"]):
        ax.text(hi + 0.006, y_, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ypos, abl["Feature set"])
    ax.set_xlim(0.5, 0.9)
    ax.set_ylim(-0.7, len(abl) - 0.3)
    ax.set(xlabel="Test AUROC (dot) with bootstrap 95% CI",
           title=f"Ablation - {final} retrained without each feature group\n"
                 f"dashed line = all features ({full_auc:.3f})")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def ablation(final_model, X_train, y_train, X_test, y_test):
    """Retrain the final model type with one feature group removed at a time."""
    rows = []
    groups = dict(config.FEATURE_GROUPS)
    groups["Everything except baseline PROMIS"] = [c for c in config.FEATURE_COLS
                                                   if c not in config.FEATURE_GROUPS["Baseline PROMIS scores"]]
    for group, cols in groups.items():
        keep_num = [c for c in config.NUMERIC_COLS if c not in cols]
        keep_cat = [c for c in config.CATEGORICAL_COLS if c not in cols]
        model = _rebuild(final_model, keep_num, keep_cat)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        lo, hi = bootstrap_auc_ci(y_test, proba, n=500)
        label = f"Only baseline PROMIS scores" if group.startswith("Everything") else f"Without {group}"
        rows.append({"Feature set": label, "AUROC": roc_auc_score(y_test, proba),
                     "CI low": lo, "CI high": hi})
    return pd.DataFrame(rows)


def _rebuild(fitted, num_cols, cat_cols):
    """Same model + tuned settings, new preprocessing restricted to given columns."""
    from sklearn.ensemble import VotingClassifier
    if isinstance(fitted, VotingClassifier):
        return VotingClassifier([(n, _rebuild(m, num_cols, cat_cols)) for n, m in fitted.estimators],
                                voting="soft")
    return make_pipeline(clone(fitted.named_steps["clf"]), num_cols, cat_cols)


def main(target=config.DEFAULT_TARGET):
    X_train, X_test, y_train, y_test = split(target)
    train_info = json.load(open(config.RESULTS_DIR / f"train_results_{target}.json"))
    final = train_info["final_model"]
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows, probas = [], {}
    for name in MODEL_ORDER:
        model = joblib.load(model_path(target, name))
        info = train_info["models"][name]
        probas[name] = model.predict_proba(X_test)[:, 1]
        m = metrics_at(y_test, probas[name], info["threshold"])
        rows.append({"Model": name + (" *" if name == final else ""),
                     "CV AUROC (train)": info["cv_auroc"], "Test AUROC": m.pop("AUROC"),
                     **m, "Threshold": info["threshold"]})
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / f"model_comparison_{target}.csv", index=False)
    with open(config.RESULTS_DIR / f"model_comparison_{target}.md", "w") as f:
        f.write(f"Target: `{target}` ({config.TARGETS[target]}), test n={len(y_test)}, "
                f"positives {y_test.mean():.1%}. * = final model (chosen by CV AUROC).\n\n")
        f.write(table.to_markdown(index=False, floatfmt=".3f"))
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    final_model = joblib.load(model_path(target, final))
    thr = train_info["models"][final]["threshold"]
    lo, hi = bootstrap_auc_ci(y_test, probas[final])
    print(f"\nFinal model {final}: test AUROC {roc_auc_score(y_test, probas[final]):.3f} "
          f"(95% CI {lo:.3f}-{hi:.3f})")

    plot_roc(y_test, probas, final, target, config.FIGURES_DIR / f"11_roc_curves_{target}.png")
    plot_confusion(y_test, (probas[final] >= thr).astype(int), final, thr,
                   config.FIGURES_DIR / f"12_confusion_matrix_{target}.png")

    # Which raw inputs matter: shuffle one column at a time, measure the AUROC lost.
    pi = permutation_importance(final_model, X_test, y_test, scoring="roc_auc", n_repeats=20,
                                random_state=config.RANDOM_STATE, n_jobs=-1)
    imp = (pd.DataFrame({"feature": X_test.columns, "importance": pi.importances_mean,
                         "std": pi.importances_std})
           .sort_values("importance", ascending=False))
    imp.to_csv(config.RESULTS_DIR / f"feature_importance_{target}.csv", index=False)
    top = imp[imp["importance"] > 0.001].head(12)  # hide features the model ignores
    plot_barh(top["feature"].tolist(), top["importance"].values,
              "Drop in test AUROC when the feature is shuffled (mean ± SD, 20 repeats)",
              f"{final} - features that matter (permutation importance)",
              config.FIGURES_DIR / f"13_feature_importance_{target}.png", errors=top["std"].values)

    abl = ablation(final_model, X_train, y_train, X_test, y_test)
    abl.to_csv(config.RESULTS_DIR / f"ablation_{target}.csv", index=False)
    full_auc = roc_auc_score(y_test, probas[final])
    plot_ablation(abl, full_auc, final, config.FIGURES_DIR / f"14_ablation_{target}.png")
    print("\nAblation:\n" + abl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nTop features:\n" + imp.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    summary = {"target": target, "final_model": final, "threshold": thr,
               "test_auroc": full_auc, "test_auroc_ci": [lo, hi],
               "metrics": {k: v for k, v in table.iloc[MODEL_ORDER.index(final)].items()
                           if k != "Model"}}
    with open(config.RESULTS_DIR / f"test_summary_{target}.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=config.DEFAULT_TARGET, choices=list(config.TARGETS))
    main(ap.parse_args().target)
