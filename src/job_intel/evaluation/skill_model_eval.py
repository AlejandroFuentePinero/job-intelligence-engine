# src/job_intel/evaluation/skill_model_eval.py

from __future__ import annotations

from typing import Dict, Optional, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import CalibrationDisplay


# -------------------------------------------------------------------
# Paths and feature configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"

ALL_FEATURES: list[str] = [
    # Company
    "size_code",
    "sector_code",
    "state_code",
    "ownership_code",
    # Role
    "seniority_code",
    "title_rich_code",
    # Skills (27 groups total including target)
    "core_programming__basic",
    "core_programming__intermediate",
    "core_programming__advanced",
    "data_engineering_pipelines__basic",
    "data_engineering_pipelines__intermediate",
    "data_engineering_pipelines__advanced",
    "ml_ai__basic",
    "ml_ai__intermediate",
    "ml_ai__advanced",
    "analytics_stats__basic",
    "analytics_stats__intermediate",
    "analytics_stats__advanced",
    "bi_viz__basic",
    "bi_viz__intermediate",
    "bi_viz__advanced",
    "cloud__basic",
    "cloud__intermediate",
    "cloud__advanced",
    "db_storage__basic",
    "db_storage__intermediate",
    "db_storage__advanced",
    "productivity_workflow__basic",
    "productivity_workflow__intermediate",
    "productivity_workflow__advanced",
    "soft_skills__core",
    "soft_skills__leadership",
    "domain_specific__none",
]


def get_feature_names_for_response(response_name: str) -> list[str]:
    """
    Return the list of predictor feature names for a given skill group.

    This is simply ALL_FEATURES minus the response_name (target).
    """
    if response_name not in ALL_FEATURES:
        raise ValueError(f"{response_name!r} is not in ALL_FEATURES.")
    return [f for f in ALL_FEATURES if f != response_name]


def load_skill_model(response_name: str):
    """
    Load the fitted LightGBM model for a given skill group from disk.
    """
    model_path = MODELS_DIR / f"{response_name}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)


# -------------------------------------------------------------------
# Main evaluation function
# -------------------------------------------------------------------


def evaluate_skill_model(
    response_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    model=None,
    show_plots: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a binary skill-requirement model on train and test sets.

    Parameters
    ----------
    response_name : str
        Name of the skill group column being modelled (target).
    X_train, X_test : pd.DataFrame
        Feature matrices. They must contain ALL_FEATURES columns.
    y_train, y_test : pd.Series or np.ndarray
        Binary targets (0/1) for the given response_name.
    model : optional
        Fitted classifier with predict_proba. If None, the function
        will load the model from disk using `response_name`.
    show_plots : bool
        Whether to display evaluation plots (ROC, PR, calibration,
        feature importances).

    Returns
    -------
    metrics : dict
        Dictionary with prevalence, ROC AUC, PR AUC (train/test),
        and Brier score (test).
    """
    # Ensure arrays for metric functions
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Determine feature set (exclude target skill group)
    feature_names = get_feature_names_for_response(response_name)

    # Subset X to the expected feature set (defensive)
    X_train_feat = X_train[feature_names]
    X_test_feat = X_test[feature_names]

    # Load model from disk if not provided
    if model is None:
        model = load_skill_model(response_name)

    # Predicted probabilities
    proba_train = model.predict_proba(X_train_feat)[:, 1]
    proba_test = model.predict_proba(X_test_feat)[:, 1]

    # Core metrics
    roc_auc_train = roc_auc_score(y_train, proba_train)
    roc_auc_test = roc_auc_score(y_test, proba_test)

    pr_auc_train = average_precision_score(y_train, proba_train)
    pr_auc_test = average_precision_score(y_test, proba_test)

    brier_test = brier_score_loss(y_test, proba_test)

    pos_frac_train = float(y_train.mean())
    pos_frac_test = float(y_test.mean())

    params = model.get_params()

    metrics = {
        "pos_frac_train": pos_frac_train,
        "pos_frac_test": pos_frac_test,
        "roc_auc_train": roc_auc_train,
        "roc_auc_test": roc_auc_test,
        "pr_auc_train": pr_auc_train,
        "pr_auc_test": pr_auc_test,
        "brier_test": brier_test,
        "learning_rate": params["learning_rate"],
        "n_estimators": params["n_estimators"],
        "num_leaves": params["num_leaves"],
        "colsample_bytree": params["colsample_bytree"],
        "subsample": params["subsample"],
    }

    # Variable importance
    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(len(importances))]

    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})

    # Plots
    if show_plots:
        _plot_roc_curve(y_test, proba_test, title=f"ROC curve – {response_name}")
        _plot_pr_curve(y_test, proba_test, title=f"PR curve – {response_name}")
        _plot_calibration_curve(
            y_test, proba_test, title=f"Calibration – {response_name}"
        )
        _plot_feature_importance(model, feature_names=feature_names)

    return metrics, df_imp


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------


def _plot_roc_curve(y_true, proba, title: str = "ROC curve") -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)

    plt.figure()
    plt.plot(fpr, tpr, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def _plot_pr_curve(y_true, proba, title: str = "Precision–Recall curve") -> None:
    precision, recall, _ = precision_recall_curve(y_true, proba)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()


def _plot_calibration_curve(
    y_true,
    proba,
    n_bins: int = 10,
    title: str = "Calibration curve",
) -> None:
    plt.figure()
    CalibrationDisplay.from_predictions(
        y_true=y_true,
        y_prob=proba,
        n_bins=n_bins,
    )
    plt.title(title)
    plt.tight_layout()


def _plot_feature_importance(
    model,
    feature_names: Optional[Sequence[str]] = None,
    top_n: int = 20,
) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(len(importances))]

    df_imp = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(8, max(4, 0.3 * len(df_imp))))
    plt.barh(df_imp["feature"][::-1], df_imp["importance"][::-1])
    plt.xlabel("Importance (gain)")
    plt.title("Top feature importances")
    plt.tight_layout()
