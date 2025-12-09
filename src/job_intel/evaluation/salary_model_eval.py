# src/job_intel/evaluation/salary_model_eval.py

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Sequence

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)

# ------------------------------------------------------------
# Load default model from the project model directory
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "salary_model_v4.pkl"


def load_default_salary_model():
    """Loads the saved salary model v4 unless another model is provided."""
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Cannot find salary model at {DEFAULT_MODEL_PATH}")
    return joblib.load(DEFAULT_MODEL_PATH)


# ------------------------------------------------------------
# Internal plotting utilities
# ------------------------------------------------------------


def _plot_residuals_hist(residuals: np.ndarray) -> None:
    plt.figure()
    plt.hist(residuals, bins=40)
    plt.title("Residuals (y_true - y_pred)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()


def _plot_residuals_vs_pred(y_pred: np.ndarray, residuals: np.ndarray) -> None:
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Predicted salary")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted Salary")
    plt.tight_layout()


def _plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    plt.figure()
    plt.scatter(y_pred, y_true, alpha=0.3)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
    plt.xlabel("Predicted salary")
    plt.ylabel("Actual salary")
    plt.title("Predicted vs Actual Salary")
    plt.tight_layout()


def _plot_feature_importance(
    model,
    feature_names: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:

    if not hasattr(model, "feature_importances_"):
        return None

    importances = np.asarray(model.feature_importances_)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 12))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Model Feature Importances")
    plt.tight_layout()

    return importance_df


# ------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------


def evaluate_salary_model(
    X_train,
    y_train,
    X_test,
    y_test,
    model=None,
    feature_names: Optional[Sequence[str]] = None,
    show_plots: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the salary model using train/test splits.
    If model is None, the saved salary_model_v4.pkl is automatically loaded.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Train/test feature matrices.
    y_train, y_test : pd.Series
        Train/test target vectors.
    model : fitted model or None
        If None, use the default saved model artefact.
    feature_names : list[str], optional
        Names for feature importance plots.
    show_plots : bool
        Whether to display evaluation plots.

    Returns
    -------
    metrics : dict
        Train/test R2, MAE, and RMSE.
    """

    # Load model if none provided
    if model is None:
        model = load_default_salary_model()

    # Convert to numpy for metrics
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    metrics = {
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "rmse_train": root_mean_squared_error(y_train, y_pred_train),
        "rmse_test": root_mean_squared_error(y_test, y_pred_test),
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
    }

    # Residuals
    residuals_test = y_test - y_pred_test

    if show_plots:
        _plot_residuals_hist(residuals_test)
        _plot_residuals_vs_pred(y_pred_test, residuals_test)
        _plot_pred_vs_actual(y_test, y_pred_test)
        _plot_feature_importance(model, feature_names=feature_names)
        plt.show()

    return metrics
