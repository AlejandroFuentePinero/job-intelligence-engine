# src/job_intel/pipelines/skill_model_pipeline.py

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
import joblib

from src.job_intel.config import MODELS_DIR, INTERIM_DATA_DIR
from src.job_intel.evaluation.skill_model_eval import evaluate_skill_model
from src.job_intel.pipelines.chapter0_build_base_dataset import (
    build_chapter0_base_dataset,
)
from src.job_intel.models.skill_prob_matrix import build_skill_probability_matrix


def run_skill_pipeline(
    show_plots_eval: bool = False,
    save_model: bool = False,
):
    """
    Train the 27 skill-requirement models and build the skill probability matrix.

    Returns
    -------
    results_df : pd.DataFrame
        One row per skill model with metrics and variable importances.
    prob_mat : pd.DataFrame
        Job x skill matrix of predicted probabilities (also saved to disk).
    """
    # ------------------------------------------------------------------
    # 1. Build dataset from Chapter 0 pipeline
    # ------------------------------------------------------------------
    df_ch0 = build_chapter0_base_dataset()

    print("✅ Data created DONE.")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    # Drop rows without salary; mirrors salary pipeline and ensures alignment.
    df_ch0 = df_ch0.dropna(subset=["sal_mean"])

    # Combine domain + job family into a richer title token
    df_ch0["title_rich"] = df_ch0["domain"] + "_" + df_ch0["job_title_family"]

    # Encode categoricals as integer codes
    df_ch0["size_code"] = df_ch0["Size"].astype("category").cat.codes
    df_ch0["sector_code"] = df_ch0["Sector"].astype("category").cat.codes
    df_ch0["state_code"] = df_ch0["state"].astype("category").cat.codes
    df_ch0["ownership_code"] = df_ch0["ownership_clean"].astype("category").cat.codes
    df_ch0["seniority_code"] = df_ch0["seniority_combined"].astype("category").cat.codes
    df_ch0["title_code"] = df_ch0["job_title_family"].astype("category").cat.codes
    df_ch0["title_rich_code"] = df_ch0["title_rich"].astype("category").cat.codes

    # ------------------------------------------------------------------
    # 3. Prepare data for LightGBM
    # ------------------------------------------------------------------
    skill_cols = [
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

    all_features = df_ch0[
        [
            # Company
            "size_code",
            "sector_code",
            "state_code",
            "ownership_code",
            # Role
            "seniority_code",
            "title_rich_code",
            # Skills
            *skill_cols,
        ]
    ].copy()

    cat_cols = [
        "size_code",
        "sector_code",
        "state_code",
        "ownership_code",
        "seniority_code",
        "title_rich_code",
    ]
    all_features[cat_cols] = all_features[cat_cols].astype("category")

    # Columns for the evaluation summary table
    cols = [
        # identifiers + metrics
        "model",
        "pos_frac_train",
        "pos_frac_test",
        "roc_auc_train",
        "roc_auc_test",
        "pr_auc_train",
        "pr_auc_test",
        "brier_test",
        # hyperparameters
        "learning_rate",
        "n_estimators",
        "num_leaves",
        "colsample_bytree",
        "subsample",
        # feature importances (company + role + skills)
        "size_code",
        "sector_code",
        "state_code",
        "ownership_code",
        "seniority_code",
        "title_rich_code",
        *skill_cols,
    ]
    print("✅ Feature engineering DONE.")

    # ------------------------------------------------------------------
    # 4. Train models with GridSearchCV (one per skill)
    # ------------------------------------------------------------------
    results_df = pd.DataFrame()  # aggregated metrics + feature importances

    for target in skill_cols:
        # -----------------------
        # Define target + predictors
        # -----------------------
        y = df_ch0[target]
        X = all_features.drop(columns=[target])

        # Guard: if only one class, skip model (stratify would fail)
        if y.nunique() < 2:
            print(f"Skipping {target}: only one class present.")
            continue

        # -----------------------
        # Train / validation split
        # -----------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # --------------------
        # Base estimator
        # --------------------
        estimator = LGBMClassifier(
            objective="binary",
            random_state=42,
            class_weight="balanced",
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            num_leaves=127,
        )

        param_grid = {
            "n_estimators": [400, 800, 1000],
            "learning_rate": [0.015, 0.025, 0.05],
        }

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            verbose=0,
            n_jobs=-1,
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        print(f"✅ Training {target} model through GridSearchCV DONE.")

        # Save individual skill model if requested
        if save_model:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, MODELS_DIR / f"{target}_model.pkl")
            print(f"✅ Model {target} saved.")

        # ------------------------------------------------------------------
        # 5. Model evaluation for this skill
        # ------------------------------------------------------------------
        metrics, var_imp = evaluate_skill_model(
            response_name=target,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model=best_model,
            show_plots=show_plots_eval,
        )

        def build_eval_row(
            metrics: dict,
            var_imp: pd.DataFrame,
            all_cols: list[str],
        ) -> pd.DataFrame:
            # Start with NaN for everything
            row = {col: np.nan for col in all_cols}

            # Model id
            row["model"] = target

            # Fill metrics
            for k, v in metrics.items():
                if k in row:
                    row[k] = v

            # Fill variable importance
            # var_imp has columns: ["feature", "importance"]
            for _, r in var_imp.iterrows():
                feat = r["feature"]
                imp = r["importance"]
                if feat in row:
                    row[feat] = imp

            return pd.DataFrame([row])

        print("✅ Model evaluation DONE")
        row_df = build_eval_row(metrics, var_imp, cols)
        results_df = pd.concat([results_df, row_df], ignore_index=True)

    # ------------------------------------------------------------------
    # 6. Build skill probability matrix
    # ------------------------------------------------------------------
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_path = INTERIM_DATA_DIR / "skill_prob_matrix_PIPELINE.csv"
    prob_mat = build_skill_probability_matrix(jobs_df=df_ch0, save_path=save_path)
    print("✅ Skill probability matrix built and saved DONE")
    print("✅ PIPELINE RUN SUCCESSFULLY.")

    return results_df, prob_mat
