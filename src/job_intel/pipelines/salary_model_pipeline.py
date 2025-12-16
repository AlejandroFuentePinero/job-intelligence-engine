# src/job_intel/pipelines/salary_model_pipeline.py

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
import joblib

from src.job_intel.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.job_intel.evaluation.salary_model_eval import evaluate_salary_model
from src.job_intel.pipelines.chapter0_build_base_dataset import (
    build_chapter0_base_dataset,
)


def run_salary_pipeline(
    do_eval: bool = False,
    show_plots_eval: bool = False,
    save_model: bool = False,
):
    """
    Train the Chapter 1 salary response model end-to-end.

    Steps
    -----
    1. Build Chapter 0 base dataset.
    2. Feature engineering (categoricals + title_rich).
    3. PCA on skill flags (10 components).
    4. Train and tune XGBoost regressor via GridSearchCV.
    5. Optionally evaluate, save artefacts, and return metrics.

    Returns
    -------
    best_model : XGBRegressor
        Fitted XGBoost salary model.
    metrics : dict | None
        Evaluation metrics if `do_eval=True`, otherwise None.
    pca : PCA
        Fitted PCA object used to compress skill flags.
    """
    # ------------------------------------------------------------------
    # 1. Build dataset from Chapter 0 pipeline
    # ------------------------------------------------------------------
    df_ch0 = build_chapter0_base_dataset()
    print("✅ Data created DONE.")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    # Drop rows without salary; we cannot train on missing target.
    df_ch0 = df_ch0.dropna(subset=["sal_mean"])

    # Combine domain + job family into a richer title token
    df_ch0["title_rich"] = df_ch0["domain"] + "_" + df_ch0["job_title_family"]

    # Encode categoricals as integer codes (later used as XGBoost categoricals)
    df_ch0["size_code"] = df_ch0["Size"].astype("category").cat.codes
    df_ch0["sector_code"] = df_ch0["Sector"].astype("category").cat.codes
    df_ch0["state_code"] = df_ch0["state"].astype("category").cat.codes
    df_ch0["ownership_code"] = df_ch0["ownership_clean"].astype("category").cat.codes
    df_ch0["seniority_code"] = df_ch0["seniority_combined"].astype("category").cat.codes
    df_ch0["title_code"] = df_ch0["job_title_family"].astype("category").cat.codes
    df_ch0["title_rich_code"] = df_ch0["title_rich"].astype("category").cat.codes
    print("✅ Feature engineering DONE.")

    # ------------------------------------------------------------------
    # 3. PCA on skill flags
    # ------------------------------------------------------------------
    skill_cols = [
        # Skills
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

    skills = df_ch0[skill_cols].copy()

    pca = PCA(n_components=10, random_state=101)
    pca.fit(skills)
    skills_pca = pca.transform(skills)  # shape: (n_jobs, 10)

    pca_cols = [f"skill_PC{i + 1}" for i in range(10)]
    df_skills_pca = pd.DataFrame(
        skills_pca,
        columns=pca_cols,
        index=skills.index,
    )

    df_ch1 = df_ch0.copy()
    df_ch1[pca_cols] = df_skills_pca

    print("✅ PCA DONE.")

    # ------------------------------------------------------------------
    # 4. Prepare data for XGBoost
    # ------------------------------------------------------------------
    response = df_ch1["sal_mean"].copy()

    feature_cols = [
        # Company
        "size_code",
        "sector_code",
        "state_code",
        "ownership_code",
        # Role
        "seniority_code",
        "title_rich_code",
        # Skills (PCA)
        "skill_PC1",
        "skill_PC2",
        "skill_PC3",
        "skill_PC4",
        "skill_PC5",
        "skill_PC6",
        "skill_PC7",
        "skill_PC8",
        "skill_PC9",
        "skill_PC10",
    ]
    features = df_ch1[feature_cols].copy()

    cat_cols = [
        "size_code",
        "sector_code",
        "state_code",
        "ownership_code",
        "seniority_code",
        "title_rich_code",
    ]
    features[cat_cols] = features[cat_cols].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        response,
        test_size=0.20,
        random_state=101,
    )
    print("✅ Preparing data for model DONE.")

    # ------------------------------------------------------------------
    # 5. Train model with GridSearchCV
    # ------------------------------------------------------------------
    xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        random_state=101,
    )

    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500, 600, 700],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.025, 0.05, 0.1, 0.125, 0.2],
    }

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="r2",
        cv=3,
        verbose=1,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("✅ Training salary model DONE.")

    # Dont forget to add index for chapter 2
    df_ch1["job_id"] = range(len(df_ch1))

    # ------------------------------------------------------------------
    # 5.1 Optional save model + PCA + training data snapshot
    # ------------------------------------------------------------------
    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(pca, MODELS_DIR / "skill_pca_v1.pkl")
        print("PCA model saved.")

        joblib.dump(best_model, MODELS_DIR / "salary_model_v4.pkl")
        print("Salary model saved.")

        df_ch1.to_csv(
            PROCESSED_DATA_DIR / "salary_model_dfv03_pca_jobid.csv", index=False
        )
        print("Clean model data saved.")

        print("✅ All models and data saved DONE.")

    # ------------------------------------------------------------------
    # 6. Optional evaluation
    # ------------------------------------------------------------------
    metrics = None
    if do_eval:
        metrics = evaluate_salary_model(
            model=best_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=features.columns,
            show_plots=show_plots_eval,
        )

        print("\n=== Salary Model Metrics ===")
        for k, v in metrics.items():
            print(f"{k:12} : {v:.4f}")

        print("✅ Model evaluation DONE.")

    print("✅ PIPELINE RUN SUCCESSFULLY.")

    return best_model, metrics, pca, df_ch1
