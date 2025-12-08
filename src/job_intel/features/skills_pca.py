# src/job_intel/features/skills_pca.py


# NOTEBOOK -> Chapter_1/08_salary_model_skill_pca.ipynb

import joblib
import numpy as np
import pandas as pd

from src.job_intel.config import MODELS_DIR


# Load saved PCA model once (fast)
PCA_MODEL = joblib.load(MODELS_DIR / "skill_pca_v1.pkl")


# IMPORTANT: the exact skill column order used during training
SKILL_COLS = [
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


def transform_skills_to_pca(skills_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform skill columns into PCA components.
    Expects a DataFrame with the 27 skill columns in SKILL_COLS order.

    Returns a DataFrame with PC1...PC10 columns.
    """
    # Select and order the skill columns
    X = skills_df[SKILL_COLS]

    # Transform using the fitted PCA model
    pca_array = PCA_MODEL.transform(X)

    # Build nice column names
    n_components = PCA_MODEL.n_components_
    pca_cols = [f"skill_PC{i+1}" for i in range(n_components)]

    return pd.DataFrame(pca_array, columns=pca_cols, index=skills_df.index)
