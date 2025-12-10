# src/job_intel/models/skill_prob_matrix.py

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from src.job_intel.config import PROCESSED_DATA_DIR, MODELS_DIR


# -------------------------------------------------------------------
# Hard-wired constants (keep consistent with training setup)
# -------------------------------------------------------------------

SKILL_GROUPS = [
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

ALL_FEATURE_COLS = [
    # Company
    "size_code",
    "sector_code",
    "state_code",
    "ownership_code",
    # Role
    "seniority_code",
    "title_rich_code",
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


def build_skill_probability_matrix(
    jobs_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build a job × skill probability matrix using saved skill models.

    Parameters
    ----------
    jobs_df : pd.DataFrame
        Full dataset with all predictor columns used in training.
        Must contain ALL_FEATURE_COLS and SKILL_GROUPS.
    save_path : Path, optional
        If provided, save the matrix to this file path.
        If None, save to PROCESSED_DATA_DIR / 'skill_prob_matrix.csv'.

    Returns
    -------
    prob_df : pd.DataFrame
        DataFrame with same index as jobs_df and one column per skill
        containing predicted probabilities.
    """

    jobs_df = jobs_df.copy()

    # ensure categorical dtypes match training
    cat_cols = [
        "size_code",
        "sector_code",
        "state_code",
        "ownership_code",
        "seniority_code",
        "title_rich_code",
    ]
    jobs_df[cat_cols] = jobs_df[cat_cols].astype("category")

    prob_df = pd.DataFrame(index=jobs_df.index)

    for skill in SKILL_GROUPS:
        model_path = MODELS_DIR / f"{skill}_model.pkl"
        model = joblib.load(model_path)

        feature_cols = [c for c in ALL_FEATURE_COLS if c != skill]
        X = jobs_df[feature_cols]

        proba = model.predict_proba(X)[:, 1]
        prob_df[f"{skill}_prob"] = proba

    if save_path is None:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DATA_DIR / "skill_prob_matrix.csv"
    else:
        output_path = save_path

    prob_df.to_csv(output_path)

    return prob_df
