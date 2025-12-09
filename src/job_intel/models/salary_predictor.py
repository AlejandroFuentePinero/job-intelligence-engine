# src/job_intel/models/salary_predictor.py

import pandas as pd
import joblib

from src.job_intel.config import MODELS_DIR
from src.job_intel.features.skills_pca import (
    transform_skills_to_pca,
    SKILL_COLS,
)


# Load fitted salary model once
SALARY_MODEL = joblib.load(MODELS_DIR / "salary_model_v4.pkl")


# Categorical columns used in the model
CAT_COLS = [
    "size_code",
    "sector_code",
    "state_code",
    "ownership_code",
    "seniority_code",
    "title_rich_code",
]


def predict_salary(record: pd.DataFrame) -> float:
    """
    Predict salary for a single job/user record.

    Expects a DataFrame with:
      - 6 categorical columns (codes)
      - 27 raw binary skill columns (SKILL_COLS)

    This function:
      1. Ensures correct dtypes
      2. Transforms skills using saved PCA model
      3. Feeds everything into the saved XGBoost model
      4. Returns a single salary prediction (float)
    """

    # Defensive copy
    record = record.copy()

    # Ensure categorical dtypes match model expectations
    record[CAT_COLS] = record[CAT_COLS].astype("category")

    # Convert skills → PCA components
    skill_pca_df = transform_skills_to_pca(record[SKILL_COLS])

    # Final model input = categorical codes + PCA components
    X = pd.concat([record[CAT_COLS], skill_pca_df], axis=1)

    # Predict (model outputs array)
    prediction = SALARY_MODEL.predict(X)[0]

    return float(prediction)
