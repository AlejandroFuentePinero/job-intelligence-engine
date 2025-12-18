# src/job_intel/features/candidate_suitability.py

from typing import Dict, Any

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def add_skill_match(
    profile: Dict[str, Any], candidates_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds:
      - skill_match_score (raw cosine similarity in PCA space, [-1, 1])
      - skill_match_norm  (mapped to [0, 1] via (s+1)/2)
    """
    candidates_df = candidates_df.copy()

    pc_cols = profile["derived"]["skill_pcs"].columns.tolist()
    missing = [c for c in pc_cols if c not in candidates_df.columns]
    if missing:
        raise KeyError(f"Candidates df missing required PCA columns: {missing}")

    user_vec = profile["derived"]["skill_pcs"].to_numpy()  # (1, 10)
    job_mat = candidates_df[pc_cols].to_numpy()  # (N, 10)

    candidates_df["skill_match_score"] = cosine_similarity(user_vec, job_mat).flatten()

    s = candidates_df["skill_match_score"]
    candidates_df["skill_match_norm"] = ((s + 1) / 2).clip(0, 1)

    return candidates_df


def add_salary_score(
    profile: Dict[str, Any], candidates_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds salary_score in [0, 1] where:
      - if target is None: salary_score = 1
      - else: min(sal_mean / target, 1)
    """
    candidates_df = candidates_df.copy()

    if "sal_mean" not in candidates_df.columns:
        raise KeyError("Candidates df missing required column: 'sal_mean'")

    target = profile["raw_inputs"]["salary_target"]
    if target is None:
        candidates_df["salary_score"] = 1.0
    else:
        candidates_df["salary_score"] = (candidates_df["sal_mean"] / target).clip(
            upper=1
        )

    return candidates_df


def add_suitability(
    profile: Dict[str, Any],
    candidates_df: pd.DataFrame,
    w_skill: float = 0.7,
    w_salary: float = 0.3,
) -> pd.DataFrame:
    """
    Adds suitability in [0, 1] as weighted sum of normalized components.
    Requires columns: skill_match_norm, salary_score.
    """
    candidates_df = candidates_df.copy()

    required = ["skill_match_norm", "salary_score"]
    missing = [c for c in required if c not in candidates_df.columns]
    if missing:
        raise KeyError(
            f"Candidates df missing required columns for suitability: {missing}"
        )

    if not (0 <= w_skill <= 1 and 0 <= w_salary <= 1):
        raise ValueError("Weights must be within [0, 1].")

    total = w_skill + w_salary
    if total == 0:
        raise ValueError("At least one weight must be > 0.")

    # enforce sum-to-1 (prevents accidental scaling)
    w_skill = w_skill / total
    w_salary = w_salary / total

    candidates_df["suitability"] = (
        w_skill * candidates_df["skill_match_norm"]
        + w_salary * candidates_df["salary_score"]
    )

    return candidates_df
