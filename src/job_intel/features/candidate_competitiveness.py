# src/job_intel/features/candidate_competitiveness.py

import pandas as pd
from typing import Dict, Any

from src.job_intel.features.skill_rarity import compute_skill_rarity_weights


def add_competitiveness(
    profile: Dict[str, Any],
    candidates_df: pd.DataFrame,
    skill_prob_matrix: pd.DataFrame,
    w_missing: float = 0.5,
    w_salary: float = 0.5,
    use_rarity: bool = True,
) -> pd.DataFrame:
    """
    Add a competitiveness index to candidate jobs.

    Competitiveness captures barrier-to-entry using:
    - Expected missing skill requirements (optionally rarity-weighted)
    - Salary percentile within the candidate set

    Higher values indicate harder-to-access jobs.
    """

    # --- 1) user missing-skill indicator ---
    user_vec = profile["derived"]["skill_vector"].iloc[0].to_numpy()  # 0/1
    missing_vec = 1 - user_vec

    skill_cols = profile["derived"]["skill_vector"].columns.tolist()
    prob_cols = [f"{s}_prob" for s in skill_cols]

    # Mismatch guard
    missing_ids = set(candidates_df["job_id"]) - set(skill_prob_matrix["job_id"])
    if missing_ids:
        sample = list(missing_ids)[:5]
        raise KeyError(
            f"job_id mismatch: {len(missing_ids)} candidate job_ids missing from skill_prob_matrix. Example: {sample}"
        )

    # --- 2) align skill-prob matrix to candidates ---
    probs = (
        skill_prob_matrix.loc[
            skill_prob_matrix["job_id"].isin(candidates_df["job_id"]),
            ["job_id"] + prob_cols,
        ]
        .set_index("job_id")
        .loc[candidates_df["job_id"]]
    )

    # --- 3) expected missingness ---
    if use_rarity:
        rarity_weights = compute_skill_rarity_weights(
            skill_prob_matrix=skill_prob_matrix,
            skill_cols=skill_cols,
        )
        w_vec = rarity_weights.to_numpy()

        expected_missing = (probs.to_numpy() * w_vec) @ missing_vec
        expected_missing_norm = expected_missing / len(skill_cols)

    else:
        expected_missing = probs.to_numpy() @ missing_vec
        expected_missing_norm = expected_missing / len(skill_cols)

    out = candidates_df.copy()
    out["expected_missing"] = expected_missing
    out["expected_missing_norm"] = expected_missing_norm

    # --- 4) salary percentile barrier ---
    out["salary_pct"] = out["sal_mean"].rank(pct=True, method="average")

    # --- 5) final competitiveness index ---
    out["competitiveness_index"] = (
        w_missing * out["expected_missing_norm"] + w_salary * out["salary_pct"]
    )

    return out
