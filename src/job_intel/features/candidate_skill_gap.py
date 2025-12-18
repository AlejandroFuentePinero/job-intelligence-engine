# src/job_intel/features/candidate_skill_gap.py

from typing import Dict, Any
import pandas as pd


def compute_skill_gaps(
    profile: Dict[str, Any],
    candidates_df: pd.DataFrame,
    skill_prob_matrix: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Compute skill gaps for a user based on the top-K most suitable jobs.

    Gap severity is defined as:
      gap(skill) = P(skill|required | top-K jobs) if user lacks skill, else 0

    Returns
    -------
    pd.DataFrame with columns:
      - skill
      - job_skill_rate   (mean probability across top-K jobs)
      - user_skill       (0/1)
      - skill_gap        (0–1)
    """

    # --- basic validation ---
    if "job_id" not in candidates_df.columns:
        raise KeyError("candidates_df must contain 'job_id'")

    if "job_id" not in skill_prob_matrix.columns:
        raise KeyError("skill_prob_matrix must contain 'job_id'")

    user_skills = profile["derived"]["skill_vector"]
    skill_cols = user_skills.columns.tolist()
    prob_cols = [f"{s}_prob" for s in skill_cols]

    missing_probs = [c for c in prob_cols if c not in skill_prob_matrix.columns]
    if missing_probs:
        raise KeyError(f"Skill probability matrix missing columns: {missing_probs}")

    # --- select top-K jobs ---
    k = min(len(candidates_df), top_k)
    top_k_jobs = candidates_df.sort_values("suitability", ascending=False).head(k)

    # --- align probabilities ---
    top_k_probs = skill_prob_matrix.loc[
        skill_prob_matrix["job_id"].isin(top_k_jobs["job_id"]),
        ["job_id"] + prob_cols,
    ]

    # --- mean probability per skill ---
    job_skill_rate = (
        top_k_probs[prob_cols].mean(axis=0).rename("job_skill_rate").reset_index()
    )

    job_skill_rate["skill"] = job_skill_rate["index"].str.replace(
        "_prob", "", regex=False
    )
    job_skill_rate = job_skill_rate.drop(columns="index")

    # --- attach user skill presence ---
    job_skill_rate["user_skill"] = user_skills.T.iloc[:, 0].values

    # --- gap severity ---
    job_skill_rate["skill_gap"] = (job_skill_rate["user_skill"] == 0) * job_skill_rate[
        "job_skill_rate"
    ]

    return job_skill_rate
