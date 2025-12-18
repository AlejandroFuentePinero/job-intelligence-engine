# src/job_intel/features/skill_rarity.py

import pandas as pd
from typing import List


def compute_skill_rarity_weights(
    skill_prob_matrix: pd.DataFrame,
    skill_cols: List[str],
    eps: float = 1e-5,
) -> pd.Series:
    """
    Compute global rarity weights for each skill based on inverse frequency.

    Returns a Series indexed by canonical skill names, mean-normalised
    (average weight = 1).
    """

    global_skill_p_mean = (
        skill_prob_matrix.drop(columns=["job_id"])
        .mean(axis=0)
        .rename("global_average")
        .reset_index()
        .rename(columns={"index": "prob_col"})
    )

    # Inverse-frequency weighting
    global_skill_p_mean["rarity_weight"] = 1 / (
        global_skill_p_mean["global_average"] + eps
    )

    # Convert "skill_prob" → "skill"
    global_skill_p_mean["skill"] = global_skill_p_mean["prob_col"].str.replace(
        "_prob", "", regex=False
    )

    # Mean-normalise for numerical stability
    global_skill_p_mean["weight_norm"] = (
        global_skill_p_mean["rarity_weight"]
        / global_skill_p_mean["rarity_weight"].mean()
    )

    # Align to canonical skill order
    w_lookup = global_skill_p_mean.set_index("skill")["weight_norm"].reindex(skill_cols)

    if w_lookup.isna().any():
        missing = w_lookup[w_lookup.isna()].index.tolist()
        raise ValueError(f"Missing rarity weights for skills: {missing}")

    return w_lookup
