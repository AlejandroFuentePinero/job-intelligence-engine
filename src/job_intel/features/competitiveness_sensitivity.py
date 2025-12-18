# src/job_intel/features/competitiveness_sensitivity.py

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from scipy.stats import spearmanr


DEFAULT_WEIGHT_GRID: List[Dict[str, float]] = [
    {"w_skill": 0.1, "w_salary": 0.9},
    {"w_skill": 0.2, "w_salary": 0.8},
    {"w_skill": 0.3, "w_salary": 0.7},
    {"w_skill": 0.4, "w_salary": 0.6},
    {"w_skill": 0.5, "w_salary": 0.5},
    {"w_skill": 0.6, "w_salary": 0.4},
    {"w_skill": 0.7, "w_salary": 0.3},  # baseline default
    {"w_skill": 0.8, "w_salary": 0.2},
    {"w_skill": 0.9, "w_salary": 0.1},
]


def compute_competitiveness_sensitivity(
    candidates_df: pd.DataFrame,
    weight_grid: Sequence[Dict[str, float]] = DEFAULT_WEIGHT_GRID,
    baseline_w_skill: float = 0.7,
    baseline_w_salary: float = 0.3,
    job_id_col: str = "job_id",
    missing_col: str = "expected_missing_rarity_norm",
    salary_pct_col: str = "salary_pct",
) -> pd.DataFrame:
    """
    Compute robustness of candidate competitiveness rankings to weight choices.

    Sensitivity is measured as Spearman rank correlation (rho) between
    the baseline competitiveness ranking and rankings recomputed under
    alternative (w_skill, w_salary) configurations.

    Returns
    -------
    pd.DataFrame with columns:
      - w_skill
      - w_salary
      - spearman_rho_vs_baseline
    """

    required = {job_id_col, missing_col, salary_pct_col}
    missing = required - set(candidates_df.columns)
    if missing:
        raise ValueError(f"candidates_df missing required columns: {sorted(missing)}")

    if candidates_df[job_id_col].duplicated().any():
        raise ValueError(
            f"Duplicate {job_id_col} values found; ranking alignment requires unique job ids."
        )

    df = candidates_df[[job_id_col, missing_col, salary_pct_col]].dropna().copy()

    # Baseline ranks (recomputed here so baseline is guaranteed)
    base_score = (
        baseline_w_skill * df[missing_col] + baseline_w_salary * df[salary_pct_col]
    )
    base_rank = base_score.rank(ascending=False, method="average")
    base_rank_by_job = pd.Series(base_rank.values, index=df[job_id_col].values)

    out: List[Dict[str, float]] = []

    for w in weight_grid:
        w_skill = float(w["w_skill"])
        w_salary = float(w["w_salary"])

        tmp_score = w_skill * df[missing_col] + w_salary * df[salary_pct_col]
        tmp_rank = tmp_score.rank(ascending=False, method="average")
        tmp_rank_by_job = pd.Series(tmp_rank.values, index=df[job_id_col].values)

        aligned = pd.concat([base_rank_by_job, tmp_rank_by_job], axis=1).dropna()
        rho, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])

        out.append(
            {
                "w_skill": w_skill,
                "w_salary": w_salary,
                "spearman_rho_vs_baseline": float(rho),
            }
        )

    return pd.DataFrame(out)
