# src/job_intel/features/artefacts_ch3.py

"""
Chapter 3 artefact loaders.

This module centralises loading of all persisted artefacts required for
individual positioning (Chapter 3), ensuring consistent sources and schemas
across suitability, gap analysis, and competitiveness components.

Artefacts loaded here are treated as read-only in the sense that:
- Source CSVs are never modified.
- Returned DataFrames may be filtered/reordered to enforce Chapter 3 eligibility
  and deterministic alignment across artefacts.
"""

import pandas as pd

from src.job_intel.config import CH2_PROCESSED_DF, SKILL_PROB_MATRIX
from src.job_intel.features.skills_pca import SKILL_COLS


def load_ch3_artefacts():
    """
    Load core Chapter 3 artefacts.

    Notes
    -----
    Applies Chapter 3 eligibility filtering:
    - Jobs must have at least one extracted skill flag present in SKILL_COLS.
    Enforces deterministic ordering and identifier alignment:
    - jobs_df is sorted by job_id after filtering
    - skill_prob_matrix is restricted to eligible job_ids and reindexed to match
      jobs_df ordering
    """

    # ------------------------------------------------------------------
    # Validate skill column contract
    # ------------------------------------------------------------------
    if len(SKILL_COLS) == 0:
        raise ValueError("SKILL_COLS is empty; cannot filter zero-skill jobs.")

    # ------------------------------------------------------------------
    # Load jobs artefact
    # ------------------------------------------------------------------
    jobs_df = pd.read_csv(CH2_PROCESSED_DF)

    if "job_id" not in jobs_df.columns:
        raise KeyError("jobs_df is missing required column: 'job_id'.")

    if jobs_df["job_id"].isna().any():
        raise ValueError("jobs_df contains NaNs in 'job_id'.")

    if jobs_df["job_id"].duplicated().any():
        dup = jobs_df.loc[jobs_df["job_id"].duplicated(), "job_id"].iloc[0]
        raise ValueError(f"jobs_df contains duplicate job_id values (e.g. {dup!r}).")

    missing_skill_cols = [c for c in SKILL_COLS if c not in jobs_df.columns]
    if missing_skill_cols:
        raise KeyError(f"Missing skill columns in jobs_df: {missing_skill_cols}")

    # ------------------------------------------------------------------
    # Chapter 3 eligibility filter: remove jobs with zero extracted skills
    # ------------------------------------------------------------------
    n_total = len(jobs_df)
    mask = jobs_df[SKILL_COLS].sum(axis=1) > 0
    jobs_df = jobs_df.loc[mask].copy()

    n_kept = len(jobs_df)
    n_dropped = n_total - n_kept
    if n_kept == 0:
        raise ValueError(
            "Chapter 3 eligibility filter removed all jobs: "
            f"kept={n_kept}, dropped={n_dropped}, total={n_total}."
        )

    # Deterministic ordering for downstream reproducibility
    jobs_df = jobs_df.sort_values("job_id").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Load + align skill probability matrix to filtered jobs_df universe
    # ------------------------------------------------------------------
    skill_prob_matrix = pd.read_csv(SKILL_PROB_MATRIX)

    if "job_id" not in skill_prob_matrix.columns:
        raise KeyError("skill_prob_matrix is missing required column: 'job_id'.")

    if skill_prob_matrix["job_id"].isna().any():
        raise ValueError("skill_prob_matrix contains NaNs in 'job_id'.")

    if skill_prob_matrix["job_id"].duplicated().any():
        dup = skill_prob_matrix.loc[
            skill_prob_matrix["job_id"].duplicated(), "job_id"
        ].iloc[0]
        raise ValueError(
            f"skill_prob_matrix contains duplicate job_id values (e.g. {dup!r})."
        )

    eligible_ids = jobs_df["job_id"].tolist()

    # Restrict to eligible universe
    skill_prob_matrix = skill_prob_matrix.loc[
        skill_prob_matrix["job_id"].isin(eligible_ids)
    ].copy()

    # Hard fail if any eligible ids are missing (avoids silent misalignment)
    missing_ids = set(eligible_ids) - set(skill_prob_matrix["job_id"].tolist())
    if missing_ids:
        ex = list(missing_ids)[:10]
        raise KeyError(
            f"skill_prob_matrix is missing {len(missing_ids)} job_ids present in jobs_df. "
            f"Examples: {ex}"
        )

    # Deterministic alignment: reorder rows to match jobs_df ordering
    skill_prob_matrix = (
        skill_prob_matrix.set_index("job_id").reindex(eligible_ids).reset_index()
    )

    return jobs_df, skill_prob_matrix
