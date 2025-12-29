# src/job_intel/features/artefacts_ch3.py

"""
Chapter 3 artefact loaders.

Centralises loading of persisted artefacts required for Chapter 3 (positioning),
enforcing consistent sources and schemas across suitability, gap analysis, and
competitiveness components.
"""

from __future__ import annotations

import pandas as pd

from src.job_intel.config import CH2_PROCESSED_DF, SKILL_PROB_MATRIX
from src.job_intel.features.skills_pca import SKILL_COLS


def _normalize_job_id(s: pd.Series, *, where: str) -> pd.Series:
    """
    Canonicalize job_id to string, stripping whitespace and fixing common float artifacts.

    Why:
    - CSV inference can produce int/float/object inconsistencies across artefacts.
    - Downstream joins/isin/reindex must operate on a stable key type.
    """
    if s.isna().any():
        raise ValueError(f"{where} contains NaNs in 'job_id'.")

    # Convert to pandas string dtype (keeps vectorized .str operations reliable)
    out = s.astype("string").str.strip()

    # Fix common case: ids read as floats and stringified like "1060.0"
    out = out.str.replace(r"\.0$", "", regex=True)

    # Guard: empty strings are invalid identifiers
    if (out == "").any():
        raise ValueError(
            f"{where} contains empty-string job_id values after normalization."
        )

    return out


def _sort_by_job_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic ordering:
    - Prefer numeric sort when possible, fall back to lexicographic tie-break.
    """
    job_id_num = pd.to_numeric(df["job_id"], errors="coerce")
    out = df.assign(_job_id_num=job_id_num).sort_values(
        by=["_job_id_num", "job_id"],
        ascending=[True, True],
        kind="mergesort",  # stable sort for reproducibility
    )
    return out.drop(columns="_job_id_num").reset_index(drop=True)


def load_ch3_artefacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load core Chapter 3 artefacts.

    Enforces:
    - Chapter 3 eligibility filtering: jobs must have >=1 extracted skill in SKILL_COLS
    - Deterministic ordering and identifier alignment between jobs_df and skill_prob_matrix
    - Strong key contract: job_id is string, non-null, unique, and aligned across artefacts
    """
    # ------------------------------------------------------------------
    # Validate skill column contract
    # ------------------------------------------------------------------
    if not SKILL_COLS:
        raise ValueError("SKILL_COLS is empty; cannot filter zero-skill jobs.")

    # ------------------------------------------------------------------
    # Load jobs artefact
    # ------------------------------------------------------------------
    jobs_df = pd.read_csv(CH2_PROCESSED_DF, low_memory=False)

    if "job_id" not in jobs_df.columns:
        raise KeyError("jobs_df is missing required column: 'job_id'.")

    jobs_df["job_id"] = _normalize_job_id(jobs_df["job_id"], where="jobs_df")

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
    skill_sum = jobs_df[SKILL_COLS].fillna(0).sum(axis=1)
    jobs_df = jobs_df.loc[skill_sum > 0].copy()

    n_kept = len(jobs_df)
    n_dropped = n_total - n_kept
    if n_kept == 0:
        raise ValueError(
            "Chapter 3 eligibility filter removed all jobs: "
            f"kept={n_kept}, dropped={n_dropped}, total={n_total}."
        )

    jobs_df = _sort_by_job_id(jobs_df)

    # ------------------------------------------------------------------
    # Load + align skill probability matrix to filtered jobs_df universe
    # ------------------------------------------------------------------
    skill_prob_matrix = pd.read_csv(SKILL_PROB_MATRIX, low_memory=False)

    if "job_id" not in skill_prob_matrix.columns:
        raise KeyError("skill_prob_matrix is missing required column: 'job_id'.")

    skill_prob_matrix["job_id"] = _normalize_job_id(
        skill_prob_matrix["job_id"], where="skill_prob_matrix"
    )

    if skill_prob_matrix["job_id"].duplicated().any():
        dup = skill_prob_matrix.loc[
            skill_prob_matrix["job_id"].duplicated(), "job_id"
        ].iloc[0]
        raise ValueError(
            f"skill_prob_matrix contains duplicate job_id values (e.g. {dup!r})."
        )

    # Ensure expected probability columns exist (competitiveness & explanations depend on this)
    expected_prob_cols = [f"{s}_prob" for s in SKILL_COLS]
    missing_prob_cols = [
        c for c in expected_prob_cols if c not in skill_prob_matrix.columns
    ]
    if missing_prob_cols:
        raise KeyError(
            f"skill_prob_matrix is missing {len(missing_prob_cols)} expected prob columns "
            f"(example: {missing_prob_cols[:10]})."
        )

    eligible_ids = jobs_df["job_id"].tolist()
    eligible_set = set(eligible_ids)

    # Restrict to eligible universe
    skill_prob_matrix = skill_prob_matrix.loc[
        skill_prob_matrix["job_id"].isin(eligible_set)
    ].copy()

    # Hard fail if any eligible ids are missing (avoids silent misalignment)
    present_set = set(skill_prob_matrix["job_id"].tolist())
    missing_ids = eligible_set - present_set
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

    # Final invariant: exact key alignment
    if skill_prob_matrix["job_id"].tolist() != eligible_ids:
        raise ValueError(
            "Final alignment invariant failed: skill_prob_matrix job_id order != jobs_df job_id order."
        )

    return jobs_df, skill_prob_matrix
