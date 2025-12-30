# src/job_intel/evaluation/chapter_4_entrypoint_eval.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.job_intel.features.artefacts_ch4 import load_ch4_context


# -----------------------------
# Helpers
# -----------------------------
_TOL = 1e-9


def _assert_or_raise(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _get_skill_prob_job_ids(skill_prob_matrix: Any) -> set:
    """
    Support both:
    - DataFrame with job_id index
    - DataFrame with job_id column
    """
    if isinstance(skill_prob_matrix, pd.DataFrame):
        if "job_id" in skill_prob_matrix.columns:
            return set(skill_prob_matrix["job_id"])
        return set(skill_prob_matrix.index)
    # fallback: try attribute access
    if hasattr(skill_prob_matrix, "index"):
        return set(skill_prob_matrix.index)
    raise TypeError(
        "skill_prob_matrix is not a supported type for extracting job_id universe."
    )


def _expected_salary_feature_cols() -> List[str]:
    pcs = [f"skill_PC{i}" for i in range(1, 11)]
    base = [
        "size_code",
        "sector_code",
        "state_code",
        "ownership_code",
        "seniority_code",
        "title_rich_code",
    ]
    return base + pcs


def _check_required_cols(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    _assert_or_raise(len(missing) == 0, f"{label}: missing required columns: {missing}")


def _check_no_nulls(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    null_counts = df[list(cols)].isna().sum()
    bad = null_counts[null_counts > 0]
    _assert_or_raise(
        bad.empty, f"{label}: nulls found in required columns: {bad.to_dict()}"
    )


# -----------------------------
# Chapter 4 Entry Evaluations
# -----------------------------
def evaluate_ch4_entrypoint(
    *,
    # keep this minimal + deterministic; adjust if this yields empty candidates in your data
    skill_text: str = "",
    current_state: str = "ALL",
    job_title_family: Optional[str] = "data_scientist",
    job_title_rich: Optional[str] = None,
    target_sectors: Optional[List[str]] = None,
    salary_target: Optional[int] = None,
    return_top_n_jobs: Optional[int] = 500,  # keep runtime low for eval
) -> Dict[str, Any]:
    """
    Minimal delta evaluation for Chapter 4 context loader.

    Validates only what Chapter 4 adds on top of Chapter 3:
    - salary feature schema + shape
    - PC broadcasting correctness
    - salary predict smoke test
    - job_id alignment to skill_prob_matrix (upskilling prerequisite)
    """
    ctx = load_ch4_context(
        skill_text=skill_text,
        current_state=current_state,
        job_title_family=job_title_family,
        job_title_rich=job_title_rich,
        target_sectors=target_sectors,
        salary_target=salary_target,
        return_top_n_jobs=return_top_n_jobs,
        run_sensitivity=False,
    )

    # unpack
    candidates_df: pd.DataFrame = ctx["candidates_df"]
    profile: Dict[str, Any] = ctx["profile"]
    skill_prob_matrix = ctx["skill_prob_matrix"]
    salary_model = ctx["salary_model"]
    X: pd.DataFrame = ctx["user_salary_model_features"]

    print("TEST - CH4 ENTRYPOINT (MINIMAL DELTA)")
    print("------------------------------------")

    # --- Check 0: non-empty candidate universe (otherwise downstream is meaningless)
    _assert_or_raise(
        len(candidates_df) > 0,
        "Candidates empty: cannot evaluate Chapter 4 entrypoint.",
    )

    # --- Check 1: salary feature schema + shape
    expected_cols = _expected_salary_feature_cols()
    _check_required_cols(X, expected_cols, label="user_salary_model_features")
    _assert_or_raise(
        X.shape[0] == candidates_df.shape[0],
        f"Row mismatch: X has {X.shape[0]} rows, candidates_df has {candidates_df.shape[0]} rows.",
    )
    _check_no_nulls(X, expected_cols, label="user_salary_model_features")
    print("✅ Salary feature matrix schema + shape: PASS")

    # --- Check 2: PC broadcasting correctness
    user_pcs = profile["derived"]["skill_pcs"]
    _assert_or_raise(
        isinstance(user_pcs, pd.DataFrame),
        "profile['derived']['skill_pcs'] must be a DataFrame.",
    )
    _assert_or_raise(
        len(user_pcs) == 1, f"Expected user_pcs to be 1 row, got {len(user_pcs)}"
    )

    for pc in [f"skill_PC{i}" for i in range(1, 11)]:
        user_val = float(user_pcs.iloc[0][pc])
        col = X[pc].astype(float)
        # all rows equal to user pc value (within tolerance)
        max_abs = float((col - user_val).abs().max())
        _assert_or_raise(
            max_abs <= _TOL,
            f"PC broadcast FAIL for {pc}: max |Δ| = {max_abs} (expected all rows == {user_val}).",
        )
    print("✅ PC broadcasting correctness: PASS")

    # --- Check 3: salary prediction smoke test
    y_hat = salary_model.predict(X)
    _assert_or_raise(
        len(y_hat) == len(X), "Predict length mismatch vs feature matrix rows."
    )
    y_hat = np.asarray(y_hat, dtype=float)
    _assert_or_raise(
        np.isfinite(y_hat).all(), "Predict contains non-finite values (NaN/inf)."
    )
    _assert_or_raise(
        (y_hat >= 0).all(), "Predict contains negative salaries (sanity check)."
    )
    print("✅ Salary predict smoke test: PASS")

    # --- Check 4: job_id alignment prerequisite for upskilling/what-if
    cand_ids = set(candidates_df["job_id"])
    skill_ids = _get_skill_prob_job_ids(skill_prob_matrix)
    _assert_or_raise(
        cand_ids.issubset(skill_ids),
        f"Alignment FAIL: {len(cand_ids - skill_ids)} candidate job_ids missing from skill_prob_matrix universe.",
    )
    print("✅ job_id alignment to skill_prob_matrix: PASS")

    return ctx
