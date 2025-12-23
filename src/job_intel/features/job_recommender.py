# src/job_intel/features/job_recommender.py

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.job_intel.config import MODELS_DIR
from src.job_intel.features.artefacts_ch4 import load_ch4_context


def job_recommender(
    skill_text: str = "",
    current_state: str = "ALL",
    job_title_family: str | None = None,
    job_title_rich: str | None = None,
    target_sectors: list[str] | None = None,
    salary_target: float | None = None,
    explain_skills: bool | None = None,
    w_skill: float = 0.7,
    w_salary: float = 0.3,
    top_k_gaps: int = 200,
    return_top_n_jobs: int | None = 6200,  # None for full-universe engine mode
    run_sensitivity: bool = False,
    salary_model_path=MODELS_DIR / "salary_model_v4.pkl",
    # Recommender controls
    s_min_base: float = 0.70,
    s_min_floor: float = 0.60,
    n_target: int = 50,
    c_max: float = 0.50,
    min_bucket_size_bestnow: int = 10,
    min_bucket_size_stretch: int = 5,
    alpha: float = 0.5,
    top_n_best: int = 10,
    top_n_stretch: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Chapter 4 — Hybrid Job Recommender (v1)

    Runs the Chapter 4 context loader (Chapter 3 positioning + salary feature prep),
    attaches salary predictions conditioned on the user's skill PCs, then produces
    two ranked recommendation buckets:

    - best_now: jobs with competitiveness_index <= c_max
    - stretch:  jobs with competitiveness_index  > c_max

    Ranking score (within each bucket):
        score = suitability - alpha * competitiveness_index

    Suitability gating:
    - First apply s_min_base; if < n_target jobs remain, fall back to s_min_floor.
    - If still < n_target jobs remain, raise ValueError (user should relax constraints or upskill).

    Returns
    -------
    dict with keys:
    - "params": all thresholds/knobs used
    - "counts": candidate counts and per-bucket counts
    - "warnings": list[str]
    - "tables":
        - "candidate_jobs": filtered + scored candidate set used for ranking
        - "top_best_now": top-N best_now recommendations (indexed by job_id)
        - "top_stretch": top-N stretch recommendations (indexed by job_id)
    - "salary_summary": mean salary comparisons for best_now vs stretch (Top-N only)
    """
    warnings: list[str] = []

    # -----------------------------
    # Load context (Ch3 positioning + salary model + feature matrix)
    # -----------------------------
    ctx = load_ch4_context(
        skill_text=skill_text,
        current_state=current_state,
        job_title_family=job_title_family,
        job_title_rich=job_title_rich,
        target_sectors=target_sectors,
        salary_target=salary_target,
        explain_skills=explain_skills,
        w_skill=w_skill,
        w_salary=w_salary,
        top_k_gaps=top_k_gaps,
        return_top_n_jobs=return_top_n_jobs,
        run_sensitivity=run_sensitivity,
        salary_model_path=salary_model_path,
    )

    can_df = ctx["candidates_df"]
    salary_model = ctx["salary_model"]
    X = ctx["user_salary_model_features"]

    # -----------------------------
    # Predict salary (row-wise aligned to candidates_df)
    # -----------------------------
    if verbose:
        print("Predicting salary based on input skills...")

    y_hat = salary_model.predict(X)

    if len(y_hat) != len(can_df):
        raise ValueError(
            f"Salary prediction length mismatch: len(pred)={len(y_hat)} vs len(candidates_df)={len(can_df)}"
        )

    can_df = can_df.copy()
    can_df["pred_sal"] = y_hat

    if can_df["pred_sal"].isna().any():
        raise ValueError(
            "Salary predictions contain NaNs; feature construction/alignment is broken."
        )

    if verbose:
        print("Acceptance shape test passed:", len(y_hat) == len(can_df))
        print("Acceptance alignment test passed:", can_df["pred_sal"].isna().sum() == 0)

    # -----------------------------
    # Suitability gating (base -> floor -> fail)
    # -----------------------------
    if verbose:
        print(f"Applying the suitability index threshold {s_min_base}...")

    try_base = can_df[can_df["suitability"] >= s_min_base]
    try_floor = can_df[can_df["suitability"] >= s_min_floor]

    if len(try_base) < n_target:
        if verbose:
            print(
                f"Total number of jobs returned is too low (<{n_target}); trying lower suitability threshold {s_min_floor}"
            )

        if len(try_floor) < n_target:
            raise ValueError(
                f"Total number of jobs returned is still too low (<{n_target}) after lowering threshold to {s_min_floor}. "
                "Reduce your constraints or switch to upskilling."
            )

        candidate_jobs = try_floor.copy()
        s_min_used = s_min_floor
    else:
        candidate_jobs = try_base.copy()
        s_min_used = s_min_base

    if verbose:
        print(f"* Returning jobs after applying s_min={s_min_used}.")
        print(f"* Suitable jobs identified = {len(candidate_jobs)}.")
        print(
            f"* {len(can_df) - len(candidate_jobs)} filtered out due to low suitability."
        )

    # -----------------------------
    # Competitiveness buckets
    # -----------------------------
    if verbose:
        print(
            'Applying competitiveness filter into 2 buckets: "best-now" and "stretch".'
        )

    candidate_jobs["competitiveness_bucket"] = np.where(
        candidate_jobs["competitiveness_index"] <= c_max, "best_now", "stretch"
    )

    vc = candidate_jobs["competitiveness_bucket"].value_counts()
    best_now = int(vc.get("best_now", 0))
    stretch = int(vc.get("stretch", 0))

    if verbose:
        print(f'Number of "Best-now" jobs = {best_now}')
        print(f'Number of "Stretch" jobs = {stretch}')

    if best_now <= min_bucket_size_bestnow:
        msg = 'Low number of "best-now" options. Consider relaxing constraints or using upskilling.'
        warnings.append(msg)
        if verbose:
            print("WARNING:", msg)

    if stretch <= min_bucket_size_stretch:
        msg = 'Low number of "stretch" options. Consider relaxing constraints or using upskilling.'
        warnings.append(msg)
        if verbose:
            print("WARNING:", msg)

    # -----------------------------
    # Ranking score and sorting
    # -----------------------------
    if verbose:
        print("Computing ranking score based on suitability and competitiveness.")

    candidate_jobs["score"] = candidate_jobs["suitability"] - (
        alpha * candidate_jobs["competitiveness_index"]
    )

    candidate_jobs = candidate_jobs.sort_values(
        by=["score", "suitability", "competitiveness_index", "job_id"],
        ascending=[False, False, True, True],
    )

    cols_out = [
        "Size",
        "Sector",
        "Industry",
        "state",
        "title_rich",
        "sal_mean",
        "pred_sal",
        "suitability",
        "competitiveness_index",
        "score",
    ]

    top_best_now = (
        candidate_jobs[candidate_jobs["competitiveness_bucket"] == "best_now"]
        .set_index("job_id")[cols_out]
        .head(top_n_best)
        .copy()
    )

    top_stretch = (
        candidate_jobs[candidate_jobs["competitiveness_bucket"] == "stretch"]
        .set_index("job_id")[cols_out]
        .head(top_n_stretch)
        .copy()
    )

    n_best_used = int(len(top_best_now))
    n_stretch_used = int(len(top_stretch))

    if n_best_used < top_n_best:
        msg = f'Only {n_best_used}/{top_n_best} "best-now" rows available for Top-N summary.'
        warnings.append(msg)
        if verbose:
            print("WARNING:", msg)

    if n_stretch_used < top_n_stretch:
        msg = f'Only {n_stretch_used}/{top_n_stretch} "stretch" rows available for Top-N summary.'
        warnings.append(msg)
        if verbose:
            print("WARNING:", msg)

    if verbose:
        print(f'Top {top_n_best} "best-now" jobs:\n')
        print(top_best_now)
        print(f'\nTop {top_n_stretch} "stretch" jobs:\n')
        print(top_stretch)

    # -----------------------------
    # Salary summary (Top-N means)
    # -----------------------------
    for df_ in (top_best_now, top_stretch):
        df_["sal_mean"] = pd.to_numeric(df_["sal_mean"], errors="coerce")
        df_["pred_sal"] = pd.to_numeric(df_["pred_sal"], errors="coerce")

    sal_mean_expected_best = float(top_best_now["sal_mean"].mean())
    sal_mean_pred_best = float(top_best_now["pred_sal"].mean())
    delta_best = sal_mean_expected_best - sal_mean_pred_best

    sal_mean_expected_stretch = float(top_stretch["sal_mean"].mean())
    sal_mean_pred_stretch = float(top_stretch["pred_sal"].mean())
    delta_stretch = sal_mean_expected_stretch - sal_mean_pred_stretch

    mean_sal_jump = delta_stretch - delta_best

    if verbose:
        print("\nBest-now salary comparison (Top-N):")
        print("---------------------------")
        print(f"Expected mean salary: {sal_mean_expected_best:.2f}")
        print(f"Predicted mean salary (based on your skills): {sal_mean_pred_best:.2f}")
        print(f"Delta (expected - predicted): {delta_best:.2f}")

        print("\nStretch salary comparison (Top-N):")
        print("---------------------------")
        print(f"Expected mean salary: {sal_mean_expected_stretch:.2f}")
        print(
            f"Predicted mean salary (based on your skills): {sal_mean_pred_stretch:.2f}"
        )
        print(f"Delta (expected - predicted): {delta_stretch:.2f}")

        print(
            f'\nMean salary jump from "Best-now" to "Stretch" jobs (Top-N): {mean_sal_jump:.2f}'
        )

    # -----------------------------
    # Package results
    # -----------------------------
    params = {
        "s_min_base": s_min_base,
        "s_min_floor": s_min_floor,
        "s_min_used": s_min_used,
        "n_target": n_target,
        "c_max": c_max,
        "min_bucket_size_bestnow": min_bucket_size_bestnow,
        "min_bucket_size_stretch": min_bucket_size_stretch,
        "alpha": alpha,
        "top_n_best": top_n_best,
        "top_n_stretch": top_n_stretch,
    }

    counts = {
        "candidate_jobs": int(len(candidate_jobs)),
        "best_now": best_now,
        "stretch": stretch,
        "n_best_used": n_best_used,
        "n_stretch_used": n_stretch_used,
    }

    tables = {
        "candidate_jobs": candidate_jobs,
        "top_best_now": top_best_now,
        "top_stretch": top_stretch,
    }

    # Backwards-compatible keys + explicit Top-N keys
    salary_summary = {
        "sal_mean_expected_best": sal_mean_expected_best,
        "sal_mean_pred_best": sal_mean_pred_best,
        "delta_best": delta_best,
        "sal_mean_expected_stretch": sal_mean_expected_stretch,
        "sal_mean_pred_stretch": sal_mean_pred_stretch,
        "delta_stretch": delta_stretch,
        "mean_sal_jump": mean_sal_jump,
        # Explicit Top-N naming (same values)
        "sal_mean_expected_best_topN": sal_mean_expected_best,
        "sal_mean_pred_best_topN": sal_mean_pred_best,
        "delta_best_topN": delta_best,
        "sal_mean_expected_stretch_topN": sal_mean_expected_stretch,
        "sal_mean_pred_stretch_topN": sal_mean_pred_stretch,
        "delta_stretch_topN": delta_stretch,
        "mean_sal_jump_topN": mean_sal_jump,
        "n_best_used": n_best_used,
        "n_stretch_used": n_stretch_used,
    }

    return {
        "params": params,
        "counts": counts,
        "warnings": warnings,
        "tables": tables,
        "salary_summary": salary_summary,
    }
