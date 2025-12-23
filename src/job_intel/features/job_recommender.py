# src/job_intel/features/job_recommender.py

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.job_intel.config import MODELS_DIR
from src.job_intel.features.artefacts_ch4 import load_ch4_context


def _require_keys(d: dict[str, Any], keys: list[str], *, where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"{where} is missing required keys: {missing}")


def _require_cols(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where} is missing required columns: {missing}")


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
        - "skill_gap": global skill-gap table (not per-job; for upskilling priorities)
        - "skill_prob_matrix": per-job skill requirement probabilities (for per-job explanations)
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

    # Minimal in-function guardrails (no separate eval script)
    _require_keys(
        ctx,
        [
            "candidates_df",
            "salary_model",
            "user_salary_model_features",
            # explanation + upskilling dependencies
            "profile",
            "gap_df",
            "skill_prob_matrix",
        ],
        where="load_ch4_context() output",
    )

    can_df = ctx["candidates_df"]
    salary_model = ctx["salary_model"]
    X = ctx["user_salary_model_features"]
    profile = ctx["profile"]
    gap_df = ctx["gap_df"]
    skill_prob_matrix = ctx["skill_prob_matrix"]

    if not isinstance(can_df, pd.DataFrame):
        raise TypeError(f"ctx['candidates_df'] must be a DataFrame, got {type(can_df)}")
    if not isinstance(skill_prob_matrix, pd.DataFrame):
        raise TypeError(
            f"ctx['skill_prob_matrix'] must be a DataFrame, got {type(skill_prob_matrix)}"
        )
    if not isinstance(gap_df, pd.DataFrame):
        raise TypeError(f"ctx['gap_df'] must be a DataFrame, got {type(gap_df)}")

    # -----------------------------
    # Validate candidates_df contract
    # -----------------------------
    required_cols = ["job_id", "suitability", "competitiveness_index", "sal_mean"]
    _require_cols(can_df, required_cols, where="candidates_df")

    display_cols = ["Size", "Sector", "Industry", "state", "title_rich"]
    _require_cols(can_df, display_cols, where="candidates_df (display columns)")

    if len(X) != len(can_df):
        raise ValueError(
            f"Feature row mismatch: len(user_salary_model_features)={len(X)} vs len(candidates_df)={len(can_df)}"
        )

    # -----------------------------
    # Validate skill_prob_matrix contract (needed for per-job skill explanations)
    # -----------------------------
    if "job_id" not in skill_prob_matrix.columns:
        raise KeyError("skill_prob_matrix must contain 'job_id' column.")

    try:
        user_skills = profile["derived"]["skill_vector"]
    except Exception as e:
        raise KeyError(
            "profile must contain profile['derived']['skill_vector'] for skill explanation."
        ) from e

    if not isinstance(user_skills, pd.DataFrame):
        raise TypeError(
            f"profile['derived']['skill_vector'] must be a DataFrame, got {type(user_skills)}"
        )
    if user_skills.shape[0] != 1:
        raise ValueError(
            f"profile['derived']['skill_vector'] must be 1 row (one user), got shape={user_skills.shape}"
        )

    skill_cols = user_skills.columns.tolist()
    prob_cols = [f"{s}_prob" for s in skill_cols]
    missing_prob_cols = [c for c in prob_cols if c not in skill_prob_matrix.columns]
    if missing_prob_cols:
        raise KeyError(
            f"skill_prob_matrix is missing {len(missing_prob_cols)} prob columns "
            f"(example: {missing_prob_cols[:5]})."
        )

    # Require non-trivial overlap between candidates and probability matrix job_ids
    cand_job_ids = set(can_df["job_id"].astype(str).tolist())
    prob_job_ids = set(skill_prob_matrix["job_id"].astype(str).tolist())
    overlap = cand_job_ids.intersection(prob_job_ids)
    if len(overlap) == 0:
        raise ValueError(
            "No overlap between candidates_df.job_id and skill_prob_matrix.job_id. "
            "Per-job skill explanations and gap computations will be invalid."
        )

    # -----------------------------
    # Validate gap_df contract (global upskilling priorities table)
    # -----------------------------
    gap_required_cols = ["skill", "job_skill_rate", "user_skill", "skill_gap"]
    _require_cols(gap_df, gap_required_cols, where="gap_df (skill gap table)")

    if gap_df["skill"].isna().any():
        raise ValueError("gap_df contains null skill values.")
    if not gap_df["skill"].is_unique:
        dup_n = int(gap_df["skill"].duplicated().sum())
        raise ValueError(f"gap_df.skill is not unique ({dup_n} duplicates).")

    # numeric ranges sanity
    for col in ["job_skill_rate", "skill_gap"]:
        s = pd.to_numeric(gap_df[col], errors="coerce")
        if s.isna().any():
            raise ValueError(f"gap_df contains non-numeric values in '{col}'.")
        if ((s < 0) | (s > 1)).any():
            raise ValueError(f"gap_df '{col}' contains values outside [0, 1].")

    us = pd.to_numeric(gap_df["user_skill"], errors="coerce")
    if us.isna().any():
        raise ValueError("gap_df.user_skill contains non-numeric values.")
    if not set(us.unique()).issubset({0, 1}):
        raise ValueError("gap_df.user_skill must be binary {0,1}.")

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

    if not np.isfinite(y_hat).all():
        raise ValueError(
            "Salary predictions contain NaN/inf; feature construction or model output is invalid."
        )

    can_df = can_df.copy()
    can_df["pred_sal"] = y_hat

    if can_df["pred_sal"].isna().any() or (
        not np.isfinite(can_df["pred_sal"].to_numpy()).all()
    ):
        raise ValueError(
            "Salary predictions contain NaN/inf after assignment; feature construction/alignment is broken."
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

    # Guardrail: job_id integrity
    if candidate_jobs["job_id"].isna().any():
        raise ValueError("candidate_jobs contains null job_id values after gating.")
    if not candidate_jobs["job_id"].is_unique:
        dup_n = int(candidate_jobs["job_id"].duplicated().sum())
        raise ValueError(
            f"candidate_jobs job_id is not unique ({dup_n} duplicates). "
            "This will break stable indexing and Top-N outputs."
        )

    if verbose:
        print(f"* Returning jobs after applying s_min={s_min_used}.")
        print(f"* Suitable jobs identified = {len(candidate_jobs)}.")
        print(
            f"* {len(can_df) - len(candidate_jobs)} filtered out due to low suitability."
        )

    # -----------------------------
    # Competitiveness buckets (canonical: 'bucket')
    # -----------------------------
    if verbose:
        print(
            'Applying competitiveness filter into 2 buckets: "best_now" and "stretch".'
        )

    candidate_jobs["bucket"] = np.where(
        candidate_jobs["competitiveness_index"] <= c_max, "best_now", "stretch"
    )
    # Backwards-compatible alias
    candidate_jobs["competitiveness_bucket"] = candidate_jobs["bucket"]

    allowed = {"best_now", "stretch"}
    actual = set(candidate_jobs["bucket"].dropna().unique())
    if not actual.issubset(allowed):
        raise ValueError(f"Invalid bucket values detected: {sorted(actual - allowed)}")

    vc = candidate_jobs["bucket"].value_counts()
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

    for col in ["suitability", "competitiveness_index", "score", "pred_sal"]:
        if candidate_jobs[col].isna().any() or (
            not np.isfinite(candidate_jobs[col].to_numpy()).all()
        ):
            raise ValueError(
                f"candidate_jobs contains NaN/inf in required column '{col}'."
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
        "bucket",
    ]

    top_best_now = (
        candidate_jobs[candidate_jobs["bucket"] == "best_now"]
        .set_index("job_id")[cols_out]
        .head(top_n_best)
        .copy()
    )

    top_stretch = (
        candidate_jobs[candidate_jobs["bucket"] == "stretch"]
        .set_index("job_id")[cols_out]
        .head(top_n_stretch)
        .copy()
    )

    # Guardrail: Top tables job_ids are subsets of candidate_jobs ids
    cand_ids = set(candidate_jobs["job_id"].tolist())
    if not set(top_best_now.index.tolist()).issubset(cand_ids):
        raise ValueError("top_best_now contains job_id not present in candidate_jobs.")
    if not set(top_stretch.index.tolist()).issubset(cand_ids):
        raise ValueError("top_stretch contains job_id not present in candidate_jobs.")

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
    if n_best_used == 0:
        raise ValueError(
            'No "best_now" rows available after gating/bucketing. Relax constraints or adjust c_max.'
        )
    if n_stretch_used == 0:
        raise ValueError(
            'No "stretch" rows available after gating/bucketing. Relax constraints or adjust c_max.'
        )

    for df_ in (top_best_now, top_stretch):
        df_["sal_mean"] = pd.to_numeric(df_["sal_mean"], errors="coerce")
        df_["pred_sal"] = pd.to_numeric(df_["pred_sal"], errors="coerce")

    if top_best_now["sal_mean"].isna().any() or top_best_now["pred_sal"].isna().any():
        raise ValueError(
            "Non-numeric sal_mean/pred_sal found in top_best_now after coercion."
        )
    if top_stretch["sal_mean"].isna().any() or top_stretch["pred_sal"].isna().any():
        raise ValueError(
            "Non-numeric sal_mean/pred_sal found in top_stretch after coercion."
        )

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
    params_out = {
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
        "skill_gap": gap_df,
        "skill_prob_matrix": skill_prob_matrix,
    }

    salary_summary = {
        "scope": "topN",
        "sal_mean_expected_best": sal_mean_expected_best,
        "sal_mean_pred_best": sal_mean_pred_best,
        "delta_best": delta_best,
        "sal_mean_expected_stretch": sal_mean_expected_stretch,
        "sal_mean_pred_stretch": sal_mean_pred_stretch,
        "delta_stretch": delta_stretch,
        "mean_sal_jump": mean_sal_jump,
        "n_best_used": n_best_used,
        "n_stretch_used": n_stretch_used,
    }

    return {
        "params": params_out,
        "counts": counts,
        "warnings": warnings,
        "tables": tables,
        "salary_summary": salary_summary,
        "profile": profile,
    }
