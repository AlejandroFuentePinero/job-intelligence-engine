# src/job_intel/features/artefacts_ch4.py

from typing import Optional, Any, Dict
import pandas as pd
import joblib

from src.job_intel.positioning import run_positioning
from src.job_intel.features.artefacts_ch3 import load_ch3_artefacts
from src.job_intel.config import MODELS_DIR


def _require_cols(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where} is missing required columns: {missing}")


def load_ch4_context(
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
    return_top_n_jobs: int | None = 6200,  # set None for full-universe engine mode
    run_sensitivity: bool = False,
    salary_model_path=MODELS_DIR / "salary_model_v4.pkl",
    candidate_override_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Chapter 4 — Context Loader / Wrapper
    """

    profile, candidates_df, gap_df, sensitivity_out = run_positioning(
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
        candidate_override_df=candidate_override_df,
    )

    jobs_df, skill_prob_matrix = load_ch3_artefacts()
    salary_model = joblib.load(salary_model_path)

    # Candidate-level salary features (job/company attributes).
    salary_feature_cols = [
        "size_code",
        "sector_code",
        "state_code",
        "ownership_code",
        "seniority_code",
        "title_rich_code",
    ]
    _require_cols(
        candidates_df, salary_feature_cols, where="candidates_df (salary features)"
    )

    model_features = candidates_df[salary_feature_cols].copy()

    # Keep dtype convention consistent with the trained model.
    model_features[salary_feature_cols] = model_features[salary_feature_cols].astype(
        "category"
    )

    # Broadcast user skill PCs across rows
    if "derived" not in profile or "skill_pcs" not in profile["derived"]:
        raise KeyError(
            "profile['derived']['skill_pcs'] is required for salary feature construction."
        )

    user_pcs = profile["derived"]["skill_pcs"]
    if not isinstance(user_pcs, pd.DataFrame):
        raise TypeError(
            f"profile['derived']['skill_pcs'] must be a DataFrame, got {type(user_pcs)}"
        )
    if len(user_pcs) != 1:
        raise ValueError(f"Expected user_pcs to be 1 row, got {len(user_pcs)}")

    if "dummy" in model_features.columns or "dummy" in user_pcs.columns:
        raise ValueError(
            "Internal column name 'dummy' must not exist in model_features or user_pcs."
        )

    overlap = set(model_features.columns).intersection(set(user_pcs.columns))
    if overlap:
        raise ValueError(
            f"user_pcs columns overlap with model_features columns: {sorted(overlap)}. "
            "This would silently corrupt the salary feature matrix."
        )

    mf = model_features.assign(dummy=1)
    up = user_pcs.assign(dummy=1)
    user_salary_model_features = mf.merge(up, on="dummy", how="left").drop(
        columns="dummy"
    )

    # Enforce row alignment with candidates_df
    if len(user_salary_model_features) != len(candidates_df):
        raise ValueError(
            "Salary feature construction produced row mismatch: "
            f"len(features)={len(user_salary_model_features)} vs len(candidates_df)={len(candidates_df)}"
        )
    user_salary_model_features.index = candidates_df.index

    return {
        "profile": profile,
        "candidates_df": candidates_df,
        "gap_df": gap_df,
        "sensitivity_out": sensitivity_out,
        "jobs_df": jobs_df,
        "skill_prob_matrix": skill_prob_matrix,
        "salary_model": salary_model,
        "user_salary_model_features": user_salary_model_features,
    }
