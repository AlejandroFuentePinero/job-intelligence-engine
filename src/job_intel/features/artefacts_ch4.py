# src/job_intel/features/artefacts_ch4.py

import joblib

from src.job_intel.positioning import run_positioning
from src.job_intel.features.artefacts_ch3 import load_ch3_artefacts
from src.job_intel.config import MODELS_DIR


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
) -> dict:
    """
    Chapter 4 — Context Loader / Wrapper

    Purpose
    - Provides a single, canonical entrypoint to obtain everything Chapter 4 needs:
      (1) Chapter 3 user positioning outputs (profile, candidates, gaps, sensitivity),
      (2) aligned Chapter 3 artefacts (refined jobs df + skill probability matrix),
      (3) the trained salary model artefact, and
      (4) a ready-to-predict salary feature matrix for the given user profile.

    Key behavior
    - Calls `run_positioning(...)` to build the candidate universe and user-derived features.
    - Loads persisted Chapter 3 artefacts via `load_ch3_artefacts()` (no expensive Chapter 1 reruns).
    - Broadcasts the user's skill PCs (1×10) across the candidate feature rows to form a design matrix
      suitable for salary prediction.

    Parameters
    - The parameters mirror `run_positioning(...)`. The caller should choose `return_top_n_jobs=None`
      when Chapter 4 needs a full candidate universe (recommended for recommender engine mode).

    Returns
    - dict with:
        - "profile": user profile dict including derived fields (e.g., skill PCs)
        - "candidates_df": candidate jobs after filtering + positioning indices
        - "gap_df": skill gap summary outputs (as produced by Chapter 3)
        - "sensitivity_out": sensitivity outputs (or None)
        - "jobs_df": refined, feature-complete jobs dataframe (aligned artefact)
        - "skill_prob_matrix": job×skill requirement probability matrix (aligned artefact)
        - "salary_model": loaded salary model artefact
        - "user_salary_model_features": candidate-level salary model feature matrix (with user PCs)
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
    )

    jobs_df, skill_prob_matrix = load_ch3_artefacts()
    salary_model = joblib.load(salary_model_path)

    # Candidate-level salary features (job/company attributes).
    model_features = candidates_df[
        [
            "size_code",
            "sector_code",
            "state_code",
            "ownership_code",
            "seniority_code",
            "title_rich_code",
        ]
    ].copy()

    # If the salary model was trained with categorical handling, keep the same dtype convention.
    cat_cols = [
        "size_code",
        "sector_code",
        "state_code",
        "ownership_code",
        "seniority_code",
        "title_rich_code",
    ]
    model_features[cat_cols] = model_features[cat_cols].astype("category")

    # Broadcast user skill PCs across rows
    user_pcs = profile["derived"]["skill_pcs"]

    assert len(user_pcs) == 1, f"Expected user_pcs to be 1 row, got {len(user_pcs)}"
    assert "dummy" not in model_features.columns
    assert "dummy" not in user_pcs.columns

    mf = model_features.assign(dummy=1)
    up = user_pcs.assign(dummy=1)
    user_salary_model_features = mf.merge(up, on="dummy", how="left").drop(
        columns="dummy"
    )

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
