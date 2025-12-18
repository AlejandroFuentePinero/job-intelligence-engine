# src/job_intel/positioning.py

from typing import Optional, Union, List, Dict, Any, Tuple

import pandas as pd

from src.job_intel.features.artefacts_ch3 import load_ch3_artefacts
from src.job_intel.features.candidate_selection import candidate_set_construction
from src.job_intel.features.candidate_skill_gap import compute_skill_gaps
from src.job_intel.features.candidate_suitability import (
    add_skill_match,
    add_salary_score,
    add_suitability,
)
from src.job_intel.features.candidate_competitiveness import add_competitiveness
from src.job_intel.features.competitiveness_sensitivity import (
    compute_competitiveness_sensitivity,
)
from src.job_intel.features.suitability_sensitivity import (
    compute_suitability_sensitivity,
)


def run_positioning(
    skill_text: str = "",
    current_state: Optional[str] = "ALL",
    job_title_family: Optional[str] = None,
    job_title_rich: Optional[str] = None,
    target_sectors: Optional[Union[str, List[str]]] = None,
    salary_target: Optional[Union[int, float, str]] = None,
    explain_skills: bool = False,
    w_skill: float = 0.7,
    w_salary: float = 0.3,
    top_k_gaps: int = 10,
    return_top_n_jobs: Optional[int] = 10,
    run_sensitivity: bool = False,
) -> Tuple[
    Dict[str, Any], pd.DataFrame, pd.DataFrame, Optional[Dict[str, pd.DataFrame]]
]:
    """
    Chapter 3 public API.

    Returns:
      - profile: validated UserProfile dict
      - candidates_df: ranked jobs with suitability + competitiveness scores
      - gap_df: user-level skill gap diagnostics
      - sensitivity_out: dict of sensitivity tables (or None)
    """

    # --- load artefacts ---
    jobs_df, skill_prob_matrix = load_ch3_artefacts()

    # --- candidate selection + profile ---
    profile, candidates_df = candidate_set_construction(
        df=jobs_df,
        skill_text=skill_text,
        current_state=current_state,
        job_title_family=job_title_family,
        job_title_rich=job_title_rich,
        target_sectors=target_sectors,
        salary_target=salary_target,
        explain_skills=explain_skills,
    )

    # --- suitability scoring ---
    candidates_df = add_skill_match(profile, candidates_df)
    candidates_df = add_salary_score(profile, candidates_df)
    candidates_df = add_suitability(profile, candidates_df, w_skill, w_salary)

    # --- competitiveness ---
    candidates_df = add_competitiveness(
        profile, candidates_df, skill_prob_matrix, use_rarity=True
    )

    # --- sensitivity analyses (optional) ---
    sensitivity_out: Optional[Dict[str, pd.DataFrame]] = None
    if run_sensitivity:
        suit_sens_df = compute_suitability_sensitivity(candidates_df)
        comp_sens_df = compute_competitiveness_sensitivity(candidates_df)

        sensitivity_out = {
            "suitability": suit_sens_df,
            "competitiveness": comp_sens_df,
        }

    # --- final ranking (primary sort = suitability) ---
    candidates_df = candidates_df.sort_values("suitability", ascending=False)

    if return_top_n_jobs is not None:
        candidates_df = candidates_df.head(return_top_n_jobs)

    # --- skill gap analysis ---
    gap_df = compute_skill_gaps(
        profile,
        candidates_df,
        skill_prob_matrix,
        top_k=top_k_gaps,
    ).sort_values("skill_gap", ascending=False)

    return profile, candidates_df, gap_df, sensitivity_out
