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


def _normalize_job_id_series(s: pd.Series, *, where: str) -> pd.Series:
    """
    Canonicalize job_id for stable set comparisons:
      - drop NA (caller decides if NA is allowed)
      - pandas string dtype + strip whitespace
      - remove common float artifact suffix '.0'
      - reject empty strings
    """
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)

    if out.isna().any():
        raise ValueError(f"{where} contains NaNs in 'job_id' after normalization.")
    if (out == "").any():
        raise ValueError(
            f"{where} contains empty-string job_id values after normalization."
        )

    return out


def _coerce_job_ids(x: pd.Series, *, where: str) -> List[str]:
    """Return normalized job_id list for robust comparisons."""
    x = x.dropna()
    if x.empty:
        return []
    return _normalize_job_id_series(x, where=where).tolist()


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
    run_sensitivity: bool = True,
    candidate_override_df: Optional[pd.DataFrame] = None,
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
        candidate_override_df=candidate_override_df,
    )

    # --- override integrity check setup (critical for frozen-universe counterfactuals) ---
    override_ids: Optional[set[str]] = None
    if candidate_override_df is not None:
        if not isinstance(candidate_override_df, pd.DataFrame):
            raise TypeError(
                f"candidate_override_df must be a DataFrame, got {type(candidate_override_df)}"
            )
        if "job_id" not in candidate_override_df.columns:
            raise KeyError("candidate_override_df must contain a 'job_id' column.")
        if "job_id" not in candidates_df.columns:
            raise KeyError("candidates_df must contain a 'job_id' column.")

        override_list = _coerce_job_ids(
            candidate_override_df["job_id"], where="candidate_override_df"
        )
        if len(override_list) == 0:
            raise ValueError("candidate_override_df contains no valid job_id values.")

        # Guardrail: do NOT allow truncation when universe is frozen
        if return_top_n_jobs is not None:
            raise ValueError(
                "return_top_n_jobs must be None when candidate_override_df is provided "
                "(frozen-universe mode). Truncation would break scenario comparability."
            )

        override_ids = set(override_list)

        returned_ids = set(
            _coerce_job_ids(
                candidates_df["job_id"], where="candidates_df (post-selection)"
            )
        )

        missing = override_ids - returned_ids
        extra = returned_ids - override_ids

        if missing or extra:
            msg = (
                "Override candidate universe mismatch detected (post-selection).\n"
                f"- override_ids: {len(override_ids)}\n"
                f"- returned_ids: {len(returned_ids)}\n"
                f"- missing_from_returned: {len(missing)} (example: {list(missing)[:5]})\n"
                f"- extra_in_returned: {len(extra)} (example: {list(extra)[:5]})\n"
            )
            raise ValueError(msg)

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

    # NOTE: truncation is disabled in override mode (guard above).
    if return_top_n_jobs is not None:
        candidates_df = candidates_df.head(return_top_n_jobs)

    # --- final override integrity check (post-ranking/truncation) ---
    if override_ids is not None:
        final_ids = set(
            _coerce_job_ids(candidates_df["job_id"], where="candidates_df (final)")
        )
        missing = override_ids - final_ids
        extra = final_ids - override_ids
        if missing or extra:
            msg = (
                "Override candidate universe mismatch detected (final output).\n"
                f"- override_ids: {len(override_ids)}\n"
                f"- final_ids: {len(final_ids)}\n"
                f"- missing_from_final: {len(missing)} (example: {list(missing)[:5]})\n"
                f"- extra_in_final: {len(extra)} (example: {list(extra)[:5]})\n"
            )
            raise ValueError(msg)

    # --- skill gap analysis ---
    gap_df = compute_skill_gaps(
        profile,
        candidates_df,
        skill_prob_matrix,
        top_k=top_k_gaps,
    ).sort_values("skill_gap", ascending=False)

    return profile, candidates_df, gap_df, sensitivity_out
