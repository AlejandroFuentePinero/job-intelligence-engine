# src/job_intel/features/candidate_selection.py

from typing import Optional, Union, List, Dict, Any, Tuple

import pandas as pd

from src.job_intel.schemas import build_user_profile


REQUIRED_FILTER_COLS = ["state", "Sector", "title_rich", "job_title_family"]


def candidate_set_construction(
    df: pd.DataFrame,
    skill_text: str = "",
    current_state: Optional[str] = "ALL",
    job_title_family: Optional[str] = None,
    job_title_rich: Optional[str] = None,
    target_sectors: Optional[Union[str, List[str]]] = None,
    salary_target: Optional[Union[int, float, str]] = None,
    explain_skills: bool = False,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Construct a candidate job set by applying hard filters derived from a user profile.

    This function:
      1) builds a validated UserProfile (schemas.py)
      2) filters the provided jobs dataframe by state / sector / title constraints
      3) returns (profile, candidates_df)

    Notes
    -----
    This function performs *no scoring*. Suitability, gaps, and other analyses are
    applied downstream to the returned candidates dataframe.
    """

    missing = [c for c in REQUIRED_FILTER_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Candidate selection requires columns missing from df: {missing}"
        )

    profile = build_user_profile(
        skill_text=skill_text,
        current_state=current_state,
        job_title_family=job_title_family,
        job_title_rich=job_title_rich,
        target_sectors=target_sectors,
        salary_target=salary_target,
        explain_skills=explain_skills,
    )

    out = df

    # 1) state
    if profile["raw_inputs"]["current_state"] is not None:
        out = out[out["state"] == profile["raw_inputs"]["current_state"]]

    # 2) sectors
    if profile["raw_inputs"]["target_sectors"] is not None:
        out = out[out["Sector"].isin(profile["raw_inputs"]["target_sectors"])]

    # 3) title_rich
    if profile["raw_inputs"]["job_title_rich"] is not None:
        out = out[out["title_rich"] == profile["raw_inputs"]["job_title_rich"]]

    # 4) title_family
    if profile["raw_inputs"]["job_title_family"] is not None:
        out = out[out["job_title_family"] == profile["raw_inputs"]["job_title_family"]]

    if verbose:
        print(f"Initial number of jobs available: {len(df)}.")
        print(f"Current number of jobs available: {len(out)}.")

    if out.empty:
        raise ValueError(
            "No jobs available within the current constraints. Please widen your filters."
        )

    return profile, out.copy()
