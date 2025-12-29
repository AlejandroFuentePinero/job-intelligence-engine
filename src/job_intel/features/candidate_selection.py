# src/job_intel/features/candidate_selection.py

from typing import Optional, Union, List, Dict, Any, Tuple

import pandas as pd

from src.job_intel.schemas import build_user_profile


REQUIRED_FILTER_COLS = ["state", "Sector", "title_rich", "job_title_family"]


def _normalize_job_id_series(s: pd.Series, *, where: str) -> pd.Series:
    """
    Canonicalize job_id to string, strip whitespace, fix float artifacts like '1060.0'.
    """
    if s.isna().any():
        raise ValueError(f"{where} contains NaNs in 'job_id'.")

    out = s.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)

    if (out == "").any():
        raise ValueError(
            f"{where} contains empty-string job_id values after normalization."
        )

    return out


def candidate_set_construction(
    df: pd.DataFrame,
    skill_text: str = "",
    current_state: Optional[str] = "ALL",
    job_title_family: Optional[str] = None,
    job_title_rich: Optional[str] = None,
    target_sectors: Optional[Union[str, List[str]]] = None,
    salary_target: Optional[Union[int, float, str]] = None,
    explain_skills: bool = False,
    candidate_override_df: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    # --- base df contract ---
    missing = [c for c in REQUIRED_FILTER_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Candidate selection requires columns missing from df: {missing}"
        )

    if "job_id" not in df.columns:
        raise KeyError("Candidate selection requires 'job_id' column in df.")

    # Normalize df job_id to ensure reliable isin() behavior
    df = df.copy()
    df["job_id"] = _normalize_job_id_series(df["job_id"], where="df (jobs artefact)")

    if df["job_id"].duplicated().any():
        dup = df.loc[df["job_id"].duplicated(), "job_id"].iloc[0]
        raise ValueError(f"df contains duplicate job_id values (e.g. {dup!r}).")

    # Build profile always (skill_text changes between upskilling runs)
    profile = build_user_profile(
        skill_text=skill_text,
        current_state=current_state,
        job_title_family=job_title_family,
        job_title_rich=job_title_rich,
        target_sectors=target_sectors,
        salary_target=salary_target,
        explain_skills=explain_skills,
    )

    # --- OVERRIDE MODE: freeze candidates by job_id set ---
    if candidate_override_df is not None:
        if "job_id" not in candidate_override_df.columns:
            raise KeyError("candidate_override_df must contain a 'job_id' column.")

        # Dropna BEFORE converting; then normalize
        ov = candidate_override_df["job_id"].dropna()
        if ov.empty:
            raise ValueError("candidate_override_df contains no valid job_id values.")

        ov = _normalize_job_id_series(ov, where="candidate_override_df")
        override_ids = ov.unique().tolist()
        override_set = set(override_ids)

        out = df[df["job_id"].isin(override_set)].copy()

        if out.empty:
            raise ValueError(
                "Override candidate set produced 0 rows after subsetting df by job_id. "
                "Likely job_id mismatch between override and artefacts."
            )

        # Strict freeze invariant: all requested ids must be present
        present = set(out["job_id"].tolist())
        missing_override = override_set - present
        if missing_override:
            ex = sorted(list(missing_override))[:10]
            raise KeyError(
                f"Override candidate set is missing {len(missing_override)} requested job_ids "
                f"after alignment to df. Examples: {ex}"
            )

        # Optional strictness: ensure no duplicates after filtering (should already hold)
        if out["job_id"].duplicated().any():
            dup = out.loc[out["job_id"].duplicated(), "job_id"].iloc[0]
            raise ValueError(
                f"Override output contains duplicate job_id values (e.g. {dup!r})."
            )

        if verbose:
            print(
                f"[override] requested ids: {len(override_ids)} | returned rows: {len(out)}"
            )

        return profile, out

    # --- NORMAL MODE: apply constraints ---
    out = df

    st = profile["raw_inputs"].get("current_state", None)
    if st not in (None, "ALL"):
        out = out[out["state"] == st]

    sectors = profile["raw_inputs"].get("target_sectors", None)
    if sectors is not None:
        out = out[out["Sector"].isin(sectors)]

    tr = profile["raw_inputs"].get("job_title_rich", None)
    if tr is not None:
        out = out[out["title_rich"] == tr]

    tf = profile["raw_inputs"].get("job_title_family", None)
    if tf is not None:
        out = out[out["job_title_family"] == tf]

    if out.empty:
        raise ValueError(
            "No jobs available within the current constraints. Please widen your filters."
        )

    if verbose:
        print(f"Initial number of jobs available: {len(df)}.")
        print(f"Current number of jobs available: {len(out)}.")

    return profile, out.copy()
