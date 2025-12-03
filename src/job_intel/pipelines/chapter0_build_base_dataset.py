# src/job_intel/pipelines/chapter0_build_base_dataset.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from job_intel.config import (
    RAW_DA_JOBS_FILE,
    RAW_DS_JOBS_FILE,
    CH0_PROCESSED_JOBS_FILE,
    CH0_DOMAIN_LOOKUP_FILE,
)
from job_intel.features.titles import add_title_features
from job_intel.features.salary import add_salary_features
from job_intel.features.text_cleaning import add_description_features
from job_intel.features.domain import add_domain_from_lookup
from job_intel.features.skill_extractor import extract_domain_level_flags

DROP_COLUMNS = [
    "Job Title",  # Only need the clean title to check for family and domain.
    "job_title_for_skills",  # Already incorporated in title_plus_description if we want to check for skills
    "job_title_raw",  # Intermediate strip of Job Title, not needed for checks
    "Salary Estimate",  # Raw text; we trust the parsed salary features
    "seniority_roman",  # Partial extraction from Roman numerals only
    "seniority_title",  # Partial extraction from title only
    "seniority_description",  # Partial extraction from description only
    "state_hq",  # Noisy; we keep only the role location
]


def load_raw_jobs() -> pd.DataFrame:
    """Load raw Data Analyst and Data Scientist job CSVs and add a role_source column."""
    da = pd.read_csv(RAW_DA_JOBS_FILE)
    ds = pd.read_csv(RAW_DS_JOBS_FILE)

    da = da.copy()
    ds = ds.copy()

    # ----
    # EXTRA ADDED: Ensure columns align

    ds = ds.drop(["Unnamed: 0", "index"], axis=1)
    da = da.drop(["Unnamed: 0"], axis=1)

    # ----

    da["role_source"] = "data_analyst"
    ds["role_source"] = "data_scientist"

    df = pd.concat([ds, da], axis=0, ignore_index=True)

    # ----
    # EXTRA ADDED: Input true NAs
    # Remove NAs

    df = df.replace(-1, np.nan)
    df = df.replace("-1", np.nan)
    # ----

    return df


def minor_feature_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Add state and state_hq columns, leaving original Location/Headquarters intact."""
    df = df.copy()

    if "Location" in df.columns:
        df["state"] = df["Location"].str.extract(r"(?<=,\s)([^,]+)$")
    else:
        df["state"] = None

    if "Headquarters" in df.columns:
        df["state_hq"] = df["Headquarters"].str.extract(r"(?<=,\s)([^,]+)$")
        df["state_hq"] = df["state_hq"].str.replace("061", "NY")
    else:
        df["state_hq"] = None

    # Pull international listings together (same logic as notebook)
    df["state"] = df["state"].apply(
        lambda x: "international" if isinstance(x, str) and len(x) > 2 else x
    )
    df["state_hq"] = df["state_hq"].apply(
        lambda x: "international" if isinstance(x, str) and len(x) > 2 else x
    )

    # ----
    # EXTRA ADDED: OWNERSHIP, Year, and dropping unneccesary features
    # Simplify ownership

    df["ownership_clean"] = df["Type of ownership"].str.lower()

    df["ownership_clean"] = (
        df["ownership_clean"]
        .str.replace("company - private", "private")
        .str.replace("private practice / firm", "private")
        .str.replace("self-employed", "private")
        .str.replace("company - public", "public")
        .str.replace("subsidiary or business segment", "public")
        .str.replace("franchise", "public")
        .str.replace("nonprofit organization", "nonprofit")
        .str.replace("college / university", "nonprofit")
        .str.replace("school / school district", "nonprofit")
        .str.replace("hospital", "nonprofit")
        .str.replace("government", "government")
        # everything else becomes unknown
        .apply(
            lambda x: (
                x
                if x in ["private", "public", "nonprofit", "government", "unknown"]
                else "unknown"
            )
        )
    )

    # Year founded
    df["Founded"] = pd.to_numeric(df["Founded"], errors="coerce").astype("Int64")

    # Drop Easy Apply and Competitors due to the very large proportion of missing data
    df = df.drop(
        [
            "Easy Apply",
            "Competitors",
            "Company Name",
            "Revenue",
            "Type of ownership",
            "Location",
            "Headquarters",
        ],
        axis=1,
    )
    # ----

    return df


def add_skill_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the existing skill_extractor to create skill flag columns.

    Assumes:
        - df has 'job_title_raw' and 'job_description_clean'.
        - job_intel.features.skill_extractor.extract_domain_level_flags(text)
          returns a dict of {skill_name: 0/1 or bool}.
    """
    df = df.copy()

    missing_cols = [
        c for c in ["job_title_raw", "job_description_clean"] if c not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"add_skill_features: missing required columns: {missing_cols}"
        )

    # 1) Build a *skills-friendly* title:
    #    - start from RAW title
    #    - lowercase
    #    - normalise separators and basic punctuation
    #    - DO NOT strip skills/noise_terms or seniority words
    df["job_title_for_skills"] = (
        df["job_title_raw"]
        .fillna("")
        .str.lower()
        # basic separators → space
        .str.replace(r"[-/|\\,]", " ", regex=True)
        .str.replace(r"[()\[\]{}]", " ", regex=True)
        # other punctuation → space (keep word chars, whitespace, + . # for things like c++ / .net)
        .str.replace(r"[^\w\s+.#]", " ", regex=True)
    )

    # 2) Combine this with the cleaned description
    df["title_plus_description"] = (
        df["job_title_for_skills"].astype(str)
        + " "
        + df["job_description_clean"].astype(str)
    )

    # 3) Extract skill flags
    df["skill_flags"] = df["title_plus_description"].apply(extract_domain_level_flags)
    features_df = pd.DataFrame(df["skill_flags"].tolist())

    # Avoid collisions if any skill flag has same name as an existing column
    for col in list(features_df.columns):
        if col in df.columns:
            features_df = features_df.rename(columns={col: f"{col}_skill"})

    df = pd.concat([df.drop(columns=["skill_flags"]), features_df], axis=1)

    return df


def build_chapter0_base_dataset(
    save: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """
    Full Chapter 0 pipeline:

    1. Load raw CSVs and stack them.
    2. Add location features.
    3. Add title / family / seniority features.
    4. Add cleaned description.
    5. Add domain from the pre-computed lookup (if present).
    6. Add salary features.
    7. Add skill features.
    8. Save to CH0_PROCESSED_JOBS_FILE (parquet or csv, based on extension).

    Returns the final DataFrame.
    """
    if verbose:
        print("Loading raw jobs ...")
    df = load_raw_jobs()
    if verbose:
        print(f"Raw combined shape: {df.shape}")
        print("Replaced -1 with NAs")

    # Location
    df = minor_feature_cleaning(df)
    if verbose:
        print("Added location features.")
        print("Simplified ownership.")
        print("Converted year to integer.")
        print(
            "Dropped uneccesary features (Nas and extremely complex) - Easy Apply, Competitors, Company Name and Revenue."
        )

    # Clean job description first (needed for correct seniority-from-description)
    desc_col = "Job Description" if "Job Description" in df.columns else None
    if desc_col is not None:
        df = add_description_features(
            df, raw_col=desc_col, out_col="job_description_clean"
        )
        if verbose:
            print("Added job_description_clean.")

    # Titles + seniority (now using the CLEAN description)
    title_col = "Job Title" if "Job Title" in df.columns else df.columns[0]

    df = add_title_features(
        df,
        title_col=title_col,
        description_col="job_description_clean",  # <-- THIS IS THE KEY FIX
    )
    if verbose:
        print("Added title/seniority/family features (using cleaned description).")

    # Domain from lookup (optional)
    df = add_domain_from_lookup(df, lookup_path=CH0_DOMAIN_LOOKUP_FILE)
    if verbose:
        if "domain" in df.columns:
            n_missing = df["domain"].isna().sum()
            print(f"Added domain from lookup (missing={n_missing}).")
        else:
            print("Domain lookup not applied (no 'domain' column created).")

    # Salary
    if "Salary Estimate" in df.columns:
        df = add_salary_features(
            df,
            salary_col="Salary Estimate",
            prefix="sal",
            drop_intermediate=True,
        )
        if verbose:
            print("Added salary features from 'Salary Estimate'.")

    # Skills
    df = add_skill_features(df)
    if verbose:
        print("Added skill flag features.")

    # -------------------------------------------------
    # NEW BLOCK: NA handling (rows + Industry/Sector/Size)
    # -------------------------------------------------
    # 1) Drop rows with no description AND no salary
    if "job_description_clean" in df.columns and "sal_mean" in df.columns:
        mask_drop = df["job_description_clean"].isna() & df["sal_mean"].isna()
        n_drop = int(mask_drop.sum())
        if n_drop > 0 and verbose:
            print(f"Dropping {n_drop} rows with no description AND no salary.")
        df = df[~mask_drop].copy()

    # 2) Fill missing Industry / Sector /Size with 'Unknown'
    if "Industry" in df.columns:
        df["Industry"] = df["Industry"].fillna("Unknown")
    if "Sector" in df.columns:
        df["Sector"] = df["Sector"].fillna("Unknown")
    if "Size" in df.columns:
        df["Size"] = df["Size"].fillna("Unknown")
    # -------------------------------------------------

    # Final column pruning
    drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    if drop_cols and verbose:
        print(f"Dropping columns: {drop_cols}")
    df = df.drop(columns=drop_cols)

    # Save (CSV only)
    if save:
        out_path = Path(CH0_PROCESSED_JOBS_FILE)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Force .csv extension
        if out_path.suffix.lower() != ".csv":
            out_path = out_path.with_suffix(".csv")

        df.to_csv(out_path, index=False)

        if verbose:
            print(
                f"Saved Chapter 0 processed dataset to: {out_path} (shape={df.shape})"
            )

    return df
