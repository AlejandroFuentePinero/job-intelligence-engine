# src/job_intel/features/salary.py

from __future__ import annotations

import numpy as np
import pandas as pd


def add_salary_features(
    df: pd.DataFrame,
    *,
    salary_col: str = "Salary Estimate",
    prefix: str = "sal",
    drop_intermediate: bool = False,
) -> pd.DataFrame:
    """
    Add salary-related features derived from the raw salary text column.

    Based on the Chapter 0 notebook logic:
    - strips 'Glassdoor est.' / 'Employer est.' labels
    - removes '$', 'K', and spaces
    - detects hourly vs yearly via 'PerHour'
    - parses min/max from patterns like '80-100'
    - converts:
        - hourly:  min/max * 2080 (40h/week * 52 weeks)
        - yearly (in K): min/max * 1000
    - creates:
        - <prefix>_is_hourly
        - <prefix>_min
        - <prefix>_max
        - <prefix>_mean
    """
    df = df.copy()

    # 1) basic cleaning of the raw text
    clean_col = f"{prefix}_clean"

    df[clean_col] = (
        df[salary_col]
        .astype(str)
        .str.replace(" (Glassdoor est.)", "", regex=False)
        .str.replace("(Glassdoor est.)", "", regex=False)
        .str.replace("(Employer est.)", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("K", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    # 2) detect hourly vs yearly
    is_hourly_col = f"{prefix}_is_hourly"
    df[is_hourly_col] = df[clean_col].str.contains("PerHour", case=False, na=False)
    df[clean_col] = df[clean_col].str.replace("PerHour", "", regex=False)

    # 3) split into min / max raw
    min_raw_col = f"{prefix}_min_raw"
    max_raw_col = f"{prefix}_max_raw"

    df[[min_raw_col, max_raw_col]] = df[clean_col].str.split("-", expand=True)
    df[min_raw_col] = pd.to_numeric(df[min_raw_col], errors="coerce")
    df[max_raw_col] = pd.to_numeric(df[max_raw_col], errors="coerce")

    # 4) scale to annual dollars
    min_col = f"{prefix}_min"
    max_col = f"{prefix}_max"
    mean_col = f"{prefix}_mean"

    df[min_col] = np.where(
        df[is_hourly_col],
        df[min_raw_col] * 2080,  # hourly → annual
        df[min_raw_col] * 1000,  # K → dollars
    )

    df[max_col] = np.where(
        df[is_hourly_col],
        df[max_raw_col] * 2080,
        df[max_raw_col] * 1000,
    )

    df[mean_col] = (df[min_col] + df[max_col]) / 2

    if drop_intermediate:
        df = df.drop(columns=[clean_col, min_raw_col, max_raw_col])

    return df
