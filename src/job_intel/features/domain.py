# src/job_intel/features/domain.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def add_domain_from_lookup(
    df: pd.DataFrame,
    *,
    lookup_path: Optional[Path] = None,
    title_key_col: str = "job_title_base",
    domain_col: str = "domain",
) -> pd.DataFrame:
    """
    Attach a 'domain' column to df based on a pre-computed lookup table.

    The lookup file is expected to have at least:
        - title_key_col (default 'job_title_base')
        - domain_col (default 'domain')

    If lookup_path is None or does not exist, the function will:
        - leave df unchanged if domain_col already exists, or
        - create df[domain_col] filled with NaN otherwise.

    The lookup file is created in the file 01_title_normalisation notebook.
        1. Retrieved unique titles
        2. Fit a sentence transformer (SBERT - pre-trained model : "all-MiniLM-L6-v2")
        3. Derived title embeddings
        4. SBERT model valiation
        5. HDBSCAN clustering - Failed, embeddings are a continuum
        6. KMean clustering - Succeded.
            - Extracted 40 clusters
            - Labelled them based on domain
        7. Save lookup file in interim/domain_lookup_ch0.csv

    """
    df = df.copy()

    if lookup_path is None or not Path(lookup_path).exists():
        # Ensure the column exists, but don't fail hard
        if domain_col not in df.columns:
            df[domain_col] = pd.NA
        return df

    lookup = pd.read_csv(lookup_path)

    if title_key_col not in lookup.columns or domain_col not in lookup.columns:
        raise ValueError(
            f"Domain lookup file must contain columns '{title_key_col}' and '{domain_col}'. "
            f"Found columns: {list(lookup.columns)}"
        )

    # Drop duplicates in lookup to avoid exploding the join
    lookup_small = lookup[[title_key_col, domain_col]].drop_duplicates()

    df = df.merge(
        lookup_small,
        on=title_key_col,
        how="left",
        suffixes=("", "_domain_lookup"),
    )

    # If df already had a 'domain' column, prefer the one from lookup when not missing
    if domain_col + "_domain_lookup" in df.columns:
        df[domain_col] = df[domain_col].fillna(df[domain_col + "_domain_lookup"])
        df = df.drop(columns=[domain_col + "_domain_lookup"])

    return df
