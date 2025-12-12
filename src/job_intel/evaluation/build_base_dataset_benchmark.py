# src/job_intel/evaluation/build_base_dataset_benchmark.py

"""
Benchmark validator for Chapter 0 processed dataset.

This module:
1. Regenerates the Chapter 0 dataset using the pipeline.
2. Loads the stored benchmark dataset.
3. Compares all critical columns (numeric + categorical + skills).
4. Reports mismatches, match ratios, and missing columns.

Designed to verify pipeline integrity after refactors or changes in features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"

BENCHMARK_PATH = INTERIM_DATA_DIR / "05_skills_extracted.csv"


# ---------------------------------------------------------------------
# Columns to benchmark
# ---------------------------------------------------------------------

COLUMNS_TO_CHECK = [
    "Rating",
    "Size",
    "Founded",
    "Industry",
    "Sector",
    "state",
    "ownership_clean",
    "job_description_clean",
    "job_title_base",
    "seniority_combined",
    "job_title_norm",
    "job_title_family",
    "domain",
    "sal_mean",
    "core_programming__basic",
    "core_programming__intermediate",
    "core_programming__advanced",
    "data_engineering_pipelines__basic",
    "data_engineering_pipelines__intermediate",
    "data_engineering_pipelines__advanced",
    "ml_ai__basic",
    "ml_ai__intermediate",
    "ml_ai__advanced",
    "analytics_stats__basic",
    "analytics_stats__intermediate",
    "analytics_stats__advanced",
    "bi_viz__basic",
    "bi_viz__intermediate",
    "bi_viz__advanced",
    "cloud__basic",
    "cloud__intermediate",
    "cloud__advanced",
    "db_storage__basic",
    "db_storage__intermediate",
    "db_storage__advanced",
    "productivity_workflow__basic",
    "productivity_workflow__intermediate",
    "productivity_workflow__advanced",
    "soft_skills__core",
    "soft_skills__leadership",
    "domain_specific__none",
]


# ---------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------


def benchmark_ch0_dataset(show_plots: bool = True) -> pd.DataFrame:
    """
    Run the Chapter 0 benchmark test.

    Steps:
    --------
    1. Build dataset using pipeline.
    2. Load benchmark dataset.
    3. Compare each column listed in COLUMNS_TO_CHECK.
    4. Compute match ratios + mismatch counts.
    5. (Optional) Plot summary of match ratios.

    Returns
    -------
    pd.DataFrame with:
        column, match_ratio, mismatches, status
    """

    # Lazy import to avoid circular dep on import
    from src.job_intel.pipelines.chapter0_build_base_dataset import (
        build_chapter0_base_dataset,
    )

    # -----------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------
    df_new = build_chapter0_base_dataset(save=False)
    df_bench = pd.read_csv(BENCHMARK_PATH)

    results = []

    # -----------------------------------------------------------------
    # Compare columns
    # -----------------------------------------------------------------
    for col in COLUMNS_TO_CHECK:

        if col not in df_new.columns:
            results.append(
                {
                    "column": col,
                    "match_ratio": 0.0,
                    "mismatches": None,
                    "status": "MISSING_NEW",
                }
            )
            continue

        if col not in df_bench.columns:
            results.append(
                {
                    "column": col,
                    "match_ratio": 0.0,
                    "mismatches": None,
                    "status": "MISSING_BENCH",
                }
            )
            continue

        if len(df_new) != len(df_bench):
            results.append(
                {
                    "column": col,
                    "match_ratio": None,
                    "mismatches": None,
                    "status": f"LENGTH_MISMATCH ({len(df_new)} vs {len(df_bench)})",
                }
            )
            continue

        s1 = df_new[col]
        s2 = df_bench[col]

        # Numeric columns → tolerant comparison
        if pd.api.types.is_numeric_dtype(s1):
            mask = np.isclose(s1, s2, equal_nan=True)
        else:
            mask = (s1 == s2) | (s1.isna() & s2.isna())

        match_ratio = mask.mean()
        mismatches = int((~mask).sum())

        status = "OK" if match_ratio == 1.0 else "FAIL"

        results.append(
            {
                "column": col,
                "match_ratio": float(match_ratio),
                "mismatches": mismatches,
                "status": status,
            }
        )

    results_df = pd.DataFrame(results)

    # -----------------------------------------------------------------
    # Optional diagnostic plot
    # -----------------------------------------------------------------
    if show_plots:
        _plot_match_summary(results_df)

    return results_df


# ---------------------------------------------------------------------
# Plotting utils
# ---------------------------------------------------------------------


def _plot_match_summary(df: pd.DataFrame) -> None:
    """
    Plot match ratios for all benchmarked columns.
    """
    ok_df = df[df["match_ratio"].notna()]

    plt.figure(figsize=(8, 12))
    sns.barplot(
        data=ok_df.sort_values("match_ratio"),
        x="match_ratio",
        y="column",
        hue="status",
        dodge=False,
        palette={"OK": "green", "FAIL": "red"},
    )
    plt.title("Column Match Ratio (New vs Benchmark)")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Convenience CLI-like method
# ---------------------------------------------------------------------


def run_benchmark(show_plots: bool = True):
    """
    Convenience function: prints a readable benchmark summary.
    """
    df = benchmark_ch0_dataset(show_plots=show_plots)

    print("\n=== BENCHMARK SUMMARY ===")
    print(df.to_string(index=False))

    return df
