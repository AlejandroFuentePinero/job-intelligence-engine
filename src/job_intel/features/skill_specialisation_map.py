# src/job_intel/features/skill_specialisation_map.py

"""
Skill Specialisation Maps (Lift vs Global Mean)

This module computes *skill specialisation* for a chosen grouping variable
(e.g., job_family_id, Sector, job_title_family, ownership_clean, Size, state).

Core idea
---------
1) Compute global mean skill probability across all jobs.
2) Compute group mean skill probability within each category.
3) Lift = (group mean) - (global mean)

Interpretation
--------------
- Positive lift: skill is over-represented in that group (specialised).
- Negative lift: skill is under-represented in that group.
- Near zero: skill is broadly generic in the dataset.

Outputs
-------
- A lift table: (groups × skills) of float values.
- Optional plots: global skill bar chart + lift heatmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.job_intel.config import PROJECT_ROOT, CH2_PROCESSED_DF, PROCESSED_DATA_DIR

# Default figure output location for narrative docs
OUT_DIR = PROJECT_ROOT / "docs/narrative/figures/ch2_specialisation_map"


def compute_skill_specialisation_lift(
    base_df: Optional[pd.DataFrame] = None,
    skill_prob_df: Optional[pd.DataFrame] = None,
    job_families: Optional[pd.DataFrame] = None,
    group_col: str = "job_family_id",
    *,
    # I/O controls
    save_data: bool = False,
    output_path: Optional[Path] = None,
    save_plots: bool = False,
    show_plots: bool = False,
    plots_dir: Path = OUT_DIR,
    # Plot controls
    heatmap_figsize: Tuple[int, int] = (14, 6),
    global_bar_figsize: Tuple[int, int] = (10, 8),
    cmap: str = "vlag",
    # Validation / behaviour
    expected_n_skills: Optional[int] = 27,
    min_group_n: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute a skill specialisation map (lift table) for a given grouping variable.

    Parameters
    ----------
    base_df:
        The Chapter-2 modelling dataframe containing `job_id` (column or index),
        plus the grouping column (e.g. "Sector", "Size", "ownership_clean",
        "job_title_family", "job_family_id", etc).
        If None, loads CH2_PROCESSED_DF from disk.
    skill_prob_df:
        Job × skill probability matrix. Must contain `job_id` (column or index).
        If None, tries to load from PROCESSED_DATA_DIR / "skill_prob_matrix.csv".
    job_families:
        Optional mapping of `job_id -> job_family_id`. Only required if
        group_col == "job_family_id" and that column is not already present in base_df.
        If None and needed, loads from PROCESSED_DATA_DIR / "job_families_graph_embeddings.csv".
    group_col:
        Column in the merged base table to group by.

    save_data:
        If True, saves the lift table as a CSV.
    output_path:
        Optional explicit path for the lift CSV. If None, uses:
        PROCESSED_DATA_DIR / f"{group_col}_skill_specialisation.csv"
    save_plots / show_plots:
        Controls plot saving and display.
    plots_dir:
        Directory for plots if save_plots=True.

    expected_n_skills:
        If set, validates that the probability matrix contains this many skill columns.
    min_group_n:
        If set, filters out groups with fewer than this many jobs (stability guardrail).
    verbose:
        Print basic progress and sanity checks.

    Returns
    -------
    pd.DataFrame
        Lift table indexed by group value, with one column per skill probability.
    """
    # ------------------------------------------------------------
    # 0) Load inputs (if missing)
    # ------------------------------------------------------------
    if base_df is None:
        if verbose:
            print(f"Loading base_df from disk: {CH2_PROCESSED_DF}")
        base_df = pd.read_csv(CH2_PROCESSED_DF)
    else:
        base_df = base_df.copy()

    if skill_prob_df is None:
        default_skill_path = PROCESSED_DATA_DIR / "skill_prob_matrix.csv"
        if verbose:
            print(f"Loading skill_prob_df from disk: {default_skill_path}")
        skill_prob_df = pd.read_csv(default_skill_path)
    else:
        skill_prob_df = skill_prob_df.copy()

    # ------------------------------------------------------------
    # 1) Standardise job_id handling (column-based merge contract)
    # ------------------------------------------------------------
    def _ensure_job_id_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
        df = df.copy()
        if "job_id" in df.columns:
            return df
        if df.index.name == "job_id":
            return df.reset_index()
        raise ValueError(
            f"{name} must contain 'job_id' as a column, or have index.name == 'job_id'."
        )

    base_df = _ensure_job_id_column(base_df, "base_df")
    skill_prob_df = _ensure_job_id_column(skill_prob_df, "skill_prob_df")

    # ------------------------------------------------------------
    # 2) If needed, attach job_family_id
    # ------------------------------------------------------------
    if group_col == "job_family_id" and "job_family_id" not in base_df.columns:
        if job_families is None:
            default_fam_path = PROCESSED_DATA_DIR / "job_families_graph_embeddings.csv"
            if verbose:
                print(f"Loading job_families from disk: {default_fam_path}")
            job_families = pd.read_csv(default_fam_path)
        else:
            job_families = job_families.copy()

        job_families = _ensure_job_id_column(job_families, "job_families")

        # Guardrail: must contain job_family_id
        if "job_family_id" not in job_families.columns:
            raise ValueError("job_families must contain column 'job_family_id'.")

        base_df = base_df.merge(
            job_families[["job_id", "job_family_id"]],
            on="job_id",
            how="inner",
        )

    # ------------------------------------------------------------
    # 3) Sanity checks
    # ------------------------------------------------------------
    if "job_id" not in base_df.columns or "job_id" not in skill_prob_df.columns:
        raise ValueError("Both base_df and skill_prob_df must contain 'job_id' column.")

    if not base_df["job_id"].is_unique:
        raise ValueError(
            "base_df has non-unique job_id values; expected 1 row per job."
        )

    if not skill_prob_df["job_id"].is_unique:
        raise ValueError(
            "skill_prob_df has non-unique job_id values; expected 1 row per job."
        )

    if group_col not in base_df.columns:
        raise ValueError(f"group_col='{group_col}' not found in base_df columns.")

    # Identify skill probability columns
    prob_cols = [c for c in skill_prob_df.columns if c != "job_id"]
    if expected_n_skills is not None and len(prob_cols) != expected_n_skills:
        raise ValueError(
            f"Expected {expected_n_skills} skill probability columns, found {len(prob_cols)}."
        )

    # Probability bounds check (allow tiny float drift)
    prob_values = skill_prob_df[prob_cols].to_numpy(dtype=float)
    if np.nanmin(prob_values) < -1e-9 or np.nanmax(prob_values) > 1 + 1e-9:
        raise ValueError(
            "skill_prob_df contains values outside [0, 1] (beyond tolerance)."
        )

    # ------------------------------------------------------------
    # 4) Build base table (join skills into base_df)
    # ------------------------------------------------------------
    base = base_df.merge(skill_prob_df, on="job_id", how="inner")

    if verbose:
        print("Base join complete.")
        print(f"  base rows: {len(base):,}")
        print(f"  columns: {base.shape[1]:,}")
        print(f"  unique job_id: {base['job_id'].is_unique}")
        print(f"  missing {group_col}: {int(base[group_col].isna().sum())}")

    # Optional: filter tiny groups for stability
    if min_group_n is not None:
        group_counts = base[group_col].value_counts(dropna=False)
        keep = group_counts[group_counts >= min_group_n].index
        before = len(base)
        base = base[base[group_col].isin(keep)].copy()
        if verbose:
            print(
                f"Filtered groups with n < {min_group_n}. Rows: {before:,} -> {len(base):,}."
            )

    # ------------------------------------------------------------
    # 5) Global skill probability mean
    # ------------------------------------------------------------
    if verbose:
        print("Computing global mean skill probabilities...")
    skill_global = base[prob_cols].mean(axis=0)  # Series indexed by prob_cols

    # Optional global bar plot
    if show_plots or save_plots:
        global_df = (
            skill_global.rename("global_mean_prob")
            .reset_index()
            .rename(columns={"index": "skill"})
            .sort_values("global_mean_prob", ascending=False)
        )

        plt.figure(figsize=global_bar_figsize)
        ax = sns.barplot(data=global_df, x="skill", y="global_mean_prob")
        ax.set_title("Global Skill Presence Probability")
        ax.set_xlabel("Skill")
        ax.set_ylabel("Global mean probability")
        plt.xticks(rotation=90)
        plt.tight_layout()

        if save_plots:
            plots_dir.mkdir(parents=True, exist_ok=True)
            out_path = plots_dir / "global_skill_probability.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"Saved: {out_path}")

        if not show_plots:
            plt.close()

    # ------------------------------------------------------------
    # 6) Group mean and lift table
    # ------------------------------------------------------------
    if verbose:
        print(f"Computing group means and lift for group_col='{group_col}'...")

    group_means = base.groupby(group_col, dropna=False)[prob_cols].mean()
    lift = group_means.subtract(skill_global, axis=1)

    # ------------------------------------------------------------
    # 7) Optional heatmap
    # ------------------------------------------------------------
    if show_plots or save_plots:
        vmax = float(np.nanmax(np.abs(lift.to_numpy())))
        vmin = -vmax

        plt.figure(figsize=heatmap_figsize)
        ax = sns.heatmap(
            lift,
            center=0,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Skill lift (group mean − global mean)"},
        )
        ax.set_title(f"Skill specialisation by {group_col} (lift vs global mean)")
        ax.set_xlabel("Skill probability column")
        ax.set_ylabel(group_col)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_plots:
            plots_dir.mkdir(parents=True, exist_ok=True)
            out_path = plots_dir / f"{group_col}_skill_probability_heatmap.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"Saved: {out_path}")

        if not show_plots:
            plt.close()

    # ------------------------------------------------------------
    # 8) Optional save lift table
    # ------------------------------------------------------------
    if save_data:
        if output_path is None:
            output_path = PROCESSED_DATA_DIR / f"{group_col}_skill_specialisation.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lift.to_csv(output_path)  # keep index: it encodes the group labels
        if verbose:
            print(f"Saved: {output_path}")

    return lift
