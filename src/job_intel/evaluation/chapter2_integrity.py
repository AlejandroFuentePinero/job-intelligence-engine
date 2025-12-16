# src/job_intel/evaluation/chapter2_integrity.py
"""
Chapter 2 — Integrity Checks

Purpose
-------
Lightweight, deterministic integrity checks for Chapter 2 artefacts:
- Job–skill bipartite graph (built from Chapter 1 outputs)
- Node2Vec embeddings (jobs + skills)
- Job family clustering output
- Skill similarity edge list (k-NN undirected pairs)
- Skill specialisation tables (lift matrices)

Design principles
-----------------
- Fast: checks invariants (shapes, ID alignment, numeric ranges) without heavy computation.
- Non-destructive: never overwrites outputs; reads artefacts or accepts in-memory objects.
- Explicit failure: raises AssertionError / ValueError with actionable messages.

Usage
-----
1) After running the Chapter 2 pipeline:
    km_jobs_df, undirected_edges = run_chapter2_hidden_structures(...)
    report = run_chapter2_integrity_checks(
        df_ch1=df_ch1,
        km_jobs_df=km_jobs_df,
        undirected_edges=undirected_edges,
        job_emb=job_emb,
        skill_emb=skill_emb,
        prob_mat=prob_mat,
        threshold=0.5,
        k_skill=5,
        embed_dim_expected=64,
        seed_expected=42,
    )
    print(report)

2) Or purely from saved files (if you prefer later):
   - pass file paths into the loader helpers you already have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _is_numeric_df(df: pd.DataFrame) -> bool:
    return all(np.issubdtype(t, np.number) for t in df.dtypes)


def _check_range(df: pd.DataFrame, lo: float, hi: float, name: str) -> None:
    _assert(
        _is_numeric_df(df), f"{name} must be numeric. Got dtypes={df.dtypes.to_dict()}"
    )
    vmin = float(np.nanmin(df.to_numpy()))
    vmax = float(np.nanmax(df.to_numpy()))
    _assert(vmin >= lo - 1e-12, f"{name} min={vmin} < {lo}")
    _assert(vmax <= hi + 1e-12, f"{name} max={vmax} > {hi}")


def _check_unique_index(df: pd.DataFrame, idx_name: str) -> None:
    _assert(
        df.index.name == idx_name,
        f"Expected index name '{idx_name}', got '{df.index.name}'",
    )
    _assert(df.index.is_unique, f"Index '{idx_name}' must be unique")


def _check_no_missing(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    miss = df.loc[:, cols].isna().sum()
    bad = miss[miss > 0]
    _assert(bad.empty, f"{name}: missing values detected in columns: {bad.to_dict()}")


def _check_expected_columns(
    df: pd.DataFrame, required: Sequence[str], name: str
) -> None:
    missing = [c for c in required if c not in df.columns]
    _assert(len(missing) == 0, f"{name}: missing required columns: {missing}")


# ---------------------------------------------------------------------
# Report object
# ---------------------------------------------------------------------
@dataclass
class Chapter2IntegrityReport:
    ok: bool
    checks: Dict[str, str]

    def __str__(self) -> str:
        lines = ["Chapter 2 Integrity Report", "-" * 28]
        for k, v in self.checks.items():
            lines.append(f"{k}: {v}")
        lines.append("-" * 28)
        lines.append("OK" if self.ok else "FAILED")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Core checks
# ---------------------------------------------------------------------
def check_ch1_dataframe_contract(df_ch1: pd.DataFrame) -> None:
    """
    Chapter 2 assumes df_ch1 contains a stable 'job_id' identifier.
    We accept either:
      A) index is 'job_id'
      B) 'job_id' column exists and is unique

    This function enforces that *at least one* holds, and normalises to index.
    """
    if df_ch1.index.name == "job_id":
        _assert(df_ch1.index.is_unique, "df_ch1 job_id index must be unique")
    else:
        _assert(
            "job_id" in df_ch1.columns,
            "df_ch1 must contain a 'job_id' column if index is not job_id",
        )
        _assert(df_ch1["job_id"].is_unique, "df_ch1['job_id'] must be unique")
        _assert(
            df_ch1["job_id"].notna().all(),
            "df_ch1['job_id'] must not contain NA values",
        )


def normalise_job_id_index(df_ch1: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with job_id set as index (without mutating original)."""
    if df_ch1.index.name == "job_id":
        out = df_ch1.copy()
        return out
    out = df_ch1.copy()
    out = out.set_index("job_id", drop=False)
    out.index.name = "job_id"
    return out


def check_probability_matrix(
    prob_mat: pd.DataFrame, df_ch1: pd.DataFrame, n_skills_expected: int = 27
) -> None:
    _check_unique_index(prob_mat, "job_id")
    _check_unique_index(df_ch1, "job_id")

    _assert(
        prob_mat.index.equals(df_ch1.index),
        "prob_mat index must match df_ch1 index exactly (same job_id order)",
    )

    _assert(
        prob_mat.shape[1] == n_skills_expected,
        f"Expected prob_mat to have {n_skills_expected} skill columns, got {prob_mat.shape[1]}",
    )
    _check_range(prob_mat, 0.0, 1.0, "prob_mat")


def check_embeddings(
    job_emb: pd.DataFrame,
    skill_emb: pd.DataFrame,
    df_ch1: pd.DataFrame,
    prob_mat: pd.DataFrame,
    embed_dim_expected: int = 64,
) -> None:
    # job embeddings
    _assert(isinstance(job_emb, pd.DataFrame), "job_emb must be a pandas DataFrame")
    _assert(job_emb.index.is_unique, "job_emb index must be unique")
    _assert(
        job_emb.shape[1] == embed_dim_expected,
        f"job_emb dim expected {embed_dim_expected}, got {job_emb.shape[1]}",
    )
    _assert(
        job_emb.index.equals(df_ch1.index),
        "job_emb index must match df_ch1 job_id index exactly",
    )
    _assert(np.isfinite(job_emb.to_numpy()).all(), "job_emb contains NaN/inf")

    # skill embeddings
    _assert(isinstance(skill_emb, pd.DataFrame), "skill_emb must be a pandas DataFrame")
    _assert(skill_emb.index.is_unique, "skill_emb index must be unique")
    _assert(
        skill_emb.shape[1] == embed_dim_expected,
        f"skill_emb dim expected {embed_dim_expected}, got {skill_emb.shape[1]}",
    )
    _assert(
        list(skill_emb.index) == list(prob_mat.columns),
        "skill_emb index must match prob_mat skill columns (same names and order)",
    )
    _assert(np.isfinite(skill_emb.to_numpy()).all(), "skill_emb contains NaN/inf")


def check_job_families(
    km_jobs_df: pd.DataFrame, df_ch1: pd.DataFrame, k_expected: int = 20
) -> None:
    _check_expected_columns(km_jobs_df, ["job_id", "job_family_id"], "km_jobs_df")

    _assert(km_jobs_df["job_id"].is_unique, "km_jobs_df job_id must be unique")
    _assert(km_jobs_df["job_id"].notna().all(), "km_jobs_df job_id must not contain NA")
    _assert(
        km_jobs_df["job_family_id"].notna().all(),
        "km_jobs_df job_family_id must not contain NA",
    )

    # Coverage: must assign every job
    _assert(
        set(km_jobs_df["job_id"]) == set(df_ch1.index),
        "km_jobs_df must cover exactly the same job_ids as df_ch1",
    )

    # Basic cluster sanity
    n_clusters = int(km_jobs_df["job_family_id"].nunique())
    _assert(
        n_clusters == k_expected, f"Expected {k_expected} clusters, got {n_clusters}"
    )
    _assert(
        km_jobs_df["job_family_id"].min() >= 0, "job_family_id must be non-negative"
    )


def check_skill_similarity_edges(
    undirected_edges: pd.DataFrame,
    prob_mat: pd.DataFrame,
    k_skill: int = 5,
) -> None:
    """
    Expects undirected edge list produced by your similarity builder:
      columns: ['skill_1', 'skill_2', 'similarity']
    where skill_1 < skill_2 lexicographically due to min/max pairing.
    """
    _check_expected_columns(
        undirected_edges, ["skill_1", "skill_2", "similarity"], "undirected_edges"
    )

    # Skill names must be valid
    skills = set(prob_mat.columns)
    _assert(
        undirected_edges["skill_1"].isin(skills).all(),
        "undirected_edges contains unknown skill_1 values",
    )
    _assert(
        undirected_edges["skill_2"].isin(skills).all(),
        "undirected_edges contains unknown skill_2 values",
    )

    # No self edges
    _assert(
        (undirected_edges["skill_1"] != undirected_edges["skill_2"]).all(),
        "Self-edges detected in undirected_edges",
    )

    # Similarity range: cosine similarity in [-1, 1] after normalisation; typically >=0 in your case but allow [-1,1]
    sim = undirected_edges["similarity"].to_numpy()
    _assert(np.isfinite(sim).all(), "undirected_edges similarity contains NaN/inf")
    _assert(sim.min() >= -1.0 - 1e-12, f"similarity min out of range: {sim.min()}")
    _assert(sim.max() <= 1.0 + 1e-12, f"similarity max out of range: {sim.max()}")

    # Duplicate undirected pairs should not exist (after grouping)
    pairs = undirected_edges[["skill_1", "skill_2"]].astype(str).agg("|".join, axis=1)
    _assert(pairs.is_unique, "Duplicate undirected (skill_1, skill_2) pairs found")

    # Soft coverage check: with k-NN per node, unique undirected edges should be roughly <= (n_skills*k)/2
    n_skills = len(skills)
    upper = (n_skills * k_skill) / 2.0
    _assert(
        len(undirected_edges) <= int(np.ceil(upper)) + 10,  # tiny slack
        f"Too many undirected edges for k={k_skill}: got {len(undirected_edges)}, expected <= ~{upper}",
    )


def check_specialisation_lift_table(
    lift_df: pd.DataFrame,
    expected_skill_cols: Sequence[str],
    group_name: str,
) -> None:
    """
    Lift tables are (group × skills) matrices of mean(prob) - global_mean(prob).
    Invariants:
    - columns match expected skills (same names, any order acceptable)
    - numeric, finite
    - mean across groups is not constrained; no strict sum rule.
    """
    _assert(_is_numeric_df(lift_df), f"{group_name} lift table must be numeric")
    _assert(
        np.isfinite(lift_df.to_numpy()).all(),
        f"{group_name} lift table contains NaN/inf",
    )

    missing = [c for c in expected_skill_cols if c not in lift_df.columns]
    _assert(len(missing) == 0, f"{group_name} lift table missing skill cols: {missing}")


# ---------------------------------------------------------------------
# One entry point for Chapter 2 checks
# ---------------------------------------------------------------------
def run_chapter2_integrity_checks(
    df_ch1: pd.DataFrame,
    prob_mat: Optional[pd.DataFrame] = None,
    job_emb: Optional[pd.DataFrame] = None,
    skill_emb: Optional[pd.DataFrame] = None,
    km_jobs_df: Optional[pd.DataFrame] = None,
    undirected_edges: Optional[pd.DataFrame] = None,
    specialisation_tables: Optional[Dict[str, pd.DataFrame]] = None,
    threshold: float = 0.5,
    k_job_families: int = 20,
    k_skill: int = 5,
    embed_dim_expected: int = 64,
) -> Chapter2IntegrityReport:
    checks: Dict[str, str] = {}
    ok = True

    try:
        check_ch1_dataframe_contract(df_ch1)
        df_ch1_idx = normalise_job_id_index(df_ch1)
        checks["df_ch1_contract"] = "PASS"

        if prob_mat is not None:
            _check_unique_index(df_ch1_idx, "job_id")
            check_probability_matrix(
                prob_mat=prob_mat, df_ch1=df_ch1_idx, n_skills_expected=27
            )
            checks["prob_mat"] = "PASS"
        else:
            checks["prob_mat"] = "SKIP (not provided)"

        if (job_emb is not None) and (skill_emb is not None) and (prob_mat is not None):
            check_embeddings(
                job_emb=job_emb,
                skill_emb=skill_emb,
                df_ch1=df_ch1_idx,
                prob_mat=prob_mat,
                embed_dim_expected=embed_dim_expected,
            )
            checks["embeddings"] = "PASS"
        else:
            checks["embeddings"] = "SKIP (requires job_emb, skill_emb, prob_mat)"

        if km_jobs_df is not None:
            check_job_families(
                km_jobs_df=km_jobs_df, df_ch1=df_ch1_idx, k_expected=k_job_families
            )
            checks["job_families"] = "PASS"
        else:
            checks["job_families"] = "SKIP (not provided)"

        if (undirected_edges is not None) and (prob_mat is not None):
            check_skill_similarity_edges(
                undirected_edges=undirected_edges, prob_mat=prob_mat, k_skill=k_skill
            )
            checks["skill_similarity_edges"] = "PASS"
        else:
            checks["skill_similarity_edges"] = (
                "SKIP (requires undirected_edges + prob_mat)"
            )

        if (specialisation_tables is not None) and (prob_mat is not None):
            expected_cols = list(prob_mat.columns)
            for name, df in specialisation_tables.items():
                check_specialisation_lift_table(
                    df, expected_skill_cols=expected_cols, group_name=name
                )
            checks["specialisation_tables"] = (
                f"PASS ({len(specialisation_tables)} tables)"
            )
        elif specialisation_tables is not None:
            checks["specialisation_tables"] = (
                "SKIP (prob_mat required for expected columns)"
            )
        else:
            checks["specialisation_tables"] = "SKIP (not provided)"

        # Threshold sanity
        _assert(
            0.0 <= float(threshold) <= 1.0,
            f"threshold must be in [0,1], got {threshold}",
        )
        checks["threshold"] = "PASS"

    except Exception as e:
        ok = False
        checks["error"] = f"{type(e).__name__}: {e}"

    return Chapter2IntegrityReport(ok=ok, checks=checks)
