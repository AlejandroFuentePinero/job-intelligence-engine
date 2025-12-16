# src/job_intel/features/skill_embedding_similarity.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from src.job_intel.config import PROCESSED_DATA_DIR


def build_skill_similarity_edges_from_embeddings(
    skill_emb: pd.DataFrame,
    *,
    k: int = 5,
    norm: str = "l2",
    atol: float = 1e-12,
    rtol: float = 1e-12,
    save_output: bool = False,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build an undirected skill–skill similarity edge list from skill embeddings.

    Overview
    --------
    1) L2-normalise embeddings (enables cosine similarity via dot product)
    2) Compute full cosine similarity matrix (dense)
    3) Sparsify by keeping top-k neighbours per skill (directed edges)
    4) Canonicalise (skill_1, skill_2) and deduplicate into an undirected edge list

    Parameters
    ----------
    skill_emb : pd.DataFrame
        Skill embedding matrix indexed by skill_id (rows) with numeric embedding dims (cols).
    k : int
        Top-k neighbours retained per skill (default 5).
    norm : str
        Normalisation norm passed to sklearn.preprocessing.normalize (default "l2").
    atol, rtol : float
        Tolerances for symmetry check on the similarity matrix (np.allclose).
    save_output : bool
        If True, save the undirected edge list CSV.
    output_path : Path or None
        Output path for CSV. If None, defaults to:
        PROCESSED_DATA_DIR / "skill_similarity_edges_k5_embeddings.csv"
    verbose : bool
        If True, print progress and lightweight diagnostics.

    Returns
    -------
    pd.DataFrame
        Undirected edge list with columns: ["skill_1", "skill_2", "similarity"].
        Each row is a canonical skill pair (skill_1 < skill_2 lexicographically).
    """
    # ------------------------
    # Basic validation
    # ------------------------
    if not isinstance(skill_emb, pd.DataFrame):
        raise TypeError("skill_emb must be a pandas DataFrame.")

    if skill_emb.empty:
        raise ValueError("skill_emb is empty.")

    if skill_emb.index.has_duplicates:
        raise ValueError(
            "skill_emb index contains duplicates; expected unique skill_id index."
        )

    if k < 1:
        raise ValueError(f"k must be >= 1. Got k={k}.")

    if k >= len(skill_emb):
        raise ValueError(
            f"k must be < number of skills. Got k={k} with n_skills={len(skill_emb)}."
        )

    if norm not in {"l2", "l1", "max"}:
        raise ValueError(f"Unsupported norm='{norm}'. Use one of: 'l2', 'l1', 'max'.")

    if output_path is None:
        output_path = PROCESSED_DATA_DIR / f"skill_similarity_edges_k{k}_embeddings.csv"

    # Ensure numeric content (fail fast if not)
    if not np.all([np.issubdtype(dt, np.number) for dt in skill_emb.dtypes]):
        raise TypeError(
            "skill_emb must contain only numeric columns (embedding dimensions)."
        )

    # ------------------------
    # 1) Normalise embeddings
    # ------------------------
    if verbose:
        print("Normalising skill embeddings...")

    # normalize expects an array-like; we keep index separately for later
    skill_emb_norm = normalize(skill_emb.values, norm=norm)

    # Normalisation can create NaNs if there are zero vectors
    if np.isnan(skill_emb_norm).any():
        raise ValueError(
            "Normalised skill embeddings contain NaNs. This usually indicates at least one "
            "all-zero embedding vector."
        )

    if verbose:
        na_raw = int(skill_emb.isna().sum().sum())
        na_norm = int(np.isnan(skill_emb_norm).sum())
        shape_ok = skill_emb.shape == skill_emb_norm.shape
        print("Sanity checks for normalised embeddings...")
        print(f"  NAs in raw skill embeddings: {na_raw}")
        print(f"  NAs in normalised embeddings: {na_norm}")
        print(f"  Shape maintained? {shape_ok}")
        if norm == "l2":
            norms = np.sqrt((skill_emb_norm[:10] ** 2).sum(axis=1))
            print(f"  Unit norm check (first 10 rows): {norms}")

    # ------------------------
    # 2) Similarity matrix (cosine via dot product)
    # ------------------------
    if verbose:
        print("Computing skill similarity matrix...")

    # With L2 norm, cosine_similarity(u, v) == dot(u, v)
    mat_dot = skill_emb_norm @ skill_emb_norm.T

    # Wrap in DataFrame for convenient per-row top-k selection
    similarity_matrix = pd.DataFrame(
        mat_dot, index=skill_emb.index, columns=skill_emb.index
    )

    if verbose:
        print("Sanity checks for the similarity matrix...")
        print(f"  Similarity matrix shape: {similarity_matrix.shape}")

        is_symmetric = np.allclose(mat_dot, mat_dot.T, atol=atol, rtol=rtol)
        print(f"  Symmetry check (allclose) = {is_symmetric}")

        diag = np.diagonal(mat_dot)
        print(f"  Average diagonal value: {diag.mean()}")
        print(f"  Min diagonal value: {diag.min()}")
        print(f"  Max diagonal value: {diag.max()}")

    # ------------------------
    # 3) Sparsify: top-k neighbours per skill
    # ------------------------
    if verbose:
        print(f"Sparsifying to top-{k} neighbours per skill...")

    # Exclude self similarity from being selected as a neighbour
    np.fill_diagonal(similarity_matrix.values, -np.inf)

    # Build directed edge list (one row per source→target)
    res: list[dict] = []
    skills = similarity_matrix.index
    n = len(skills)

    for i in range(n):
        # Take the top-k most similar skills for this source skill
        top_k = similarity_matrix.iloc[i, :].nlargest(k)

        source = skills[i]
        for j in range(k):
            res.append(
                {
                    "skill_source": source,
                    "skill_target": top_k.index[j],
                    "similarity": float(top_k.iloc[j]),
                }
            )

    top_k_edges = pd.DataFrame(res)

    if verbose:
        n_self = int((top_k_edges["skill_source"] == top_k_edges["skill_target"]).sum())
        print("Directed neighbour edge list checks...")
        print(
            f"  Rows (expected n_skills*k): {top_k_edges.shape[0]} (expected {n * k})"
        )
        print(f"  Self-edges present? {n_self} (expected 0)")
        print("  Preview (first 10):")
        print(top_k_edges.head(10))

    # ------------------------
    # 4) Deduplicate to undirected edges
    # ------------------------
    if verbose:
        print("Deduplicating to an undirected edge list...")

    # Canonical pair representation ensures (A,B) == (B,A)
    top_k_edges["skill_1"] = top_k_edges[["skill_source", "skill_target"]].min(axis=1)
    top_k_edges["skill_2"] = top_k_edges[["skill_source", "skill_target"]].max(axis=1)

    undirected_edges = (
        top_k_edges.groupby(["skill_1", "skill_2"], as_index=False)
        .agg(similarity=("similarity", "max"))
        .sort_values("similarity", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        # Undirected self-edges should be impossible given diagonal masking
        n_self_undir = int(
            (undirected_edges["skill_1"] == undirected_edges["skill_2"]).sum()
        )
        print("Undirected edge list checks...")
        print(f"  Rows: {undirected_edges.shape[0]}")
        print(f"  Self-edges present? {n_self_undir} (expected 0)")
        print("  Preview (top 10 by similarity):")
        print(undirected_edges.head(10))

    # ------------------------
    # Optional save
    # ------------------------
    if save_output:
        undirected_edges.to_csv(output_path, index=False)
        if verbose:
            print(f"Output saved to: {output_path}")

    return undirected_edges
