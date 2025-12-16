# src/job_intel/features/job_families_clustering.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from src.job_intel.config import PROCESSED_DATA_DIR
from src.job_intel.features.embedding_loader import load_node2vec_embeddings


def build_job_families_from_embeddings(
    embedding: Optional[pd.DataFrame] = None,
    embedding_tag: str = "v01",
    k: int = 20,
    norm: str = "l2",
    random_state: int = 42,
    n_init: str | int = "auto",
    max_iter: int = 1000,
    save_output: bool = False,
    output_filename: str = "job_families_graph_embeddings.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build job-family assignments from Node2Vec job embeddings using KMeans.

    Purpose
    -------
    Produces a stable mapping `job_id -> job_family_id` for Chapter 2 and
    downstream aggregation modules (e.g., skill distributions by job family).

    The function supports two modes:
    - Injection mode: use a provided embedding DataFrame directly
    - Load mode: load embeddings from disk using `embedding_tag`

    Steps
    -----
    1) Obtain job embeddings (either injected or loaded from disk)
    2) L2-normalise embeddings so Euclidean distance reflects directional similarity
    3) Fit KMeans and assign each job_id to a job_family_id
    4) Optionally save the mapping to PROCESSED_DATA_DIR

    Parameters
    ----------
    embedding : pd.DataFrame or None
        Optional job embedding matrix indexed by job_id.
        If provided, `embedding_tag` is ignored.
    embedding_tag : str
        Tag/version for embeddings to load if `embedding` is None (e.g., "v01").
    k : int
        Number of clusters (job families).
    norm : str
        Normalisation norm passed to sklearn.preprocessing.normalize
        (default "l2").
    random_state : int
        Reproducibility seed for KMeans.
    n_init : str | int
        KMeans n_init parameter ("auto" recommended for modern scikit-learn).
    max_iter : int
        Maximum iterations for KMeans optimisation.
    save_output : bool
        If True, saves the output CSV into PROCESSED_DATA_DIR/output_filename.
    output_filename : str
        Output filename if save_output is True.
    verbose : bool
        If True, prints basic progress and sanity checks.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ["job_id", "job_family_id"].
    """

    # ------------------------
    # Basic parameter checks
    # ------------------------
    if k < 2:
        raise ValueError(f"k must be >= 2. Got k={k}.")

    if norm not in {"l2", "l1", "max"}:
        raise ValueError(f"Unsupported norm='{norm}'. Use one of: 'l2', 'l1', 'max'.")

    if embedding is None:
        if not isinstance(embedding_tag, str) or not embedding_tag.strip():
            raise ValueError(
                "embedding_tag must be a non-empty string when embedding is None."
            )

    # ------------------------
    # 1) Obtain job embeddings
    # ------------------------
    if embedding is not None:
        # Injection mode: use provided embeddings directly
        if not isinstance(embedding, pd.DataFrame):
            raise TypeError("`embedding` must be a pandas DataFrame.")

        job_emb = embedding.copy()

        if job_emb.index.has_duplicates:
            raise ValueError("Injected embedding index contains duplicate job_ids.")

    else:
        # Load mode: load embeddings from disk
        job_emb, _ = load_node2vec_embeddings(tag=embedding_tag)

        if not isinstance(job_emb, pd.DataFrame):
            raise TypeError(
                "load_node2vec_embeddings() must return job_emb as a pandas DataFrame."
            )

        if job_emb.index.has_duplicates:
            raise ValueError(
                "Loaded job_emb index contains duplicates; expected unique job_id index."
            )

    if verbose:
        src = "injected" if embedding is not None else f"loaded (tag='{embedding_tag}')"
        print(f"Job embeddings {src}. Shape={job_emb.shape}")

    # ------------------------
    # 2) Normalise embeddings
    # ------------------------
    # L2-normalise so Euclidean distance reflects directional similarity
    job_emb_norm = normalize(job_emb.values, norm=norm)

    # Fail fast if normalisation produced NaNs (usually zero vectors)
    if np.isnan(job_emb_norm).any():
        raise ValueError(
            "Normalised embeddings contain NaNs. This usually indicates zero vectors "
            "or non-numeric values in the embeddings."
        )

    if verbose:
        na_raw = int(job_emb.isna().sum().sum())
        na_norm = int(np.isnan(job_emb_norm).sum())
        shape_ok = job_emb.shape == job_emb_norm.shape

        print(f"Embeddings normalised using '{norm}' normalisation.")
        print("Checking data and normalisation...")
        print(f"  NAs in raw job embeddings: {na_raw}")
        print(f"  NAs in normalised embeddings: {na_norm}")
        print(f"  Shape maintained? {shape_ok}")

        if norm == "l2":
            norms = np.sqrt((job_emb_norm[:10] ** 2).sum(axis=1))
            print(f"  Unit norm check (first 10 rows): {norms}")

    # ------------------------
    # 3) Fit KMeans
    # ------------------------
    km_model_jobs = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )

    labels_km_jobs = km_model_jobs.fit_predict(job_emb_norm)

    km_jobs_df = pd.DataFrame(
        {
            "job_id": job_emb.index,
            "job_family_id": labels_km_jobs,
        }
    )

    if verbose:
        print(f"KMeans fitted. k={k}, random_state={random_state}.")
        print(
            f"Output rows={len(km_jobs_df)} "
            f"(should equal n_jobs={job_emb.shape[0]})."
        )

    # ------------------------
    # 4) Optional save
    # ------------------------
    if save_output:
        output_path: Path = PROCESSED_DATA_DIR / output_filename
        km_jobs_df.to_csv(output_path, index=False)
        if verbose:
            print(f"Job families saved to: {output_path}")

    return km_jobs_df
