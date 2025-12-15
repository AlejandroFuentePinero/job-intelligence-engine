# src/job_intel/models/sbert_clustering_training_title.py

"""
Train SBERT embeddings + KMeans clusters over unique cleaned job titles
and export a deterministic title → domain lookup table for Chapter 0.

This module is intentionally "offline/training-only":
- It is NOT used at runtime by the Chapter 0 pipeline.
- The pipeline should consume the saved lookup artefact (CSV) deterministically.

Outputs
-------
- CH0_DOMAIN_LOOKUP_FILE: CSV with columns ["job_title_base", "domain"]

Notes
-----
- This script encodes unique titles with a SentenceTransformer model.
- Clustering (KMeans) is used to group semantically similar titles.
- Domains are assigned via a manual cluster→domain mapping (curated once, then reused).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.cluster import KMeans

# Optional diagnostics imports (only needed if sbert_eval=True)
import matplotlib.pyplot as plt
import umap

from src.job_intel.config import INTERIM_DATA_DIR, CH0_DOMAIN_LOOKUP_FILE


@dataclass(frozen=True)
class SBERTTitleClusterParams:
    """Configuration for title embedding + clustering."""

    sbert_model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize_embeddings: bool = True
    n_clusters: int = 40
    random_state: int = 42
    n_init: str = "auto"  # sklearn >= 1.4 supports "auto"


def load_titles(
    titles_path: Path = INTERIM_DATA_DIR / "02_job_titles_cleaned.csv",
    title_col: str = "job_title_base",
) -> List[str]:
    """
    Load the cleaned titles file and return a de-duplicated list of unique titles.
    """
    df = pd.read_csv(titles_path)
    if title_col not in df.columns:
        raise ValueError(f"Expected column '{title_col}' not found in {titles_path}.")
    titles = (
        df[title_col]
        .astype(str)
        .str.strip()
        .replace({"": np.nan})
        .dropna()
        .unique()
        .tolist()
    )
    return titles


def embed_titles(
    titles: List[str],
    params: SBERTTitleClusterParams,
) -> np.ndarray:
    """
    Compute SBERT embeddings for a list of titles.

    Returns
    -------
    embeddings : np.ndarray of shape (n_titles, embedding_dim)
    """
    model = SentenceTransformer(params.sbert_model_name)
    embeddings = model.encode(
        titles,
        batch_size=params.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=params.normalize_embeddings,
    )
    return embeddings


def kmeans_cluster_titles(
    embeddings: np.ndarray,
    params: SBERTTitleClusterParams,
) -> np.ndarray:
    """
    Cluster title embeddings with KMeans and return cluster labels.
    """
    km = KMeans(
        n_clusters=params.n_clusters,
        random_state=params.random_state,
        n_init=params.n_init,
    )
    labels = km.fit_predict(embeddings)
    return labels


def nearest_titles(
    query: str,
    titles: List[str],
    embeddings: np.ndarray,
    k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Lightweight semantic sanity check:
    return the top-k nearest titles to 'query' in embedding space.

    Notes
    -----
    - Requires query to be present in titles.
    - Uses cosine similarity (embeddings are usually normalized).
    """
    if query not in titles:
        raise ValueError(
            f"Query title '{query}' not found in title list. "
            "Pick a query that exists in df['job_title_base']."
        )

    idx = titles.index(query)
    query_emb = embeddings[idx]
    sims = cos_sim(query_emb, embeddings)[0].cpu().numpy()

    # sort descending; skip itself at index idx
    top_idx = sims.argsort()[::-1]
    top_idx = [i for i in top_idx if i != idx][:k]

    return [(titles[i], float(sims[i])) for i in top_idx]


def umap_plot_embeddings(
    embeddings: np.ndarray,
    title: str = "SBERT Title Embeddings (2D UMAP)",
    random_state: int = 42,
) -> None:
    """
    Optional diagnostic: 2D UMAP projection of embeddings.
    """
    reducer = umap.UMAP(random_state=random_state)
    emb_2d = reducer.fit_transform(embeddings)

    plt.figure()
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=5)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def build_domain_lookup(
    titles: List[str],
    labels: np.ndarray,
    cluster_domains: Dict[int, str],
    out_path: Path = CH0_DOMAIN_LOOKUP_FILE,
) -> pd.DataFrame:
    """
    Build and save title→domain lookup table from title clusters and a manual mapping.
    """
    df = pd.DataFrame({"job_title_base": titles, "cluster": labels}).sort_values(
        "cluster"
    )

    # Map cluster id → domain label
    df["domain"] = df["cluster"].map(cluster_domains)

    # Guardrail: ensure mapping is complete (no NaNs)
    if df["domain"].isna().any():
        missing = sorted(df.loc[df["domain"].isna(), "cluster"].unique().tolist())
        raise ValueError(
            f"cluster_domains mapping is incomplete. Missing clusters: {missing}"
        )

    # Persist only the deterministic lookup used by Chapter 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lookup = df[["job_title_base", "domain"]].drop_duplicates()
    lookup.to_csv(out_path, index=False)

    return lookup


def train_title_sbert_kmeans_domain_lookup(
    sbert_eval: bool = False,
    title_eval: str = "data engineer",
    params: Optional[SBERTTitleClusterParams] = None,
    cluster_domains: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    End-to-end trainer: titles → embeddings → clusters → domain lookup CSV.

    Returns
    -------
    lookup_df : pd.DataFrame
        DataFrame with columns ["job_title_base", "domain"]
    """
    params = params or SBERTTitleClusterParams()

    # 1) Load unique titles
    titles = load_titles()

    # 2) Embed with SBERT
    embeddings = embed_titles(titles, params)

    # 3) Optional diagnostics
    if sbert_eval:
        # Nearest neighbour check (prints nothing by default; return list if needed)
        nn = nearest_titles(title_eval, titles, embeddings, k=10)
        print(f"Nearest titles to '{title_eval}':")
        for t, s in nn:
            print(f"  {t:40s}  sim={s:.3f}")

        # Embedding norms (useful if normalize_embeddings=False)
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"Embedding norms: mean={norms.mean():.3f}, std={norms.std():.3f}")

        # 2D UMAP plot
        umap_plot_embeddings(embeddings, random_state=params.random_state)

    # 4) Cluster
    labels = kmeans_cluster_titles(embeddings, params)

    # 5) Build/save lookup
    if cluster_domains is None:
        # Keep your curated mapping local so the trainer is self-contained.
        cluster_domains = {
            0: "health",
            1: "business",
            2: "general_data",
            3: "business",
            4: "health",
            5: "business",
            6: "research",
            7: "business",
            8: "ML_AI",
            9: "ML_AI",
            10: "health",
            11: "general_data",
            12: "business",
            13: "business",
            14: "general_data",
            15: "general_data",
            16: "general_data",
            17: "health",
            18: "research",
            19: "business",
            20: "research",
            21: "health",
            22: "health",
            23: "general_data",
            24: "business",
            25: "general_data",
            26: "research",
            27: "business",
            28: "health",
            29: "business",
            30: "sport",
            31: "research",
            32: "research",
            33: "security",
            34: "health",
            35: "ML_AI",
            36: "research",
            37: "research",
            38: "business",
            39: "health",
        }

    lookup_df = build_domain_lookup(
        titles=titles,
        labels=labels,
        cluster_domains=cluster_domains,
        out_path=CH0_DOMAIN_LOOKUP_FILE,
    )

    print(f"Saved domain lookup to: {CH0_DOMAIN_LOOKUP_FILE}")
    print(f"Lookup rows: {len(lookup_df)}")

    return lookup_df


if __name__ == "__main__":
    # Typical usage: run as a script to regenerate the lookup artefact.
    train_title_sbert_kmeans_domain_lookup(sbert_eval=False)
