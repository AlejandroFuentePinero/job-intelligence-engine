# src/job_intel/pipelines/chapter2_hidden_structures.py

"""
Chapter 2 — Hidden Structures (Pipeline Orchestrator)

This pipeline stitches together Chapter 2’s core artefacts:

1) Build job–skill bipartite graph from Chapter 1 skill probabilities
2) Train Node2Vec on the weighted graph and extract embeddings
3) Cluster job embeddings into job families (KMeans)
4) Convert skill embeddings into an undirected similarity edge list (top-k per skill)

Design notes
------------
- This orchestrator is intentionally lightweight: it wires together existing features/models.
- Saving is OFF by default to avoid overwriting artefacts during development.
- When save_* flags are enabled, you should pass explicit filenames/paths to make
  versioning intentional (e.g., v01, v02).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.job_intel.features.graph_job_skill import build_job_skill_bipartite_graph
from src.job_intel.models.node2vec_trainer import (
    train_node2vec_model,
    extract_embeddings,
)
from src.job_intel.features.job_families_clustering import (
    build_job_families_from_embeddings,
)
from src.job_intel.features.skill_embedding_similarity import (
    build_skill_similarity_edges_from_embeddings,
)
from src.job_intel.config import PROCESSED_DATA_DIR


@dataclass(frozen=True)
class Node2VecParams:
    """Parameters container passed into train_node2vec_model()."""

    dimensions: int = 64
    walk_length: int = 20
    num_walks: int = 40
    workers: int = 4
    weight_key: str = "weight"
    seed: Optional[int] = 42  # set None for fully stochastic runs

    # gensim/Word2Vec params
    window: int = 10
    min_count: int = 1
    batch_words: int = 4


def run_chapter2_hidden_structures(
    # Graph
    jobs_df: Optional[pd.DataFrame] = None,
    threshold: float = 0.5,
    save_graph_pickle: bool = False,
    # Node2Vec
    node2vec_params: Node2VecParams = Node2VecParams(),
    # Clustering
    k_job_families: int = 20,
    # Skill similarity
    k_skill_neighbours: int = 5,
    atol: float = 1e-12,
    rtol: float = 1e-12,
    # Saving (OFF by default)
    save_outputs: bool = False,
    job_families_filename: str = "job_families_graph_embeddings.csv",
    skill_edges_filename: str = "skill_similarity_edges_k5_embeddings.csv",
    # Misc
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Chapter 2 end-to-end and return the core outputs.

    Parameters
    ----------
    jobs_df : pd.DataFrame | None
        Optional jobs dataframe from Chapter 1. If provided, must contain `job_id`
        either as a column or index. If None, the graph builder will load from disk.
    threshold : float
        Probability threshold for adding job–skill edges to the bipartite graph.
    save_graph_pickle : bool
        If True, persist the bipartite graph (graph builder handles path/versioning).
    node2vec_params : Node2VecParams
        Hyperparameters for Node2Vec training.
    k_job_families : int
        Number of KMeans clusters for job-family clustering.
    k_skill_neighbours : int
        Top-k neighbours retained per skill when building the skill similarity edge list.
    atol, rtol : float
        Tolerances used in similarity-matrix symmetry checks.
    save_outputs : bool
        If True, saves job families + skill edges to PROCESSED_DATA_DIR using filenames below.
        (Files are not overwritten automatically here; caller should version filenames.)
    job_families_filename : str
        Output CSV name for job families (when save_outputs=True).
    skill_edges_filename : str
        Output CSV name for skill similarity edges (when save_outputs=True).
    verbose : bool
        Print progress messages.

    Returns
    -------
    km_jobs_df : pd.DataFrame
        Columns: ["job_id", "job_family_id"].
    undirected_edges : pd.DataFrame
        Columns: ["skill_1", "skill_2", "similarity"] (undirected, deduplicated).
    """
    # ------------------------------------------------------------------
    # 1) Build job–skill bipartite graph
    # ------------------------------------------------------------------
    if verbose:
        print("=== Chapter 2: Build job–skill bipartite graph ===")

    G, prob_mat, used_threshold, df_used = build_job_skill_bipartite_graph(
        df=jobs_df,
        threshold=threshold,
        save_graph_pickle=save_graph_pickle,
    )

    if verbose:
        print(
            f"Graph ready. Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}, "
            f"threshold={used_threshold}"
        )

    # ------------------------------------------------------------------
    # 2) Train Node2Vec + extract embeddings
    # ------------------------------------------------------------------
    if verbose:
        print("=== Chapter 2: Train Node2Vec + extract embeddings ===")

    model = train_node2vec_model(G=G, params=node2vec_params)

    job_emb, skill_emb = extract_embeddings(model, G=G)

    if verbose:
        print(
            f"Embeddings extracted: job_emb={job_emb.shape}, skill_emb={skill_emb.shape}"
        )

    # ------------------------------------------------------------------
    # 3) Job family clustering (KMeans)
    # ------------------------------------------------------------------
    if verbose:
        print("=== Chapter 2: Cluster job embeddings into job families ===")

    km_jobs_df = build_job_families_from_embeddings(
        embedding=job_emb,
        embedding_tag="v01",  # unused when embedding is provided; kept for logging consistency
        k=k_job_families,
        norm="l2",
        random_state=42,
        n_init="auto",
        max_iter=1000,
        save_output=False,  # orchestrator controls saving below
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 4) Skill similarity edges (top-k per skill, undirected + deduped)
    # ------------------------------------------------------------------
    if verbose:
        print(
            "=== Chapter 2: Build skill similarity edge list from skill embeddings ==="
        )

    undirected_edges = build_skill_similarity_edges_from_embeddings(
        skill_emb=skill_emb,
        k=k_skill_neighbours,
        norm="l2",
        atol=atol,
        rtol=rtol,
        save_output=False,  # orchestrator controls saving below
        output_path=None,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 5) Optional save (OFF by default to avoid overwrites)
    # ------------------------------------------------------------------
    if save_outputs:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        job_out = PROCESSED_DATA_DIR / job_families_filename
        skill_out = PROCESSED_DATA_DIR / skill_edges_filename

        # Fail-fast to avoid accidental overwrite during development
        if job_out.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {job_out}")
        if skill_out.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {skill_out}")

        km_jobs_df.to_csv(job_out, index=False)
        undirected_edges.to_csv(skill_out, index=False)

        if verbose:
            print(f"Saved job families: {job_out}")
            print(f"Saved skill similarity edges: {skill_out}")

    return km_jobs_df, undirected_edges
