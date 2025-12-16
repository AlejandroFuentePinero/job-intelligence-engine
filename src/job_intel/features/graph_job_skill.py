# src/job_intel/features/graph_job_skill.py

"""
Job–Skill Bipartite Graph Builder

Constructs a bipartite graph linking jobs to skills using the skill probability
matrix produced in Chapter 1.

Nodes
-----
- Job nodes: one per job_id
- Skill nodes: one per skill group (columns of the probability matrix)

Edges
-----
- An edge exists if P(skill|required|job) >= threshold
- Edge weight = that probability (float in [0, 1])

This graph is a structural artefact used downstream in Chapter 2 (Node2Vec,
clustering, skill ecosystem analysis).
"""

from __future__ import annotations

from typing import Optional, Tuple

import pickle
import pandas as pd
import networkx as nx

from src.job_intel.models.skill_prob_matrix import build_skill_probability_matrix
from src.job_intel.config import CH2_PROCESSED_DF, MODELS_DIR


def build_job_skill_bipartite_graph(
    df: Optional[pd.DataFrame] = None,
    threshold: float = 0.5,
    save_graph_pickle: bool = False,
) -> Tuple[nx.Graph, pd.DataFrame, float, pd.DataFrame]:
    """
    Build a job–skill bipartite graph from the Chapter 1 skill probability matrix.

    Parameters
    ----------
    df : pd.DataFrame | None
        Source modelling dataframe. Must contain `job_id` either as a column or
        already set as the index. If None, loads from CH2_PROCESSED_DF.
    threshold : float
        Minimum probability required to include a job–skill edge. Must be in [0, 1].
    save_graph_pickle : bool
        If True, persist the graph to MODELS_DIR.

    Returns
    -------
    G : networkx.Graph
        Bipartite graph with job and skill nodes and weighted edges.
    prob_mat : pd.DataFrame
        Skill probability matrix (jobs × skills).
    threshold : float
        Threshold used to create edges.
    df : pd.DataFrame
        Source dataframe indexed by job_id (enforced here).
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"threshold must be in [0, 1]. Got threshold={threshold}")

    # ------------------------------------------------------------------
    # Load processed modelling data and enforce identifier contract
    # ------------------------------------------------------------------
    if df is not None:
        df = df.copy()
    else:
        # Expect job_id to be a column in the saved artefact
        df = pd.read_csv(CH2_PROCESSED_DF)

    # If job_id is a column, promote it to the index (Chapter 2 identifier boundary)
    if "job_id" in df.columns:
        if df["job_id"].isna().any():
            raise ValueError("job_id column contains NaNs.")
        if df["job_id"].duplicated().any():
            raise ValueError("job_id column contains duplicates.")
        df = df.set_index("job_id")

    # If job_id is not a column, it must already be the index
    if df.index.name != "job_id":
        raise ValueError(
            "DataFrame must have job_id as index or as a column named 'job_id'. "
            f"Got index name={df.index.name!r}."
        )
    if not df.index.is_unique:
        raise ValueError("job_id index must be unique.")

    # ------------------------------------------------------------------
    # Build skill probability matrix
    # ------------------------------------------------------------------
    print("✅ Building the job probability matrix.")
    prob_mat = build_skill_probability_matrix(jobs_df=df)
    print("✅ Job probability matrix built.")

    # Ensure alignment between dataframe and probability matrix
    if not prob_mat.index.equals(df.index):
        raise ValueError("Index mismatch between dataframe and probability matrix.")

    # ------------------------------------------------------------------
    # Create empty bipartite graph
    # ------------------------------------------------------------------
    print("✅ Creating graph...")
    G = nx.Graph()

    # Add job + skill nodes
    print("✅ Adding nodes...")
    G.add_nodes_from(prob_mat.index, bipartite="job")
    G.add_nodes_from(prob_mat.columns, bipartite="skill")

    expected_nodes = len(prob_mat.index) + len(prob_mat.columns)
    if G.number_of_nodes() != expected_nodes:
        raise ValueError(
            f"Unexpected number of nodes. Expected {expected_nodes}, got {G.number_of_nodes()}."
        )

    # ------------------------------------------------------------------
    # Add weighted edges using threshold rule
    # ------------------------------------------------------------------
    print(f"✅ Adding edges using threshold = {threshold}.")
    for job_id in prob_mat.index:
        row = prob_mat.loc[job_id]
        for skill, prob in row.items():
            if prob >= threshold:
                G.add_edge(job_id, skill, weight=float(prob))

    print(f"✅ Edges added. Total edges = {G.number_of_edges()}")

    # ------------------------------------------------------------------
    # Optional: persist graph to disk
    # ------------------------------------------------------------------
    if save_graph_pickle:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        graph_path = MODELS_DIR / f"job_skill_bipartite_thres{threshold}.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        print(f"✅ Graph saved to: {graph_path}")

    print("✅ Graph successfully built!")
    return G, prob_mat, threshold, df
