# src/job_intel/features/graph_job_skill.py

"""
Job–Skill Bipartite Graph Builder

This module constructs a bipartite graph linking jobs to skills using
the skill probability matrix produced in Chapter 1.

Nodes:
- Job nodes: one per job_id (index of the processed modelling dataframe)
- Skill nodes: one per skill group (columns of the probability matrix)

Edges:
- An edge exists if the skill probability >= threshold
- Edge weight = skill probability (float in [0, 1])

The resulting graph is a structural artefact used downstream in Chapter 2
(e.g. Node2Vec embeddings, clustering, skill co-occurrence analysis).
"""

import pandas as pd
import networkx as nx
import pickle
from pathlib import Path

from src.job_intel.models.skill_prob_matrix import build_skill_probability_matrix
from src.job_intel.config import CH2_PROCESSED_DF, MODELS_DIR


def build_job_skill_bipartite_graph(
    threshold: float = 0.5,
    save_graph_pickle: bool = False,
):
    """
    Build a job–skill bipartite graph from the skill probability matrix.

    Parameters
    ----------
    threshold : float
        Minimum probability required to include a job–skill edge.
    save_graph_pickle : bool, default False
        Whether to persist the graph to disk as a pickle file.

    Returns
    -------
    G : networkx.Graph
        Bipartite graph with job and skill nodes and weighted edges.
    prob_mat : pd.DataFrame
        Skill probability matrix (jobs × skills).
    threshold : float
        Threshold used to create edges.
    df : pd.DataFrame
        Source modelling dataframe indexed by job_id.
    """

    # ------------------------------------------------------------------
    # Load processed modelling data (job_id must already be the index)
    # ------------------------------------------------------------------
    df = pd.read_csv(CH2_PROCESSED_DF, index_col=0)

    # Basic safety checks on the identifier contract
    assert df.index.name == "job_id", "DataFrame index must be 'job_id'"
    assert df.index.is_unique, "job_id index must be unique"

    # ------------------------------------------------------------------
    # Build skill probability matrix
    # ------------------------------------------------------------------
    print("✅ Building the job probability matrix.")
    prob_mat = build_skill_probability_matrix(jobs_df=df)
    print("✅ Job probability matrix built.")

    # Ensure alignment between dataframe and probability matrix
    assert prob_mat.index.equals(
        df.index
    ), "Index mismatch between dataframe and probability matrix"

    # ------------------------------------------------------------------
    # Create empty bipartite graph
    # ------------------------------------------------------------------
    print("✅ Creating graph...")
    G = nx.Graph()

    # ------------------------------------------------------------------
    # Add job nodes
    # ------------------------------------------------------------------
    print("✅ Adding job nodes...")
    for job_id in prob_mat.index:
        G.add_node(job_id, bipartite="job")

    # ------------------------------------------------------------------
    # Add skill nodes
    # ------------------------------------------------------------------
    print("✅ Adding skill nodes...")
    for skill in prob_mat.columns:
        G.add_node(skill, bipartite="skill")

    expected_nodes = len(prob_mat.index) + len(prob_mat.columns)
    if G.number_of_nodes() == expected_nodes:
        print("✅ Nodes added successfully.")
    else:
        raise ValueError("Unexpected number of nodes in the graph.")

    # ------------------------------------------------------------------
    # Add weighted edges using threshold rule
    # ------------------------------------------------------------------
    print(f"✅ Adding edges using threshold = {threshold}.")
    for job_id in prob_mat.index:
        for skill in prob_mat.columns:
            prob = prob_mat.loc[job_id, skill]
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

    # ------------------------------------------------------------------
    # Return artefacts
    # ------------------------------------------------------------------
    return G, prob_mat, threshold, df
