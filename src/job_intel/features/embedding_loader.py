# src/job_intel/features/embedding_loader.py

"""
Utility functions for loading Node2Vec embeddings produced in Chapter 2.

This module provides a lightweight, consistent interface for loading
job and skill embedding tables from disk with basic validation.
"""

from pathlib import Path
import pandas as pd

from src.job_intel.config import PROCESSED_DATA_DIR


def load_node2vec_embeddings(
    tag: str = "v01",
    embeddings_dir: Path = PROCESSED_DATA_DIR,
    expected_dim: int = 64,
):
    """
    Load Node2Vec job and skill embeddings from CSV files.

    Parameters
    ----------
    tag : str
        Version tag used when saving embeddings (e.g. "v01").
    embeddings_dir : Path
        Directory where embedding CSV files are stored.
    expected_dim : int
        Expected embedding dimensionality (number of columns).

    Returns
    -------
    job_emb : pd.DataFrame
        Job embeddings indexed by job_id.
    skill_emb : pd.DataFrame
        Skill embeddings indexed by skill name.
    """
    job_path = embeddings_dir / f"job_embeddings_node2vec_{tag}.csv"
    skill_path = embeddings_dir / f"skill_embeddings_node2vec_{tag}.csv"

    if not job_path.exists():
        raise FileNotFoundError(f"Job embeddings file not found: {job_path}")
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill embeddings file not found: {skill_path}")

    job_emb = pd.read_csv(job_path, index_col="job_id")
    skill_emb = pd.read_csv(skill_path, index_col="skill")

    # Enforce index types
    job_emb.index = job_emb.index.astype(int)
    skill_emb.index = skill_emb.index.astype(str)

    # Basic shape validation
    if job_emb.shape[1] != expected_dim:
        raise ValueError(
            f"Job embeddings have {job_emb.shape[1]} dimensions; "
            f"expected {expected_dim}."
        )

    if skill_emb.shape[1] != expected_dim:
        raise ValueError(
            f"Skill embeddings have {skill_emb.shape[1]} dimensions; "
            f"expected {expected_dim}."
        )

    # Guard against silent corruption
    if job_emb.index.has_duplicates:
        raise ValueError("Duplicate job_id values found in job embeddings.")

    if skill_emb.index.has_duplicates:
        raise ValueError("Duplicate skill names found in skill embeddings.")

    return job_emb, skill_emb
