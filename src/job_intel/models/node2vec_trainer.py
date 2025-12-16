# src/job_intel/models/node2vec_trainer.py

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Iterable, Optional, Tuple

import networkx as nx
import pandas as pd
from node2vec import Node2Vec

from src.job_intel.config import PROCESSED_DATA_DIR


@dataclass(frozen=True)
class Node2VecParams:
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


def train_node2vec_model(G: nx.Graph, params: Node2VecParams):
    """
    Train a Node2Vec model over a weighted graph.

    Returns the fitted gensim model (model.wv contains node vectors).
    """
    if G.number_of_edges() <= 0:
        raise ValueError("Graph has 0 edges; cannot train Node2Vec.")

    node2vec = Node2Vec(
        G,
        dimensions=params.dimensions,
        walk_length=params.walk_length,
        num_walks=params.num_walks,
        workers=params.workers,
        weight_key=params.weight_key,
        seed=params.seed,
    )

    model = node2vec.fit(
        window=params.window,
        min_count=params.min_count,
        batch_words=params.batch_words,
    )

    # Coverage sanity check
    if len(model.wv) < G.number_of_nodes():
        # Not always fatal, but should be rare; fail early to avoid silent partial artefacts.
        raise ValueError(
            f"Model vocabulary ({len(model.wv)}) < graph nodes ({G.number_of_nodes()})."
        )

    return model


def extract_embeddings(
    model,
    G: nx.Graph,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract job and skill embeddings from a fitted Node2Vec gensim model.

    Node keys are typically stored as strings in gensim, so we use model.wv[str(node)].
    """
    job_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "job"]
    skill_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "skill"]

    if not job_nodes or not skill_nodes:
        raise ValueError(
            f"Empty node split: jobs={len(job_nodes)}, skills={len(skill_nodes)}. "
            "Check that bipartite attributes were set correctly."
        )

    dim = model.wv.vector_size
    cols = [f"emb_{i}" for i in range(dim)]

    job_emb = pd.DataFrame(
        [model.wv[str(n)] for n in job_nodes],
        index=job_nodes,
        columns=cols,
    )
    skill_emb = pd.DataFrame(
        [model.wv[str(n)] for n in skill_nodes],
        index=skill_nodes,
        columns=cols,
    )

    job_emb.index.name = "job_id"
    skill_emb.index.name = "skill"

    return job_emb, skill_emb


def stability_overlap_topn(
    model_a,
    model_b,
    anchors: Iterable[str],
    topn: int = 10,
) -> dict[str, int]:
    """
    Simple stability diagnostic: overlap in top-N nearest neighbours across two fits.
    """

    def top_neighbors(wv, node: str, topn: int) -> list[str]:
        return [k for k, _ in wv.most_similar(node, topn=topn)]

    anchors = [str(a) for a in anchors]

    nbrs_a = {a: set(top_neighbors(model_a.wv, a, topn)) for a in anchors}
    nbrs_b = {a: set(top_neighbors(model_b.wv, a, topn)) for a in anchors}

    return {a: len(nbrs_a[a].intersection(nbrs_b[a])) for a in anchors}


def save_embeddings(
    job_emb: pd.DataFrame,
    skill_emb: pd.DataFrame,
    out_dir: Path = PROCESSED_DATA_DIR,
    tag: str = "v01",
    metadata: Optional[dict] = None,
) -> dict[str, Path]:
    """
    Save embeddings as CSV plus a metadata JSON alongside them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    job_path = out_dir / f"job_embeddings_node2vec_{tag}.csv"
    skill_path = out_dir / f"skill_embeddings_node2vec_{tag}.csv"
    meta_path = out_dir / f"node2vec_embeddings_{tag}_metadata.json"

    job_emb.to_csv(job_path)
    skill_emb.to_csv(skill_path)

    meta = metadata or {}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "job_embeddings": job_path,
        "skill_embeddings": skill_path,
        "metadata": meta_path,
    }
