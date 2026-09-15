"""Hybrid retrieval via Reciprocal Rank Fusion.

Why RRF and not score averaging
-------------------------------
Cosine similarity lives in [-1, 1] and is dense around 0.2-0.8; BM25 is
unbounded and corpus-dependent. Averaging or min-max normalising them makes the
combination depend on the score *distribution* of a particular query, which is
unstable: one outlier BM25 hit can dominate the blend.

RRF discards magnitudes and uses only ranks:

    RRF(d) = sum_i  w_i / (k + rank_i(d))

``k`` (default 60, from Cormack et al. 2009) damps the influence of the very top
ranks so that a document ranked #1 by one retriever cannot single-handedly win
against a document ranked #2-#3 by both. Documents found by both retrievers
accumulate two terms and rise — which is exactly the agreement signal we want.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.document import Chunk, RetrievedChunk
from app.retrieval.base import BaseRetriever


@dataclass(frozen=True)
class HybridConfig:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 20
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists into one ranking.

    Args:
        ranked_lists: retriever name -> IDs ordered best-first (rank 1 = first).
        k: RRF damping constant.
        weights: optional per-retriever weight, default 1.0.

    Returns:
        ``(id, score)`` sorted by descending score. Ties are broken by ID so
        the output is deterministic and therefore testable.
    """
    if k < 1:
        raise ValueError("rrf_k must be >= 1")
    weights = weights or {}
    scores: dict[str, float] = {}
    for retriever, ids in ranked_lists.items():
        weight = float(weights.get(retriever, 1.0))
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self, dense: BaseRetriever, sparse: BaseRetriever,
                 config: HybridConfig | None = None) -> None:
        self.dense = dense
        self.sparse = sparse
        self.config = config or HybridConfig()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        cfg = self.config
        limit = top_k or cfg.fusion_top_k

        dense_hits = self.dense.retrieve(query, cfg.dense_top_k)
        sparse_hits = self.sparse.retrieve(query, cfg.sparse_top_k)
        if not dense_hits and not sparse_hits:
            return []

        chunks: dict[str, Chunk] = {}
        component_scores: dict[str, dict[str, float]] = {}
        component_ranks: dict[str, dict[str, int]] = {}
        ranked_lists: dict[str, list[str]] = {"dense": [], "sparse": []}

        for retriever_name, hits in (("dense", dense_hits), ("sparse", sparse_hits)):
            for hit in hits:
                chunks.setdefault(hit.chunk_id, hit.chunk)
                ranked_lists[retriever_name].append(hit.chunk_id)
                component_scores.setdefault(hit.chunk_id, {})[retriever_name] = hit.score
                component_ranks.setdefault(hit.chunk_id, {})[retriever_name] = hit.rank

        fused = reciprocal_rank_fusion(
            ranked_lists,
            k=cfg.rrf_k,
            weights={"dense": cfg.dense_weight, "sparse": cfg.sparse_weight},
        )[:limit]

        return [
            RetrievedChunk(
                chunk=chunks[chunk_id], score=score, rank=rank, retriever=self.name,
                component_scores=component_scores.get(chunk_id, {}),
                component_ranks=component_ranks.get(chunk_id, {}),
            )
            for rank, (chunk_id, score) in enumerate(fused, start=1)
        ]
