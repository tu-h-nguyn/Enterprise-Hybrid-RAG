"""Deterministic retrieval metrics, implemented directly.

Definitions used here (stated explicitly because "recall@k" is overloaded):

* ``recall@k``     fraction of a question's relevant chunks that appear in the
                   top k. With a single relevant chunk this equals hit@k.
* ``hit@k``        1 if at least one relevant chunk is in the top k, else 0.
* ``precision@k``  fraction of the top k that is relevant.
* ``MRR``          mean of 1/rank of the first relevant chunk (0 if absent).
* ``nDCG@k``       binary-gain DCG normalised by the ideal ordering.
* ``max_recall@k`` the highest ``recall@k`` any retriever could achieve on this
                   question set. Reported because Recall@k is bounded by
                   ``min(k, |relevant|) / |relevant|``: a question with three
                   relevant chunks caps Recall@1 at 0.3333, and reading 0.4524
                   against an imagined 1.0 makes a saturated category look like
                   the weakest one. This project made exactly that mistake and
                   repeated it in four documents before computing the bound.

Questions with no relevant chunks (unanswerable) are excluded from all of the
above and scored separately by ``no_answer_rate``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def max_recall_at_k(relevant: set[str], k: int) -> float:
    """The best ``recall@k`` obtainable, given how many chunks are relevant."""
    if not relevant:
        return 0.0
    return min(k, len(relevant)) / len(relevant)


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    window = retrieved[:k]
    return len(set(window) & relevant) / len(window)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(position + 1)
              for position, chunk_id in enumerate(retrieved[:k], start=1)
              if chunk_id in relevant)
    ideal = sum(1.0 / math.log2(position + 1)
                for position in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


@dataclass
class RetrievalMetrics:
    ks: tuple[int, ...] = DEFAULT_KS
    _recall: dict[int, list[float]] = field(default_factory=dict)
    _hit: dict[int, list[float]] = field(default_factory=dict)
    _precision: dict[int, list[float]] = field(default_factory=dict)
    _ndcg: dict[int, list[float]] = field(default_factory=dict)
    _max_recall: dict[int, list[float]] = field(default_factory=dict)
    _rr: list[float] = field(default_factory=list)
    n_scored: int = 0
    n_skipped: int = 0

    def __post_init__(self) -> None:
        for k in self.ks:
            self._recall.setdefault(k, [])
            self._hit.setdefault(k, [])
            self._precision.setdefault(k, [])
            self._ndcg.setdefault(k, [])
            self._max_recall.setdefault(k, [])

    def update(self, retrieved: list[str], relevant: set[str]) -> None:
        if not relevant:
            self.n_skipped += 1
            return
        self.n_scored += 1
        for k in self.ks:
            self._recall[k].append(recall_at_k(retrieved, relevant, k))
            self._hit[k].append(hit_at_k(retrieved, relevant, k))
            self._precision[k].append(precision_at_k(retrieved, relevant, k))
            self._ndcg[k].append(ndcg_at_k(retrieved, relevant, k))
            self._max_recall[k].append(max_recall_at_k(relevant, k))
        self._rr.append(reciprocal_rank(retrieved, relevant))

    @staticmethod
    def _mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    def summary(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {"n_scored": self.n_scored, "n_skipped": self.n_skipped}
        for k in self.ks:
            out[f"recall@{k}"] = self._mean(self._recall[k])
            out[f"hit@{k}"] = self._mean(self._hit[k])
            out[f"precision@{k}"] = self._mean(self._precision[k])
            out[f"ndcg@{k}"] = self._mean(self._ndcg[k])
            out[f"max_recall@{k}"] = self._mean(self._max_recall[k])
        out["mrr"] = self._mean(self._rr)
        return out
