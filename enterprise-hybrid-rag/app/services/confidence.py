"""No-answer / abstention gate (Phase 10).

Retrieval always returns *something*: a nearest neighbour exists even when the
corpus contains nothing relevant. Without a gate the LLM is handed irrelevant
context and invited to be creative, which is the most common source of
confident nonsense in RAG systems.

Two independent defences are used, because neither is sufficient alone:

1. **This gate** - a score threshold on the best candidate after the final
   ranking stage. Deterministic, cheap, and it prevents the LLM being called
   at all when nothing relevant was found.
2. **The prompt** - instructs the model to emit ``INSUFFICIENT_CONTEXT`` when
   the supplied sources do not support an answer.

Thresholds are **per stage** because the score scales are not comparable:

    cosine similarity          [-1, 1]
    RRF fusion mass            ~[0, 0.03]  (sum of 1/(k+rank))
    BM25                       unbounded, corpus dependent
    cross-encoder logit        ~[-11, +11]
    lexical fallback score     ~[0, 5]

Neither defence eliminates hallucination: the thresholds are tuned on one
corpus and prompt compliance is not guaranteed. This is documented as a
limitation rather than presented as a fix.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import Settings
from app.models.document import RetrievedChunk


@dataclass(frozen=True)
class GateDecision:
    abstain: bool
    reason: str | None = None
    top_score: float | None = None
    threshold: float | None = None


class ConfidenceGate:
    """Decides whether to abstain, given the final candidate list."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(self, candidates: list[RetrievedChunk], *, method: str,
                 reranked: bool, reranker_is_neural: bool) -> GateDecision:
        if not candidates:
            return GateDecision(abstain=True, reason="no_candidates")

        top = float(candidates[0].score)
        if not self.settings.no_answer_enabled:
            return GateDecision(abstain=False, top_score=top)

        threshold, reason = self._threshold(method, reranked, reranker_is_neural)
        if threshold is None:
            return GateDecision(abstain=False, top_score=top)
        if top < threshold:
            return GateDecision(abstain=True, reason=reason, top_score=top, threshold=threshold)
        return GateDecision(abstain=False, top_score=top, threshold=threshold)

    def _threshold(self, method: str, reranked: bool,
                   reranker_is_neural: bool) -> tuple[float | None, str]:
        settings = self.settings
        if reranked:
            if reranker_is_neural:
                return settings.min_rerank_score, "low_rerank_score"
            return settings.min_lexical_rerank_score, "low_rerank_score"
        if method == "dense":
            return settings.min_dense_similarity, "low_dense_similarity"
        if method == "sparse":
            return settings.min_sparse_score, "low_sparse_score"
        return settings.min_fusion_score, "low_fusion_score"
