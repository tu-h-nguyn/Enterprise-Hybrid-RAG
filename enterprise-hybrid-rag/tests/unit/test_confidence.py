"""The no-answer gate. Thresholds are per stage because the score scales are
not comparable, so each stage is tested against its own scale."""

from __future__ import annotations

from app.config.settings import Settings
from app.models.document import Chunk, RetrievedChunk
from app.services.confidence import ConfidenceGate


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


def _candidate(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id="c1", document_id="doc", source="doc.md", text="Some text."),
        score=score, rank=1, retriever="hybrid",
    )


def test_no_candidates_always_abstains() -> None:
    decision = ConfidenceGate(_settings()).evaluate([], method="hybrid", reranked=False,
                                                    reranker_is_neural=False)
    assert decision.abstain is True
    assert decision.reason == "no_candidates"


def test_dense_below_its_own_threshold_abstains() -> None:
    gate = ConfidenceGate(_settings(min_dense_similarity=0.15))

    low = gate.evaluate([_candidate(0.05)], method="dense", reranked=False,
                        reranker_is_neural=False)
    high = gate.evaluate([_candidate(0.40)], method="dense", reranked=False,
                         reranker_is_neural=False)

    assert low.abstain is True
    assert low.reason == "low_dense_similarity"
    assert low.threshold == 0.15
    assert high.abstain is False


def test_sparse_uses_the_bm25_threshold() -> None:
    gate = ConfidenceGate(_settings(min_sparse_score=0.5))
    assert gate.evaluate([_candidate(0.2)], method="sparse", reranked=False,
                         reranker_is_neural=False).reason == "low_sparse_score"
    assert gate.evaluate([_candidate(3.0)], method="sparse", reranked=False,
                         reranker_is_neural=False).abstain is False


def test_fusion_uses_the_rrf_threshold() -> None:
    gate = ConfidenceGate(_settings(min_fusion_score=0.01))
    assert gate.evaluate([_candidate(0.001)], method="hybrid", reranked=False,
                         reranker_is_neural=False).reason == "low_fusion_score"


def test_neural_and_lexical_rerankers_use_different_scales() -> None:
    """A cross-encoder logit of -3 is acceptable; a lexical score of -3 is not.
    Sharing one threshold across both would abstain on everything or nothing."""
    gate = ConfidenceGate(_settings(min_rerank_score=-6.0, min_lexical_rerank_score=1.25))

    neural = gate.evaluate([_candidate(-3.0)], method="hybrid", reranked=True,
                           reranker_is_neural=True)
    lexical = gate.evaluate([_candidate(-3.0)], method="hybrid", reranked=True,
                            reranker_is_neural=False)

    assert neural.abstain is False
    assert neural.threshold == -6.0
    assert lexical.abstain is True
    assert lexical.threshold == 1.25
    assert lexical.reason == "low_rerank_score"


def test_disabling_the_gate_never_abstains() -> None:
    gate = ConfidenceGate(_settings(no_answer_enabled=False, min_dense_similarity=0.9))
    decision = gate.evaluate([_candidate(0.0)], method="dense", reranked=False,
                             reranker_is_neural=False)

    assert decision.abstain is False
    assert decision.top_score == 0.0


def test_score_exactly_at_the_threshold_is_accepted() -> None:
    gate = ConfidenceGate(_settings(min_dense_similarity=0.15))
    assert gate.evaluate([_candidate(0.15)], method="dense", reranked=False,
                         reranker_is_neural=False).abstain is False
