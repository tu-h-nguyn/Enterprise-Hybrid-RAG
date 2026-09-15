"""Reranker ordering and the no-op used to isolate the reranker in ablations."""

from __future__ import annotations

import pytest

from app.models.document import Chunk, RetrievedChunk
from app.retrieval.reranker import LexicalReranker, NoOpReranker


def _candidate(chunk_id: str, text: str, score: float, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, document_id="doc", source="doc.md", text=text),
        score=score, rank=rank, retriever="hybrid",
        component_scores={"hybrid": score}, component_ranks={"hybrid": rank},
    )


@pytest.fixture
def candidates() -> list[RetrievedChunk]:
    # Deliberately mis-ordered: the chunk that answers the question is last.
    return [
        _candidate("c1", "The office kitchen is stocked daily and post is sorted each morning.",
                   0.9, 1),
        _candidate("c2", "Parking spaces are allocated by lottery each quarter.", 0.8, 2),
        _candidate("c3", "Employees are entitled to 22 days of paid annual leave per year.",
                   0.7, 3),
    ]


def test_lexical_reranker_promotes_the_relevant_chunk(candidates: list[RetrievedChunk]) -> None:
    reranked = LexicalReranker().rerank("How many days of paid annual leave?", candidates, top_k=3)

    assert reranked[0].chunk_id == "c3", "the answering chunk must be promoted to rank 1"
    assert [c.rank for c in reranked] == [1, 2, 3]
    assert reranked[0].score >= reranked[1].score >= reranked[2].score


def test_reranker_respects_top_k(candidates: list[RetrievedChunk]) -> None:
    assert len(LexicalReranker().rerank("annual leave", candidates, top_k=2)) == 2


def test_reranker_records_its_own_score_without_losing_provenance(
        candidates: list[RetrievedChunk]) -> None:
    reranked = LexicalReranker().rerank("annual leave", candidates, top_k=1)[0]

    assert "lexical" in reranked.component_scores
    assert "hybrid" in reranked.component_scores, "the upstream score must survive"
    assert reranked.retriever == "hybrid+rerank"


def test_idf_weighting_prefers_the_rare_term() -> None:
    """A term appearing in every chunk should not decide the ranking."""
    corpus = [["policy", "common"] for _ in range(20)] + [["policy", "sev1"]]
    reranker = LexicalReranker(corpus)
    assert reranker._term_idf("sev1") > reranker._term_idf("policy")


def test_noop_reranker_preserves_order(candidates: list[RetrievedChunk]) -> None:
    """The ablation depends on this: 'no reranker' must change nothing but length."""
    kept = NoOpReranker().rerank("annual leave", candidates, top_k=2)

    assert [c.chunk_id for c in kept] == ["c1", "c2"]
    assert [c.score for c in kept] == [0.9, 0.8]


def test_reranker_handles_no_candidates() -> None:
    assert LexicalReranker().rerank("anything", [], top_k=5) == []


def test_reranker_is_declared_non_neural() -> None:
    """Evaluation artefacts key off this flag to avoid reporting fallback numbers
    as cross-encoder numbers."""
    assert LexicalReranker().is_neural is False
    assert LexicalReranker().describe() == {"backend": "lexical", "neural": False}
