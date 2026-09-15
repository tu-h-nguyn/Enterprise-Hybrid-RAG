"""End-to-end: ingest -> chunk -> index -> RagService.query.

This is the path the API and the evaluation harness both take, so a break here
invalidates every number the project reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import Settings
from app.evaluation.dataset import EvalDataset, RelevanceResolver
from app.evaluation.evaluator import RetrievalConfig, RetrievalEvaluator
from app.indexing.index_builder import IndexBuilder
from app.ingestion.pipeline import IngestionPipeline
from app.models.document import Chunk
from app.services.rag_service import IndexNotReadyError, RagService


def test_ingestion_produces_chunks_with_provenance(settings: Settings,
                                                   chunks: list[Chunk]) -> None:
    assert len(chunks) >= 4
    assert {c.source for c in chunks} == {"leave_policy.md", "security_policy.md",
                                          "expense_policy.md", "incident_runbook.md"}
    assert all(c.chunk_id and c.document_id and c.text for c in chunks)
    assert all(c.section for c in chunks), "every chunk should carry a section label"


def test_chunks_round_trip_through_disk(settings: Settings, chunks: list[Chunk]) -> None:
    pipeline = IngestionPipeline(settings)
    pipeline.save_chunks(chunks)
    reloaded = pipeline.load_chunks()

    assert [c.chunk_id for c in reloaded] == [c.chunk_id for c in chunks]
    assert [c.text for c in reloaded] == [c.text for c in chunks]


def test_index_is_built_from_the_same_chunks_for_both_backends(settings: Settings,
                                                               chunks: list[Chunk]) -> None:
    bundle = IndexBuilder(settings).build(chunks, reset=True)

    assert bundle.n_chunks == len(chunks)
    assert bundle.bm25.count() == len(chunks)
    assert bundle.manifest["n_documents"] == 4
    assert bundle.manifest["embedder"]["backend"] == "tfidf_svd"
    # Dense and sparse must address the same IDs or fusion is meaningless.
    assert {c.chunk_id for c in bundle.vector_store.get_all()} == {c.chunk_id for c in chunks}


@pytest.mark.parametrize("method", ["dense", "sparse", "hybrid"])
def test_every_retrieval_method_finds_the_leave_policy(service: RagService, method: str) -> None:
    hits, trace, _ = service.retrieve("How many days of paid annual leave?", method, 5, False)

    assert hits, f"{method} returned nothing"
    assert trace.retrieval_method == method
    assert any(h.source == "leave_policy.md" for h in hits[:3])


def test_bm25_finds_an_exact_reference_code(service: RagService) -> None:
    hits, _, _ = service.retrieve("ISP-2024-03", "sparse", 5, False)
    assert hits[0].source == "security_policy.md"


def test_query_returns_a_grounded_cited_answer(service: RagService) -> None:
    answer = service.query("How many days of paid annual leave are employees entitled to?",
                           top_k=5, method="hybrid", rerank=True)

    assert answer.is_no_answer is False
    assert "22 days" in answer.answer
    assert answer.citations, "a grounded answer must carry at least one citation"

    # Every citation must resolve to a chunk that was actually in the context.
    context_ids = {c.chunk_id for c in answer.contexts}
    for citation in answer.citations:
        assert citation.chunk_id in context_ids
        assert 1 <= citation.source_index <= len(answer.contexts)
    assert any(c.source == "leave_policy.md" for c in answer.citations)


def test_query_populates_the_observability_trace(service: RagService) -> None:
    answer = service.query("What is the mileage rate?", top_k=5, method="hybrid", rerank=True)
    trace = answer.trace

    assert trace.retrieval_method == "hybrid"
    assert trace.reranked is True
    assert trace.reranker == "lexical"
    assert trace.n_final_contexts > 0
    assert trace.context_tokens > 0
    assert trace.latency_ms > 0
    assert "retrieval" in trace.stage_latency_ms


def test_unanswerable_question_abstains(service: RagService) -> None:
    """The gate must fire before generation, not after."""
    answer = service.query("What is the vesting schedule for employee stock options?",
                           top_k=5, method="hybrid", rerank=True)

    assert answer.is_no_answer is True
    assert answer.citations == []
    assert answer.trace.no_answer_reason is not None


def test_service_without_an_index_raises(settings: Settings) -> None:
    with pytest.raises(IndexNotReadyError):
        RagService(settings).query("anything")


def test_stats_describe_the_loaded_index(service: RagService) -> None:
    stats = service.stats()

    assert stats.n_documents == 4
    assert stats.n_chunks > 0
    assert stats.embedding_dim > 0
    assert {d["source"] for d in stats.documents} == {
        "leave_policy.md", "security_policy.md", "expense_policy.md", "incident_runbook.md"}


def test_evaluator_runs_over_the_same_service(service: RagService, chunks: list[Chunk],
                                              tmp_path: Path) -> None:
    """The evaluation harness must drive the production path, and its labels
    must resolve against the chunks that were actually indexed."""
    dataset = EvalDataset.model_validate({
        "name": "tiny",
        "questions": [
            {"id": "t1", "question": "How many days of paid annual leave?",
             "ground_truth": "22 days.", "question_type": "factual",
             "relevant_documents": ["leave_policy.md"],
             "answer_spans": ["22 days of paid annual leave"]},
            {"id": "t2", "question": "What is reference ISP-2024-03?",
             "ground_truth": "The security policy.", "question_type": "keyword",
             "relevant_documents": ["security_policy.md"],
             "answer_spans": ["reference ISP-2024-03"]},
            {"id": "t3", "question": "What is the stock option vesting schedule?",
             "question_type": "unanswerable"},
        ],
    })
    resolver = RelevanceResolver(chunks)
    assert resolver.audit(dataset) == [], "fixture labels must resolve"

    result = RetrievalEvaluator(service, resolver, ks=(1, 3)).run(
        dataset, RetrievalConfig("Hybrid", "hybrid", False))

    assert result.metrics["n_scored"] == 2
    assert result.metrics["n_skipped"] == 1     # the unanswerable one
    assert result.metrics["recall@1"] == 1.0
    assert result.latency_ms_mean > 0
