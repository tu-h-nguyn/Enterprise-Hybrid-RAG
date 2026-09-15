"""Dense, sparse and hybrid retrieval.

Dense retrieval is tested against a *stub* embedder with hand-placed vectors,
so the assertion is about the retriever's ranking logic and not about whatever
an embedding model happens to think today.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.indexing.bm25_store import BM25Store
from app.indexing.embeddings import BaseEmbedder
from app.indexing.vector_store import NumpyVectorStore
from app.models.document import Chunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridConfig, HybridRetriever
from app.retrieval.sparse import SparseRetriever


class StubEmbedder(BaseEmbedder):
    """Maps known texts to fixed unit vectors; anything else lands on the origin-ish axis."""

    name = "stub"

    def __init__(self, table: dict[str, list[float]], dim: int = 3) -> None:
        self._table = table
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        rows = [self._table.get(text, [0.0] * self._dim) for text in texts]
        return np.asarray(rows, dtype=np.float32)


def _chunk(chunk_id: str, text: str, source: str = "doc.md") -> Chunk:
    return Chunk(chunk_id=chunk_id, document_id=source.split(".")[0], source=source,
                 text=text, n_tokens=len(text.split()))


@pytest.fixture
def dense_retriever() -> DenseRetriever:
    chunks = [_chunk("c1", "leave"), _chunk("c2", "security"), _chunk("c3", "expenses")]
    table = {
        "leave": [1.0, 0.0, 0.0],
        "security": [0.0, 1.0, 0.0],
        "expenses": [0.0, 0.0, 1.0],
        "annual leave question": [0.9, 0.1, 0.0],
    }
    embedder = StubEmbedder(table)
    store = NumpyVectorStore()
    texts = [c.text for c in chunks]
    store.upsert(chunks, embedder.embed_documents(texts), texts)
    return DenseRetriever(embedder, store)


def test_dense_retrieval_ranks_by_cosine_similarity(dense_retriever: DenseRetriever) -> None:
    hits = dense_retriever.retrieve("annual leave question", top_k=3)

    assert [h.chunk_id for h in hits] == ["c1", "c2", "c3"]
    assert hits[0].score == pytest.approx(0.9 / np.linalg.norm([0.9, 0.1, 0.0]), abs=1e-5)
    assert hits[0].rank == 1
    assert hits[0].retriever == "dense"
    assert hits[0].component_scores == {"dense": pytest.approx(hits[0].score)}


def test_dense_retrieval_respects_top_k(dense_retriever: DenseRetriever) -> None:
    assert len(dense_retriever.retrieve("annual leave question", top_k=2)) == 2


def test_dense_retrieval_on_empty_query_returns_nothing(dense_retriever: DenseRetriever) -> None:
    assert dense_retriever.retrieve("   ", top_k=5) == []


def test_dense_retrieval_on_empty_store_returns_nothing() -> None:
    retriever = DenseRetriever(StubEmbedder({}), NumpyVectorStore())
    assert retriever.retrieve("anything", top_k=5) == []


@pytest.fixture
def sparse_retriever() -> SparseRetriever:
    chunks = [
        _chunk("c1", "Employees receive 22 days of paid annual leave per calendar year."),
        _chunk("c2", "This policy is reference ISP-2024-03 and passwords must be 14 characters."),
        _chunk("c3", "Mileage for private cars is reimbursed at 0.38 euro per kilometre."),
    ]
    store = BM25Store()
    store.build(chunks, [c.text for c in chunks])
    return SparseRetriever(store)


def test_bm25_matches_an_exact_rare_token(sparse_retriever: SparseRetriever) -> None:
    hits = sparse_retriever.retrieve("ISP-2024-03", top_k=3)

    assert hits, "BM25 should match the reference code exactly"
    assert hits[0].chunk_id == "c2"
    assert hits[0].retriever == "sparse"
    assert hits[0].score > 0.0


def test_bm25_ranks_the_relevant_chunk_first(sparse_retriever: SparseRetriever) -> None:
    hits = sparse_retriever.retrieve("how many days of annual leave", top_k=3)
    assert hits[0].chunk_id == "c1"


def test_bm25_drops_zero_scoring_chunks(sparse_retriever: SparseRetriever) -> None:
    """Chunks sharing no query term carry no evidence and must not enter fusion."""
    hits = sparse_retriever.retrieve("mileage kilometre", top_k=10)
    assert [h.chunk_id for h in hits] == ["c3"]
    assert all(h.score > 0.0 for h in hits)


def test_bm25_survives_a_save_load_round_trip(tmp_path, sparse_retriever: SparseRetriever) -> None:
    sparse_retriever.store.save(tmp_path)
    reloaded = BM25Store()

    assert reloaded.load(tmp_path) is True
    assert reloaded.count() == 3
    assert SparseRetriever(reloaded).retrieve("ISP-2024-03", top_k=1)[0].chunk_id == "c2"


def test_hybrid_combines_both_arms(dense_retriever: DenseRetriever,
                                   sparse_retriever: SparseRetriever) -> None:
    hybrid = HybridRetriever(dense_retriever, sparse_retriever,
                             HybridConfig(dense_top_k=3, sparse_top_k=3, fusion_top_k=5, rrf_k=60))
    hits = hybrid.retrieve("annual leave question")

    assert hits
    assert all(h.retriever == "hybrid" for h in hits)
    assert all(h.rank == i for i, h in enumerate(hits, start=1))
    # Component provenance is what makes a hybrid result explainable.
    assert any("dense" in h.component_scores for h in hits)
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


def test_hybrid_returns_nothing_when_both_arms_are_empty() -> None:
    empty_dense = DenseRetriever(StubEmbedder({}), NumpyVectorStore())
    empty_sparse = SparseRetriever(BM25Store())
    assert HybridRetriever(empty_dense, empty_sparse).retrieve("anything") == []
