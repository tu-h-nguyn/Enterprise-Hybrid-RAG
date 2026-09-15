"""Sparse (BM25) retrieval.

BM25 scores exact term matches with TF saturation and length normalisation. It
is unbeatable for identifiers, acronyms and quoted terminology, and it needs no
training — but it cannot match paraphrases.
"""

from __future__ import annotations

from app.indexing.bm25_store import BM25Store
from app.models.document import RetrievedChunk
from app.retrieval.base import BaseRetriever


class SparseRetriever(BaseRetriever):
    name = "sparse"

    def __init__(self, store: BM25Store) -> None:
        self.store = store

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query = (query or "").strip()
        if not query or not self.store.is_built:
            return []
        return self.to_results(self.store.query(query, top_k=top_k), self.name)
