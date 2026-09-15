"""Retriever contract. Every retriever returns the same normalised objects so
that fusion, reranking and the context builder never branch on retriever type."""

from __future__ import annotations

import abc

from app.models.document import Chunk, RetrievedChunk


class BaseRetriever(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...

    @staticmethod
    def to_results(pairs: list[tuple[Chunk, float]], retriever: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(chunk=chunk, score=float(score), rank=rank, retriever=retriever,
                           component_scores={retriever: float(score)},
                           component_ranks={retriever: rank})
            for rank, (chunk, score) in enumerate(pairs, start=1)
        ]
