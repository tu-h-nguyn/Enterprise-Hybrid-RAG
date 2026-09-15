"""Dense (bi-encoder) retrieval.

The query is embedded with the *same* embedder used at index time and compared
by cosine similarity. Dense retrieval generalises across vocabulary — "time
off" finds "annual leave" — but is weak on rare exact tokens such as policy
codes, which is precisely the gap BM25 fills.
"""

from __future__ import annotations

import logging

from app.indexing.embeddings import BaseEmbedder
from app.indexing.vector_store import BaseVectorStore
from app.models.document import RetrievedChunk
from app.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    name = "dense"

    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query = (query or "").strip()
        if not query or self.vector_store.count() == 0:
            return []
        try:
            embedding = self.embedder.embed_query(query)
        except Exception as exc:
            logger.error("dense_embed_failed", extra={"error": str(exc)})
            return []
        pairs = self.vector_store.query(embedding, top_k=top_k)
        return self.to_results(pairs, self.name)
