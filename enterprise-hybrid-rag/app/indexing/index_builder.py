"""Builds and loads the two indexes as one consistent unit.

The dense and sparse indexes are independent structures but must always be
built from the *same* chunk list, using the same ``index_text`` (title+section
header prepended). A mismatch would silently corrupt fusion, so both are always
written together and a manifest records what produced them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.config.settings import Settings
from app.indexing.bm25_store import BM25Store
from app.indexing.embeddings import BaseEmbedder, build_embedder
from app.indexing.vector_store import BaseVectorStore, build_vector_store
from app.ingestion.chunking.chunker import Chunker
from app.models.document import Chunk

logger = logging.getLogger(__name__)


@dataclass
class IndexBundle:
    embedder: BaseEmbedder
    vector_store: BaseVectorStore
    bm25: BM25Store
    manifest: dict

    @property
    def n_chunks(self) -> int:
        return self.vector_store.count()


class IndexBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def manifest_path(self) -> Path:
        return Path(self.settings.processed_dir) / "index_manifest.json"

    @property
    def embedder_dir(self) -> Path:
        return Path(self.settings.processed_dir) / "embedder"

    def build(self, chunks: list[Chunk], reset: bool = True) -> IndexBundle:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")

        index_texts = [Chunker.index_text(c) for c in chunks]
        embedder = build_embedder(self.settings)
        if embedder.requires_fit:
            embedder.fit(index_texts)
            embedder.save(self.embedder_dir)

        vector_store = build_vector_store(self.settings)
        if reset:
            vector_store.reset()
        embeddings = embedder.embed_documents(index_texts)
        vector_store.upsert(chunks, embeddings, index_texts)

        bm25 = BM25Store(k1=self.settings.bm25_k1, b=self.settings.bm25_b)
        bm25.build(chunks, index_texts)
        bm25.save(self.settings.bm25_dir)

        manifest = {
            "n_chunks": len(chunks),
            "n_documents": len({c.document_id for c in chunks}),
            "documents": sorted({c.source for c in chunks}),
            "embedder": embedder.describe(),
            "vector_backend": vector_store.name,
            "chunk_size_tokens": self.settings.chunk_size_tokens,
            "chunk_overlap_tokens": self.settings.chunk_overlap_tokens,
            "bm25": {"k1": self.settings.bm25_k1, "b": self.settings.bm25_b},
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("index_built", extra={k: v for k, v in manifest.items() if k != "documents"})
        return IndexBundle(embedder, vector_store, bm25, manifest)

    def load(self) -> IndexBundle | None:
        """Attach to an index previously written by :meth:`build`."""
        if not self.manifest_path.exists():
            return None
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("manifest_unreadable", extra={"error": str(exc)})
            return None

        embedder = build_embedder(self.settings)
        if embedder.requires_fit and not embedder.load(self.embedder_dir):
            logger.error("embedder_state_missing", extra={"dir": str(self.embedder_dir)})
            return None

        vector_store = build_vector_store(self.settings)
        bm25 = BM25Store(k1=self.settings.bm25_k1, b=self.settings.bm25_b)
        if not bm25.load(self.settings.bm25_dir):
            logger.error("bm25_index_missing")
            return None
        if vector_store.count() == 0:
            logger.error("vector_index_empty")
            return None
        return IndexBundle(embedder, vector_store, bm25, manifest)
