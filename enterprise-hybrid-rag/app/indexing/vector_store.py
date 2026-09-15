"""Vector storage abstraction.

``ChromaVectorStore`` is the default persistent backend. ``NumpyVectorStore``
is an in-memory exact-search implementation used by unit tests and as an escape
hatch when a Chroma install is unavailable; keeping the interface narrow (five
methods) is what makes swapping them safe.

Scores returned by both backends are **cosine similarities in [-1, 1]**, higher
is better. Chroma natively returns a distance, so it is converted once here
instead of leaking a backend detail into the retrievers.
"""

from __future__ import annotations

import abc
import contextlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import Settings
from app.models.document import Chunk

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    pass


class BaseVectorStore(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray, index_texts: list[str]) -> None: ...

    @abc.abstractmethod
    def query(self, embedding: np.ndarray, top_k: int,
              where: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]: ...

    @abc.abstractmethod
    def count(self) -> int: ...

    @abc.abstractmethod
    def reset(self) -> None: ...

    @abc.abstractmethod
    def get_all(self) -> list[Chunk]: ...

    def delete_document(self, document_id: str) -> int:
        raise NotImplementedError


class ChromaVectorStore(BaseVectorStore):
    name = "chroma"

    def __init__(self, directory: Path, collection_name: str) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("chromadb is not installed") from exc

        Path(directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(directory),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # HNSW defaults to L2; cosine is the right metric for normalised
            # sentence embeddings.
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray, index_texts: list[str]) -> None:
        if not chunks:
            return
        if len(chunks) != len(index_texts) or embeddings.shape[0] != len(chunks):
            raise VectorStoreError("chunks, embeddings and index_texts must be the same length")
        batch = 512
        for start in range(0, len(chunks), batch):
            window = slice(start, start + batch)
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks[window]],
                embeddings=[e.tolist() for e in embeddings[window]],
                documents=[c.text for c in chunks[window]],
                metadatas=[c.to_flat_metadata() for c in chunks[window]],
            )

    def query(self, embedding: np.ndarray, top_k: int,
              where: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=min(top_k, self.count()),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma types every field of a result as optional, so narrow once here
        # rather than indexing through Optionals at each use.
        ids = result.get("ids") or [[]]
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        out: list[tuple[Chunk, float]] = []
        for i, chunk_id in enumerate(ids[0]):
            text = str(documents[0][i] or "")
            meta = dict(metadatas[0][i] or {})
            distance = float(distances[0][i])
            out.append((Chunk.from_flat(chunk_id, text, meta), 1.0 - distance))
        return out

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        with contextlib.suppress(Exception):  # collection may not exist yet
            self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"})

    def get_all(self) -> list[Chunk]:
        if self.count() == 0:
            return []
        data = self._collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        return [Chunk.from_flat(cid, str(doc or ""), dict(meta or {}))
                for cid, doc, meta in zip(ids, documents, metadatas, strict=True)]

    def delete_document(self, document_id: str) -> int:
        before = self.count()
        self._collection.delete(where={"document_id": document_id})
        return before - self.count()


class NumpyVectorStore(BaseVectorStore):
    """Exact brute-force cosine search. O(N) per query; fine below ~100k chunks."""

    name = "numpy"

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._chunks: dict[str, Chunk] = {}
        self._matrix: np.ndarray | None = None

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray, index_texts: list[str]) -> None:
        if not chunks:
            return
        vectors = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
        for chunk, vector in zip(chunks, vectors, strict=True):
            if chunk.chunk_id in self._chunks:
                position = self._ids.index(chunk.chunk_id)
                assert self._matrix is not None
                self._matrix[position] = vector
            else:
                self._ids.append(chunk.chunk_id)
                self._matrix = vector[None, :] if self._matrix is None else np.vstack(
                    [self._matrix, vector[None, :]])
            self._chunks[chunk.chunk_id] = chunk

    def query(self, embedding: np.ndarray, top_k: int,
              where: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self._ids:
            return []
        vector = np.asarray(embedding, dtype=np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = self._matrix @ vector
        order = np.argsort(-scores)[: max(top_k, 1)]
        out: list[tuple[Chunk, float]] = []
        for position in order:
            chunk = self._chunks[self._ids[int(position)]]
            if where and any(getattr(chunk, k, chunk.metadata.get(k)) != v for k, v in where.items()):
                continue
            out.append((chunk, float(scores[int(position)])))
        return out

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._ids, self._chunks, self._matrix = [], {}, None

    def get_all(self) -> list[Chunk]:
        return [self._chunks[i] for i in self._ids]

    def delete_document(self, document_id: str) -> int:
        keep = [i for i in self._ids if self._chunks[i].document_id != document_id]
        removed = len(self._ids) - len(keep)
        if removed and self._matrix is not None:
            positions = [self._ids.index(i) for i in keep]
            self._matrix = self._matrix[positions] if positions else None
        self._chunks = {i: self._chunks[i] for i in keep}
        self._ids = keep
        return removed


def build_vector_store(settings: Settings) -> BaseVectorStore:
    if settings.vector_backend == "numpy":
        return NumpyVectorStore()
    try:
        return ChromaVectorStore(settings.chroma_dir, settings.chroma_collection)
    except VectorStoreError as exc:  # pragma: no cover
        logger.warning("vector_backend_fallback", extra={"error": str(exc), "fallback": "numpy"})
        return NumpyVectorStore()
