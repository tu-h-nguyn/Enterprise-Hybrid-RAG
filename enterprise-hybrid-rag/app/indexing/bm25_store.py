"""Persistent BM25 (Okapi) sparse index.

Chunk IDs are the join key with the dense index: the same ``chunk_id`` must
address the same text in Chroma and here, otherwise Reciprocal Rank Fusion
would be fusing two different documents. ``save()`` writes the tokenised corpus
so the index can be rebuilt deterministically without re-parsing the sources.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.ingestion.text_utils import tokenize
from app.models.document import Chunk

logger = logging.getLogger(__name__)


class BM25Store:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    # ------------------------------------------------------------------ build
    def build(self, chunks: list[Chunk], index_texts: list[str] | None = None) -> "BM25Store":
        texts = index_texts if index_texts is not None else [c.text for c in chunks]
        if len(texts) != len(chunks):
            raise ValueError("index_texts must align with chunks")
        self._chunks = list(chunks)
        self._tokens = [tokenize(t, remove_stopwords=True) for t in texts]
        self._fit()
        return self

    def _fit(self) -> None:
        usable = [t if t else ["__empty__"] for t in self._tokens]
        self._bm25 = BM25Okapi(usable, k1=self.k1, b=self.b) if usable else None

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None and bool(self._chunks)

    def count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------ query
    def query(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if not self.is_built:
            return []
        tokens = tokenize(query, remove_stopwords=True) or tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda kv: -kv[1])[: max(top_k, 1)]
        # BM25 returns 0.0 for chunks sharing no query term; those carry no
        # evidence and would only pollute the fusion candidate pool.
        return [(self._chunks[i], float(s)) for i, s in ranked if s > 0.0]

    # ------------------------------------------------------------- persistence
    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "bm25_index.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"__meta__": {"k1": self.k1, "b": self.b,
                                              "n": len(self._chunks)}}) + "\n")
            for chunk, tokens in zip(self._chunks, self._tokens):
                fh.write(json.dumps({"chunk": chunk.model_dump(), "tokens": tokens},
                                    ensure_ascii=False) + "\n")
        return path

    def load(self, directory: Path) -> bool:
        path = Path(directory) / "bm25_index.jsonl"
        if not path.exists():
            return False
        chunks: list[Chunk] = []
        tokens: list[list[str]] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if i == 0 and "__meta__" in payload:
                        meta = payload["__meta__"]
                        self.k1, self.b = float(meta.get("k1", self.k1)), float(meta.get("b", self.b))
                        continue
                    chunks.append(Chunk.model_validate(payload["chunk"]))
                    tokens.append(payload["tokens"])
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.error("bm25_load_failed", extra={"error": str(exc)})
            return False
        if not chunks:
            return False
        self._chunks, self._tokens = chunks, tokens
        self._fit()
        return True

    def reset(self) -> None:
        self._chunks, self._tokens, self._bm25 = [], [], None
