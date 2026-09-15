"""Ingestion orchestration: files -> Documents -> Chunks (+ on-disk manifest).

Keeping the parsed corpus on disk (``data/processed/chunks.jsonl``) means the
BM25 index can be rebuilt, evaluation can resolve chunk IDs, and experiments
with different chunk sizes are reproducible without re-parsing PDFs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config.settings import Settings
from app.ingestion.chunking.chunker import Chunker, ChunkingConfig
from app.ingestion.loaders.base import LoaderError
from app.ingestion.loaders.registry import LoaderRegistry
from app.models.document import Chunk, Document

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, settings: Settings, registry: LoaderRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry or LoaderRegistry()
        self.chunker = Chunker(
            ChunkingConfig(
                chunk_size_tokens=settings.chunk_size_tokens,
                chunk_overlap_tokens=settings.chunk_overlap_tokens,
                min_chunk_tokens=settings.min_chunk_tokens,
                chars_per_token=settings.chars_per_token,
                merge_small_sections=settings.merge_small_sections,
            )
        )

    @property
    def chunks_path(self) -> Path:
        return Path(self.settings.processed_dir) / "chunks.jsonl"

    def load_file(self, path: Path) -> Document:
        return self.registry.load(Path(path))

    def discover(self, directory: Path | None = None) -> list[Path]:
        directory = Path(directory or self.settings.raw_dir)
        suffixes = set(self.registry.supported_suffixes)
        return sorted(p for p in directory.rglob("*")
                      if p.is_file() and p.suffix.lower() in suffixes)

    def run(self, paths: list[Path] | None = None,
            directory: Path | None = None) -> tuple[list[Document], list[Chunk]]:
        targets = [Path(p) for p in paths] if paths else self.discover(directory)
        documents: list[Document] = []
        chunks: list[Chunk] = []
        for path in targets:
            try:
                document = self.load_file(path)
            except LoaderError as exc:
                logger.error("ingestion_failed", extra={"file": str(path), "error": str(exc)})
                continue
            doc_chunks = self.chunker.chunk_document(document)
            if not doc_chunks:
                logger.warning("no_chunks_produced", extra={"file": str(path)})
                continue
            documents.append(document)
            chunks.extend(doc_chunks)
            logger.info("document_ingested", extra={"document_id": document.document_id,
                                                    "chunks": len(doc_chunks)})
        return documents, chunks

    # ------------------------------------------------------------- persistence
    def save_chunks(self, chunks: list[Chunk], path: Path | None = None) -> Path:
        target = Path(path or self.chunks_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
        return target

    def load_chunks(self, path: Path | None = None) -> list[Chunk]:
        target = Path(path or self.chunks_path)
        if not target.exists():
            return []
        with target.open("r", encoding="utf-8") as fh:
            return [Chunk.model_validate_json(line) for line in fh if line.strip()]
