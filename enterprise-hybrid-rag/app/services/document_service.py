"""Document lifecycle: upload -> parse -> chunk -> index.

Indexing is a full rebuild rather than an incremental upsert. With a corpus of
this size a rebuild takes seconds and removes a whole class of bugs (stale BM25
statistics, orphaned vectors, drifting IDs). The trade-off is documented in the
README; incremental indexing is listed as future work.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.config.settings import Settings
from app.indexing.index_builder import IndexBuilder, IndexBundle
from app.ingestion.loaders.base import LoaderError
from app.ingestion.loaders.registry import LoaderRegistry
from app.ingestion.pipeline import IngestionPipeline
from app.models.document import Chunk

logger = logging.getLogger(__name__)


class DocumentServiceError(RuntimeError):
    pass


class DocumentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline = IngestionPipeline(settings)
        self.builder = IndexBuilder(settings)
        self.registry = LoaderRegistry()

    @property
    def supported_suffixes(self) -> tuple[str, ...]:
        return self.registry.supported_suffixes

    # ------------------------------------------------------------------ upload
    def save_upload(self, filename: str, payload: bytes) -> Path:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.supported_suffixes:
            raise DocumentServiceError(
                f"Unsupported file type '{suffix}'. Supported: {', '.join(self.supported_suffixes)}")
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(payload) > max_bytes:
            raise DocumentServiceError(f"File exceeds the {self.settings.max_upload_mb} MB limit")
        if not payload:
            raise DocumentServiceError("Uploaded file is empty")

        safe_name = Path(filename).name.replace("/", "_").replace("\\", "_")
        target = Path(self.settings.raw_dir) / safe_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

        try:  # fail fast: reject unparsable files at upload time, not at index time
            self.registry.load(target)
        except LoaderError as exc:
            target.unlink(missing_ok=True)
            raise DocumentServiceError(f"File could not be parsed: {exc}") from exc

        logger.info("document_uploaded", extra={"file": safe_name, "bytes": len(payload)})
        return target

    def list_raw_files(self) -> list[dict]:
        return [
            {"source": path.name, "size_bytes": path.stat().st_size,
             "suffix": path.suffix.lower()}
            for path in self.pipeline.discover()
        ]

    def delete_raw_file(self, source: str) -> bool:
        target = Path(self.settings.raw_dir) / Path(source).name
        if not target.exists():
            return False
        target.unlink()
        return True

    # ----------------------------------------------------------------- index
    def ingest_and_index(self, paths: list[Path] | None = None) -> tuple[IndexBundle, list[Chunk]]:
        documents, chunks = self.pipeline.run(paths=paths)
        if not chunks:
            raise DocumentServiceError(
                f"No indexable content found in {self.settings.raw_dir}. "
                f"Supported types: {', '.join(self.supported_suffixes)}")
        self.pipeline.save_chunks(chunks)
        bundle = self.builder.build(chunks, reset=True)
        logger.info("index_rebuilt", extra={"documents": len(documents), "chunks": len(chunks)})
        return bundle, chunks

    def clear_index(self) -> None:
        for directory in (self.settings.chroma_dir, self.settings.bm25_dir,
                          Path(self.settings.processed_dir) / "embedder"):
            shutil.rmtree(directory, ignore_errors=True)
        (Path(self.settings.processed_dir) / "index_manifest.json").unlink(missing_ok=True)
        (Path(self.settings.processed_dir) / "chunks.jsonl").unlink(missing_ok=True)
        self.settings.ensure_dirs()
