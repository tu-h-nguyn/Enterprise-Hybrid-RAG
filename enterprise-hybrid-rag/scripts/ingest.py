#!/usr/bin/env python3
"""Ingest documents and build both indexes.

    python scripts/ingest.py                      # everything in data/raw
    python scripts/ingest.py --file a.pdf b.md    # specific files
    python scripts/ingest.py --chunk-size 300 --chunk-overlap 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings  # noqa: E402
from app.indexing.index_builder import IndexBuilder  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents and build indexes")
    parser.add_argument("--file", nargs="*", type=Path, default=None)
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--no-reset", action="store_true",
                        help="Add to the existing index instead of rebuilding it")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, json_output=not args.quiet)
    if args.chunk_size:
        settings.chunk_size_tokens = args.chunk_size
    if args.chunk_overlap is not None:
        settings.chunk_overlap_tokens = args.chunk_overlap

    pipeline = IngestionPipeline(settings)
    documents, chunks = pipeline.run(paths=args.file, directory=args.dir)
    if not chunks:
        print("No documents found to ingest. Put files in data/raw/ first.", file=sys.stderr)
        return 1
    pipeline.save_chunks(chunks)

    bundle = IndexBuilder(settings).build(chunks, reset=not args.no_reset)

    print(f"\nIngested {len(documents)} documents -> {len(chunks)} chunks")
    print(f"  chunk size / overlap : {settings.chunk_size_tokens} / {settings.chunk_overlap_tokens} tokens")
    print(f"  embedder             : {bundle.manifest['embedder']}")
    print(f"  vector backend       : {bundle.manifest['vector_backend']} ({bundle.vector_store.count()} vectors)")
    print(f"  bm25                 : {bundle.bm25.count()} chunks")
    for document in documents:
        n = sum(1 for c in chunks if c.document_id == document.document_id)
        print(f"    - {document.source:<40} {n:>4} chunks, {document.n_pages or '-'} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
