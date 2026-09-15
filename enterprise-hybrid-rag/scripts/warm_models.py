#!/usr/bin/env python3
"""Pre-download the neural models into the image's cache at build time.

Kept as a script rather than inlined in the Dockerfile so it needs no
BuildKit heredoc support, and so it can be run by hand on a machine that
gains hub access later:

    python scripts/warm_models.py

Exits 0 even when the hub is unreachable: a build must not fail because a
network is restricted. The application already degrades to its offline
backends and reports that fact in /health and in every evaluation artefact,
so a missed warmup is visible rather than silent.
"""

from __future__ import annotations

import os
import sys

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def main() -> int:
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        SentenceTransformer(EMBEDDING_MODEL)
        CrossEncoder(RERANKER_MODEL)
    except Exception as exc:
        print(f"WARNING: model warmup skipped ({type(exc).__name__}: {exc}). "
              f"The runtime will fall back to the offline backends and will say so "
              f"in /health.", file=sys.stderr)
        return 0
    print(f"Cached {EMBEDDING_MODEL} and {RERANKER_MODEL} into {os.environ.get('HF_HOME')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
