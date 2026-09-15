"""Application state and FastAPI dependencies.

The embedder, indexes and reranker are expensive to construct, so a single
``AppState`` is built at startup and shared. Rebuilding the index swaps the
bundle atomically on the live service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from app.config.settings import Settings, get_settings
from app.services.document_service import DocumentService
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"


@dataclass
class AppState:
    settings: Settings
    rag: RagService
    documents: DocumentService

    @classmethod
    def build(cls, settings: Settings | None = None) -> "AppState":
        settings = settings or get_settings()
        rag = RagService.from_disk(settings)
        if not rag.is_ready:
            logger.warning("index_not_loaded",
                           extra={"hint": "POST /documents/index or run scripts/ingest.py"})
        return cls(settings=settings, rag=rag, documents=DocumentService(settings))


def get_state(request: Request) -> AppState:
    state: AppState | None = getattr(request.app.state, "app_state", None)
    if state is None:  # pragma: no cover - only if startup failed
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Application state is not initialised")
    return state
