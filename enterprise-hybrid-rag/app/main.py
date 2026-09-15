"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.deps import API_VERSION, AppState
from app.api.routes import documents, health, query
from app.config.settings import get_settings
from app.observability.logging import configure_logging

logger = logging.getLogger(__name__)

DESCRIPTION = """
Document-grounded question answering over PDF, DOCX and Markdown.

Pipeline: **ingest -> chunk -> dense + BM25 index -> RRF hybrid retrieval ->
cross-encoder reranking -> context building -> grounded generation with citations.**

Answers are restricted to the indexed corpus; when retrieval confidence is too
low the service returns an explicit "insufficient information" response instead
of guessing.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    app.state.app_state = AppState.build(settings)
    logger.info("api_started", extra={"index_ready": app.state.app_state.rag.is_ready})
    yield
    logger.info("api_stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, description=DESCRIPTION, version=API_VERSION,
                  lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"name": settings.app_name, "version": API_VERSION, "docs": "/docs"}

    return app


app = create_app()
