from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import AppState, get_state
from app.api.schemas.documents import (
    DocumentListResponse,
    DocumentSummary,
    IndexRequest,
    IndexResponse,
    UploadResponse,
)
from app.services.document_service import DocumentServiceError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED,
             summary="Upload a PDF, DOCX or Markdown file")
async def upload_document(file: UploadFile = File(...),
                          state: AppState = Depends(get_state)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "A filename is required")
    payload = await file.read()
    try:
        path = state.documents.save_upload(file.filename, payload)
    except DocumentServiceError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return UploadResponse(filename=path.name, size_bytes=len(payload), indexed=False,
                          message="Uploaded. Call POST /documents/index to make it searchable.")


@router.post("/index", response_model=IndexResponse, summary="(Re)build the dense and sparse indexes")
def index_documents(payload: IndexRequest | None = None,
                    state: AppState = Depends(get_state)) -> IndexResponse:
    started = time.perf_counter()
    try:
        bundle, _chunks = state.documents.ingest_and_index()
    except DocumentServiceError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("indexing_failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Indexing failed: {exc}") from exc

    state.rag.attach(bundle)
    stats = state.rag.stats()
    return IndexResponse(n_documents=stats.n_documents, n_chunks=stats.n_chunks,
                         embedding_model=stats.embedding_model, embedding_dim=stats.embedding_dim,
                         vector_backend=stats.vector_backend,
                         elapsed_ms=round((time.perf_counter() - started) * 1000, 2))


@router.get("", response_model=DocumentListResponse, summary="List indexed documents")
def list_documents(state: AppState = Depends(get_state)) -> DocumentListResponse:
    stats = state.rag.stats()
    indexed_sources = {d["source"] for d in stats.documents}
    pending = [f["source"] for f in state.documents.list_raw_files()
               if f["source"] not in indexed_sources]
    return DocumentListResponse(
        n_documents=stats.n_documents, n_chunks=stats.n_chunks,
        documents=[DocumentSummary(**d) for d in stats.documents],
        pending_files=pending,
    )


@router.delete("/{source}", summary="Remove a source file (requires re-indexing)")
def delete_document(source: str, state: AppState = Depends(get_state)) -> dict[str, str]:
    if not state.documents.delete_raw_file(source):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No such file: {source}")
    return {"status": "deleted", "source": source,
            "message": "Call POST /documents/index to refresh the indexes."}
