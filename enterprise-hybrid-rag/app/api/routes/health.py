from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import API_VERSION, AppState, get_state
from app.api.schemas.documents import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and index readiness")
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    stats = state.rag.stats()
    return HealthResponse(
        status="ok",
        index_ready=state.rag.is_ready,
        n_chunks=stats.n_chunks,
        embedding_backend=stats.embedding_model or "unavailable",
        llm_provider=f"{state.rag.llm.provider}:{state.rag.llm.model}",
        reranker=state.rag._reranker.name,
        version=API_VERSION,
    )
