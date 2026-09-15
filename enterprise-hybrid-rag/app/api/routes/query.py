from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import AppState, get_state
from app.api.schemas.query import (QueryMetadata, QueryRequest, QueryResponse, RetrievedChunkOut)
from app.services.rag_service import IndexNotReadyError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse,
             summary="Ask a question grounded in the indexed documents")
def query(payload: QueryRequest, state: AppState = Depends(get_state)) -> QueryResponse:
    try:
        result = state.rag.query(
            question=payload.question, top_k=payload.top_k, method=payload.retrieval_method,
            rerank=payload.rerank, include_chunks=payload.include_chunks,
        )
    except IndexNotReadyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("query_failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Query failed: {exc}") from exc

    trace = result.trace
    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        retrieved_chunks=[
            RetrievedChunkOut(
                chunk_id=c.chunk.chunk_id, document_id=c.chunk.document_id, source=c.chunk.source,
                page=c.chunk.page, section=c.chunk.section, score=round(c.score, 6), rank=c.rank,
                retriever=c.retriever, component_scores=c.component_scores, text=c.chunk.text,
            ) for c in result.retrieved_chunks
        ],
        metadata=QueryMetadata(
            retrieval_method=trace.retrieval_method, reranker=trace.reranker,
            reranked=trace.reranked,
            n_candidates=max(trace.n_fused_candidates,
                             trace.n_dense_candidates + trace.n_sparse_candidates),
            n_final_contexts=trace.n_final_contexts, context_tokens=trace.context_tokens,
            top_score=trace.top_score, no_answer=trace.no_answer,
            no_answer_reason=trace.no_answer_reason, embedding_backend=trace.embedding_backend,
            llm_provider=trace.llm_provider, model=trace.model,
            latency_ms=trace.latency_ms, stage_latency_ms=trace.stage_latency_ms,
        ),
        extra=result.metadata,
    )
