"""Top-level result objects returned by the RAG service and the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.document import Citation, RetrievedChunk


class QueryTrace(BaseModel):
    """Per-query observability record (see app/observability)."""

    retrieval_method: str = "hybrid"
    reranker: str = "none"
    reranked: bool = False
    n_dense_candidates: int = 0
    n_sparse_candidates: int = 0
    n_fused_candidates: int = 0
    n_final_contexts: int = 0
    context_tokens: int = 0
    dropped_duplicates: int = 0
    dropped_budget: int = 0
    top_score: float | None = None
    no_answer: bool = False
    no_answer_reason: str | None = None
    embedding_backend: str = ""
    llm_provider: str = ""
    model: str = ""
    prompt_version: str = ""
    latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class RagAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    is_no_answer: bool = False
    trace: QueryTrace = Field(default_factory=QueryTrace)
    metadata: dict[str, Any] = Field(default_factory=dict)
