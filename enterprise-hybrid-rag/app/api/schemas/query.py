"""Request/response schemas for the query endpoint."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.document import Citation


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000,
                          description="Natural-language question about the indexed documents.")
    top_k: int = Field(5, ge=1, le=20, description="Number of contexts passed to the LLM.")
    retrieval_method: Literal["dense", "sparse", "hybrid"] | None = Field(
        None, description="Overrides the configured default method.")
    rerank: bool | None = Field(None, description="Overrides the configured reranking setting.")
    include_chunks: bool = Field(True, description="Return the full retrieved chunk list.")

    model_config = {"json_schema_extra": {"examples": [
        {"question": "How many days of annual leave do employees get?", "top_k": 5}]}}


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    page: int | None = None
    section: str | None = None
    score: float
    rank: int
    retriever: str
    component_scores: dict[str, float] = Field(default_factory=dict)
    text: str


class QueryMetadata(BaseModel):
    retrieval_method: str
    reranker: str
    reranked: bool
    n_candidates: int
    n_final_contexts: int
    context_tokens: int
    top_score: float | None = None
    no_answer: bool = False
    no_answer_reason: str | None = None
    embedding_backend: str = ""
    llm_provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkOut] = Field(default_factory=list)
    metadata: QueryMetadata
    extra: dict[str, Any] = Field(default_factory=dict)
