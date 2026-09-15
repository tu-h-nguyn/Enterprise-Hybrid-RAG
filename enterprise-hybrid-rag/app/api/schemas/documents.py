"""Request/response schemas for document endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str
    size_bytes: int
    indexed: bool = False
    message: str


class IndexRequest(BaseModel):
    rebuild: bool = Field(True, description="Rebuild the index from every file in data/raw.")


class IndexResponse(BaseModel):
    n_documents: int
    n_chunks: int
    embedding_model: str
    embedding_dim: int
    vector_backend: str
    elapsed_ms: float


class DocumentSummary(BaseModel):
    document_id: str
    source: str
    title: str | None = None
    n_chunks: int = 0
    pages: int = 0


class DocumentListResponse(BaseModel):
    n_documents: int
    n_chunks: int
    documents: list[DocumentSummary] = Field(default_factory=list)
    pending_files: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    n_chunks: int
    embedding_backend: str
    llm_provider: str
    reranker: str
    version: str
