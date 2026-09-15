"""Internal representation shared by every stage of the pipeline.

Loaders normalise PDF/DOCX/Markdown into ``RawBlock`` objects, the chunker turns
blocks into ``Chunk`` objects, and every retriever returns ``RetrievedChunk``.
Having one schema end-to-end is what makes citations reliable: page numbers and
sources are attached at load time and are never re-derived later.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal["pdf", "docx", "markdown", "text"]


class RawBlock(BaseModel):
    """A contiguous piece of text as produced by a loader (pre-chunking)."""

    text: str
    page: int | None = None
    section: str | None = None
    block_type: Literal["heading", "paragraph", "list", "table", "code"] = "paragraph"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A loaded source document, normalised into ordered blocks."""

    document_id: str
    source: str
    source_type: SourceType
    title: str | None = None
    blocks: list[RawBlock] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def n_pages(self) -> int:
        pages = [b.page for b in self.blocks if b.page is not None]
        return max(pages) if pages else 0


class Chunk(BaseModel):
    """The atomic unit that is embedded, indexed, retrieved and cited."""

    chunk_id: str
    document_id: str
    source: str
    text: str
    page: int | None = None
    page_end: int | None = None
    section: str | None = None
    chunk_index: int = 0
    n_tokens: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def citation_label(self) -> str:
        parts = [self.metadata.get("title") or self.document_id]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(self.section)
        return " — ".join(str(p) for p in parts)

    def to_flat_metadata(self) -> dict[str, Any]:
        """Chroma only accepts scalar metadata values, so flatten here."""
        flat: dict[str, Any] = {
            "document_id": self.document_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "n_tokens": self.n_tokens,
        }
        if self.page is not None:
            flat["page"] = self.page
        if self.page_end is not None:
            flat["page_end"] = self.page_end
        if self.section:
            flat["section"] = self.section
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                flat[f"meta_{key}"] = value
        return flat

    @classmethod
    def from_flat(cls, chunk_id: str, text: str, flat: dict[str, Any]) -> Chunk:
        metadata = {k[5:]: v for k, v in flat.items() if k.startswith("meta_")}
        return cls(
            chunk_id=chunk_id,
            document_id=str(flat.get("document_id", "")),
            source=str(flat.get("source", "")),
            text=text,
            page=flat.get("page"),
            page_end=flat.get("page_end"),
            section=flat.get("section"),
            chunk_index=int(flat.get("chunk_index", 0)),
            n_tokens=int(flat.get("n_tokens", 0)),
            metadata=metadata,
        )


class RetrievedChunk(BaseModel):
    """A chunk plus the score/rank assigned by a retrieval stage."""

    chunk: Chunk
    score: float
    rank: int
    retriever: str
    component_scores: dict[str, float] = Field(default_factory=dict)
    component_ranks: dict[str, int] = Field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def page(self) -> int | None:
        return self.chunk.page

    @property
    def metadata(self) -> dict[str, Any]:
        return self.chunk.metadata


class Citation(BaseModel):
    """A resolved reference from an answer back to an indexed chunk."""

    source_index: int
    chunk_id: str
    document_id: str
    source: str
    page: int | None = None
    section: str | None = None
    score: float | None = None
    snippet: str | None = None


class IndexStats(BaseModel):
    n_documents: int = 0
    n_chunks: int = 0
    embedding_model: str = ""
    embedding_dim: int = 0
    vector_backend: str = ""
    documents: list[dict[str, Any]] = Field(default_factory=list)
