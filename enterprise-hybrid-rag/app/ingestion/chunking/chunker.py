"""Metadata-aware chunking.

Strategy
--------
1. Blocks are grouped into *sections* (heading + following body blocks).
2. Inside a section, sentences are packed greedily until the token budget is
   reached, then a sentence-aligned overlap window is carried into the next
   chunk. Overlap is sentence-aligned rather than character-aligned so a chunk
   never starts mid-sentence, which measurably helps the cross-encoder.
3. A chunk never spans two documents and never merges two sections. Sections
   are the strongest semantic boundary available for free.
4. Page provenance is tracked as ``page`` (first page touched) and ``page_end``
   (last page touched), so citations stay correct for chunks that straddle a
   page break.

``chunk_id`` follows ``{document_id}_{page:03d}_{index:03d}`` so that IDs are
human-readable in evaluation files and identical across the dense and sparse
indexes.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.text_utils import estimate_tokens, split_sentences
from app.models.document import Chunk, Document, RawBlock


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100
    min_chunk_tokens: int = 24
    chars_per_token: float = 4.0
    merge_small_sections: bool = True

    def __post_init__(self) -> None:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")


@dataclass
class _Piece:
    """Intermediate chunk payload before IDs and metadata are assigned."""

    text: str
    pages: list[int]
    sections: list[str]
    n_tokens: int


@dataclass
class _Section:
    title: str | None
    blocks: list[RawBlock]

    @property
    def pages(self) -> list[int]:
        return [b.page for b in self.blocks if b.page is not None]


class Chunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    # ------------------------------------------------------------------ API
    def chunk_document(self, document: Document) -> list[Chunk]:
        pieces: list[_Piece] = []
        for section in self._sections(document):
            for text, pages in self._pack(section):
                n_tokens = estimate_tokens(text, self.config.chars_per_token)
                pieces.append(_Piece(text=text, pages=pages,
                                     sections=[section.title] if section.title else [],
                                     n_tokens=n_tokens))

        if self.config.merge_small_sections:
            pieces = self._merge_adjacent(pieces)

        chunks: list[Chunk] = []
        for index, piece in enumerate(pieces):
            if piece.n_tokens < self.config.min_chunk_tokens and chunks:
                continue  # drop stray fragments, but never drop the only chunk
            page = min(piece.pages) if piece.pages else None
            page_end = max(piece.pages) if piece.pages else None
            metadata: dict = {"title": document.title or document.document_id,
                              "source_type": document.source_type}
            if len(piece.sections) > 1:
                metadata["sections"] = "; ".join(piece.sections)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}_{(page or 0):03d}_{index:03d}",
                    document_id=document.document_id,
                    source=document.source,
                    text=piece.text,
                    page=page,
                    page_end=page_end,
                    section=piece.sections[0] if piece.sections else None,
                    chunk_index=index,
                    n_tokens=piece.n_tokens,
                    metadata=metadata,
                )
            )
        return chunks

    def _merge_adjacent(self, pieces: list[_Piece]) -> list[_Piece]:
        """Pack consecutive short sections into one chunk.

        Without this, a document made of many short headed sections produces
        many tiny chunks and ``chunk_size_tokens`` becomes decorative. Merging
        only ever joins *adjacent* pieces of the same document, and the merged
        section labels are preserved in metadata so citations stay truthful.
        """
        merged: list[_Piece] = []
        for piece in pieces:
            if merged and merged[-1].n_tokens + piece.n_tokens <= self.config.chunk_size_tokens:
                previous = merged[-1]
                previous.text = f"{previous.text}\n\n{piece.text}"
                previous.pages = sorted(set(previous.pages) | set(piece.pages))
                previous.sections = previous.sections + [s for s in piece.sections
                                                         if s not in previous.sections]
                previous.n_tokens = estimate_tokens(previous.text, self.config.chars_per_token)
            else:
                merged.append(_Piece(text=piece.text, pages=list(piece.pages),
                                     sections=list(piece.sections), n_tokens=piece.n_tokens))
        return merged

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for document in documents:
            out.extend(self.chunk_document(document))
        return out

    @staticmethod
    def index_text(chunk: Chunk) -> str:
        """Text actually fed to the embedder / BM25.

        The document title and section are prepended so that a chunk which says
        "Employees are entitled to 18 days" is still retrievable by a query
        about "annual leave policy" even when those words appear only in the
        heading. The stored ``chunk.text`` is left untouched for citation.
        """
        header_parts = [str(chunk.metadata.get("title") or chunk.document_id)]
        if chunk.section:
            header_parts.append(chunk.section)
        return f"{' > '.join(header_parts)}\n{chunk.text}"

    # -------------------------------------------------------------- internals
    def _sections(self, document: Document) -> list[_Section]:
        sections: list[_Section] = []
        current = _Section(title=None, blocks=[])
        for block in document.blocks:
            if block.block_type == "heading":
                if current.blocks:
                    sections.append(current)
                current = _Section(title=block.section or block.text, blocks=[])
                continue
            if current.title is None and block.section:
                current.title = block.section
            current.blocks.append(block)
        if current.blocks:
            sections.append(current)
        return sections

    def _pack(self, section: _Section) -> list[tuple[str, list[int]]]:
        """Greedy sentence packing with sentence-aligned overlap."""
        units: list[tuple[str, int | None, int]] = []
        for block in section.blocks:
            for sentence in split_sentences(block.text):
                units.append((sentence, block.page,
                              estimate_tokens(sentence, self.config.chars_per_token)))
        if not units:
            return []

        out: list[tuple[str, list[int]]] = []
        buffer: list[tuple[str, int | None, int]] = []
        buffer_tokens = 0

        for unit in units:
            sentence, _, tokens = unit
            if buffer and buffer_tokens + tokens > self.config.chunk_size_tokens:
                out.append(self._materialise(buffer))
                buffer = self._overlap_window(buffer)
                buffer_tokens = sum(u[2] for u in buffer)
            if tokens > self.config.chunk_size_tokens:
                # A single oversized sentence (e.g. a table row): emit it alone
                # rather than silently truncating evidence.
                if buffer:
                    out.append(self._materialise(buffer))
                    buffer, buffer_tokens = [], 0
                out.append(self._materialise([unit]))
                continue
            buffer.append(unit)
            buffer_tokens += tokens

        if buffer:
            out.append(self._materialise(buffer))
        return out

    def _overlap_window(self, buffer: list[tuple[str, int | None, int]]) -> list[tuple[str, int | None, int]]:
        if self.config.chunk_overlap_tokens <= 0:
            return []
        window: list[tuple[str, int | None, int]] = []
        total = 0
        for unit in reversed(buffer):
            if total + unit[2] > self.config.chunk_overlap_tokens and window:
                break
            window.insert(0, unit)
            total += unit[2]
        return window

    @staticmethod
    def _materialise(buffer: list[tuple[str, int | None, int]]) -> tuple[str, list[int]]:
        text = " ".join(u[0].strip() for u in buffer).strip()
        pages = sorted({u[1] for u in buffer if u[1] is not None})
        return text, pages
