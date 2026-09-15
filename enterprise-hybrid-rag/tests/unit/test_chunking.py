"""Chunking: page and section provenance must survive, because citations are
built from it. A chunk that loses its page number produces a citation that
points at a document but not at a place in it."""

from __future__ import annotations

from app.ingestion.chunking.chunker import Chunker, ChunkingConfig
from app.models.document import Document, RawBlock


def _document(blocks: list[RawBlock], title: str = "Handbook") -> Document:
    return Document(document_id="handbook", source="handbook.pdf", source_type="pdf",
                    title=title, blocks=blocks)


def test_sections_are_not_merged_across_headings() -> None:
    document = _document([
        RawBlock(text="Annual Leave", block_type="heading", page=1, section="Annual Leave"),
        RawBlock(text="Employees receive 22 days of paid annual leave per calendar year. "
                      "Leave accrues monthly for each completed month of service. "
                      "A maximum of five unused days may be carried into the next year.",
                 page=1, section="Annual Leave"),
        RawBlock(text="Sick Leave", block_type="heading", page=2, section="Sick Leave"),
        RawBlock(text="Employees receive 10 paid sick days per calendar year. "
                      "Unused sick days do not carry over and are not paid out. "
                      "A medical certificate is required from the fourth consecutive day.",
                 page=2, section="Sick Leave"),
    ])
    chunks = Chunker(ChunkingConfig(chunk_size_tokens=200, chunk_overlap_tokens=20,
                                    merge_small_sections=False)).chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0].section == "Annual Leave"
    assert chunks[0].page == 1
    assert chunks[1].section == "Sick Leave"
    assert chunks[1].page == 2
    assert "sick" not in chunks[0].text.lower()


def test_page_range_is_tracked_across_a_page_break() -> None:
    """A chunk straddling a page break must report both ends of the range."""
    document = _document([
        RawBlock(text="Overview", block_type="heading", page=1, section="Overview"),
        RawBlock(text="The policy begins on the first page and sets out the scope "
                      "that applies to every permanent employee of the company.",
                 page=1, section="Overview"),
        RawBlock(text="It continues onto the second page, where the remaining "
                      "obligations and the review cadence are described in full.",
                 page=2, section="Overview"),
    ])
    chunks = Chunker(ChunkingConfig(chunk_size_tokens=400, chunk_overlap_tokens=20)).chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].page_end == 2


def test_chunk_ids_are_stable_and_carry_page() -> None:
    document = _document([
        RawBlock(text="Overview", block_type="heading", page=3, section="Overview"),
        RawBlock(text="Some policy text on page three that is long enough to survive "
                      "the minimum chunk size and therefore produces a real chunk.",
                 page=3, section="Overview"),
    ])
    config = ChunkingConfig(chunk_size_tokens=200, chunk_overlap_tokens=20)

    first = Chunker(config).chunk_document(document)
    second = Chunker(config).chunk_document(document)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunk_id == "handbook_003_000"
    assert first[0].document_id == "handbook"
    assert first[0].source == "handbook.pdf"


def test_long_section_is_split_with_sentence_aligned_overlap() -> None:
    sentences = [f"Sentence number {i} describes a distinct policy detail." for i in range(60)]
    document = _document([
        RawBlock(text="Details", block_type="heading", page=1, section="Details"),
        RawBlock(text=" ".join(sentences), page=1, section="Details"),
    ])
    chunks = Chunker(ChunkingConfig(chunk_size_tokens=120, chunk_overlap_tokens=30)).chunk_document(document)

    assert len(chunks) > 1
    assert all(c.section == "Details" for c in chunks)
    # Overlap is sentence-aligned, so no chunk starts mid-sentence.
    assert all(c.text[0].isupper() for c in chunks)
    # Consecutive chunks share text — that is the overlap doing its job.
    assert set(chunks[0].text.split(".")) & set(chunks[1].text.split("."))


def test_small_adjacent_sections_merge_but_keep_their_labels() -> None:
    document = _document([
        RawBlock(text="One", block_type="heading", page=1, section="One"),
        RawBlock(text="First section body, written long enough to clear the minimum "
                      "chunk size so that it is not discarded as a stray fragment.",
                 page=1, section="One"),
        RawBlock(text="Two", block_type="heading", page=1, section="Two"),
        RawBlock(text="Second section body, also long enough to clear the minimum "
                      "chunk size and therefore eligible to be merged with the first.",
                 page=1, section="Two"),
    ])
    chunks = Chunker(ChunkingConfig(chunk_size_tokens=400, chunk_overlap_tokens=20,
                                    merge_small_sections=True)).chunk_document(document)

    assert len(chunks) == 1
    # The merged chunk is labelled with the first section, and the full list is
    # preserved in metadata so a citation never claims the wrong section.
    assert chunks[0].section == "One"
    assert chunks[0].metadata["sections"] == "One; Two"


def test_index_text_prepends_title_and_section() -> None:
    document = _document([
        RawBlock(text="Annual Leave", block_type="heading", page=1, section="Annual Leave"),
        RawBlock(text="Employees receive 22 days of paid annual leave each year, "
                      "accruing monthly across the whole of the calendar year.",
                 page=1, section="Annual Leave"),
    ])
    chunk = Chunker(ChunkingConfig(chunk_size_tokens=200, chunk_overlap_tokens=20)).chunk_document(document)[0]
    index_text = Chunker.index_text(chunk)

    assert index_text.startswith("Handbook > Annual Leave")
    assert "Employees receive 22 days of paid annual leave" in index_text
    # The stored text stays clean so citations quote the document, not the header.
    assert not chunk.text.startswith("Handbook")


def test_empty_document_yields_no_chunks() -> None:
    assert Chunker().chunk_document(_document([])) == []
