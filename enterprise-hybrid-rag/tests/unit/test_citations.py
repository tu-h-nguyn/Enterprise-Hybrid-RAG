"""Citation parsing and the dropping of unresolvable citations.

An unresolvable citation is a hallucinated provenance claim. It must never
reach the user as though it were real, so these tests pin both the parser and
the stripper.
"""

from __future__ import annotations

from app.generation.answer import parse_citation_indices, resolve_citations, strip_invalid_citations
from app.models.document import Chunk, RetrievedChunk


def _context(chunk_id: str, text: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, document_id="handbook", source="handbook.pdf",
                    text=text, page=page, section="Leave"),
        score=0.9, rank=1, retriever="hybrid",
    )


def test_parses_single_and_grouped_markers() -> None:
    assert parse_citation_indices("Employees get 22 days [Source 2].") == [2]
    assert parse_citation_indices("Both apply [Sources 1, 3].") == [1, 3]
    assert parse_citation_indices("Joined [Sources 2 and 4].") == [2, 4]
    assert parse_citation_indices("Ampersand [Sources 1 & 5].") == [1, 5]


def test_parsing_is_case_insensitive_and_tolerates_spacing() -> None:
    assert parse_citation_indices("text [ source  7 ]") == [7]
    assert parse_citation_indices("text [SOURCE 7]") == [7]


def test_indices_are_deduplicated_in_order_of_appearance() -> None:
    assert parse_citation_indices("a [Source 3] b [Source 1] c [Source 3]") == [3, 1]


def test_no_markers_yields_no_indices() -> None:
    assert parse_citation_indices("An answer with no citations at all.") == []
    assert parse_citation_indices("") == []


def test_resolution_maps_numbers_to_chunks_and_reports_invalid_ones() -> None:
    contexts = [_context("c1", "Employees receive 22 days.", 1),
                _context("c2", "Sick leave is 10 days.", 2)]

    citations, invalid = resolve_citations("First [Source 1]. Second [Source 2]. Ghost [Source 9].",
                                           contexts)

    assert [c.source_index for c in citations] == [1, 2]
    assert [c.chunk_id for c in citations] == ["c1", "c2"]
    assert citations[0].page == 1 and citations[1].page == 2
    assert citations[0].source == "handbook.pdf"
    assert invalid == [9]


def test_zero_and_negative_indices_are_invalid() -> None:
    contexts = [_context("c1", "Body.", 1)]
    _, invalid = resolve_citations("Text [Source 0].", contexts)
    assert invalid == [0]


def test_strip_removes_only_the_unresolvable_numbers() -> None:
    text = "Employees get 22 days [Sources 1, 9]. Sick leave differs [Source 9]."
    stripped = strip_invalid_citations(text, {1})

    assert "[Source 1]" in stripped
    assert "9" not in stripped
    assert "Sick leave differs" in stripped


def test_strip_collapses_the_whitespace_it_leaves_behind() -> None:
    stripped = strip_invalid_citations("A claim [Source 4]. Another claim.", set())
    assert "[Source" not in stripped
    assert "  " not in stripped


def test_valid_citations_survive_stripping_untouched() -> None:
    text = "Employees get 22 days [Source 1]."
    assert strip_invalid_citations(text, {1}) == text
