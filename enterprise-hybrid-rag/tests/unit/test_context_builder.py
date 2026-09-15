"""Context building: deduplication, token budget and the [Source N] contract."""

from __future__ import annotations

from app.models.document import Chunk, RetrievedChunk
from app.retrieval.context_builder import ContextBuilder


def _candidate(chunk_id: str, text: str, rank: int, page: int | None = 1,
               section: str | None = "Leave") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, document_id="handbook", source="handbook.pdf",
                    text=text, page=page, section=section,
                    metadata={"title": "Employee Handbook"}),
        score=1.0 / rank, rank=rank, retriever="hybrid",
    )


def test_numbered_source_blocks_carry_document_page_and_section() -> None:
    built = ContextBuilder().build([_candidate("c1", "Employees receive 22 days.", 1)])

    assert "[Source 1]" in built.text
    assert "Document: Employee Handbook" in built.text
    assert "File: handbook.pdf" in built.text
    assert "Page: 1" in built.text
    assert "Section: Leave" in built.text
    assert built.sources[0].chunk_id == "c1"


def test_page_range_is_rendered_when_a_chunk_straddles_pages() -> None:
    candidate = _candidate("c1", "Text spanning a page break.", 1)
    candidate.chunk.page_end = 2
    assert "Pages: 1-2" in ContextBuilder().build([candidate]).text


def test_near_duplicates_are_dropped() -> None:
    text = "Employees are entitled to 22 days of paid annual leave per calendar year."
    built = ContextBuilder(dedup_threshold=0.92).build([
        _candidate("c1", text, 1),
        _candidate("c2", text, 2),          # identical — chunk overlap produces these
        _candidate("c3", "Mileage is reimbursed at 0.38 euro per kilometre.", 3),
    ])

    assert [s.chunk_id for s in built.sources] == ["c1", "c3"]
    assert built.dropped_duplicates == 1
    assert built.dropped_budget == 0


def test_token_budget_stops_the_loop_but_always_keeps_the_top_hit() -> None:
    # Distinct vocabulary, so the second chunk is rejected by the budget rather
    # than by the dedup filter that runs before it.
    first = "Paid annual leave entitlement accrues monthly for permanent staff. " * 60
    second = "Mileage reimbursement covers each kilometre driven on business. " * 60
    built = ContextBuilder(token_budget=120).build([
        _candidate("c1", first, 1),
        _candidate("c2", second, 2),
    ])

    assert [s.chunk_id for s in built.sources] == ["c1"], "the top-ranked chunk is never dropped"
    assert built.dropped_duplicates == 0
    assert built.dropped_budget == 1


def test_source_numbering_is_contiguous_after_drops() -> None:
    """Citation resolution indexes into this list, so gaps would mis-resolve."""
    text = "Identical text that will be deduplicated away."
    built = ContextBuilder().build([
        _candidate("c1", text, 1),
        _candidate("c2", text, 2),
        _candidate("c3", "A completely different passage about mileage and kilometres.", 3),
    ])

    assert "[Source 1]" in built.text
    assert "[Source 2]" in built.text
    assert "[Source 3]" not in built.text
    assert len(built.sources) == 2


def test_empty_candidate_list_yields_empty_context() -> None:
    built = ContextBuilder().build([])
    assert built.is_empty
    assert built.text == ""
    assert built.n_tokens == 0
