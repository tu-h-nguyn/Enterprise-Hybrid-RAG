"""Loader tests. The PDF case runs against a PDF generated in the test itself,
so it exercises real PyMuPDF parsing rather than a mocked page object."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from app.ingestion.loaders.base import LoaderError
from app.ingestion.loaders.registry import LoaderRegistry
from app.ingestion.loaders.pdf_loader import PDFLoader


def _write_pdf(path: Path, pages: list[tuple[str, list[str]]]) -> Path:
    """Write a PDF whose headings are visually larger than its body text."""
    doc = fitz.open()
    for heading, body in pages:
        page = doc.new_page()
        page.insert_text((72, 90), heading, fontsize=18, fontname="helv")
        y = 130
        for line in body:
            page.insert_text((72, y), line, fontsize=10, fontname="helv")
            y += 16
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    return _write_pdf(tmp_path / "handbook.pdf", [
        ("Annual Leave", ["Employees receive 22 days of paid annual leave.",
                          "Leave accrues monthly over the calendar year."]),
        ("Sick Leave", ["Employees receive 10 paid sick days each year.",
                        "A certificate is required from the fourth day."]),
    ])


def test_pdf_loader_extracts_text_and_pages(sample_pdf: Path) -> None:
    document = PDFLoader().load(sample_pdf)

    assert document.source == "handbook.pdf"
    assert document.source_type == "pdf"
    assert document.blocks
    assert "22 days of paid annual leave" in document.text
    assert "10 paid sick days" in document.text

    # Page provenance is what citations depend on; it must come from the loader.
    pages = {block.page for block in document.blocks}
    assert pages == {1, 2}
    assert document.n_pages == 2


def test_pdf_loader_detects_headings_as_sections(sample_pdf: Path) -> None:
    document = PDFLoader().load(sample_pdf)

    headings = [b.text for b in document.blocks if b.block_type == "heading"]
    assert "Annual Leave" in headings
    assert "Sick Leave" in headings

    # Body text on page 2 must be labelled with the page-2 heading, not page 1's.
    page_two_body = [b for b in document.blocks if b.page == 2 and b.block_type == "paragraph"]
    assert page_two_body
    assert all(b.section == "Sick Leave" for b in page_two_body)


def test_pdf_loader_assigns_a_stable_document_id(sample_pdf: Path) -> None:
    assert PDFLoader().load(sample_pdf).document_id == "handbook"
    assert PDFLoader().load(sample_pdf, document_id="custom").document_id == "custom"


def test_registry_dispatches_by_suffix(tmp_path: Path, sample_pdf: Path) -> None:
    registry = LoaderRegistry()
    markdown = tmp_path / "note.md"
    markdown.write_text("# Title\n\nSome body text.\n", encoding="utf-8")

    assert registry.load(sample_pdf).source_type == "pdf"
    assert registry.load(markdown).source_type == "markdown"
    assert ".pdf" in registry.supported_suffixes
    assert ".docx" in registry.supported_suffixes
    assert ".md" in registry.supported_suffixes


def test_registry_rejects_unknown_suffix(tmp_path: Path) -> None:
    unsupported = tmp_path / "data.xlsx"
    unsupported.write_bytes(b"not a document")
    with pytest.raises(LoaderError):
        LoaderRegistry().load(unsupported)


def test_markdown_loader_keeps_heading_structure(tmp_path: Path) -> None:
    """Nested headings become a breadcrumb, so a chunk's section says where in
    the document it came from and not merely which heading was nearest."""
    path = tmp_path / "policy.md"
    path.write_text("# Policy\n\nBody one.\n\n## Details\n\nBody two.\n", encoding="utf-8")
    document = LoaderRegistry().load(path)

    sections = {b.section for b in document.blocks if b.block_type == "paragraph"}
    assert sections == {"Policy", "Policy > Details"}
