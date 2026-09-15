"""PDF loader built directly on PyMuPDF.

Design notes
------------
* Text is extracted **per page** so that page numbers survive into citations.
* Headings are inferred from font size relative to the document's median body
  size. This is a heuristic, but it is what allows chunks to carry a
  ``section`` label for PDFs, which have no semantic structure of their own.
* Running headers/footers repeated across pages are detected and stripped,
  because they otherwise dominate BM25 term statistics.
"""

from __future__ import annotations

import statistics
from pathlib import Path

try:  # PyMuPDF >= 1.24 exposes the modern module name
    import pymupdf as fitz
except ImportError:  # pragma: no cover - older PyMuPDF
    import fitz  # type: ignore[no-redef]  # the older name is the fallback

from app.ingestion.cleaners.text_cleaner import clean_text, detect_boilerplate, strip_lines
from app.ingestion.loaders.base import BaseLoader, LoaderError
from app.models.document import Document, RawBlock


class PDFLoader(BaseLoader):
    suffixes = (".pdf",)

    def __init__(self, heading_size_ratio: float = 1.15, max_heading_words: int = 14) -> None:
        self.heading_size_ratio = heading_size_ratio
        self.max_heading_words = max_heading_words

    def load(self, path: Path, document_id: str | None = None) -> Document:
        path = Path(path)
        try:
            doc = fitz.open(path)
        except Exception as exc:  # pragma: no cover - depends on corrupt input
            raise LoaderError(f"Could not open PDF {path}: {exc}") from exc

        try:
            pages = [self._page_spans(doc, i) for i in range(doc.page_count)]
            sizes = [s["size"] for page in pages for s in page if s["text"].strip()]
            body_size = statistics.median(sizes) if sizes else 10.0

            plain_pages = ["\n".join(s["text"] for s in page) for page in pages]
            boilerplate = detect_boilerplate(plain_pages)

            blocks: list[RawBlock] = []
            current_section: str | None = None
            for page_no, spans in enumerate(pages, start=1):
                buffer: list[str] = []

                def flush() -> None:
                    nonlocal buffer
                    if not buffer:
                        return
                    text = clean_text(strip_lines("\n".join(buffer), boilerplate))
                    if text:
                        # B023 is suppressed deliberately below: flush() is only
                        # ever called inside this same iteration, and current_section
                        # must be read late — binding it early would file a flushed
                        # buffer under the wrong heading.
                        blocks.append(RawBlock(text=text, page=page_no,  # noqa: B023
                                               section=current_section,  # noqa: B023
                                               block_type="paragraph"))
                    buffer = []

                for span in spans:
                    text = span["text"].strip()
                    if not text or text in boilerplate:
                        continue
                    if self._is_heading(span, body_size, text):
                        flush()
                        current_section = clean_text(text) or current_section
                        blocks.append(RawBlock(text=clean_text(text), page=page_no,
                                               section=current_section, block_type="heading"))
                    else:
                        buffer.append(text)
                flush()

            title = (doc.metadata or {}).get("title") or path.stem
            return Document(
                document_id=document_id or self.default_document_id(path),
                source=path.name,
                source_type="pdf",
                title=title.strip() or path.stem,
                blocks=blocks,
                metadata={"n_pages": doc.page_count, "title": title.strip() or path.stem},
            )
        finally:
            doc.close()

    def _page_spans(self, doc: fitz.Document, index: int) -> list[dict]:
        """Return line-level spans with their dominant font size."""
        page = doc.load_page(index)
        try:
            data = page.get_text("dict")
        except Exception as exc:  # pragma: no cover
            raise LoaderError(f"Could not extract text from page {index + 1}: {exc}") from exc

        spans: list[dict] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue  # image block
            for line in block.get("lines", []):
                pieces = [s.get("text", "") for s in line.get("spans", [])]
                text = "".join(pieces)
                if not text.strip():
                    continue
                font_sizes = [s.get("size", 0.0) for s in line.get("spans", []) if s.get("text", "").strip()]
                flags = [s.get("flags", 0) for s in line.get("spans", []) if s.get("text", "").strip()]
                spans.append({
                    "text": text,
                    "size": max(font_sizes) if font_sizes else 0.0,
                    "bold": any(f & 2 ** 4 for f in flags),
                })
        return spans

    def _is_heading(self, span: dict, body_size: float, text: str) -> bool:
        if len(text.split()) > self.max_heading_words:
            return False
        if span["size"] >= body_size * self.heading_size_ratio:
            return True
        # Bold short lines that do not end with sentence punctuation read as headings.
        return bool(span.get("bold")) and not text.endswith((".", ",", ";"))
