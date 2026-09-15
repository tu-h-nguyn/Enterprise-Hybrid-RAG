"""DOCX loader using python-docx.

DOCX carries explicit style information, so sections come from real heading
styles rather than heuristics. Word has no fixed pagination in the XML, so
``page`` is left as ``None`` and citations fall back to the section label —
this is preferable to inventing page numbers.
"""

from __future__ import annotations

from pathlib import Path

import docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingestion.cleaners.text_cleaner import clean_text
from app.ingestion.loaders.base import BaseLoader, LoaderError
from app.models.document import Document, RawBlock


def _iter_block_items(parent: DocxDocument):
    """Yield paragraphs and tables in true document order."""
    body = parent.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


class DocxLoader(BaseLoader):
    suffixes = (".docx",)

    def load(self, path: Path, document_id: str | None = None) -> Document:
        path = Path(path)
        try:
            source_doc = docx.Document(str(path))
        except Exception as exc:
            raise LoaderError(f"Could not open DOCX {path}: {exc}") from exc

        blocks: list[RawBlock] = []
        current_section: str | None = None
        title: str | None = None

        for item in _iter_block_items(source_doc):
            if isinstance(item, Table):
                text = self._table_to_text(item)
                if text:
                    blocks.append(RawBlock(text=text, section=current_section, block_type="table"))
                continue

            text = clean_text(item.text)
            if not text:
                continue
            style = (item.style.name or "") if item.style is not None else ""
            if style.startswith("Heading") or style in {"Title", "Subtitle"}:
                current_section = text
                if style == "Title" and title is None:
                    title = text
                blocks.append(RawBlock(text=text, section=current_section, block_type="heading",
                                       metadata={"style": style}))
            else:
                block_type = "list" if "List" in style else "paragraph"
                blocks.append(RawBlock(text=text, section=current_section, block_type=block_type))

        core_title = None
        try:
            core_title = (source_doc.core_properties.title or "").strip() or None
        except Exception:  # pragma: no cover - malformed core properties
            core_title = None
        resolved_title = core_title or title or path.stem

        return Document(
            document_id=document_id or self.default_document_id(path),
            source=path.name,
            source_type="docx",
            title=resolved_title,
            blocks=blocks,
            metadata={"title": resolved_title},
        )

    @staticmethod
    def _table_to_text(table: Table) -> str:
        rows: list[str] = []
        for row in table.rows:
            cells = [clean_text(c.text).replace("\n", " ") for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)
