"""Format dispatch. Adding a new file type means adding one loader here."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.loaders.base import BaseLoader, LoaderError
from app.ingestion.loaders.docx_loader import DocxLoader
from app.ingestion.loaders.markdown_loader import MarkdownLoader
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.models.document import Document

DEFAULT_LOADERS: tuple[BaseLoader, ...] = (PDFLoader(), DocxLoader(), MarkdownLoader())


class LoaderRegistry:
    def __init__(self, loaders: tuple[BaseLoader, ...] = DEFAULT_LOADERS) -> None:
        self._loaders = loaders

    @property
    def supported_suffixes(self) -> tuple[str, ...]:
        return tuple(sorted({s for loader in self._loaders for s in loader.suffixes}))

    def get(self, path: Path) -> BaseLoader:
        for loader in self._loaders:
            if loader.supports(Path(path)):
                return loader
        raise LoaderError(
            f"Unsupported file type '{Path(path).suffix}'. "
            f"Supported: {', '.join(self.supported_suffixes)}"
        )

    def load(self, path: Path, document_id: str | None = None) -> Document:
        return self.get(path).load(Path(path), document_id=document_id)
