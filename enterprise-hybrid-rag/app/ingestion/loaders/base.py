"""Loader contract: every format is normalised into the same ``Document``."""

from __future__ import annotations

import abc
from pathlib import Path

from app.models.document import Document


class LoaderError(RuntimeError):
    """Raised when a file cannot be parsed. Always carries the file path."""


class BaseLoader(abc.ABC):
    #: file suffixes this loader claims, lowercase and dot-prefixed
    suffixes: tuple[str, ...] = ()

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.suffixes

    @abc.abstractmethod
    def load(self, path: Path, document_id: str | None = None) -> Document:
        """Parse ``path`` into a normalised Document."""

    @staticmethod
    def default_document_id(path: Path) -> str:
        from app.ingestion.text_utils import slugify

        return slugify(path.stem)
