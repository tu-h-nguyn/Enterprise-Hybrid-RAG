"""Markdown loader implemented with the standard library only.

Headings build a hierarchical section path (``Parent > Child``) which gives the
chunker and the citation layer much better labels than a flat title. Fenced
code blocks are preserved verbatim and never split on '#' characters.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.cleaners.text_cleaner import clean_text
from app.ingestion.loaders.base import BaseLoader, LoaderError
from app.models.document import Document, RawBlock

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


class MarkdownLoader(BaseLoader):
    suffixes = (".md", ".markdown", ".txt")

    def load(self, path: Path, document_id: str | None = None) -> Document:
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise LoaderError(f"Could not read {path}: {exc}") from exc

        blocks: list[RawBlock] = []
        heading_stack: list[tuple[int, str]] = []
        buffer: list[str] = []
        in_fence = False
        title: str | None = None

        def section_path() -> str | None:
            return " > ".join(h for _, h in heading_stack) if heading_stack else None

        def flush(block_type: str = "paragraph") -> None:
            nonlocal buffer
            text = clean_text("\n".join(buffer)) if block_type != "code" else "\n".join(buffer).strip()
            if text:
                blocks.append(RawBlock(text=text, section=section_path(), block_type=block_type))
            buffer = []

        for line in raw.split("\n"):
            if _FENCE_RE.match(line):
                if in_fence:
                    buffer.append(line)
                    flush("code")
                    in_fence = False
                else:
                    flush()
                    in_fence = True
                    buffer.append(line)
                continue
            if in_fence:
                buffer.append(line)
                continue

            match = _ATX_RE.match(line)
            if match:
                flush()
                level, heading = len(match.group(1)), match.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading))
                if level == 1 and title is None:
                    title = heading
                blocks.append(RawBlock(text=heading, section=section_path(),
                                       block_type="heading", metadata={"level": level}))
                continue

            if not line.strip():
                flush()
            else:
                buffer.append(line)
        flush("code" if in_fence else "paragraph")

        resolved_title = title or path.stem
        return Document(
            document_id=document_id or self.default_document_id(path),
            source=path.name,
            source_type="markdown",
            title=resolved_title,
            blocks=blocks,
            metadata={"title": resolved_title},
        )
