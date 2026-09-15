"""Text normalisation applied between loading and chunking.

The cleaner is intentionally conservative: it fixes artefacts that hurt
retrieval (hyphenation across line breaks, repeated headers/footers, control
characters) but never rewrites content, because every cleaned character must
still be quotable as evidence in a citation.
"""

from __future__ import annotations

import re
from collections import Counter

from app.ingestion.text_utils import normalize_unicode

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_BULLET_RE = re.compile(r"^[\s]*[\u2022\u25cf\u25aa\u2023\u2043\-\*]\s+", re.MULTILINE)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_unicode(text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)          # de-hyphenate line breaks
    text = _BULLET_RE.sub("- ", text)                    # normalise bullet glyphs
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def detect_boilerplate(page_texts: list[str], min_ratio: float = 0.6,
                       max_line_len: int = 90) -> set[str]:
    """Find short lines repeated on most pages (running headers/footers).

    Returns the set of offending lines so the caller can strip them. Requires at
    least 3 pages, otherwise legitimate repeated content would be removed.
    """
    if len(page_texts) < 3:
        return set()
    counter: Counter[str] = Counter()
    for page in page_texts:
        seen = {ln.strip() for ln in page.split("\n") if 0 < len(ln.strip()) <= max_line_len}
        counter.update(seen)
    threshold = max(2, int(len(page_texts) * min_ratio))
    return {line for line, count in counter.items() if count >= threshold}


def strip_lines(text: str, blocked: set[str]) -> str:
    if not blocked:
        return text
    kept = [ln for ln in text.split("\n") if ln.strip() not in blocked]
    return "\n".join(kept)
