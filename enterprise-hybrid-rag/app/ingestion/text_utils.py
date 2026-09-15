"""Small, dependency-free text helpers used across the pipeline.

Token counting is deliberately an *estimate*: the project must run without a
model-specific tokenizer (and without network access to download one). All
chunk sizes and context budgets are therefore approximate, which is documented
in the README as a known limitation.
"""

from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\[])|\n+")

# Deliberately small: aggressive stop-word removal hurts BM25 on policy text
# where terms like "not" and "no" carry meaning.
STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be by for from has have he in is it its of on or that the
    to was were will with this these those there their them they you your we our i""".split()
)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """Lowercased word tokenizer shared by BM25 and the lexical scorers."""
    tokens = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Approximate LLM token count.

    Uses the max of a character-based and a word-based estimate so that both
    dense prose and token-heavy tabular text are treated conservatively.
    """
    if not text:
        return 0
    char_estimate = len(text) / chars_per_token
    word_estimate = len(text.split()) * 1.3
    return max(1, round(max(char_estimate, word_estimate)))


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter. Good enough for chunk boundaries and for
    extractive answer selection; it is not a linguistic tokenizer."""
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_overlap_ratio(candidate: list[str], reference: set[str]) -> float:
    """Fraction of candidate tokens that also appear in the reference set."""
    if not candidate:
        return 0.0
    return sum(1 for t in candidate if t in reference) / len(candidate)


def slugify(value: str, max_len: int = 64) -> str:
    value = normalize_unicode(value)
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    value = re.sub(r"[\s_-]+", "_", value)
    return value[:max_len] or "document"
