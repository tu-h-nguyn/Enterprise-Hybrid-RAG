"""Turns reranked chunks into the exact string handed to the LLM.

Responsibilities
----------------
1. Drop near-duplicates (chunk overlap and repeated boilerplate mean the top-5
   often contains the same sentence twice, which wastes budget and biases the
   model toward whatever is repeated).
2. Enforce a token budget so the prompt cannot silently overflow the context
   window; chunks are added in rank order and the first one that does not fit
   stops the loop.
3. Emit a numbered ``[Source N]`` block per chunk carrying document, page and
   section, because the answer layer resolves citations by that number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ingestion.text_utils import estimate_tokens, jaccard, tokenize
from app.models.document import RetrievedChunk


@dataclass
class BuiltContext:
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    n_tokens: int = 0
    dropped_duplicates: int = 0
    dropped_budget: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.sources


class ContextBuilder:
    def __init__(self, token_budget: int = 2400, dedup_threshold: float = 0.92,
                 chars_per_token: float = 4.0) -> None:
        self.token_budget = token_budget
        self.dedup_threshold = dedup_threshold
        self.chars_per_token = chars_per_token

    def build(self, candidates: list[RetrievedChunk]) -> BuiltContext:
        selected: list[RetrievedChunk] = []
        seen_tokens: list[set[str]] = []
        used_tokens = 0
        dropped_duplicates = 0
        dropped_budget = 0

        for candidate in candidates:
            tokens = set(tokenize(candidate.chunk.text))
            if any(jaccard(tokens, prior) >= self.dedup_threshold for prior in seen_tokens):
                dropped_duplicates += 1
                continue
            block = self._format_block(len(selected) + 1, candidate)
            block_tokens = estimate_tokens(block, self.chars_per_token)
            if selected and used_tokens + block_tokens > self.token_budget:
                dropped_budget += 1
                continue
            selected.append(candidate)
            seen_tokens.append(tokens)
            used_tokens += block_tokens

        text = "\n\n".join(self._format_block(i, c) for i, c in enumerate(selected, start=1))
        return BuiltContext(text=text, sources=selected,
                            n_tokens=estimate_tokens(text, self.chars_per_token),
                            dropped_duplicates=dropped_duplicates,
                            dropped_budget=dropped_budget)

    @staticmethod
    def _format_block(index: int, candidate: RetrievedChunk) -> str:
        chunk = candidate.chunk
        lines = [f"[Source {index}]",
                 f"Document: {chunk.metadata.get('title') or chunk.document_id}",
                 f"File: {chunk.source}"]
        if chunk.page is not None:
            page = f"Page: {chunk.page}"
            if chunk.page_end and chunk.page_end != chunk.page:
                page = f"Pages: {chunk.page}-{chunk.page_end}"
            lines.append(page)
        if chunk.section:
            lines.append(f"Section: {chunk.section}")
        lines.append("")
        lines.append(chunk.text)
        return "\n".join(lines)
