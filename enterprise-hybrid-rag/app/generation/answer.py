"""Answer synthesis: prompt -> LLM -> validated, cited answer.

Citation handling is deliberately strict. The model writes ``[Source 3]``; this
layer resolves 3 back to the actual chunk that occupied slot 3 in the context.
Numbers that do not correspond to a supplied source are *dropped* and counted,
because an unresolvable citation is a hallucinated one and must never be
rendered to the user as if it were real provenance.
"""

from __future__ import annotations

import logging
import re

from app.generation.llm import BaseLLM, LLMError, LLMRequest
from app.generation.prompts import NO_ANSWER_MARKER, PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from app.models.document import Citation, RetrievedChunk

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[\s*Sources?\s*([0-9]+(?:\s*(?:,|and|&)\s*[0-9]+)*)\s*\]", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+")


def parse_citation_indices(text: str) -> list[int]:
    """Extract source numbers from ``[Source 2]`` / ``[Sources 1, 3]`` markers."""
    found: list[int] = []
    for match in _CITATION_RE.finditer(text or ""):
        for number in _NUMBER_RE.findall(match.group(1)):
            value = int(number)
            if value not in found:
                found.append(value)
    return found


def resolve_citations(text: str, contexts: list[RetrievedChunk],
                      snippet_chars: int = 240) -> tuple[list[Citation], list[int]]:
    """Map cited numbers to chunks. Returns (valid citations, invalid numbers)."""
    citations: list[Citation] = []
    invalid: list[int] = []
    for index in parse_citation_indices(text):
        if 1 <= index <= len(contexts):
            candidate = contexts[index - 1]
            chunk = candidate.chunk
            citations.append(Citation(
                source_index=index, chunk_id=chunk.chunk_id, document_id=chunk.document_id,
                source=chunk.source, page=chunk.page, section=chunk.section,
                score=round(candidate.score, 6),
                snippet=chunk.text[:snippet_chars].strip(),
            ))
        else:
            invalid.append(index)
    return citations, invalid


def strip_invalid_citations(text: str, valid_indices: set[int]) -> str:
    def _replace(match: re.Match[str]) -> str:
        numbers = [int(n) for n in _NUMBER_RE.findall(match.group(1))]
        kept = [n for n in numbers if n in valid_indices]
        if not kept:
            return ""
        label = "Source" if len(kept) == 1 else "Sources"
        return f"[{label} {', '.join(str(n) for n in kept)}]"

    return re.sub(r"\s{2,}", " ", _CITATION_RE.sub(_replace, text)).strip()


class AnswerSynthesizer:
    def __init__(self, llm: BaseLLM, no_answer_message: str) -> None:
        self.llm = llm
        self.no_answer_message = no_answer_message

    def synthesize(self, question: str, context_text: str,
                   contexts: list[RetrievedChunk]) -> dict:
        """Returns a dict with answer/citations/no-answer flags and LLM metadata."""
        request = LLMRequest(
            system=SYSTEM_PROMPT,
            prompt=build_user_prompt(question, context_text),
            question=question,
            context=context_text,
        )
        try:
            response = self.llm.generate(request)
        except LLMError as exc:
            logger.error("generation_failed", extra={"error": str(exc)})
            return {"answer": "The answer service is temporarily unavailable. Please retry.",
                    "citations": [], "is_no_answer": True, "no_answer_reason": "llm_error",
                    "model": self.llm.model, "provider": self.llm.provider,
                    "latency_ms": 0.0, "prompt_version": PROMPT_VERSION, "error": str(exc)}

        text = (response.text or "").strip()
        if not text or NO_ANSWER_MARKER in text.upper():
            return {"answer": self.no_answer_message, "citations": [], "is_no_answer": True,
                    "no_answer_reason": "model_declined", "model": response.model,
                    "provider": response.provider, "latency_ms": response.latency_ms,
                    "usage": response.usage, "prompt_version": PROMPT_VERSION}

        citations, invalid = resolve_citations(text, contexts)
        if invalid:
            logger.warning("invalid_citations_dropped", extra={"indices": invalid})
            text = strip_invalid_citations(text, {c.source_index for c in citations})

        return {"answer": text, "citations": citations, "is_no_answer": False,
                "no_answer_reason": None, "model": response.model, "provider": response.provider,
                "latency_ms": response.latency_ms, "usage": response.usage,
                "invalid_citations": invalid, "prompt_version": PROMPT_VERSION}
