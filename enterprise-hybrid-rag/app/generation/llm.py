"""Modular LLM providers.

The pipeline depends only on ``BaseLLM.generate(LLMRequest) -> LLMResponse``.
Providers are selected from configuration, so swapping OpenAI for a local vLLM
server, Anthropic, or the offline extractive backend is a one-line env change
and touches no pipeline code.

``ExtractiveLLM`` deserves a note: it is *not* a language model. It selects the
sentences from the supplied context that best match the question and emits them
with citations, or the no-answer marker when nothing matches. It exists so the
full pipeline — including citation resolution and no-answer control — can be
run, tested and demonstrated without an API key or network access. Answers it
produces are extractive and read as such; it is never presented as a generative
result in evaluation.
"""

from __future__ import annotations

import abc
import logging
import re
import time

import httpx
from pydantic import BaseModel, Field

from app.config.settings import Settings
from app.generation.prompts import NO_ANSWER_MARKER
from app.ingestion.text_utils import split_sentences, tokenize

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMRequest(BaseModel):
    system: str
    prompt: str
    question: str = ""
    context: str = ""
    temperature: float = 0.0
    max_tokens: int = 700


class LLMResponse(BaseModel):
    text: str
    model: str
    provider: str
    latency_ms: float = 0.0
    usage: dict[str, int] = Field(default_factory=dict)


class BaseLLM(abc.ABC):
    provider: str = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    def _complete(self, request: LLMRequest) -> tuple[str, dict[str, int]]: ...

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        text, usage = self._complete(request)
        elapsed = (time.perf_counter() - started) * 1000
        return LLMResponse(text=text.strip(), model=self.model, provider=self.provider,
                           latency_ms=round(elapsed, 2), usage=usage)

    def describe(self) -> dict[str, object]:
        return {"provider": self.provider, "model": self.model}


class OpenAICompatibleLLM(BaseLLM):
    """Works with OpenAI, Azure-style gateways, Together, Groq, vLLM, Ollama…

    Anything exposing ``POST {base_url}/chat/completions``.
    """

    provider = "openai_compatible"

    def __init__(self, model: str, base_url: str, api_key: str, timeout_s: float = 60.0) -> None:
        super().__init__(model)
        if not api_key:
            raise LLMError("No API key configured for the OpenAI-compatible provider")
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.timeout_s = timeout_s

    def _complete(self, request: LLMRequest) -> tuple[str, dict[str, int]]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": request.system},
                         {"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(f"{self.base_url}/chat/completions",
                                       json=payload, headers=self._headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {str(data)[:200]}") from exc
        usage = {k: int(v) for k, v in (data.get("usage") or {}).items() if isinstance(v, int)}
        return text, usage


class AnthropicLLM(BaseLLM):
    """Second real provider, included to prove the abstraction is not
    OpenAI-shaped by accident (different endpoint, headers and body)."""

    provider = "anthropic"

    def __init__(self, model: str, base_url: str, api_key: str, timeout_s: float = 60.0) -> None:
        super().__init__(model)
        if not api_key:
            raise LLMError("No API key configured for the Anthropic provider")
        self.base_url = base_url.rstrip("/")
        self._headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"}
        self.timeout_s = timeout_s

    def _complete(self, request: LLMRequest) -> tuple[str, dict[str, int]]:
        payload = {"model": self.model, "system": request.system,
                   "messages": [{"role": "user", "content": request.prompt}],
                   "max_tokens": request.max_tokens, "temperature": request.temperature}
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(f"{self.base_url}/messages", json=payload,
                                       headers=self._headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage_raw = data.get("usage") or {}
        usage = {"prompt_tokens": int(usage_raw.get("input_tokens", 0)),
                 "completion_tokens": int(usage_raw.get("output_tokens", 0))}
        return text, usage


_SOURCE_BLOCK_RE = re.compile(r"\[Source (\d+)\]\n(.*?)(?=\n\[Source \d+\]|\Z)", re.DOTALL)


class ExtractiveLLM(BaseLLM):
    """Deterministic, offline, non-generative fallback (see module docstring)."""

    provider = "extractive"

    def __init__(self, max_sentences: int = 3, min_coverage: float = 0.34) -> None:
        super().__init__("extractive-offline")
        self.max_sentences = max_sentences
        self.min_coverage = min_coverage

    def _complete(self, request: LLMRequest) -> tuple[str, dict[str, int]]:
        context = request.context or request.prompt
        question_tokens = tokenize(request.question, remove_stopwords=True) or tokenize(request.question)
        if not question_tokens or not context.strip():
            return NO_ANSWER_MARKER, {}
        question_set = set(question_tokens)

        scored: list[tuple[float, int, str]] = []
        for match in _SOURCE_BLOCK_RE.finditer(context):
            source_index = int(match.group(1))
            body = match.group(2)
            body = body.split("\n\n", 1)[1] if "\n\n" in body else body
            for sentence in split_sentences(body):
                tokens = set(tokenize(sentence, remove_stopwords=True))
                if not tokens:
                    continue
                coverage = len(question_set & tokens) / len(question_set)
                # Prefer sentences that are informative but not whole paragraphs.
                brevity = 1.0 / (1.0 + abs(len(sentence.split()) - 28) / 28)
                scored.append((coverage + 0.25 * brevity, source_index, sentence.strip()))

        scored.sort(key=lambda item: -item[0])
        best = [s for s in scored if s[0] - 0.25 >= self.min_coverage][: self.max_sentences]
        if not best:
            return NO_ANSWER_MARKER, {}

        # Present in source order so the answer reads coherently.
        best.sort(key=lambda item: item[1])
        parts = [f"{sentence.rstrip('.')} [Source {idx}]." for _, idx, sentence in best]
        return " ".join(parts), {}


def build_llm(settings: Settings) -> BaseLLM:
    provider = settings.llm_provider
    if provider == "extractive":
        return ExtractiveLLM()
    if provider in {"auto", "openai_compatible"}:
        try:
            return OpenAICompatibleLLM(settings.llm_model, settings.llm_base_url,
                                       settings.llm_api_key, settings.llm_timeout_s)
        except LLMError as exc:
            if provider == "openai_compatible":
                raise
            logger.warning("llm_provider_unavailable",
                           extra={"provider": "openai_compatible", "error": str(exc)})
    if provider in {"auto", "anthropic"}:
        try:
            return AnthropicLLM(settings.llm_model, settings.anthropic_base_url,
                                settings.anthropic_api_key, settings.llm_timeout_s)
        except LLMError as exc:
            if provider == "anthropic":
                raise
            logger.warning("llm_provider_unavailable",
                           extra={"provider": "anthropic", "error": str(exc)})
    logger.warning("llm_fallback_extractive",
                   extra={"reason": "no API key configured; answers will be extractive"})
    return ExtractiveLLM()
