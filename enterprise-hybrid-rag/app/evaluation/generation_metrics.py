"""Generation metrics.

Two clearly separated families:

**Deterministic metrics** (this module's default). Computed from string overlap
only, so they are reproducible, free, and comparable across runs. They measure
*lexical* support, which is a proxy for faithfulness: an answer can be fully
grounded yet score below 1.0 because it paraphrases. They are floors, not
truth.

**LLM-judge metrics** (``LLMJudge``, opt-in). A model scores faithfulness and
relevance. This correlates better with human judgement but is approximate,
non-deterministic, sensitive to the judge model, and biased toward the style of
the generator. Judge scores are always reported in a separate block and never
merged into the deterministic numbers.
"""

from __future__ import annotations

import json
import logging
import re

from app.evaluation.dataset import EvalQuestion, normalize
from app.generation.llm import BaseLLM, LLMError, LLMRequest
from app.ingestion.text_utils import split_sentences, tokenize
from app.models.rag import RagAnswer

logger = logging.getLogger(__name__)

_CITATION_MARKER_RE = re.compile(r"\[\s*Sources?[^\]]*\]", re.IGNORECASE)


def _content_tokens(text: str) -> list[str]:
    return tokenize(_CITATION_MARKER_RE.sub(" ", text), remove_stopwords=True)


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 between an answer and the ground truth."""
    predicted, expected = _content_tokens(prediction), _content_tokens(reference)
    if not predicted or not expected:
        return 0.0
    common = 0
    remaining = list(expected)
    for token in predicted:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def groundedness(answer: str, context: str) -> float:
    """Share of the answer's content bigrams that occur in the context.

    Bigrams rather than unigrams: unigram overlap is trivially high for any
    answer that reuses domain vocabulary, so it would flatter a hallucination
    that recombines real terms into a false claim.
    """
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = _content_tokens(context)
    if not context_tokens:
        return 0.0
    if len(answer_tokens) < 2:
        return 1.0 if answer_tokens[0] in set(context_tokens) else 0.0
    context_bigrams = {(context_tokens[i], context_tokens[i + 1])
                       for i in range(len(context_tokens) - 1)}
    answer_bigrams = [(answer_tokens[i], answer_tokens[i + 1])
                      for i in range(len(answer_tokens) - 1)]
    return sum(1 for bg in answer_bigrams if bg in context_bigrams) / len(answer_bigrams)


def answer_span_coverage(answer: str, question: EvalQuestion) -> float:
    """Fraction of the labelled answer spans that appear in the answer.

    This is the strictest available check: it asks whether the specific fact
    (e.g. "22 days of paid annual leave") actually made it into the output.
    """
    spans = [normalize(s) for s in question.answer_spans if s.strip()]
    if not spans:
        return 0.0
    normalized_answer = normalize(answer)
    return sum(1 for span in spans if span in normalized_answer) / len(spans)


def citation_support(result: RagAnswer, min_overlap: float = 0.15) -> dict[str, float]:
    """Do the cited chunks actually support the sentences that cite them?

    For each sentence carrying ``[Source N]``, measure the content-token
    overlap between that sentence and chunk N. A sentence citing a chunk it
    shares almost nothing with is a mis-citation even if chunk N exists.
    """
    if not result.citations:
        return {"n_citations": 0, "citation_precision": 0.0, "uncited_sentence_rate": 1.0}

    by_index = {c.source_index: c for c in result.citations}
    context_text = {c.source_index: ctx.chunk.text
                    for c, ctx in zip(result.citations, [result.contexts[c.source_index - 1]
                                                         for c in result.citations
                                                         if 0 < c.source_index <= len(result.contexts)])} \
        if result.contexts else {}

    supported, checked, uncited = 0, 0, 0
    for sentence in split_sentences(result.answer):
        markers = re.findall(r"\[\s*Sources?\s*([0-9,\s]+)\]", sentence, re.IGNORECASE)
        if not markers:
            if _content_tokens(sentence):
                uncited += 1
            continue
        indices = {int(n) for group in markers for n in re.findall(r"\d+", group)}
        sentence_tokens = set(_content_tokens(sentence))
        for index in indices:
            if index not in by_index:
                checked += 1
                continue
            chunk_text = context_text.get(index) or (by_index[index].snippet or "")
            chunk_tokens = set(_content_tokens(chunk_text))
            checked += 1
            if sentence_tokens and chunk_tokens:
                overlap = len(sentence_tokens & chunk_tokens) / len(sentence_tokens)
                if overlap >= min_overlap:
                    supported += 1

    n_sentences = max(1, len(split_sentences(result.answer)))
    return {
        "n_citations": float(len(result.citations)),
        "citation_precision": round(supported / checked, 4) if checked else 0.0,
        "uncited_sentence_rate": round(uncited / n_sentences, 4),
    }


def score_answer(result: RagAnswer, question: EvalQuestion, context_text: str) -> dict[str, float]:
    metrics: dict[str, float] = {
        "groundedness": round(groundedness(result.answer, context_text), 4),
        "answer_f1": round(token_f1(result.answer, question.ground_truth), 4),
        "span_coverage": round(answer_span_coverage(result.answer, question), 4),
        "refused": 1.0 if result.is_no_answer else 0.0,
    }
    metrics.update(citation_support(result))
    if question.is_unanswerable:
        # For an unanswerable question the only correct behaviour is refusal.
        metrics["correct_refusal"] = 1.0 if result.is_no_answer else 0.0
    else:
        metrics["over_refusal"] = 1.0 if result.is_no_answer else 0.0
    return metrics


JUDGE_SYSTEM = """You are a strict evaluator of retrieval-augmented answers.
You receive a question, the retrieved context, and a candidate answer.
Score two dimensions from 1 to 5:
- faithfulness: is every claim in the answer supported by the context?
- relevance: does the answer address the question that was asked?
Reply with JSON only: {"faithfulness": <1-5>, "relevance": <1-5>, "reason": "<one sentence>"}"""

JUDGE_TEMPLATE = """Question: {question}

Context:
{context}

Candidate answer:
{answer}

JSON:"""


class LLMJudge:
    """Optional LLM-as-judge scorer. Approximate by construction."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    @property
    def is_available(self) -> bool:
        # The offline extractive backend cannot judge anything; refuse rather
        # than emit meaningless scores.
        return self.llm.provider not in {"extractive"}

    def score(self, question: str, context: str, answer: str) -> dict[str, float | str] | None:
        if not self.is_available:
            return None
        request = LLMRequest(
            system=JUDGE_SYSTEM,
            prompt=JUDGE_TEMPLATE.format(question=question, context=context[:6000], answer=answer),
            question=question, context=context, max_tokens=200,
        )
        try:
            response = self.llm.generate(request)
        except LLMError as exc:
            logger.warning("judge_failed", extra={"error": str(exc)})
            return None
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return {
            "judge_faithfulness": float(payload.get("faithfulness", 0)),
            "judge_relevance": float(payload.get("relevance", 0)),
            "judge_reason": str(payload.get("reason", ""))[:300],
            "judge_model": self.llm.model,
        }
