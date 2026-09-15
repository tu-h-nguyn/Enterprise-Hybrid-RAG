"""Reranking stage.

Retrieval optimises recall over a large candidate pool; reranking optimises
precision over a small one. A cross-encoder reads ``(query, chunk)`` *jointly*
with full attention, so it can judge whether the chunk actually answers the
question — something a bi-encoder cannot do because it must compress the chunk
into a vector before ever seeing the query. The cost is quadratic attention per
pair, which is why it only ever sees ~20 candidates, not the whole corpus.

Backends
--------
``CrossEncoderReranker``   production default (ms-marco-MiniLM-L-6-v2).
``LexicalReranker``        offline fallback. **Not a neural reranker.** It is a
                           transparent feature-based scorer (term coverage,
                           bigram/phrase hits, term proximity, heading match).
                           It is reported separately in evaluation so its
                           numbers are never mistaken for cross-encoder numbers.
``NoOpReranker``           keeps fusion order; used to isolate the reranker's
                           contribution in ablations.
"""

from __future__ import annotations

import abc
import logging
import math

from app.config.settings import Settings
from app.ingestion.text_utils import tokenize
from app.models.document import RetrievedChunk

logger = logging.getLogger(__name__)


class RerankerError(RuntimeError):
    pass


class BaseReranker(abc.ABC):
    name: str = "base"
    is_neural: bool = False

    @abc.abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]: ...

    def rerank(self, query: str, candidates: list[RetrievedChunk],
               top_k: int = 5) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self.score(query, [c.chunk.text for c in candidates])
        order = sorted(zip(candidates, scores), key=lambda pair: -pair[1])[: max(top_k, 1)]
        out: list[RetrievedChunk] = []
        for rank, (candidate, score) in enumerate(order, start=1):
            component_scores = dict(candidate.component_scores)
            component_scores[self.name] = float(score)
            component_ranks = dict(candidate.component_ranks)
            component_ranks[candidate.retriever] = candidate.rank
            out.append(RetrievedChunk(
                chunk=candidate.chunk, score=float(score), rank=rank,
                retriever=f"{candidate.retriever}+rerank",
                component_scores=component_scores, component_ranks=component_ranks,
            ))
        return out

    def describe(self) -> dict[str, object]:
        return {"backend": self.name, "neural": self.is_neural}


class CrossEncoderReranker(BaseReranker):
    name = "cross_encoder"
    is_neural = True

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RerankerError("sentence-transformers is not installed") from exc
        try:
            self._model = CrossEncoder(model_name)
        except Exception as exc:
            raise RerankerError(f"Could not load reranker '{model_name}': {exc}") from exc
        self.model_name = model_name
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    def describe(self) -> dict[str, object]:
        return {"backend": self.name, "model": self.model_name, "neural": True}


class LexicalReranker(BaseReranker):
    """Feature-based fallback reranker (explicitly non-neural).

    Combines four cheap signals that a cross-encoder would otherwise learn:
      * coverage   - fraction of distinct query terms present in the chunk
      * idf_weight - rare query terms count more than common ones
      * bigram     - consecutive query-term pairs appearing in order
      * proximity  - how tightly matched terms cluster inside the chunk
    """

    name = "lexical"
    is_neural = False

    def __init__(self, corpus_tokens: list[list[str]] | None = None) -> None:
        self._idf: dict[str, float] = {}
        if corpus_tokens:
            self.fit(corpus_tokens)

    def fit(self, corpus_tokens: list[list[str]]) -> "LexicalReranker":
        n_docs = len(corpus_tokens)
        document_frequency: dict[str, int] = {}
        for tokens in corpus_tokens:
            for term in set(tokens):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        self._idf = {t: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                     for t, df in document_frequency.items()}
        return self

    def _term_idf(self, term: str) -> float:
        return self._idf.get(term, math.log(1 + len(self._idf) + 0.5) if self._idf else 1.0)

    def score(self, query: str, texts: list[str]) -> list[float]:
        query_tokens = tokenize(query, remove_stopwords=True) or tokenize(query)
        if not query_tokens:
            return [0.0] * len(texts)
        unique_terms = list(dict.fromkeys(query_tokens))
        total_idf = sum(self._term_idf(t) for t in unique_terms) or 1.0
        query_bigrams = {(query_tokens[i], query_tokens[i + 1]) for i in range(len(query_tokens) - 1)}

        scores: list[float] = []
        for text in texts:
            tokens = tokenize(text, remove_stopwords=True)
            if not tokens:
                scores.append(0.0)
                continue
            positions: dict[str, list[int]] = {}
            for i, token in enumerate(tokens):
                positions.setdefault(token, []).append(i)

            matched = [t for t in unique_terms if t in positions]
            coverage = len(matched) / len(unique_terms)
            idf_weight = sum(self._term_idf(t) for t in matched) / total_idf

            text_bigrams = {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}
            bigram = (len(query_bigrams & text_bigrams) / len(query_bigrams)) if query_bigrams else 0.0

            hit_positions = sorted(p for t in matched for p in positions[t][:1])
            if len(hit_positions) > 1:
                span = hit_positions[-1] - hit_positions[0] + 1
                proximity = len(hit_positions) / span
            else:
                proximity = 1.0 if hit_positions else 0.0

            # Weights chosen so coverage dominates and the rest break ties.
            scores.append(round(2.0 * coverage + 1.5 * idf_weight + 1.0 * bigram + 0.5 * proximity, 6))
        return scores


class NoOpReranker(BaseReranker):
    name = "none"

    def score(self, query: str, texts: list[str]) -> list[float]:
        # Descending pseudo-scores preserve the incoming order.
        return [float(len(texts) - i) for i in range(len(texts))]

    def rerank(self, query: str, candidates: list[RetrievedChunk],
               top_k: int = 5) -> list[RetrievedChunk]:
        return candidates[: max(top_k, 1)]


def build_reranker(settings: Settings,
                   corpus_tokens: list[list[str]] | None = None) -> BaseReranker:
    backend = settings.reranker_backend
    if not settings.rerank_enabled or backend == "none":
        return NoOpReranker()
    if backend in {"auto", "cross_encoder"}:
        try:
            return CrossEncoderReranker(settings.reranker_model, settings.rerank_batch_size)
        except RerankerError as exc:
            if backend == "cross_encoder":
                raise
            logger.warning("reranker_backend_fallback",
                           extra={"requested": settings.reranker_model,
                                  "error": str(exc), "fallback": "lexical"})
    return LexicalReranker(corpus_tokens)
