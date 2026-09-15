"""The RAG pipeline, assembled.

    query -> [dense | sparse | hybrid(RRF)] -> rerank -> context -> LLM -> cited answer

This class owns stage sequencing, the no-answer gate and the observability
trace. It holds no HTTP or UI concerns so it is directly usable from the API,
the scripts and the evaluation harness — the evaluation numbers therefore come
from exactly the same code path a user hits.
"""

from __future__ import annotations

import logging
from typing import Literal

from app.config.settings import Settings
from app.generation.answer import AnswerSynthesizer
from app.generation.llm import build_llm
from app.indexing.index_builder import IndexBuilder, IndexBundle
from app.ingestion.text_utils import tokenize
from app.models.document import IndexStats, RetrievedChunk
from app.models.rag import QueryTrace, RagAnswer
from app.observability.logging import Stopwatch
from app.retrieval.base import BaseRetriever
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridConfig, HybridRetriever
from app.retrieval.reranker import BaseReranker, NoOpReranker, build_reranker
from app.retrieval.sparse import SparseRetriever
from app.services.confidence import ConfidenceGate

logger = logging.getLogger(__name__)

RetrievalMethod = Literal["dense", "sparse", "hybrid"]


class IndexNotReadyError(RuntimeError):
    """Raised when a query arrives before any document has been indexed."""


class RagService:
    def __init__(self, settings: Settings, bundle: IndexBundle | None = None,
                 reranker: BaseReranker | None = None) -> None:
        self.settings = settings
        self.context_builder = ContextBuilder(
            token_budget=settings.context_token_budget,
            dedup_threshold=settings.context_dedup_threshold,
            chars_per_token=settings.chars_per_token,
        )
        self.llm = build_llm(settings)
        self.synthesizer = AnswerSynthesizer(self.llm, settings.no_answer_message)
        self._gate = ConfidenceGate(settings)
        self._bundle: IndexBundle | None = None
        self._reranker: BaseReranker = reranker or NoOpReranker()
        self._dense: BaseRetriever | None = None
        self._sparse: BaseRetriever | None = None
        self._hybrid: HybridRetriever | None = None
        if bundle is not None:
            self.attach(bundle, reranker)

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def from_disk(cls, settings: Settings) -> RagService:
        service = cls(settings)
        bundle = IndexBuilder(settings).load()
        if bundle is not None:
            service.attach(bundle)
        return service

    def attach(self, bundle: IndexBundle, reranker: BaseReranker | None = None) -> None:
        self._bundle = bundle
        self._dense = DenseRetriever(bundle.embedder, bundle.vector_store)
        self._sparse = SparseRetriever(bundle.bm25)
        self._hybrid = HybridRetriever(
            self._dense, self._sparse,
            HybridConfig(
                dense_top_k=self.settings.dense_top_k,
                sparse_top_k=self.settings.sparse_top_k,
                fusion_top_k=self.settings.fusion_top_k,
                rrf_k=self.settings.rrf_k,
                dense_weight=self.settings.rrf_dense_weight,
                sparse_weight=self.settings.rrf_sparse_weight,
            ),
        )
        if reranker is not None:
            self._reranker = reranker
        else:
            corpus_tokens = [tokenize(c.text, remove_stopwords=True)
                             for c in bundle.vector_store.get_all()]
            self._reranker = build_reranker(self.settings, corpus_tokens)

    @property
    def is_ready(self) -> bool:
        return self._bundle is not None and self._bundle.n_chunks > 0

    @property
    def reranker(self) -> BaseReranker:
        """The reranker actually in use — evaluation artefacts must record it."""
        return self._reranker

    @property
    def bundle(self) -> IndexBundle:
        if self._bundle is None:
            raise IndexNotReadyError("No index is loaded. Ingest and index documents first.")
        return self._bundle

    def stats(self) -> IndexStats:
        if self._bundle is None:
            return IndexStats()
        manifest = self._bundle.manifest
        embedder = manifest.get("embedder", {})
        chunks = self._bundle.vector_store.get_all()
        by_document: dict[str, dict] = {}
        for chunk in chunks:
            entry = by_document.setdefault(chunk.document_id, {
                "document_id": chunk.document_id, "source": chunk.source,
                "title": chunk.metadata.get("title"), "n_chunks": 0, "pages": 0})
            entry["n_chunks"] += 1
            if chunk.page:
                entry["pages"] = max(entry["pages"], chunk.page)
        return IndexStats(
            n_documents=len(by_document), n_chunks=len(chunks),
            embedding_model=str(embedder.get("model", embedder.get("backend", ""))),
            embedding_dim=int(embedder.get("dimension") or 0),
            vector_backend=str(manifest.get("vector_backend", "")),
            documents=sorted(by_document.values(), key=lambda d: d["document_id"]),
        )

    # -------------------------------------------------------------- retrieval
    def retrieve(self, question: str, method: RetrievalMethod, top_k: int,
                 rerank: bool) -> tuple[list[RetrievedChunk], QueryTrace, Stopwatch]:
        if not self.is_ready:
            raise IndexNotReadyError("No index is loaded. Ingest and index documents first.")

        watch = Stopwatch()
        trace = QueryTrace(
            retrieval_method=method,
            embedding_backend=str(self.bundle.manifest.get("embedder", {}).get("backend", "")),
            llm_provider=self.llm.provider, model=self.llm.model,
        )

        assert self._dense and self._sparse and self._hybrid
        if method == "dense":
            with watch("retrieval"):
                candidates = self._dense.retrieve(question, self.settings.dense_top_k)
            trace.n_dense_candidates = len(candidates)
        elif method == "sparse":
            with watch("retrieval"):
                candidates = self._sparse.retrieve(question, self.settings.sparse_top_k)
            trace.n_sparse_candidates = len(candidates)
        else:
            with watch("retrieval"):
                candidates = self._hybrid.retrieve(question)
            trace.n_dense_candidates = sum(1 for c in candidates if "dense" in c.component_scores)
            trace.n_sparse_candidates = sum(1 for c in candidates if "sparse" in c.component_scores)
            trace.n_fused_candidates = len(candidates)

        use_rerank = rerank and not isinstance(self._reranker, NoOpReranker)
        if use_rerank and candidates:
            with watch("rerank"):
                candidates = self._reranker.rerank(question, candidates, top_k)
            trace.reranked = True
            trace.reranker = self._reranker.name
        else:
            candidates = candidates[:top_k]
            trace.reranker = "none"

        trace.top_score = round(candidates[0].score, 6) if candidates else None
        return candidates, trace, watch

    # ------------------------------------------------------------------ query
    def query(self, question: str, top_k: int | None = None,
              method: RetrievalMethod | None = None, rerank: bool | None = None,
              include_chunks: bool = True) -> RagAnswer:
        question = (question or "").strip()
        method = method or self.settings.retrieval_method
        top_k = top_k or self.settings.rerank_top_k
        rerank = self.settings.rerank_enabled if rerank is None else rerank

        candidates, trace, watch = self.retrieve(question, method, top_k, rerank)

        gate_reason = self._no_answer_reason(candidates, trace)
        if gate_reason is not None:
            trace.no_answer = True
            trace.no_answer_reason = gate_reason
            trace.latency_ms = watch.total_ms
            trace.stage_latency_ms = watch.stages
            self._log(question, trace)
            return RagAnswer(question=question, answer=self.settings.no_answer_message,
                             citations=[], retrieved_chunks=candidates if include_chunks else [],
                             contexts=[], is_no_answer=True, trace=trace)

        with watch("context"):
            context = self.context_builder.build(candidates)
        trace.n_final_contexts = len(context.sources)
        trace.context_tokens = context.n_tokens
        trace.dropped_duplicates = context.dropped_duplicates
        trace.dropped_budget = context.dropped_budget

        with watch("generation"):
            result = self.synthesizer.synthesize(question, context.text, context.sources)

        trace.model = str(result.get("model", trace.model))
        trace.llm_provider = str(result.get("provider", trace.llm_provider))
        trace.prompt_version = str(result.get("prompt_version", ""))
        trace.no_answer = bool(result.get("is_no_answer"))
        trace.no_answer_reason = result.get("no_answer_reason")
        trace.error = result.get("error")
        trace.latency_ms = watch.total_ms
        trace.stage_latency_ms = watch.stages
        self._log(question, trace)

        return RagAnswer(
            question=question, answer=str(result["answer"]), citations=result["citations"],
            retrieved_chunks=candidates if include_chunks else [],
            contexts=context.sources, is_no_answer=bool(result.get("is_no_answer")), trace=trace,
            metadata={"usage": result.get("usage", {}),
                      "invalid_citations": result.get("invalid_citations", [])},
        )

    # -------------------------------------------------------------- internals
    def _no_answer_reason(self, candidates: list[RetrievedChunk], trace: QueryTrace) -> str | None:
        """Confidence gate applied *before* paying for generation.

        Thresholds are per-stage because score scales differ: cosine similarity,
        RRF mass and reranker logits are not comparable quantities.
        """
        decision = self._gate.evaluate(
            candidates,
            method=trace.retrieval_method,
            reranked=trace.reranked,
            reranker_is_neural=self._reranker.is_neural,
        )
        return decision.reason if decision.abstain else None

    def _log(self, question: str, trace: QueryTrace) -> None:
        payload = trace.model_dump()
        payload["question_chars"] = len(question)
        if self.settings.log_chunk_text:
            payload["question"] = question
        logger.info("query_completed", extra=payload)
