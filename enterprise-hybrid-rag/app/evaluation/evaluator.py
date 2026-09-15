"""Evaluation harness.

The retrieval evaluator calls the *same* ``RagService.retrieve`` used to serve
traffic, so nothing measured here is a reimplementation that could drift from
production behaviour.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.evaluation.dataset import EvalDataset, RelevanceResolver
from app.evaluation.generation_metrics import LLMJudge, score_answer
from app.evaluation.retrieval_metrics import DEFAULT_KS, RetrievalMetrics
from app.retrieval.context_builder import ContextBuilder
from app.services.rag_service import RagService, RetrievalMethod

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    """One row of the comparison table."""

    label: str
    method: RetrievalMethod
    rerank: bool = False

    def as_dict(self) -> dict:
        return {"label": self.label, "method": self.method, "rerank": self.rerank}


DEFAULT_CONFIGS: tuple[RetrievalConfig, ...] = (
    RetrievalConfig("Dense only", "dense", False),
    RetrievalConfig("BM25 only", "sparse", False),
    RetrievalConfig("Hybrid (RRF)", "hybrid", False),
    RetrievalConfig("Hybrid + Reranker", "hybrid", True),
)

@dataclass
class RetrievalRunResult:
    config: RetrievalConfig
    metrics: dict[str, float | int]
    per_question: list[dict] = field(default_factory=list)
    latency_ms_mean: float = 0.0
    latency_ms_p95: float = 0.0


class RetrievalEvaluator:
    def __init__(self, service: RagService, resolver: RelevanceResolver,
                 ks: tuple[int, ...] = DEFAULT_KS) -> None:
        self.service = service
        self.resolver = resolver
        self.ks = ks
        self.top_k = max(ks)

    def run(self, dataset: EvalDataset, config: RetrievalConfig) -> RetrievalRunResult:
        metrics = RetrievalMetrics(ks=self.ks)
        per_question: list[dict] = []
        latencies: list[float] = []

        for question in dataset.questions:
            relevant = self.resolver.resolve(question)
            started = time.perf_counter()
            # retrieve() returns (candidates, QueryTrace, Stopwatch) — see HANDOFF §4.
            final, _, _ = self.service.retrieve(question.question, config.method,
                                                self.top_k, config.rerank)
            latencies.append((time.perf_counter() - started) * 1000)
            retrieved_ids = [hit.chunk_id for hit in final]
            metrics.update(retrieved_ids, relevant)
            per_question.append({
                "id": question.id, "question_type": question.question_type,
                "n_relevant": len(relevant), "retrieved": retrieved_ids[:self.top_k],
                "first_hit_rank": next((i for i, cid in enumerate(retrieved_ids, 1)
                                        if cid in relevant), None),
                "top_score": round(final[0].score, 6) if final else None,
            })

        latencies.sort()
        return RetrievalRunResult(
            config=config, metrics=metrics.summary(), per_question=per_question,
            latency_ms_mean=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            latency_ms_p95=round(latencies[int(len(latencies) * 0.95) - 1], 2) if latencies else 0.0,
        )

    def run_all(self, dataset: EvalDataset,
                configs: Iterable[RetrievalConfig] = DEFAULT_CONFIGS) -> list[RetrievalRunResult]:
        results: list[RetrievalRunResult] = []
        for config in configs:
            logger.info("retrieval_eval_started", extra={"config": config.label})
            results.append(self.run(dataset, config))
        return results

    def metrics_by_question_type(self, dataset: EvalDataset,
                                 result: RetrievalRunResult) -> dict[str, dict]:
        """Per-type breakdown: where each retriever actually wins or loses."""
        by_id = {q.id: q for q in dataset.questions}
        buckets: dict[str, RetrievalMetrics] = {}
        for record in result.per_question:
            question = by_id[record["id"]]
            if question.is_unanswerable:
                continue
            bucket = buckets.setdefault(question.question_type, RetrievalMetrics(ks=self.ks))
            bucket.update(record["retrieved"], self.resolver.resolve(question))
        return {name: metric.summary() for name, metric in sorted(buckets.items())}


@dataclass
class GenerationRunResult:
    metrics: dict[str, float]
    per_question: list[dict] = field(default_factory=list)
    judge_used: bool = False
    provider: str = ""
    model: str = ""


class GenerationEvaluator:
    def __init__(self, service: RagService, judge: LLMJudge | None = None) -> None:
        self.service = service
        self.judge = judge

    def run(self, dataset: EvalDataset, top_k: int = 5,
            method: RetrievalMethod | None = None,
            rerank: bool | None = None) -> GenerationRunResult:
        builder: ContextBuilder = self.service.context_builder
        records: list[dict] = []
        judge_used = False

        for question in dataset.questions:
            answer = self.service.query(question.question, top_k=top_k,
                                        method=method, rerank=rerank)
            context_text = builder.build(answer.contexts).text if answer.contexts else ""
            scores = score_answer(answer, question, context_text)

            record = {
                "id": question.id, "question": question.question,
                "question_type": question.question_type,
                "answer": answer.answer, "is_no_answer": answer.is_no_answer,
                "citations": [c.model_dump() for c in answer.citations],
                "n_contexts": len(answer.contexts),
                "latency_ms": answer.trace.latency_ms,
                **scores,
            }
            if self.judge is not None:
                verdict = self.judge.score(question.question, context_text, answer.answer)
                if verdict:
                    judge_used = True
                    record.update(verdict)
            records.append(record)

        return GenerationRunResult(
            metrics=self._aggregate(records), per_question=records, judge_used=judge_used,
            provider=self.service.llm.provider, model=self.service.llm.model,
        )

    @staticmethod
    def _aggregate(records: list[dict]) -> dict[str, float]:
        def mean(key: str, subset: list[dict]) -> float:
            values = [r[key] for r in subset if isinstance(r.get(key), (int, float))]
            return round(sum(values) / len(values), 4) if values else 0.0

        answerable = [r for r in records if r["question_type"] != "unanswerable"]
        unanswerable = [r for r in records if r["question_type"] == "unanswerable"]

        summary = {
            "n_questions": len(records),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "groundedness": mean("groundedness", answerable),
            "answer_f1": mean("answer_f1", answerable),
            "span_coverage": mean("span_coverage", answerable),
            "citation_precision": mean("citation_precision", answerable),
            "uncited_sentence_rate": mean("uncited_sentence_rate", answerable),
            "mean_citations": mean("n_citations", answerable),
            "over_refusal_rate": mean("over_refusal", answerable),
            "correct_refusal_rate": mean("correct_refusal", unanswerable),
            "mean_latency_ms": mean("latency_ms", records),
        }
        judged = [r for r in records if "judge_faithfulness" in r]
        if judged:
            summary["judge_faithfulness"] = mean("judge_faithfulness", judged)
            summary["judge_relevance"] = mean("judge_relevance", judged)
            summary["n_judged"] = len(judged)
        return summary
