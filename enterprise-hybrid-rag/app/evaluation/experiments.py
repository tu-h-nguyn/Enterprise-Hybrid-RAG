"""Experiment orchestration (Phase 14).

This module answers the two questions the README has to answer with numbers:

1. **Which retrieval strategy wins?** Four configurations are run over the same
   dataset, through the same ``RagService.retrieve`` the API uses:
   ``A`` dense only, ``B`` BM25 only, ``C`` hybrid RRF, ``D`` hybrid + reranker.
2. **Does chunk size matter?** The corpus is re-ingested and both indexes are
   rebuilt at each chunk size, then all four configurations are re-run.

Two design points are load-bearing:

*   The ablation **re-resolves ground truth at every chunk size**. Labels are
    answer spans, not chunk IDs (see ``app.evaluation.dataset``), so a new
    ``RelevanceResolver`` is constructed from the newly produced chunks. Reusing
    one resolver across sizes would compare retrieval against labels that no
    longer address anything and would read as a catastrophic regression.
*   The ablation builds into a **scratch index directory**, never the serving
    one. An evaluation run must not leave ``data/processed`` holding an index
    built at 256 tokens that the API then quietly serves.

Every artefact records which backends actually produced it, because in a
restricted environment ``auto`` silently degrades to offline fallbacks and
those numbers must never be read as neural-model numbers.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import sys
import tempfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from app.config.settings import Settings
from app.evaluation.dataset import EvalDataset, RelevanceResolver
from app.evaluation.evaluator import (
    DEFAULT_CONFIGS,
    GenerationEvaluator,
    RetrievalConfig,
    RetrievalEvaluator,
    RetrievalRunResult,
)
from app.evaluation.generation_metrics import LLMJudge
from app.evaluation.retrieval_metrics import DEFAULT_KS
from app.indexing.index_builder import IndexBuilder, IndexBundle
from app.ingestion.pipeline import IngestionPipeline
from app.models.document import Chunk
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

RESULTS_FILENAME = "results.json"
REPORT_FILENAME = "report.md"

#: Chunk sizes for the ablation, with an overlap held at a constant 20% so the
#: comparison isolates chunk size rather than confounding it with overlap.
DEFAULT_CHUNK_SIZES: tuple[int, ...] = (256, 500, 800)
OVERLAP_RATIO = 0.2


class ExperimentSuiteConfig(BaseModel):
    """Declarative description of a benchmark run."""

    dataset_path: Path
    ks: tuple[int, ...] = DEFAULT_KS
    chunk_sizes: tuple[int, ...] = DEFAULT_CHUNK_SIZES
    run_chunk_ablation: bool = True
    run_generation: bool = True
    generation_top_k: int = Field(5, ge=1)
    use_llm_judge: bool = False
    out_dir: Path | None = None

    model_config = {"arbitrary_types_allowed": True}


def _overlap_for(chunk_size: int) -> int:
    return max(0, round(chunk_size * OVERLAP_RATIO))


def describe_backends(service: RagService) -> dict[str, object]:
    """Capture exactly what produced a set of numbers.

    ``embedding_backend`` of ``tfidf_svd``, ``reranker.neural == False`` or
    ``llm.provider == 'extractive'`` all mean the run used an offline fallback.
    """
    manifest = service.bundle.manifest
    embedder = dict(manifest.get("embedder", {}))
    reranker = dict(service.reranker.describe())
    llm = dict(service.llm.describe())
    return {
        "embedder": embedder,
        "reranker": reranker,
        "llm": llm,
        "vector_backend": manifest.get("vector_backend", ""),
        "is_neural_embedder": embedder.get("backend") == "sentence_transformers",
        "is_neural_reranker": bool(reranker.get("neural")),
        "is_generative_llm": llm.get("provider") != "extractive",
    }


class ExperimentRunner:
    """Runs the comparison table and the chunk-size ablation, and writes artefacts."""

    def __init__(self, settings: Settings, config: ExperimentSuiteConfig) -> None:
        self.settings = settings
        self.config = config
        self.out_dir = Path(config.out_dir or settings.evaluation_dir)
        self.dataset = EvalDataset.load(config.dataset_path)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _result_payload(result: RetrievalRunResult,
                        by_type: dict[str, dict] | None = None,
                        include_per_question: bool = True) -> dict:
        payload: dict = {
            **result.config.as_dict(),
            "metrics": result.metrics,
            "latency_ms_mean": result.latency_ms_mean,
            "latency_ms_p95": result.latency_ms_p95,
        }
        if by_type is not None:
            payload["by_question_type"] = by_type
        if include_per_question:
            payload["per_question"] = result.per_question
        return payload

    def _evaluate(self, service: RagService, chunks: list[Chunk],
                  configs: Sequence[RetrievalConfig],
                  include_per_question: bool = True,
                  include_by_type: bool = True) -> list[dict]:
        """Run every configuration against one index.

        The resolver is rebuilt from ``chunks`` on purpose — see module docstring.
        """
        resolver = RelevanceResolver(chunks)
        evaluator = RetrievalEvaluator(service, resolver, ks=self.config.ks)
        rows: list[dict] = []
        for retrieval_config in configs:
            result = evaluator.run(self.dataset, retrieval_config)
            by_type = (evaluator.metrics_by_question_type(self.dataset, result)
                       if include_by_type else None)
            rows.append(self._result_payload(result, by_type, include_per_question))
            logger.info("experiment_config_done",
                        extra={"config": retrieval_config.label,
                               "recall@1": result.metrics.get("recall@1"),
                               "mrr": result.metrics.get("mrr")})
        return rows

    # ------------------------------------------------------- main comparison
    def run_retrieval_comparison(
            self, service: RagService, chunks: list[Chunk],
            configs: Iterable[RetrievalConfig] = DEFAULT_CONFIGS) -> list[dict]:
        return self._evaluate(service, chunks, tuple(configs))

    # -------------------------------------------------------- chunk ablation
    def run_chunk_size_ablation(
            self, configs: Iterable[RetrievalConfig] = DEFAULT_CONFIGS) -> list[dict]:
        """Re-ingest and rebuild the indexes at each chunk size.

        Runs entirely inside a temporary directory so the serving index in
        ``data/processed`` is untouched.
        """
        configs = tuple(configs)
        rows: list[dict] = []
        scratch = Path(tempfile.mkdtemp(prefix="rag_ablation_"))
        try:
            for chunk_size in self.config.chunk_sizes:
                overlap = _overlap_for(chunk_size)
                run_dir = scratch / f"chunk_{chunk_size}"
                ablation_settings = self.settings.model_copy(update={
                    "chunk_size_tokens": chunk_size,
                    "chunk_overlap_tokens": overlap,
                    "processed_dir": run_dir,
                    "chroma_dir": run_dir / "chroma",
                    "bm25_dir": run_dir / "bm25",
                })
                ablation_settings.ensure_dirs()

                pipeline = IngestionPipeline(ablation_settings)
                documents, chunks = pipeline.run()
                if not chunks:
                    raise RuntimeError(f"Ablation at chunk_size={chunk_size} produced no chunks")
                bundle: IndexBundle = IndexBuilder(ablation_settings).build(chunks, reset=True)
                service = RagService(ablation_settings, bundle)

                rows.append({
                    "chunk_size_tokens": chunk_size,
                    "chunk_overlap_tokens": overlap,
                    "n_documents": len(documents),
                    "n_chunks": len(chunks),
                    "embedding_dim": int(bundle.manifest.get("embedder", {}).get("dimension") or 0),
                    "mean_chunk_tokens": round(sum(c.n_tokens for c in chunks) / len(chunks), 1),
                    "configs": self._evaluate(service, chunks, configs,
                                              include_per_question=False,
                                              include_by_type=False),
                })
                logger.info("ablation_size_done",
                            extra={"chunk_size": chunk_size, "n_chunks": len(chunks)})
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        return rows

    # ------------------------------------------------------------ generation
    def run_generation(self, service: RagService) -> dict:
        judge = LLMJudge(service.llm) if self.config.use_llm_judge else None
        if judge is not None and not judge.is_available:
            logger.warning("judge_unavailable",
                           extra={"provider": service.llm.provider,
                                  "reason": "offline backend cannot judge"})
            judge = None
        result = GenerationEvaluator(service, judge).run(
            self.dataset, top_k=self.config.generation_top_k)
        return {
            "provider": result.provider,
            "model": result.model,
            "judge_used": result.judge_used,
            "top_k": self.config.generation_top_k,
            "metrics": result.metrics,
            "per_question": result.per_question,
        }

    # ------------------------------------------------------------------- run
    def run(self, service: RagService, chunks: list[Chunk]) -> dict:
        backends = describe_backends(service)
        payload: dict = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "dataset": {
                "name": self.dataset.name,
                "path": str(self.config.dataset_path),
                "n_questions": len(self.dataset),
                "n_answerable": len(self.dataset.answerable),
                "n_unanswerable": len(self.dataset.unanswerable),
                "counts_by_type": self.dataset.counts_by_type(),
            },
            "index": {
                "n_chunks": len(chunks),
                "n_documents": len({c.document_id for c in chunks}),
                "chunk_size_tokens": self.settings.chunk_size_tokens,
                "chunk_overlap_tokens": self.settings.chunk_overlap_tokens,
            },
            "backends": backends,
            "retrieval_settings": {
                "ks": list(self.config.ks),
                "dense_top_k": self.settings.dense_top_k,
                "sparse_top_k": self.settings.sparse_top_k,
                "fusion_top_k": self.settings.fusion_top_k,
                "rrf_k": self.settings.rrf_k,
                "rrf_dense_weight": self.settings.rrf_dense_weight,
                "rrf_sparse_weight": self.settings.rrf_sparse_weight,
                "rerank_top_k": self.settings.rerank_top_k,
            },
        }

        logger.info("experiment_started", extra={"n_questions": len(self.dataset)})
        payload["retrieval"] = self.run_retrieval_comparison(service, chunks)

        if self.config.run_chunk_ablation:
            payload["chunk_size_ablation"] = self.run_chunk_size_ablation()

        if self.config.run_generation:
            payload["generation"] = self.run_generation(service)

        return payload

    # ------------------------------------------------------------ artefacts
    def write_results(self, payload: dict, path: Path | None = None) -> Path:
        target = Path(path or (self.out_dir / RESULTS_FILENAME))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def write_report(self, payload: dict, path: Path | None = None) -> Path:
        target = Path(path or (self.out_dir / REPORT_FILENAME))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_report(payload), encoding="utf-8")
        return target


# --------------------------------------------------------------- rendering
def _fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return "-" if value is None else str(value)


def backend_caveat(backends: dict) -> list[str]:
    """Explicit warnings when a run used an offline fallback.

    Returned as lines rather than printed so the same text appears in the
    report, on stdout and nowhere else by accident.
    """
    notes: list[str] = []
    if not backends.get("is_neural_embedder"):
        embedder = backends.get("embedder", {})
        notes.append(
            f"Embeddings came from the **offline `{embedder.get('backend')}` fallback** "
            f"(`{embedder.get('model')}`, dim {embedder.get('dimension')}), not a neural "
            "sentence encoder. Dense numbers here are lexical-semantic and are a floor, "
            "not a measurement of `all-MiniLM-L6-v2`."
        )
    if not backends.get("is_neural_reranker"):
        notes.append(
            f"The reranker was the **non-neural `{backends.get('reranker', {}).get('backend')}` "
            "fallback**, a feature-based scorer. It is not a cross-encoder and its numbers "
            "must not be reported as `ms-marco-MiniLM-L-6-v2` numbers."
        )
    if not backends.get("is_generative_llm"):
        notes.append(
            "Generation used the **offline extractive backend**, which selects supporting "
            "sentences from the context rather than generating text. Generation metrics "
            "below describe that behaviour, not an LLM's."
        )
    return notes


def _metric_table(rows: list[dict], ks: list[int]) -> list[str]:
    header = ["Configuration", "R@1", "R@3", "R@5", "R@10", "MRR", "nDCG@5", "P@5",
              "mean ms", "p95 ms"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        metrics = row["metrics"]
        lines.append("| " + " | ".join([
            row["label"],
            _fmt(metrics.get("recall@1")), _fmt(metrics.get("recall@3")),
            _fmt(metrics.get("recall@5")), _fmt(metrics.get("recall@10")),
            _fmt(metrics.get("mrr")), _fmt(metrics.get("ndcg@5")),
            _fmt(metrics.get("precision@5")),
            _fmt(row.get("latency_ms_mean"), 2), _fmt(row.get("latency_ms_p95"), 2),
        ]) + " |")
    return lines


def render_report(payload: dict) -> str:
    backends = payload.get("backends", {})
    dataset = payload.get("dataset", {})
    index = payload.get("index", {})
    ks = payload.get("retrieval_settings", {}).get("ks", list(DEFAULT_KS))

    lines: list[str] = [
        "# Retrieval benchmark report",
        "",
        f"Generated: `{payload.get('generated_at')}`  ",
        f"Dataset: `{dataset.get('name')}` — {dataset.get('n_questions')} questions "
        f"({dataset.get('n_answerable')} answerable, {dataset.get('n_unanswerable')} unanswerable)  ",
        f"Index: {index.get('n_documents')} documents, {index.get('n_chunks')} chunks at "
        f"`chunk_size_tokens={index.get('chunk_size_tokens')}` / "
        f"`chunk_overlap_tokens={index.get('chunk_overlap_tokens')}`",
        "",
        "## Backends that produced these numbers",
        "",
        "| Component | Backend | Detail |",
        "|---|---|---|",
        f"| Embedding | `{backends.get('embedder', {}).get('backend')}` | "
        f"{backends.get('embedder', {}).get('model')}, dim "
        f"{backends.get('embedder', {}).get('dimension')} |",
        f"| Reranker | `{backends.get('reranker', {}).get('backend')}` | "
        f"neural: {backends.get('reranker', {}).get('neural')} |",
        f"| LLM | `{backends.get('llm', {}).get('provider')}` | "
        f"{backends.get('llm', {}).get('model')} |",
        f"| Vector store | `{backends.get('vector_backend')}` | |",
        "",
    ]

    caveats = backend_caveat(backends)
    if caveats:
        lines.append("> **Read this before quoting any number below.**")
        lines.append(">")
        for note in caveats:
            lines.append(f"> * {note}")
        lines.append("")

    lines += [
        "## Question type mix",
        "",
        "| Type | Count |",
        "|---|---|",
    ]
    for name, count in (dataset.get("counts_by_type") or {}).items():
        lines.append(f"| {name} | {count} |")
    lines.append("")

    lines += ["## Retrieval comparison", "",
              "Recall is computed over the answer-span ground truth resolved at evaluation "
              "time. Unanswerable questions carry no relevant chunks and are excluded from "
              "these metrics (they are scored by refusal behaviour under *Generation*).", ""]
    lines += _metric_table(payload.get("retrieval", []), ks)
    lines.append("")

    by_type_rows = [r for r in payload.get("retrieval", []) if r.get("by_question_type")]
    if by_type_rows:
        types = sorted({t for r in by_type_rows for t in r["by_question_type"]})
        lines += ["### Recall@1 by question type", "",
                  "This is where the retrievers actually differ: keyword questions turn on "
                  "rare reference codes, paraphrased questions share no vocabulary with the "
                  "source passage.", "",
                  "| Configuration | " + " | ".join(types) + " |",
                  "|" + "|".join(["---"] * (len(types) + 1)) + "|"]
        for row in by_type_rows:
            cells = [_fmt(row["by_question_type"].get(t, {}).get("recall@1")) for t in types]
            lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
        lines.append("")

        lines += ["### MRR by question type", "",
                  "| Configuration | " + " | ".join(types) + " |",
                  "|" + "|".join(["---"] * (len(types) + 1)) + "|"]
        for row in by_type_rows:
            cells = [_fmt(row["by_question_type"].get(t, {}).get("mrr")) for t in types]
            lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
        lines.append("")

    ablation = payload.get("chunk_size_ablation") or []
    if ablation:
        lines += ["## Chunk-size ablation", "",
                  "The corpus is re-ingested and both indexes are rebuilt at each size, and "
                  "ground truth is re-resolved against the new chunks (labels are answer "
                  "spans, not chunk IDs, precisely so this comparison stays valid).", "",
                  "| Chunk size | Overlap | Chunks | Mean tokens | Embed dim | Configuration | "
                  "R@1 | R@5 | MRR | nDCG@5 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in ablation:
            for i, cfg in enumerate(row.get("configs", [])):
                metrics = cfg["metrics"]
                prefix = ([str(row["chunk_size_tokens"]), str(row["chunk_overlap_tokens"]),
                           str(row["n_chunks"]), str(row["mean_chunk_tokens"]),
                           str(row["embedding_dim"])] if i == 0 else ["", "", "", "", ""])
                lines.append("| " + " | ".join([
                    *prefix,
                    cfg["label"], _fmt(metrics.get("recall@1")), _fmt(metrics.get("recall@5")),
                    _fmt(metrics.get("mrr")), _fmt(metrics.get("ndcg@5")),
                ]) + " |")
        lines.append("")

    generation = payload.get("generation")
    if generation:
        metrics = generation.get("metrics", {})
        lines += ["## Generation", "",
                  f"Provider: `{generation.get('provider')}` / `{generation.get('model')}`, "
                  f"top_k={generation.get('top_k')}, LLM judge used: "
                  f"{generation.get('judge_used')}", "",
                  "| Metric | Value |", "|---|---|"]
        for key in ("n_questions", "n_answerable", "n_unanswerable", "groundedness",
                    "answer_f1", "span_coverage", "citation_precision",
                    "uncited_sentence_rate", "mean_citations", "over_refusal_rate",
                    "correct_refusal_rate", "mean_latency_ms"):
            if key in metrics:
                lines.append(f"| {key} | {_fmt(metrics[key])} |")
        if "judge_faithfulness" in metrics:
            lines.append(f"| judge_faithfulness | {_fmt(metrics['judge_faithfulness'])} |")
            lines.append(f"| judge_relevance | {_fmt(metrics['judge_relevance'])} |")
        lines.append("")

    lines += ["---", "",
              "Produced by `scripts/benchmark.py`. Raw per-question records are in "
              f"`{RESULTS_FILENAME}`.", ""]
    return "\n".join(lines)
