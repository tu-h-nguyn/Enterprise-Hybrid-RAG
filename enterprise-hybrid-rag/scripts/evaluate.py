#!/usr/bin/env python3
"""Evaluate retrieval (and optionally generation) against a labelled dataset.

    python scripts/evaluate.py                          # all four configs
    python scripts/evaluate.py --audit-only             # check labels, run nothing
    python scripts/evaluate.py --method hybrid --rerank # one config
    python scripts/evaluate.py --generation --out data/evaluation/gen.json

This is a thin CLI over ``RetrievalEvaluator`` / ``GenerationEvaluator``. It
runs the audit first and refuses to report metrics over labels that do not
resolve, because an unresolvable label drives recall to zero and is
indistinguishable from a retrieval failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings
from app.evaluation.dataset import EvalDataset, RelevanceResolver
from app.evaluation.evaluator import (
    DEFAULT_CONFIGS,
    GenerationEvaluator,
    RetrievalConfig,
    RetrievalEvaluator,
)
from app.evaluation.experiments import backend_caveat, describe_backends
from app.evaluation.generation_metrics import LLMJudge
from app.ingestion.pipeline import IngestionPipeline
from app.observability.logging import configure_logging
from app.services.rag_service import RagService


def print_comparison(rows: list[dict]) -> None:
    header = f"{'Configuration':<22}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'nDCG@5':>9}{'ms':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        metrics = row["metrics"]
        print(f"{row['label']:<22}"
              f"{metrics['recall@1']:>8.4f}{metrics['recall@3']:>8.4f}"
              f"{metrics['recall@5']:>8.4f}{metrics['recall@10']:>8.4f}"
              f"{metrics['mrr']:>8.4f}{metrics['ndcg@5']:>9.4f}"
              f"{row['latency_ms_mean']:>9.2f}")


def print_by_type(rows: list[dict]) -> None:
    types = sorted({t for row in rows for t in row.get("by_question_type", {})})
    if not types:
        return
    print("\nRecall@1 by question type")
    print(f"{'Configuration':<22}" + "".join(f"{t:>14}" for t in types))
    print("-" * (22 + 14 * len(types)))
    for row in rows:
        cells = "".join(f"{row['by_question_type'].get(t, {}).get('recall@1', 0.0):>14.4f}"
                        for t in types)
        print(f"{row['label']:<22}{cells}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and generation")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Defaults to data/evaluation/demo_questions.json")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Largest k scored; metrics are reported at 1/3/5/top-k")
    parser.add_argument("--method", choices=["dense", "sparse", "hybrid"], default=None,
                        help="Run a single method instead of the full comparison")
    parser.add_argument("--rerank", action="store_true", help="Rerank (with --method)")
    parser.add_argument("--generation", action="store_true",
                        help="Also score generated answers")
    parser.add_argument("--judge", action="store_true",
                        help="Use the LLM judge (ignored on offline backends)")
    parser.add_argument("--audit-only", action="store_true",
                        help="Audit dataset labels and exit")
    parser.add_argument("--out", type=Path, default=None, help="Write results JSON here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING" if args.quiet else settings.log_level, json_output=False)

    dataset_path = args.dataset or (Path(settings.evaluation_dir) / "demo_questions.json")
    dataset = EvalDataset.load(dataset_path)
    chunks = IngestionPipeline(settings).load_chunks()
    if not chunks:
        print("No chunks found. Run scripts/ingest.py first.", file=sys.stderr)
        return 1

    resolver = RelevanceResolver(chunks)
    problems = resolver.audit(dataset)
    print(f"Dataset : {dataset.name} — {len(dataset)} questions "
          f"({len(dataset.answerable)} answerable, {len(dataset.unanswerable)} unanswerable)")
    print(f"Types   : {json.dumps(dataset.counts_by_type())}")
    print(f"Corpus  : {len(chunks)} chunks")
    print(f"Audit   : {'OK — every label resolves' if not problems else f'{len(problems)} PROBLEM(S)'}")
    for problem in problems:
        print(f"          {problem}")
    if problems:
        print("\nRefusing to report metrics over labels that do not resolve.", file=sys.stderr)
        return 2
    if args.audit_only:
        return 0

    service = RagService.from_disk(settings)
    if not service.is_ready:
        print("No index loaded. Run scripts/ingest.py first.", file=sys.stderr)
        return 1

    backends = describe_backends(service)
    print(f"\nBackends: embedder={backends['embedder']} reranker={backends['reranker']} "
          f"llm={backends['llm']}")
    for note in backend_caveat(backends):
        print(f"  ! {note}")

    ks = tuple(sorted({1, 3, 5, args.top_k}))
    evaluator = RetrievalEvaluator(service, resolver, ks=ks)
    configs = ((RetrievalConfig(f"{args.method}{'+rerank' if args.rerank else ''}",
                                args.method, args.rerank),)
               if args.method else DEFAULT_CONFIGS)

    rows: list[dict] = []
    for config in configs:
        result = evaluator.run(dataset, config)
        rows.append({
            **config.as_dict(), "metrics": result.metrics,
            "latency_ms_mean": result.latency_ms_mean,
            "latency_ms_p95": result.latency_ms_p95,
            "by_question_type": evaluator.metrics_by_question_type(dataset, result),
            "per_question": result.per_question,
        })

    print()
    print_comparison(rows)
    print_by_type(rows)

    payload: dict = {"dataset": dataset.name, "n_questions": len(dataset),
                     "ks": list(ks), "backends": backends, "retrieval": rows}

    if args.generation:
        judge = LLMJudge(service.llm) if args.judge else None
        if judge is not None and not judge.is_available:
            print("\n! LLM judge unavailable on an offline backend; skipping judge scores.")
            judge = None
        generation = GenerationEvaluator(service, judge).run(dataset, top_k=5)
        payload["generation"] = {"provider": generation.provider, "model": generation.model,
                                 "judge_used": generation.judge_used,
                                 "metrics": generation.metrics,
                                 "per_question": generation.per_question}
        print("\nGeneration")
        for key, value in generation.metrics.items():
            print(f"  {key:<24}{value}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
