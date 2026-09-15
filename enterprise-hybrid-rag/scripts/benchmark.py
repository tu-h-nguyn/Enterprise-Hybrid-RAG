#!/usr/bin/env python3
"""Run the full benchmark suite and write results.json + report.md.

    python scripts/benchmark.py                       # everything
    python scripts/benchmark.py --no-ablation         # comparison table only
    python scripts/benchmark.py --chunk-sizes 256 500 800
    python scripts/benchmark.py --out data/evaluation

Thin CLI over ``app.evaluation.experiments``. The chunk-size ablation rebuilds
the indexes in a temporary directory, so the index in ``data/processed`` that
the API serves is left exactly as it was found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings  # noqa: E402
from app.evaluation.dataset import EvalDataset, RelevanceResolver  # noqa: E402
from app.evaluation.experiments import (DEFAULT_CHUNK_SIZES, ExperimentRunner,  # noqa: E402
                                        ExperimentSuiteConfig, backend_caveat,
                                        describe_backends)
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402


def print_comparison(rows: list[dict]) -> None:
    header = (f"{'Configuration':<22}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'R@10':>8}"
              f"{'MRR':>8}{'nDCG@5':>9}{'P@5':>8}{'ms':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        m = row["metrics"]
        print(f"{row['label']:<22}{m['recall@1']:>8.4f}{m['recall@3']:>8.4f}"
              f"{m['recall@5']:>8.4f}{m['recall@10']:>8.4f}{m['mrr']:>8.4f}"
              f"{m['ndcg@5']:>9.4f}{m['precision@5']:>8.4f}{row['latency_ms_mean']:>9.2f}")


def print_ablation(rows: list[dict]) -> None:
    header = (f"{'Chunk size':>11}{'Chunks':>8}{'Dim':>6}  {'Configuration':<22}"
              f"{'R@1':>8}{'R@5':>8}{'MRR':>8}")
    print(header)
    print("-" * len(header))
    for row in rows:
        for i, cfg in enumerate(row["configs"]):
            m = cfg["metrics"]
            left = (f"{row['chunk_size_tokens']:>11}{row['n_chunks']:>8}{row['embedding_dim']:>6}"
                    if i == 0 else " " * 25)
            print(f"{left}  {cfg['label']:<22}{m['recall@1']:>8.4f}"
                  f"{m['recall@5']:>8.4f}{m['mrr']:>8.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Full retrieval benchmark and ablation")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default data/evaluation)")
    parser.add_argument("--chunk-sizes", nargs="*", type=int, default=list(DEFAULT_CHUNK_SIZES))
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING" if args.quiet else settings.log_level, json_output=False)

    dataset_path = args.dataset or (Path(settings.evaluation_dir) / "demo_questions.json")
    chunks = IngestionPipeline(settings).load_chunks()
    if not chunks:
        print("No chunks found. Run scripts/ingest.py first.", file=sys.stderr)
        return 1

    dataset = EvalDataset.load(dataset_path)
    problems = RelevanceResolver(chunks).audit(dataset)
    if problems:
        print(f"Dataset audit failed with {len(problems)} problem(s); refusing to benchmark:",
              file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 2

    service = RagService.from_disk(settings)
    if not service.is_ready:
        print("No index loaded. Run scripts/ingest.py first.", file=sys.stderr)
        return 1

    config = ExperimentSuiteConfig(
        dataset_path=dataset_path,
        ks=tuple(sorted({1, 3, 5, args.top_k})),
        chunk_sizes=tuple(args.chunk_sizes),
        run_chunk_ablation=not args.no_ablation,
        run_generation=not args.no_generation,
        use_llm_judge=args.judge,
        out_dir=args.out,
    )
    runner = ExperimentRunner(settings, config)

    backends = describe_backends(service)
    print(f"Dataset : {dataset.name} — {len(dataset)} questions")
    print(f"Corpus  : {len(chunks)} chunks at chunk_size={settings.chunk_size_tokens}")
    print(f"Embedder: {backends['embedder']}")
    print(f"Reranker: {backends['reranker']}")
    print(f"LLM     : {backends['llm']}")
    for note in backend_caveat(backends):
        print(f"  ! {note}")
    print()

    payload = runner.run(service, chunks)

    print("\n=== Retrieval comparison ===")
    print_comparison(payload["retrieval"])
    if payload.get("chunk_size_ablation"):
        print("\n=== Chunk-size ablation ===")
        print_ablation(payload["chunk_size_ablation"])
    if payload.get("generation"):
        print("\n=== Generation ===")
        for key, value in payload["generation"]["metrics"].items():
            print(f"  {key:<24}{value}")

    results_path = runner.write_results(payload)
    report_path = runner.write_report(payload)
    print(f"\nWrote {results_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
