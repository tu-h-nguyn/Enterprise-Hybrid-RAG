#!/usr/bin/env python3
"""Does the hybrid + reranker advantage survive a bigger haystack?

    python scripts/scale_experiment.py                      # 44 -> ~5000 chunks
    python scripts/scale_experiment.py --sizes 0 140        # quick check
    python scripts/scale_experiment.py --out data/evaluation

The question
------------
The labelled corpus is 22 documents / 44 chunks. At that size ``Recall@5`` is
close to meaningless — five chunks is more than a tenth of everything there is —
and a reader is entitled to ask whether the headline table only holds because
the haystack is a haybale.

The experiment
--------------
The 50 questions, their answer spans and every retrieval setting are held fixed.
The only variable is how much *other* plausible material sits in the index:
in-domain distractor documents from ``scripts/make_distractor_corpus.py``, on
the same topics and in the same register as the real corpus, for other fictional
companies. Corpus sizes are nested prefixes of one deterministic distractor set,
so each larger corpus strictly contains the smaller one.

Everything is built in a temporary directory. The index in ``data/processed``
that the API serves is never touched.

Why the numbers can be trusted
------------------------------
*   **Ground truth cannot drift.** Relevance is resolved per source document
    (``app.evaluation.dataset``), so a distractor is structurally incapable of
    being labelled relevant. The run additionally refuses to start if any
    distractor chunk contains a labelled answer span — that would be a correct
    answer scored as a miss.
*   **The gold chunks stay identical.** The real corpus is re-ingested at every
    size with unchanged chunking settings; the run asserts that the resolved
    gold chunk set is byte-identical to the one at size zero. If it were not,
    a fall in Recall@1 could be a labelling artefact rather than a scale effect.
*   **Distractor hardness is measured, not asserted.** Each row reports how
    often a distractor takes rank 1 and what share of the top 5 they occupy. A
    corpus of easy negatives would show near-zero interference and the
    experiment would prove nothing; that is visible in the output rather than
    hidden by it.
*   **The backend is recorded.** ``auto`` degrades to offline fallbacks without
    network access, and those numbers must never be read as MiniLM numbers.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

# The project root for ``app.*`` and this directory for the sibling generator,
# so the module imports the same way whether it is run as a script or
# imported by a test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config.settings import Settings, get_settings
from app.evaluation.dataset import EvalDataset, RelevanceResolver, normalize
from app.evaluation.evaluator import (
    DEFAULT_CONFIGS,
    RetrievalConfig,
    RetrievalEvaluator,
)
from app.evaluation.experiments import backend_caveat, describe_backends
from app.indexing.index_builder import IndexBuilder
from app.ingestion.pipeline import IngestionPipeline
from app.models.document import Chunk
from app.observability.logging import configure_logging
from app.services.rag_service import RagService
from make_distractor_corpus import generate as generate_distractors

VectorBackend = Literal["chroma", "numpy"]

RESULTS_FILENAME = "scale_results.json"
REPORT_FILENAME = "scale_report.md"

#: Distractor *document* counts. At ~1.5 chunks per document these land near
#: 44, 250, 1000 and 5000 total chunks — roughly a decade of corpus growth,
#: which is the range where a reader's "but does it scale?" actually bites.
DEFAULT_SIZES: tuple[int, ...] = (0, 140, 655, 3400)
KS: tuple[int, ...] = (1, 3, 5, 10)

#: Both vector stores, at every size. ``chroma`` is HNSW — an *approximate*
#: index — and ``numpy`` is exhaustive cosine. At 44 chunks they agree because
#: HNSW visits everything anyway; the gap that opens as the corpus grows is the
#: price of approximate search, and is worth reporting rather than assuming.
DEFAULT_VECTOR_BACKENDS: tuple[VectorBackend, ...] = ("numpy", "chroma")

#: Overlap as a share of chunk size, held constant so a chunk-size comparison
#: isolates chunk size rather than confounding it with overlap. Same ratio the
#: chunk-size ablation in ``app.evaluation.experiments`` uses.
OVERLAP_RATIO = 0.2

#: ``--extra-chunk-sizes`` runs against exhaustive search only. Whether the
#: approximate index costs recall is a separate question, already answered at
#: the configured chunk size, and re-asking it at every extra size would double
#: the run for nothing.
EXTRA_CHUNK_SIZE_STORE: VectorBackend = "numpy"


def _overlap_for(chunk_size: int) -> int:
    return max(0, round(chunk_size * OVERLAP_RATIO))


def _run_plan(chunk_sizes: list[int], vector_backends: list[VectorBackend],
              configured_chunk_size: int) -> list[tuple[int, VectorBackend]]:
    """Which (chunk size, vector store) pairs to run, and in what order.

    The configured chunk size is run against every requested store, because the
    exact-vs-approximate comparison is one of the things the experiment is for.
    Any *extra* chunk size is run against exhaustive search only — see
    ``EXTRA_CHUNK_SIZE_STORE``.
    """
    plan: list[tuple[int, VectorBackend]] = []
    for chunk_size in chunk_sizes:
        if chunk_size == configured_chunk_size:
            plan.extend((chunk_size, store) for store in vector_backends)
        elif EXTRA_CHUNK_SIZE_STORE in vector_backends:
            plan.append((chunk_size, EXTRA_CHUNK_SIZE_STORE))
        else:
            plan.append((chunk_size, vector_backends[0]))
    return plan


def _backend_identity(backends: dict) -> tuple[str, str, str]:
    """Which models are in play, ignoring the corpus-dependent embedding width."""
    return (str(backends.get("embedder", {}).get("backend")),
            str(backends.get("reranker", {}).get("backend")),
            str(backends.get("llm", {}).get("provider")))


class ContaminationError(RuntimeError):
    """A distractor chunk restates a labelled answer — the run would be unfair."""


def _assert_uncontaminated(distractor_chunks: list[Chunk], dataset: EvalDataset) -> None:
    spans = [normalize(span) for question in dataset.questions
             for span in question.answer_spans if span.strip()]
    for chunk in distractor_chunks:
        text = normalize(chunk.text)
        for span in spans:
            if span in text:
                raise ContaminationError(
                    f"Distractor chunk {chunk.chunk_id} ({chunk.source}) contains the "
                    f"labelled answer span {span!r}. It would be a correct answer scored "
                    "as a miss. Regenerate the distractor corpus with a different seed."
                )


def _interference(per_question: list[dict], distractor_ids: set[str],
                  answerable_ids: set[str]) -> dict[str, float]:
    """How much the distractors actually get in the way.

    ``at_rank_1`` is the share of answerable questions whose top hit is a
    distractor; ``share_at_5`` is the mean fraction of the top 5 they occupy.
    Both are zero by construction at corpus size 0.
    """
    rank_one = 0
    share_total = 0.0
    counted = 0
    for record in per_question:
        if record["id"] not in answerable_ids:
            continue
        counted += 1
        retrieved = record["retrieved"]
        if retrieved and retrieved[0] in distractor_ids:
            rank_one += 1
        top5 = retrieved[:5]
        if top5:
            share_total += sum(1 for cid in top5 if cid in distractor_ids) / len(top5)
    if not counted:
        return {"at_rank_1": 0.0, "share_at_5": 0.0}
    return {"at_rank_1": round(rank_one / counted, 4),
            "share_at_5": round(share_total / counted, 4)}


def run_one_size(settings: Settings, dataset: EvalDataset, real_paths: list[Path],
                 distractor_paths: list[Path], scratch: Path,
                 configs: tuple[RetrievalConfig, ...],
                 baseline_gold: dict[str, set[str]] | None,
                 vector_backend: VectorBackend = "chroma",
                 chunk_size: int | None = None
                 ) -> tuple[dict, dict[str, set[str]], dict]:
    """Ingest, index and evaluate one corpus size in an isolated directory."""
    chunk_size = chunk_size or settings.chunk_size_tokens
    overlap = _overlap_for(chunk_size)
    run_dir = scratch / f"{vector_backend}_c{chunk_size}_n{len(distractor_paths):05d}"
    run_settings = settings.model_copy(update={
        "vector_backend": vector_backend,
        "chunk_size_tokens": chunk_size,
        "chunk_overlap_tokens": overlap,
        "processed_dir": run_dir,
        "chroma_dir": run_dir / "chroma",
        "bm25_dir": run_dir / "bm25",
    })
    run_settings.ensure_dirs()

    pipeline = IngestionPipeline(run_settings)
    _, real_chunks = pipeline.run(paths=real_paths)
    _, distractor_chunks = (pipeline.run(paths=distractor_paths)
                            if distractor_paths else ([], []))
    _assert_uncontaminated(distractor_chunks, dataset)

    chunks = real_chunks + distractor_chunks
    resolver = RelevanceResolver(chunks)
    gold = {q.id: resolver.resolve(q) for q in dataset.answerable}
    if baseline_gold is not None and gold != baseline_gold:
        changed = sorted(qid for qid in gold if gold[qid] != baseline_gold.get(qid, set()))
        raise RuntimeError(
            "Gold chunk sets changed when the corpus grew, so the metrics would not be "
            f"comparable across sizes. Affected questions: {changed[:5]}"
            f" (chunk_size={chunk_size})"
        )

    build_started = time.perf_counter()
    bundle = IndexBuilder(run_settings).build(chunks, reset=True)
    build_seconds = round(time.perf_counter() - build_started, 2)

    service = RagService(run_settings, bundle)
    evaluator = RetrievalEvaluator(service, resolver, ks=KS)
    distractor_ids = {c.chunk_id for c in distractor_chunks}
    answerable_ids = {q.id for q in dataset.answerable}

    rows: list[dict] = []
    for config in configs:
        result = evaluator.run(dataset, config)
        rows.append({
            **config.as_dict(),
            "metrics": result.metrics,
            "latency_ms_mean": result.latency_ms_mean,
            "latency_ms_p95": result.latency_ms_p95,
            "distractor_interference": _interference(result.per_question, distractor_ids,
                                                     answerable_ids),
        })

    payload = {
        "vector_backend": vector_backend,
        "chunk_size_tokens": chunk_size,
        "chunk_overlap_tokens": overlap,
        # The offline tfidf_svd fallback sizes itself from the corpus
        # (``min(target_dim, rank - 1)``), so this is not constant across sizes
        # the way a neural encoder's 384 is. Recorded per row, and flagged in
        # the report, because a changing embedder is a confound.
        "embedding_dim": int(bundle.manifest.get("embedder", {}).get("dimension") or 0),
        "n_distractor_documents": len(distractor_paths),
        "n_documents": len({c.document_id for c in chunks}),
        "n_chunks": len(chunks),
        "n_real_chunks": len(real_chunks),
        "n_distractor_chunks": len(distractor_chunks),
        "n_gold_chunks": len({cid for ids in gold.values() for cid in ids}),
        "gold_share_of_corpus": round(
            len({cid for ids in gold.values() for cid in ids}) / len(chunks), 6),
        "index_build_seconds": build_seconds,
        "configs": rows,
    }
    # Taken from the live service rather than reloaded from disk: the reload
    # path depends on the vector backend, and the backends are a property of
    # the run, not of what happens to be on disk afterwards.
    return payload, gold, describe_backends(service)


def print_table(sizes: list[dict]) -> None:
    header = (f"{'Chunk':>6}{'Store':>8}{'Chunks':>8}  {'Configuration':<22}{'R@1':>8}"
              f"{'R@5':>8}{'MRR':>8}{'ms':>9}{'distr@1':>9}{'distr%@5':>10}")
    print(header)
    print("-" * len(header))
    for size in sizes:
        for i, config in enumerate(size["configs"]):
            m = config["metrics"]
            interference = config["distractor_interference"]
            left = (f"{size['chunk_size_tokens']:>6}{size['vector_backend']:>8}"
                    f"{size['n_chunks']:>8}" if i == 0 else " " * 22)
            print(f"{left}  {config['label']:<22}{m['recall@1']:>8.4f}{m['recall@5']:>8.4f}"
                  f"{m['mrr']:>8.4f}{config['latency_ms_mean']:>9.2f}"
                  f"{interference['at_rank_1']:>9.4f}{interference['share_at_5']:>10.4f}")


def render_report(payload: dict) -> str:
    backends = payload["backends"]
    lines = [
        "# Corpus-scale experiment",
        "",
        f"Generated {payload['generated_at']} · "
        f"{payload['dataset']['n_questions']} questions "
        f"({payload['dataset']['n_answerable']} answerable), held fixed throughout.",
        "",
        f"* Embedder: `{backends['embedder'].get('backend')}` / "
        f"`{backends['embedder'].get('model')}` (dim {backends['embedder'].get('dimension')})",
        f"* Reranker: `{backends['reranker'].get('backend')}` "
        f"(neural: {backends['reranker'].get('neural')})",
        f"* Vector stores: {', '.join(f'`{b}`' for b in payload['vector_backends'])}",
        f"* Distractor corpus: seed {payload['distractors']['seed']}, "
        f"{payload['distractors']['n_available']} documents available",
        "",
    ]
    for note in backend_caveat(backends):
        lines += [f"> ⚠️ {note}", ""]

    dims = sorted({row["embedding_dim"] for row in payload["sizes"]})
    if len(dims) > 1:
        lines += [
            f"> ⚠️ The embedding dimension was not constant across sizes ({dims}). The "
            "offline `tfidf_svd` fallback derives its width from the corpus, so these "
            "rows vary the encoder as well as the corpus and the two effects cannot be "
            "separated. A run on `sentence_transformers` holds the dimension at 384 and "
            "is the one to read.",
            "",
        ]

    lines += [
        "## Method",
        "",
        "The 50 labelled questions, their answer spans and every retrieval setting are",
        "held fixed. Only the size of the haystack changes: in-domain distractor",
        "documents on the same topics and in the same register as the real corpus, for",
        "other fictional companies. Sizes are nested, so each corpus contains the",
        "previous one. The run aborts if a distractor contains a labelled answer span,",
        "or if the resolved gold chunks differ from the smallest corpus.",
        "",
        "Every size is run against both vector stores. `numpy` is exhaustive cosine and",
        "is therefore exact; `chroma` is HNSW and is approximate. The difference between",
        "the two rows at the same size is the recall the approximate index gives up.",
        "",
        "`distr@1` is the share of answerable questions whose top hit is a distractor;",
        "`distr%@5` is the mean share of the top 5 they occupy. These are the evidence",
        "that the added documents are genuinely competitive rather than easy padding.",
        "",
        "## Results",
        "",
        "| Chunk | Store | Chunks | Gold share | Configuration | R@1 | R@5 | MRR "
        "| mean ms | distr@1 | distr%@5 |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for size in payload["sizes"]:
        for i, config in enumerate(size["configs"]):
            m = config["metrics"]
            interference = config["distractor_interference"]
            left = (f"| {size['chunk_size_tokens']} | `{size['vector_backend']}` "
                    f"| {size['n_chunks']} | "
                    f"{size['gold_share_of_corpus'] * 100:.2f}% "
                    if i == 0 else "|  |  |  |  ")
            lines.append(
                f"{left}| {config['label']} | {m['recall@1']:.4f} | {m['recall@5']:.4f} | "
                f"{m['mrr']:.4f} | {config['latency_ms_mean']:.2f} | "
                f"{interference['at_rank_1']:.4f} | {interference['share_at_5']:.4f} |"
            )

    exact_vs_ann = _ann_cost_rows(payload)
    if exact_vs_ann:
        lines += ["", "## What approximate search costs", "",
                  "Recall@1 under exhaustive cosine (`numpy`) minus Recall@1 under HNSW",
                  "(`chroma`), at the same corpus size. A positive number is recall the",
                  "approximate index lost. BM25 never touches the vector store, so its",
                  "rows must read zero — they are the control that says nothing else",
                  "changed between the two runs.", "",
                  "| Chunks | Configuration | exact R@1 | HNSW R@1 | lost |",
                  "|---:|---|---:|---:|---:|"]
        for row in exact_vs_ann:
            lines.append(f"| {row['n_chunks']} | {row['label']} | {row['exact']:.4f} | "
                         f"{row['approximate']:.4f} | {row['lost']:+.4f} |")

    chunk_rows = _chunk_size_rows(payload)
    if chunk_rows:
        configured = payload["chunking"]["configured_chunk_size_tokens"]
        lines += ["", "## Does the chunk size that wins on a small corpus still win?", "",
                  "Recall@1 for the production configuration (hybrid + reranker) under",
                  "exhaustive search, at each chunk size. Corpus sizes are matched by how",
                  "many distractor *documents* were added, because a different chunk size",
                  "turns the same documents into a different number of chunks — that",
                  "count is shown for each.", "",
                  "| Distractor docs | "
                  + " | ".join(f"{c} tok — chunks / R@1 / MRR" for c in chunk_rows["sizes"])
                  + " | best |",
                  "|---:|" + "---:|" * len(chunk_rows["sizes"]) + "---|"]
        for row in chunk_rows["rows"]:
            cells = []
            for chunk_size in chunk_rows["sizes"]:
                cell = row["by_chunk"].get(chunk_size)
                cells.append("—" if cell is None else
                             f"{cell['n_chunks']} / {cell['recall@1']:.4f} / "
                             f"{cell['mrr']:.4f}")
            best = row["best"]
            lines.append(f"| {row['n_distractor_documents']} | " + " | ".join(cells)
                         + f" | {'tie' if best is None else f'{best} tok'} |")
        lines += ["",
                  f"The shipped default is {configured} tokens. A chunk size that wins "
                  "only at the smallest corpus is a finding about the corpus, not about "
                  "chunking — which is the whole reason this table is not just the "
                  "chunk-size ablation run again."]

    lines += ["", "## Index build cost", "",
              "| Store | Chunks | Documents | Embedding dim | Index build (s) |",
              "|---|---:|---:|---:|---:|"]
    for size in payload["sizes"]:
        lines.append(f"| `{size['vector_backend']}` | {size['n_chunks']} | "
                     f"{size['n_documents']} | {size['embedding_dim']} | "
                     f"{size['index_build_seconds']:.2f} |")
    lines += ["",
              "Distractors are synthetic. They are hard negatives by construction — same",
              "topics, same register, same policy vocabulary — but a real corpus of this",
              "size would contain both easier negatives (off-topic material) and harder",
              "ones (near-duplicate revisions of the same policy).",
              ""]
    return "\n".join(lines)


def _chunk_size_rows(payload: dict) -> dict | None:
    """Hybrid + reranker under exhaustive search, one row per corpus size.

    Returns ``None`` when only one chunk size was run, so the section is simply
    absent rather than a table with one column.
    """
    label = "Hybrid + Reranker"
    chunk_sizes = sorted({row["chunk_size_tokens"] for row in payload["sizes"]})
    if len(chunk_sizes) < 2:
        return None

    by_docs: dict[int, dict[int, dict]] = {}
    for row in payload["sizes"]:
        if row["vector_backend"] != "numpy":
            continue
        config = next((c for c in row["configs"] if c["label"] == label), None)
        if config is None:
            continue
        by_docs.setdefault(row["n_distractor_documents"], {})[row["chunk_size_tokens"]] = {
            "n_chunks": row["n_chunks"],
            "recall@1": config["metrics"]["recall@1"],
            "mrr": config["metrics"]["mrr"],
        }

    rows = []
    for n_docs in sorted(by_docs):
        by_chunk = by_docs[n_docs]
        top = max(by_chunk.values(), key=lambda cell: cell["recall@1"])["recall@1"]
        winners = [c for c, cell in by_chunk.items() if cell["recall@1"] == top]
        rows.append({"n_distractor_documents": n_docs, "by_chunk": by_chunk,
                     "best": winners[0] if len(winners) == 1 else None})
    return {"sizes": chunk_sizes, "rows": rows}


def _ann_cost_rows(payload: dict) -> list[dict]:
    """Pair up the exact and approximate runs at each size, where both exist."""
    exact = {(row["chunk_size_tokens"], row["n_chunks"], c["label"]):
             c["metrics"]["recall@1"]
             for row in payload["sizes"] if row["vector_backend"] == "numpy"
             for c in row["configs"]}
    approximate = {(row["chunk_size_tokens"], row["n_chunks"], c["label"]):
                   c["metrics"]["recall@1"]
                   for row in payload["sizes"] if row["vector_backend"] == "chroma"
                   for c in row["configs"]}
    rows = []
    for key in sorted(set(exact) & set(approximate)):
        _, n_chunks, label = key
        rows.append({"n_chunks": n_chunks, "label": label,
                     "exact": exact[key], "approximate": approximate[key],
                     "lost": exact[key] - approximate[key]})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval quality as the corpus grows",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--sizes", nargs="*", type=int, default=list(DEFAULT_SIZES),
                        help="Distractor document counts (nested prefixes)")
    parser.add_argument("--vector-backends", nargs="*", default=list(DEFAULT_VECTOR_BACKENDS),
                        choices=list(get_args(VectorBackend)),
                        help="Vector stores to run at every size (default: both)")
    parser.add_argument("--extra-chunk-sizes", nargs="*", type=int, default=[],
                        metavar="TOKENS",
                        help="Also run the whole sweep at these chunk sizes, in addition "
                             "to the configured one, against exhaustive search")
    parser.add_argument("--seed", type=int, default=20250915)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING" if args.quiet else settings.log_level, json_output=False)

    dataset_path = args.dataset or (Path(settings.evaluation_dir) / "demo_questions.json")
    dataset = EvalDataset.load(dataset_path)
    out_dir = Path(args.out or settings.evaluation_dir)

    real_paths = IngestionPipeline(settings).discover(settings.raw_dir)
    if not real_paths:
        print(f"No source documents in {settings.raw_dir}. Run scripts/make_demo_corpus.py.",
              file=sys.stderr)
        return 1

    sizes = sorted({max(0, n) for n in args.sizes})
    vector_backends: list[VectorBackend] = [b for b in DEFAULT_VECTOR_BACKENDS
                                            if b in set(args.vector_backends)]
    chunk_sizes = [settings.chunk_size_tokens]
    chunk_sizes += [c for c in dict.fromkeys(args.extra_chunk_sizes)
                    if c > 0 and c != settings.chunk_size_tokens]
    scratch = Path(tempfile.mkdtemp(prefix="rag_scale_"))
    try:
        distractor_dir = scratch / "distractors"
        available = generate_distractors(distractor_dir, max(sizes) or 1, args.seed)
        if max(sizes) > len(available):
            print(f"Requested {max(sizes)} distractors but only {len(available)} exist.",
                  file=sys.stderr)
            return 1

        print(f"Dataset    : {dataset.name} — {len(dataset)} questions")
        print(f"Real corpus: {len(real_paths)} documents from {settings.raw_dir}")
        print(f"Distractors: {len(available)} generated (seed {args.seed})")
        print(f"Sizes      : {sizes} distractor documents")
        print(f"Stores     : {vector_backends}")
        print(f"Chunk sizes: {chunk_sizes} "
              f"(configured {settings.chunk_size_tokens})\n")

        rows: list[dict] = []
        # Gold chunk IDs are resolved from answer spans against the chunks a
        # given chunking produced, so they legitimately differ between chunk
        # sizes and must only be held constant *within* one.
        baseline_gold: dict[int, dict[str, set[str]]] = {}
        backends: dict | None = None
        for chunk_size, vector_backend in _run_plan(chunk_sizes, vector_backends,
                                                    settings.chunk_size_tokens):
            for n in sizes:
                started = time.perf_counter()
                payload, gold, run_backends = run_one_size(
                    settings, dataset, real_paths, available[:n], scratch,
                    DEFAULT_CONFIGS, baseline_gold.get(chunk_size), vector_backend,
                    chunk_size)
                baseline_gold.setdefault(chunk_size, gold)
                # The vector store and the chunk size are the variables here;
                # which embedder, reranker and LLM are in play must not change
                # under us, or the rows stop being comparable. The embedder's
                # *dimension* is allowed to move, because the offline fallback
                # derives it from the corpus; that is recorded per row and
                # flagged in the report.
                identity = _backend_identity(run_backends)
                if backends is None:
                    backends = {k: v for k, v in run_backends.items()
                                if k != "vector_backend"}
                elif identity != _backend_identity(backends):
                    raise RuntimeError(
                        "Model backends changed between runs, so the rows are not "
                        f"comparable: {_backend_identity(backends)} -> {identity}")
                rows.append(payload)
                print(f"  chunk {chunk_size:>4}  {vector_backend:<7} "
                      f"size {n:>5} distractors -> {payload['n_chunks']:>5} chunks "
                      f"in {time.perf_counter() - started:.1f}s")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    assert backends is not None
    result = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "dataset": {"name": dataset.name, "path": str(dataset_path),
                    "n_questions": len(dataset), "n_answerable": len(dataset.answerable)},
        "distractors": {"seed": args.seed, "n_available": len(available),
                        "generator": "scripts/make_distractor_corpus.py"},
        "backends": backends,
        "vector_backends": vector_backends,
        "chunking": {"configured_chunk_size_tokens": settings.chunk_size_tokens,
                     "configured_chunk_overlap_tokens": settings.chunk_overlap_tokens,
                     "chunk_sizes_run": chunk_sizes,
                     "overlap_ratio": OVERLAP_RATIO},
        "ks": list(KS),
        "sizes": rows,
    }

    print("\n=== Retrieval quality vs corpus size ===")
    print_table(rows)
    for note in backend_caveat(backends):
        print(f"\n! {note}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / RESULTS_FILENAME
    report_path = out_dir / REPORT_FILENAME
    results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    print(f"\nWrote {results_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
