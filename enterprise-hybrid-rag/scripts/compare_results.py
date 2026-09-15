#!/usr/bin/env python3
"""Compare two result files on quality metrics only.

    python scripts/compare_results.py --baseline old.json --candidate new.json

Handles both shapes written under ``data/evaluation``: ``results.json`` from
``scripts/benchmark.py`` and ``scale_results.json`` from
``scripts/scale_experiment.py``. Sections absent from a payload are simply
empty, so the same invocation works for either.

Latency is wall-clock and differs on every run; quality metrics are
deterministic given the same corpus, dataset and backends. Comparing only the
latter turns a benchmark re-run into a reproducibility check: identical output
means nothing drifted, and a difference is a real change worth committing.

Some values are deterministic in principle but not in practice.
``--allow-drift PREFIX`` names those: differences under the prefix are printed
like any other, but do not count as a change. The one that needs it is
``scale.chroma/`` — Chroma's HNSW index is approximate, and two identical runs
of the corpus-scale experiment disagree on a handful of deep-rank values while
every exhaustive-search row reproduces exactly. Without the flag the experiment
would rewrite its own artefact on every run and the reproducibility check would
stop meaning anything; with it, the check still holds exact search to bit
equality and the drift stays visible in the log.

Exit codes:
    0  quality metrics identical (ignoring any allowed drift)
    1  quality metrics differ (the differences are printed)
    2  the files cannot be compared (different backends, missing file)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: Keys whose values are wall-clock measurements rather than quality.
LATENCY_KEYS = frozenset({
    "latency_ms_mean", "latency_ms_p95", "mean_latency_ms", "latency_ms",
    "generated_at", "environment", "index_build_seconds",
})


def quality_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Project a results payload down to its deterministic parts."""
    view: dict[str, Any] = {"retrieval": {}, "ablation": {}, "generation": {}, "scale": {}}

    for row in payload.get("retrieval", []):
        view["retrieval"][row["label"]] = {
            "metrics": row["metrics"],
            "by_question_type": row.get("by_question_type", {}),
        }

    for row in payload.get("chunk_size_ablation", []):
        key = str(row["chunk_size_tokens"])
        view["ablation"][key] = {
            "n_chunks": row["n_chunks"],
            "configs": {c["label"]: c["metrics"] for c in row.get("configs", [])},
        }

    # scale_results.json (scripts/scale_experiment.py) carries its own shape.
    # Distractor interference is deterministic too, so it belongs in the view:
    # if the same corpus stopped competing with the gold chunks, that is a real
    # change and not a timing artefact.
    for row in payload.get("sizes", []):
        # Keyed by store as well as size: the same corpus is run through both
        # the exact and the approximate index, and they must not collide.
        key = f"{row.get('vector_backend', 'chroma')}/{row['n_chunks']}"
        view["scale"][key] = {
            "n_distractor_documents": row["n_distractor_documents"],
            "n_gold_chunks": row["n_gold_chunks"],
            "embedding_dim": row.get("embedding_dim"),
            "configs": {c["label"]: {"metrics": c["metrics"],
                                     "interference": c.get("distractor_interference", {})}
                        for c in row.get("configs", [])},
        }

    generation = (payload.get("generation") or {}).get("metrics", {})
    view["generation"] = {k: v for k, v in generation.items() if k not in LATENCY_KEYS}
    return view


def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if key in LATENCY_KEYS:
                continue
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out[prefix] = node
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare benchmark results on quality only")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--allow-drift", action="append", default=[], metavar="PREFIX",
                        help="Metric-key prefix whose differences are reported but not "
                             "counted as a change. Repeatable.")
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"No baseline at {args.baseline}; treating as a first run.")
        return 1
    if not args.candidate.exists():
        print(f"Candidate missing: {args.candidate}", file=sys.stderr)
        return 2

    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())

    # Comparing across different backends would report every metric as changed
    # and say nothing useful, so refuse rather than produce a misleading diff.
    for side, payload in (("baseline", baseline), ("candidate", candidate)):
        print(f"{side:<10} embedder={payload['backends']['embedder'].get('backend')} "
              f"reranker={payload['backends']['reranker'].get('backend')}")
    if baseline["backends"]["embedder"] != candidate["backends"]["embedder"] or \
            baseline["backends"]["reranker"] != candidate["backends"]["reranker"]:
        print("\nBackends differ; these two runs are not comparable.", file=sys.stderr)
        return 2

    left, right = flatten(quality_view(baseline)), flatten(quality_view(candidate))
    keys = sorted(set(left) | set(right))
    differing = [(k, left.get(k), right.get(k)) for k in keys if left.get(k) != right.get(k)]

    allowed = [(k, a, b) for k, a, b in differing
               if any(k.startswith(prefix) for prefix in args.allow_drift)]
    changes = [row for row in differing if row not in allowed]

    print(f"\ncompared {len(keys)} quality values")
    if allowed:
        print(f"{len(allowed)} differed under an --allow-drift prefix "
              f"({', '.join(args.allow_drift)}) and are not counted:\n")
        for key, old_value, new_value in allowed[:20]:
            print(f"  {key:<60}{old_value!s:>12}{new_value!s:>12}")
        if len(allowed) > 20:
            print(f"  ... and {len(allowed) - 20} more")
        print()
    if not changes:
        print("identical — the benchmark reproduced exactly"
              + (" outside the allowed drift" if allowed else ""))
        return 0

    print(f"{len(changes)} changed:\n")
    print(f"{'metric':<62}{'baseline':>12}{'candidate':>12}")
    print("-" * 86)
    for key, old, new in changes[:60]:
        print(f"{key:<62}{old!s:>12}{new!s:>12}")
    if len(changes) > 60:
        print(f"... and {len(changes) - 60} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
