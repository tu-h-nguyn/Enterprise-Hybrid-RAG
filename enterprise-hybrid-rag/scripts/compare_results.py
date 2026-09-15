#!/usr/bin/env python3
"""Compare two benchmark result files on quality metrics only.

    python scripts/compare_results.py --baseline old.json --candidate new.json

Latency is wall-clock and differs on every run; quality metrics are
deterministic given the same corpus, dataset and backends. Comparing only the
latter turns a benchmark re-run into a reproducibility check: identical output
means nothing drifted, and a difference is a real change worth committing.

Exit codes:
    0  quality metrics identical
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
    "generated_at", "environment",
})


def quality_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Project a results payload down to its deterministic parts."""
    view: dict[str, Any] = {"retrieval": {}, "ablation": {}, "generation": {}}

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
    changes = [(k, left.get(k), right.get(k)) for k in keys if left.get(k) != right.get(k)]

    print(f"\ncompared {len(keys)} quality values")
    if not changes:
        print("identical — the benchmark reproduced exactly")
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
