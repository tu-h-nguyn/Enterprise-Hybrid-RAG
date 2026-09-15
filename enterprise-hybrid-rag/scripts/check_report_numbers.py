#!/usr/bin/env python3
"""Check that every figure in the documents traces to a committed artefact.

    python scripts/check_report_numbers.py
    python scripts/check_report_numbers.py --show-known

Prose goes stale silently. A metric moves, a default changes, a run lands on a
different runner, and a number that was true last week is now a claim nobody can
check. This script is the mechanical part of the answer: it collects every value
in ``data/evaluation/*.json`` — including the differences the prose computes
between them — and reports any number in a document that does not appear there.

Exit codes:
    0  every number is accounted for
    1  a number could not be traced (they are printed with their file)

Some figures legitimately do not come from an artefact: a Python version, a
measurement explicitly labelled as historical, an example payload captured under
an earlier default. Those live in ``KNOWN`` with the reason attached, so that
"unexplained" means unexplained rather than "not yet excused".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
EVALUATION = PROJECT_ROOT / "data" / "evaluation"

DOCUMENTS = (
    REPO_ROOT / "README.md",
    PROJECT_ROOT / "README.md",
    REPO_ROOT / "docs" / "report" / "report.tex",
)

#: Numbers that are not metrics, with why each one is allowed to be here.
KNOWN: dict[str, str] = {
    "3.11": "Python version",
    "7.29": "the image size before the CPU-only torch change, labelled as the before",
    "2.4": "a LaTeX geometry margin",
    "2.8": "a LaTeX geometry margin",
    "0.4": "a LaTeX rule width",
    "0.7": "a LaTeX rule width",
    "0.6": "a LaTeX title spacing",
    "0.8": "a LaTeX title spacing",
    "1.0": "a LaTeX spacing value",
    "0.15": "a LaTeX spacing value",
    "1.1": "a LaTeX spacing value",
    "1.2": "a LaTeX list indent",
    "1.4": "a LaTeX list indent",
    "0.86": "a LaTeX minipage width",
    "2.0": "a LaTeX spacing value",
    "2.5": "a LaTeX spacing value",
    "1.209": "mean gold chunks per question, computed from the dataset labels",
    "0.9362": "example API payload, captured at the 500-token default",
    "1.49": "example API payload, captured at the 500-token default",
    "1664.59": "example API payload, captured at the 500-token default",
    "1689.21": "example API payload, captured at the 500-token default",
    "3.3085": "example API payload, captured at the 500-token default",
    "0.3411": "an offline-fallback HNSW run quoted to show it is not reproducible",
    "0.3876": "an offline-fallback HNSW run quoted to show it is not reproducible",
    "0.3643": "an offline-fallback HNSW run quoted to show it is not reproducible",
    "0.4109": "an offline-fallback exhaustive run quoted for the same comparison",
    "1416.77": "a superseded latency, quoted in the report as an example of the mistake",
    "154.99": "a superseded index build time, quoted for the same reason",
}

NUMBER_RE = re.compile(r"\d+\.\d{2,4}")


def _add(value: Any, out: set[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    for form in (str(value), f"{value:.4f}", f"{value:.2f}", f"{value:.1f}"):
        out.add(form)
    # Prose says "34.88% of answerable questions" where the artefact says
    # 0.3488. Both are the same measurement and both must be checkable.
    if 0.0 <= float(value) <= 1.0:
        for form in (f"{value * 100:.2f}", f"{value * 100:.1f}"):
            out.add(form)


def _walk(node: Any, out: set[str]) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _walk(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk(value, out)
    else:
        _add(node, out)


def artefact_values() -> set[str]:
    """Every value in the committed artefacts, plus the deltas prose computes."""
    values: set[str] = set()
    payloads: dict[str, dict] = {}
    for path in sorted(EVALUATION.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[path.name] = payload
        _walk(payload, values)

    scale = payloads.get("scale_results.json", {})
    for row in scale.get("sizes", []):
        values.add(f"{row['gold_share_of_corpus'] * 100:.2f}")

    neural = {r["label"]: r["metrics"] for r in payloads.get("results.json", {})
              .get("retrieval", [])}
    fallback = {r["label"]: r["metrics"] for r in payloads.get("results_fallback.json", {})
                .get("retrieval", [])}
    for label, metrics in neural.items():
        other = fallback.get(label)
        if other is None:
            continue
        for key in ("recall@1", "mrr"):
            values.add(f"{abs(metrics[key] - other[key]):.4f}")

    # Differences between configurations, chunk sizes and corpus sizes: the
    # prose states these as deltas and they have to be checkable too.
    for source in (neural, fallback):
        for a in source.values():
            for b in source.values():
                for key in ("recall@1", "recall@5", "mrr", "ndcg@5"):
                    values.add(f"{abs(a[key] - b[key]):.4f}")

    rows = {(r["chunk_size_tokens"], r["vector_backend"], r["n_distractor_documents"]): r
            for r in scale.get("sizes", [])}
    for row_key, row in rows.items():
        for config in row["configs"]:
            # The same configuration at a different corpus size or chunk size:
            # the prose states these as "the change over 111x".
            for other_key, other_row in rows.items():
                if other_key == row_key:
                    continue
                twin = next((c for c in other_row["configs"]
                             if c["label"] == config["label"]), None)
                if twin is None:
                    continue
                for metric in ("recall@1", "recall@5", "mrr"):
                    values.add(
                        f"{abs(config['metrics'][metric] - twin['metrics'][metric]):.4f}")
            # A different configuration at the same size: "worth +0.3450 over
            # dense alone".
            for sibling in row["configs"]:
                for metric in ("recall@1", "recall@5", "mrr"):
                    values.add(
                        f"{abs(config['metrics'][metric] - sibling['metrics'][metric]):.4f}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show-known", action="store_true",
                        help="List the exemptions and why each one exists")
    args = parser.parse_args()

    if args.show_known:
        for value, reason in sorted(KNOWN.items()):
            print(f"  {value:<10} {reason}")
        return 0

    values = artefact_values()
    failures = 0
    for document in DOCUMENTS:
        if not document.exists():
            print(f"{document}: missing", file=sys.stderr)
            failures += 1
            continue
        text = document.read_text(encoding="utf-8")
        numbers = sorted(set(NUMBER_RE.findall(text)))
        unexplained = [n for n in numbers if n not in values and n not in KNOWN]
        label = str(document.relative_to(REPO_ROOT))
        print(f"{label:<40} {len(numbers):>4} numbers, "
              f"{len(unexplained)} unexplained")
        for number in unexplained:
            print(f"    {number}")
        failures += len(unexplained)

    if failures:
        print(f"\n{failures} number(s) do not trace to a committed artefact.",
              file=sys.stderr)
        return 1
    print("\nEvery number traces to a committed artefact or a documented exemption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
