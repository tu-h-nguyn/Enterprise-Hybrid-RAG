"""Tests for the reproducibility check that decides whether CI commits a result.

This script is the only thing standing between a wall-clock wobble and a commit
that claims a metric moved, so the parts worth testing are the ones that decide:
what counts as a quality value, what is refused outright, and what is allowed to
drift without counting as a change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compare_results import flatten, main, quality_view

BACKENDS = {
    "embedder": {"backend": "sentence_transformers", "dimension": 384},
    "reranker": {"backend": "cross_encoder", "neural": True},
}


def _scale_payload(chroma_mrr: float = 0.62, numpy_mrr: float = 0.63) -> dict:
    def size(store: str, mrr: float) -> dict:
        return {
            "vector_backend": store, "n_chunks": 4658, "n_distractor_documents": 3400,
            "n_gold_chunks": 30, "embedding_dim": 384, "index_build_seconds": 155.0,
            "configs": [{"label": "Hybrid (RRF)", "metrics": {"recall@1": 0.4457, "mrr": mrr},
                         "latency_ms_mean": 54.63,
                         "distractor_interference": {"at_rank_1": 0.3953}}],
        }
    return {"backends": BACKENDS, "generated_at": "2026-01-01T00:00:00+00:00",
            "sizes": [size("numpy", numpy_mrr), size("chroma", chroma_mrr)]}


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ------------------------------------------------------------- the projection
def test_latency_and_timestamps_are_not_quality() -> None:
    """The whole point: a slower runner must not read as a changed metric."""
    keys = flatten(quality_view(_scale_payload()))

    assert not any("latency" in key or "generated_at" in key or "build_seconds" in key
                   for key in keys)
    assert any(key.endswith("metrics.recall@1") for key in keys)


def test_scale_rows_are_keyed_by_store_as_well_as_size() -> None:
    """Both stores index the same corpus; colliding them would hide half the run."""
    view = quality_view(_scale_payload())

    assert set(view["scale"]) == {"numpy/4658", "chroma/4658"}


def test_interference_is_compared_too() -> None:
    """If the distractors stopped competing, that is a real change, not timing."""
    keys = flatten(quality_view(_scale_payload()))

    assert any("interference.at_rank_1" in key for key in keys)


# ------------------------------------------------------------------- the gate
def test_identical_files_reproduce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _scale_payload()
    baseline = _write(tmp_path, "a.json", payload)
    candidate = _write(tmp_path, "b.json", payload)
    monkeypatch.setattr("sys.argv", ["compare_results.py", "--baseline", str(baseline),
                                     "--candidate", str(candidate)])

    assert main() == 0


def test_a_moved_quality_metric_is_a_change(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _write(tmp_path, "a.json", _scale_payload(numpy_mrr=0.63))
    candidate = _write(tmp_path, "b.json", _scale_payload(numpy_mrr=0.64))
    monkeypatch.setattr("sys.argv", ["compare_results.py", "--baseline", str(baseline),
                                     "--candidate", str(candidate)])

    assert main() == 1


def test_allowed_drift_does_not_count_as_a_change(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """HNSW is approximate; only the exhaustive rows are held to bit equality."""
    baseline = _write(tmp_path, "a.json", _scale_payload(chroma_mrr=0.6227))
    candidate = _write(tmp_path, "b.json", _scale_payload(chroma_mrr=0.6203))
    monkeypatch.setattr("sys.argv", ["compare_results.py", "--baseline", str(baseline),
                                     "--candidate", str(candidate),
                                     "--allow-drift", "scale.chroma/"])

    assert main() == 0


def test_allowed_drift_does_not_excuse_the_exact_rows(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag must narrow the gate, not open it."""
    baseline = _write(tmp_path, "a.json", _scale_payload(chroma_mrr=0.6227, numpy_mrr=0.63))
    candidate = _write(tmp_path, "b.json", _scale_payload(chroma_mrr=0.6203, numpy_mrr=0.64))
    monkeypatch.setattr("sys.argv", ["compare_results.py", "--baseline", str(baseline),
                                     "--candidate", str(candidate),
                                     "--allow-drift", "scale.chroma/"])

    assert main() == 1


def test_runs_on_different_backends_are_refused(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """A fallback-vs-neural diff would report everything as changed and mean nothing."""
    fallback = _scale_payload()
    fallback["backends"] = {"embedder": {"backend": "tfidf_svd", "dimension": 43},
                            "reranker": {"backend": "lexical", "neural": False}}
    baseline = _write(tmp_path, "a.json", fallback)
    candidate = _write(tmp_path, "b.json", _scale_payload())
    monkeypatch.setattr("sys.argv", ["compare_results.py", "--baseline", str(baseline),
                                     "--candidate", str(candidate)])

    assert main() == 2


def test_a_missing_baseline_is_a_first_run_not_an_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _write(tmp_path, "b.json", _scale_payload())
    monkeypatch.setattr("sys.argv", ["compare_results.py",
                                     "--baseline", str(tmp_path / "missing.json"),
                                     "--candidate", str(candidate)])

    assert main() == 1
