"""Tests for the gate that keeps published prose honest.

This script is the only thing standing between a changed metric and a document
that quietly still quotes the old one, so what matters is that it fails when it
should — not that it passes today.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import check_report_numbers as checker


@pytest.fixture
def artefacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    evaluation = tmp_path / "data" / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "results.json").write_text(json.dumps({
        "retrieval": [
            {"label": "Dense only",
             "metrics": {"recall@1": 0.5853, "recall@5": 0.8953,
                         "mrr": 0.7740, "ndcg@5": 0.7869}},
            {"label": "Hybrid + Reranker",
             "metrics": {"recall@1": 0.7946, "recall@5": 0.9070,
                         "mrr": 0.9085, "ndcg@5": 0.8937}},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(checker, "EVALUATION", evaluation)
    return tmp_path


def _document(tmp_path: Path, text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the checker at one throwaway document and nothing else.

    COUNTED is patched too: it otherwise still names the real READMEs, and a
    test would quietly read the repository it is supposed to be isolated from.
    """
    path = tmp_path / "DOC.md"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(checker, "DOCUMENTS", (path,))
    monkeypatch.setattr(checker, "COUNTED", (path,))
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)


def test_a_number_from_an_artefact_passes(artefacts: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    _document(artefacts, "The pipeline reaches 0.7946 Recall@1.", monkeypatch)
    monkeypatch.setattr("sys.argv", ["check_report_numbers.py"])

    assert checker.main() == 0


def test_a_number_that_came_from_nowhere_fails(artefacts: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure this exists to catch: a figure nobody can trace."""
    _document(artefacts, "The pipeline reaches 0.8123 Recall@1.", monkeypatch)
    monkeypatch.setattr("sys.argv", ["check_report_numbers.py"])

    assert checker.main() == 1


def test_a_stale_number_fails_once_the_artefact_moves(
        artefacts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """0.7016 was the real answer before the chunk size changed. Now it is not."""
    _document(artefacts, "The pipeline reaches 0.7016 Recall@1.", monkeypatch)
    monkeypatch.setattr("sys.argv", ["check_report_numbers.py"])

    assert checker.main() == 1


def test_a_delta_the_prose_computes_is_accepted(artefacts: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """Prose says "+0.2093 over dense alone"; that is arithmetic, not a new claim."""
    _document(artefacts, "Reranking is worth +0.2093 over dense alone.", monkeypatch)
    monkeypatch.setattr("sys.argv", ["check_report_numbers.py"])

    assert checker.main() == 0


def test_a_percentage_of_a_rate_is_accepted(artefacts: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """The artefact says 0.7946; prose may say 79.46%."""
    _document(artefacts, "It answers 79.46% of the time.", monkeypatch)
    monkeypatch.setattr("sys.argv", ["check_report_numbers.py"])

    assert checker.main() == 0


def test_every_exemption_carries_a_reason() -> None:
    """An exemption without a reason is just a suppressed failure."""
    assert checker.KNOWN
    for value, reason in checker.KNOWN.items():
        assert reason.strip(), value
        assert len(reason) > 10, f"{value}: {reason!r} does not explain anything"


# ------------------------------------------------------------------ timings
@pytest.fixture
def timed_artefacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    evaluation = tmp_path / "data" / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "results.json").write_text(json.dumps({
        "retrieval": [{"label": "Hybrid + Reranker",
                       "metrics": {"recall@1": 0.7946, "recall@5": 0.9070,
                                   "mrr": 0.9085, "ndcg@5": 0.8937},
                       "latency_ms_mean": 693.19}],
    }), encoding="utf-8")
    monkeypatch.setattr(checker, "EVALUATION", evaluation)
    return tmp_path


def test_a_timing_near_the_artefact_passes(timed_artefacts: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """699.95 ms and 693.19 ms are the same measurement on two runners."""
    _document(timed_artefacts, "Reranking costs 699.95 ms per query.", monkeypatch)
    monkeypatch.setattr("sys.argv", ["check_report_numbers.py"])

    assert checker.main() == 0


def test_a_timing_from_a_different_pipeline_still_fails(
        timed_artefacts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tolerance is for runner variance, not for quoting the wrong thing."""
    _document(timed_artefacts, "Reranking costs 2500.00 ms per query.", monkeypatch)
    monkeypatch.setattr("sys.argv", ["check_report_numbers.py"])

    assert checker.main() == 1


def test_the_tolerance_does_not_leak_into_quality_metrics(
        timed_artefacts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """0.7946 is a claim; 0.7900 is a different claim, however close.

    Quality metrics and sub-millisecond timings share a numeric range, so the
    tolerance keys off the unit the document writes, not off magnitude. Without
    that, BM25's 0.74 ms would open a window over Recall@1.
    """
    _document(timed_artefacts, "It reaches 0.7900 Recall@1.", monkeypatch)
    monkeypatch.setattr("sys.argv", ["check_report_numbers.py"])

    assert checker.main() == 1


def test_a_metric_is_not_excused_by_a_timing_of_the_same_size(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure the unit check exists to prevent."""
    evaluation = tmp_path / "data" / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "results.json").write_text(json.dumps({
        "retrieval": [{"label": "BM25 only",
                       "metrics": {"recall@1": 0.6434, "recall@5": 0.8023,
                                   "mrr": 0.7597, "ndcg@5": 0.7660},
                       "latency_ms_mean": 0.74}],
    }), encoding="utf-8")
    monkeypatch.setattr(checker, "EVALUATION", evaluation)
    _document(tmp_path, "It reaches 0.7900 Recall@1.", monkeypatch)
    monkeypatch.setattr("sys.argv", ["check_report_numbers.py"])

    assert checker.main() == 1


# -------------------------------------------------------------- test counts
def test_a_test_count_that_disagrees_between_documents_fails(
        artefacts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The drift this catches happened five times in one day.

    The count appears in a badge, a command comment, a repository map, the
    handover document and the report. Nothing made them agree.
    """
    badge = artefacts / "A.md"
    badge.write_text("![Tests](https://img.shields.io/badge/tests-168%20passing-green)",
                     encoding="utf-8")
    prose = artefacts / "B.md"
    prose.write_text("| `tests/` | 151 unit and integration tests |", encoding="utf-8")
    monkeypatch.setattr(checker, "DOCUMENTS", (badge, prose))
    monkeypatch.setattr(checker, "COUNTED", (badge, prose))
    monkeypatch.setattr(checker, "REPO_ROOT", artefacts)
    monkeypatch.setattr("sys.argv", ["check_report_numbers.py"])

    assert checker.main() == 1


def test_agreeing_test_counts_pass(artefacts: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    badge = artefacts / "A.md"
    badge.write_text("![Tests](https://img.shields.io/badge/tests-168%20passing-green)",
                     encoding="utf-8")
    prose = artefacts / "B.md"
    prose.write_text("| `tests/` | 168 unit and integration tests |", encoding="utf-8")
    monkeypatch.setattr(checker, "DOCUMENTS", (badge, prose))
    monkeypatch.setattr(checker, "COUNTED", (badge, prose))
    monkeypatch.setattr(checker, "REPO_ROOT", artefacts)
    monkeypatch.setattr("sys.argv", ["check_report_numbers.py"])

    assert checker.main() == 0
