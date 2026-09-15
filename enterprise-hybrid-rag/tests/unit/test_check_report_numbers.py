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
    path = tmp_path / "DOC.md"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(checker, "DOCUMENTS", (path,))
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
