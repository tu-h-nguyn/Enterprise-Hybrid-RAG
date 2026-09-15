"""Reciprocal Rank Fusion — including the hand-computed case from HANDOFF §5.4.

Fusion is the one place where a silent arithmetic change would move every
downstream number without failing anything else, so it is pinned against
values worked out by hand rather than against a recorded output.
"""

from __future__ import annotations

import pytest

from app.retrieval.hybrid import reciprocal_rank_fusion


def test_hand_computed_example() -> None:
    """k=1, dense [A,B,C] + sparse [B,A,D].

    A = 1/(1+1) + 1/(1+2) = 0.5      + 0.3333 = 0.8333
    B = 1/(1+2) + 1/(1+1) = 0.3333   + 0.5    = 0.8333
    C = 1/(1+3)                              = 0.25
    D = 1/(1+3)                              = 0.25
    """
    fused = reciprocal_rank_fusion({"dense": ["A", "B", "C"], "sparse": ["B", "A", "D"]}, k=1)

    scores = dict(fused)
    assert scores["A"] == pytest.approx(1 / 2 + 1 / 3)
    assert scores["A"] == pytest.approx(0.833333, abs=1e-6)
    assert scores["B"] == pytest.approx(1 / 3 + 1 / 2)
    assert scores["C"] == pytest.approx(0.25)
    assert scores["D"] == pytest.approx(0.25)

    # A and B tie at 0.8333 and C and D tie at 0.25; ties break by ID so the
    # ordering is deterministic and therefore testable.
    assert [doc_id for doc_id, _ in fused] == ["A", "B", "C", "D"]


def test_agreement_beats_a_single_top_rank() -> None:
    """The whole point of RRF: two retrievers agreeing outranks one being sure."""
    fused = dict(reciprocal_rank_fusion(
        {"dense": ["solo", "agreed"], "sparse": ["other", "agreed"]}, k=60))
    assert fused["agreed"] > fused["solo"]


def test_weights_are_applied_per_retriever() -> None:
    fused = dict(reciprocal_rank_fusion(
        {"dense": ["A"], "sparse": ["B"]}, k=1, weights={"dense": 2.0, "sparse": 1.0}))
    assert fused["A"] == pytest.approx(2.0 / 2)
    assert fused["B"] == pytest.approx(1.0 / 2)


def test_missing_weight_defaults_to_one() -> None:
    fused = dict(reciprocal_rank_fusion({"dense": ["A"]}, k=1, weights={"sparse": 5.0}))
    assert fused["A"] == pytest.approx(0.5)


def test_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion({}, k=60) == []
    assert reciprocal_rank_fusion({"dense": [], "sparse": []}, k=60) == []


def test_rejects_k_below_one() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"dense": ["A"]}, k=0)


def test_scores_are_descending() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": ["A", "B", "C", "D"], "sparse": ["D", "C", "B", "A"]}, k=60)
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)
