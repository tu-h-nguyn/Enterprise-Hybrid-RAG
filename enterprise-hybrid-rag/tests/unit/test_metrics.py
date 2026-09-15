"""Metric arithmetic on fixed inputs.

These are pinned by hand because every headline number in the README is one of
these functions applied to retrieval output; a silent change here would move
the results without breaking anything visible.
"""

from __future__ import annotations

import math

import pytest

from app.evaluation.dataset import EvalQuestion, normalize
from app.evaluation.generation_metrics import groundedness, token_f1
from app.evaluation.retrieval_metrics import (RetrievalMetrics, hit_at_k, ndcg_at_k,
                                              precision_at_k, recall_at_k, reciprocal_rank)

RETRIEVED = ["a", "b", "c", "d", "e"]


def test_recall_at_k_is_the_fraction_of_relevant_chunks_found() -> None:
    relevant = {"b", "e", "z"}          # three relevant, one outside the list entirely
    assert recall_at_k(RETRIEVED, relevant, 1) == pytest.approx(0.0)
    assert recall_at_k(RETRIEVED, relevant, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, relevant, 5) == pytest.approx(2 / 3)


def test_recall_equals_hit_when_there_is_one_relevant_chunk() -> None:
    relevant = {"c"}
    for k in (1, 3, 5):
        assert recall_at_k(RETRIEVED, relevant, k) == hit_at_k(RETRIEVED, relevant, k)


def test_hit_at_k_is_binary() -> None:
    assert hit_at_k(RETRIEVED, {"c"}, 2) == 0.0
    assert hit_at_k(RETRIEVED, {"c"}, 3) == 1.0


def test_precision_at_k_divides_by_the_window() -> None:
    assert precision_at_k(RETRIEVED, {"a", "b"}, 2) == pytest.approx(1.0)
    assert precision_at_k(RETRIEVED, {"a", "b"}, 5) == pytest.approx(2 / 5)
    assert precision_at_k(RETRIEVED, {"a"}, 0) == 0.0


def test_reciprocal_rank_uses_the_first_hit_only() -> None:
    assert reciprocal_rank(RETRIEVED, {"a"}) == pytest.approx(1.0)
    assert reciprocal_rank(RETRIEVED, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RETRIEVED, {"c", "e"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RETRIEVED, {"zzz"}) == 0.0


def test_ndcg_is_one_when_the_ranking_is_ideal() -> None:
    assert ndcg_at_k(RETRIEVED, {"a", "b"}, 5) == pytest.approx(1.0)


def test_ndcg_hand_computed_for_a_single_hit_at_rank_three() -> None:
    # DCG = 1/log2(3+1) = 0.5 ; ideal = 1/log2(1+1) = 1.0
    assert ndcg_at_k(RETRIEVED, {"c"}, 5) == pytest.approx(1 / math.log2(4))


def test_metrics_with_no_relevant_chunks_are_zero() -> None:
    assert recall_at_k(RETRIEVED, set(), 5) == 0.0
    assert ndcg_at_k(RETRIEVED, set(), 5) == 0.0


def test_aggregator_skips_unanswerable_questions() -> None:
    """Unanswerable questions have no relevant chunks; averaging them in as zeros
    would understate every retriever equally and hide the real differences."""
    metrics = RetrievalMetrics(ks=(1, 3))
    metrics.update(["a", "b", "c"], {"a"})       # rank 1
    metrics.update(["x", "y", "b"], {"b"})       # rank 3
    metrics.update(["p", "q"], set())            # unanswerable

    summary = metrics.summary()
    assert summary["n_scored"] == 2
    assert summary["n_skipped"] == 1
    assert summary["recall@1"] == pytest.approx(0.5)
    assert summary["recall@3"] == pytest.approx(1.0)
    assert summary["mrr"] == pytest.approx((1.0 + 1 / 3) / 2, abs=1e-4)


def test_normalize_is_whitespace_and_case_insensitive() -> None:
    assert normalize("  22 DAYS   of\nLeave ") == "22 days of leave"


def test_token_f1_against_a_hand_computed_case() -> None:
    # prediction content tokens: {22, days, paid, annual, leave}
    # reference  content tokens: {22, days, annual, leave}
    # common = 4 -> P = 4/5, R = 4/4 -> F1 = 2*0.8*1.0/1.8
    score = token_f1("22 days of paid annual leave", "22 days of annual leave")
    assert score == pytest.approx(2 * 0.8 * 1.0 / 1.8, abs=1e-4)


def test_token_f1_ignores_citation_markers() -> None:
    assert token_f1("22 days of annual leave [Source 1]",
                    "22 days of annual leave") == pytest.approx(1.0)


def test_groundedness_uses_bigrams_not_unigrams() -> None:
    """An answer recombining real terms into a false claim must not score 1.0."""
    context = "Employees receive 22 days of paid annual leave each calendar year."
    assert groundedness("Employees receive 22 days of paid annual leave", context) == pytest.approx(1.0)

    recombined = groundedness("Annual employees receive paid leave days", context)
    assert recombined < 1.0


def test_groundedness_of_an_ungrounded_answer_is_zero() -> None:
    assert groundedness("Completely unrelated invented assertion here",
                        "Employees receive 22 days of leave.") == 0.0


def test_unanswerable_question_rejects_relevance_labels() -> None:
    with pytest.raises(ValueError):
        EvalQuestion(id="q1", question="?", question_type="unanswerable",
                     relevant_documents=["a.md"], answer_spans=["x"])


def test_answerable_question_requires_labels() -> None:
    with pytest.raises(ValueError):
        EvalQuestion(id="q1", question="?", question_type="factual")
