"""Tests for the corpus-scale experiment and its distractor corpus.

The scale experiment exists to make a claim about numbers, so what needs
testing is not that it runs but that its *guards* fire. Three things would
silently turn it into a lie:

*   a distractor that restates a labelled answer (a correct hit scored as a miss),
*   gold chunks that shift when the corpus grows (a labelling artefact read as a
    scale effect),
*   a distractor corpus that is not reproducible from its seed.

Each has a test below that fails when the guard is removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import Settings
from app.evaluation.dataset import EvalDataset, EvalQuestion, RelevanceResolver
from app.evaluation.evaluator import DEFAULT_CONFIGS
from app.models.document import Chunk
from make_distractor_corpus import TOPICS, generate, render_document
from scale_experiment import (
    ContaminationError,
    _assert_uncontaminated,
    _interference,
    run_one_size,
)

SEED = 4242


@pytest.fixture
def dataset() -> EvalDataset:
    """Questions whose spans really do occur in the ``conftest`` tiny corpus."""
    return EvalDataset(name="tiny", questions=[
        EvalQuestion(
            id="q1", question="How many days of annual leave are employees entitled to?",
            question_type="factual", relevant_documents=["leave_policy.md"],
            answer_spans=["22 days of paid annual leave per calendar year"],
        ),
        EvalQuestion(
            id="q2", question="How long must a password be?",
            question_type="factual", relevant_documents=["security_policy.md"],
            answer_spans=["Passwords must be at least 14 characters long"],
        ),
        EvalQuestion(
            id="q3", question="What is the capital of Peru?",
            question_type="unanswerable",
        ),
    ])


def _chunk(chunk_id: str, text: str, source: str = "distractor.md") -> Chunk:
    return Chunk(chunk_id=chunk_id, document_id=source, source=source, text=text)


# ------------------------------------------------------------ distractor corpus
def test_generation_is_reproducible_from_the_seed(tmp_path: Path) -> None:
    left = generate(tmp_path / "a", 25, SEED)
    right = generate(tmp_path / "b", 25, SEED)

    assert [p.name for p in left] == [p.name for p in right]
    assert [p.read_bytes() for p in left] == [p.read_bytes() for p in right]


def test_a_different_seed_produces_a_different_corpus(tmp_path: Path) -> None:
    left = generate(tmp_path / "a", 25, SEED)
    right = generate(tmp_path / "b", 25, SEED + 1)

    assert [p.read_bytes() for p in left] != [p.read_bytes() for p in right]


def test_a_prefix_covers_every_topic(tmp_path: Path) -> None:
    """Sizes are nested prefixes, so a prefix must be a fair cross-section.

    Documents are emitted company-major for exactly this reason: a topic-major
    order would make the smallest corpus entirely leave policies.
    """
    written = generate(tmp_path / "corpus", len(TOPICS), SEED)

    slugs = {path.name.split("_", 1)[1].removesuffix(".md") for path in written}
    assert slugs == {topic.slug for topic in TOPICS}


def test_asking_for_more_documents_than_exist_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        generate(tmp_path / "corpus", 10**6, SEED)


def test_documents_look_like_policy_documents(tmp_path: Path) -> None:
    import random

    text = render_document("Example Partners", TOPICS[0], random.Random(SEED))

    assert text.startswith("# Example Partners ")
    assert text.count("\n## ") >= 4          # several sections, not one wall of prose
    assert "{" not in text and "}" not in text  # every template slot was filled


# ------------------------------------------------------------- contamination
def test_clean_distractors_pass_the_contamination_check(dataset: EvalDataset,
                                                        tmp_path: Path) -> None:
    written = generate(tmp_path / "corpus", 120, SEED)
    chunks = [_chunk(f"c{i}", path.read_text(encoding="utf-8"), path.name)
              for i, path in enumerate(written)]

    _assert_uncontaminated(chunks, dataset)  # must not raise


def test_a_distractor_restating_an_answer_is_rejected(dataset: EvalDataset) -> None:
    planted = _chunk("bad", "Staff receive 22 DAYS   of paid\nannual leave per calendar "
                            "year under this policy.")

    with pytest.raises(ContaminationError, match="labelled answer span"):
        _assert_uncontaminated([planted], dataset)


def test_unanswerable_questions_contribute_no_spans(dataset: EvalDataset) -> None:
    """An unanswerable question has no spans, so nothing can collide with it."""
    _assert_uncontaminated([_chunk("ok", "The capital of Peru is Lima.")], dataset)


# ------------------------------------------------------------- interference
def test_interference_is_zero_without_distractors() -> None:
    per_question = [{"id": "q1", "retrieved": ["a", "b", "c", "d", "e"]}]

    assert _interference(per_question, set(), {"q1"}) == {"at_rank_1": 0.0, "share_at_5": 0.0}


def test_interference_counts_rank_one_and_top_five_share() -> None:
    per_question = [
        {"id": "q1", "retrieved": ["d1", "a", "d2", "b", "c"]},   # distractor at rank 1
        {"id": "q2", "retrieved": ["a", "b", "c", "d", "e"]},     # none
    ]

    got = _interference(per_question, {"d1", "d2"}, {"q1", "q2"})

    assert got == {"at_rank_1": 0.5, "share_at_5": 0.2}


def test_interference_ignores_unanswerable_questions() -> None:
    per_question = [
        {"id": "q1", "retrieved": ["a", "b"]},
        {"id": "unanswerable", "retrieved": ["d1", "d2"]},
    ]

    assert _interference(per_question, {"d1", "d2"}, {"q1"})["at_rank_1"] == 0.0


# --------------------------------------------------------------- run_one_size
def test_run_one_size_reports_the_corpus_it_actually_indexed(
        settings: Settings, dataset: EvalDataset, raw_dir: Path, tmp_path: Path) -> None:
    real_paths = sorted(raw_dir.glob("*.md"))
    distractors = generate(tmp_path / "distractors", 12, SEED)

    payload, gold, backends = run_one_size(settings, dataset, real_paths, distractors,
                                           tmp_path / "scratch", DEFAULT_CONFIGS, None)

    assert backends["embedder"]["backend"] == "tfidf_svd"   # the hermetic fixtures
    assert backends["is_neural_reranker"] is False

    assert payload["n_distractor_documents"] == 12
    assert payload["n_chunks"] == payload["n_real_chunks"] + payload["n_distractor_chunks"]
    assert payload["n_distractor_chunks"] > 0
    assert payload["n_gold_chunks"] == len({cid for ids in gold.values() for cid in ids})
    assert 0 < payload["gold_share_of_corpus"] < 1
    assert [row["label"] for row in payload["configs"]] == [c.label for c in DEFAULT_CONFIGS]
    for row in payload["configs"]:
        assert 0.0 <= row["metrics"]["recall@1"] <= 1.0
        assert set(row["distractor_interference"]) == {"at_rank_1", "share_at_5"}


def test_gold_chunks_are_stable_as_the_corpus_grows(
        settings: Settings, dataset: EvalDataset, raw_dir: Path, tmp_path: Path) -> None:
    """The premise of the whole experiment: only the haystack changes."""
    real_paths = sorted(raw_dir.glob("*.md"))
    distractors = generate(tmp_path / "distractors", 24, SEED)

    _, baseline, _ = run_one_size(settings, dataset, real_paths, [],
                                  tmp_path / "s0", DEFAULT_CONFIGS, None)
    _, grown, _ = run_one_size(settings, dataset, real_paths, distractors,
                               tmp_path / "s1", DEFAULT_CONFIGS, baseline)

    assert grown == baseline
    assert all(ids for ids in baseline.values())


def test_shifted_gold_chunks_abort_the_run(
        settings: Settings, dataset: EvalDataset, raw_dir: Path, tmp_path: Path) -> None:
    real_paths = sorted(raw_dir.glob("*.md"))
    wrong_baseline = {"q1": {"not-a-real-chunk-id"}, "q2": set()}

    with pytest.raises(RuntimeError, match="not be comparable"):
        run_one_size(settings, dataset, real_paths, [], tmp_path / "s0",
                     DEFAULT_CONFIGS, wrong_baseline)


def test_the_tiny_corpus_labels_resolve_at_all(settings: Settings, chunks: list[Chunk],
                                               dataset: EvalDataset) -> None:
    """Guards the fixtures themselves: unresolvable labels look like a miss."""
    assert RelevanceResolver(chunks).audit(dataset) == []
