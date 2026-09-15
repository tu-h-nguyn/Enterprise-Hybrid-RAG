"""Evaluation dataset schema and relevance resolution.

Design decision: ground truth is expressed as **answer spans plus source
documents**, not as a hard-coded list of chunk IDs.

Chunk IDs depend on the chunking configuration. If the dataset pinned chunk
IDs, changing ``chunk_size_tokens`` would silently invalidate every label and
the chunk-size ablation would be meaningless (or, worse, would look like a
retrieval regression). Instead, a chunk is judged relevant at evaluation time
when it comes from a listed source document *and* contains one of the answer
spans. Explicit ``relevant_chunk_ids`` are still honoured when supplied, so a
human-labelled dataset can be dropped in unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.models.document import Chunk

QuestionType = Literal[
    "factual", "paraphrased", "multi_step", "keyword", "terminology", "unanswerable"
]

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Whitespace- and case-insensitive form used for span matching."""
    return _WS_RE.sub(" ", (text or "").lower()).strip()


class EvalQuestion(BaseModel):
    id: str
    question: str
    ground_truth: str = ""
    question_type: QuestionType = "factual"
    relevant_documents: list[str] = Field(default_factory=list)
    answer_spans: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def is_unanswerable(self) -> bool:
        return self.question_type == "unanswerable"

    @model_validator(mode="after")
    def _check_labels(self) -> EvalQuestion:
        if self.is_unanswerable:
            if self.relevant_documents or self.answer_spans:
                raise ValueError(f"{self.id}: unanswerable questions must have no relevance labels")
        elif not (self.relevant_documents and (self.answer_spans or self.relevant_chunk_ids)):
            raise ValueError(f"{self.id}: needs relevant_documents plus answer_spans or chunk ids")
        return self


class EvalDataset(BaseModel):
    name: str = "demo"
    description: str = ""
    questions: list[EvalQuestion] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.questions)

    @property
    def answerable(self) -> list[EvalQuestion]:
        return [q for q in self.questions if not q.is_unanswerable]

    @property
    def unanswerable(self) -> list[EvalQuestion]:
        return [q for q in self.questions if q.is_unanswerable]

    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for question in self.questions:
            counts[question.question_type] = counts.get(question.question_type, 0) + 1
        return dict(sorted(counts.items()))

    @classmethod
    def load(cls, path: Path) -> EvalDataset:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):  # bare list of questions is also accepted
            raw = {"questions": raw}
        return cls.model_validate(raw)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path


class RelevanceResolver:
    """Maps each question onto the set of chunk IDs that count as correct."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._normalized = {c.chunk_id: normalize(c.text) for c in chunks}
        self._by_source: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            self._by_source.setdefault(chunk.source, []).append(chunk)

    def resolve(self, question: EvalQuestion) -> set[str]:
        if question.is_unanswerable:
            return set()
        if question.relevant_chunk_ids:
            return set(question.relevant_chunk_ids)

        relevant: set[str] = set()
        spans = [normalize(s) for s in question.answer_spans if s.strip()]
        for source in question.relevant_documents:
            for chunk in self._by_source.get(source, []):
                text = self._normalized[chunk.chunk_id]
                if any(span in text for span in spans):
                    relevant.add(chunk.chunk_id)
        return relevant

    def audit(self, dataset: EvalDataset) -> list[dict]:
        """Report questions whose labels do not resolve to any chunk.

        Run this before trusting any metric: an unresolvable label silently
        drives recall to zero and looks exactly like a retrieval failure.
        """
        problems: list[dict] = []
        known_sources = set(self._by_source)
        for question in dataset.questions:
            if question.is_unanswerable:
                continue
            missing = [d for d in question.relevant_documents if d not in known_sources]
            resolved = self.resolve(question)
            if missing or not resolved:
                problems.append({"id": question.id, "unknown_documents": missing,
                                 "n_resolved_chunks": len(resolved),
                                 "question": question.question})
        return problems
