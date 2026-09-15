"""Shared fixtures.

Every fixture builds an index in ``tmp_path`` with the offline backends
(``numpy`` vector store, ``tfidf_svd`` embedder, ``lexical`` reranker,
``extractive`` LLM) so the suite is hermetic: no network, no API key, and no
dependence on whatever happens to be sitting in ``data/processed``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import Settings  # noqa: E402
from app.indexing.index_builder import IndexBuilder, IndexBundle  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.models.document import Chunk  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402

#: Small corpus with enough shared vocabulary for TF-IDF to fit, and with
#: deliberately distinct facts so retrieval assertions are unambiguous.
TINY_CORPUS: dict[str, str] = {
    "leave_policy.md": """# Leave Policy

Employees are entitled to 22 days of paid annual leave per calendar year.
Annual leave accrues monthly for every completed month of service.
A maximum of 5 unused annual leave days may be carried over.

## Sick Leave

Employees receive 10 paid sick days per calendar year.
A medical certificate is required from the fourth consecutive day of absence.
""",
    "security_policy.md": """# Security Policy

This policy is reference ISP-2024-03 and is reviewed annually.
Passwords must be at least 14 characters long.
Multi-factor authentication is mandatory for all production systems.

## Data Classification

Restricted data includes customer personal data and authentication secrets.
Restricted data must be encrypted at rest using AES-256.
""",
    "expense_policy.md": """# Expense Policy

This policy is reference FIN-TE-4 and is owned by the finance team.
Expense claims must be submitted within 45 days of the expense being incurred.
Mileage for private cars is reimbursed at 0.38 euro per kilometre.

## Receipts

Itemised receipts are required for every expense above 25 euro.
Alcohol is not reimbursable except at approved client events.
""",
    "incident_runbook.md": """# Incident Runbook

A SEV1 incident is a total loss of service affecting customer data.
The primary on-call engineer must acknowledge a page within 5 minutes.
Internal status updates are posted every 30 minutes during a SEV1.

## Postmortems

Every SEV1 requires a written postmortem within 5 working days.
Postmortems are blameless and describe systems rather than individuals.
""",
}


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in TINY_CORPUS.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


@pytest.fixture
def settings(tmp_path: Path, raw_dir: Path) -> Settings:
    processed = tmp_path / "processed"
    return Settings(
        data_dir=tmp_path,
        raw_dir=raw_dir,
        processed_dir=processed,
        evaluation_dir=tmp_path / "evaluation",
        chroma_dir=processed / "chroma",
        bm25_dir=processed / "bm25",
        vector_backend="numpy",
        embedding_backend="tfidf_svd",
        reranker_backend="lexical",
        llm_provider="extractive",
        chunk_size_tokens=200,
        chunk_overlap_tokens=40,
        log_json=False,
    )


@pytest.fixture
def chunks(settings: Settings) -> list[Chunk]:
    _, produced = IngestionPipeline(settings).run()
    assert produced, "fixture corpus produced no chunks"
    return produced


@pytest.fixture
def bundle(settings: Settings, chunks: list[Chunk]) -> IndexBundle:
    return IndexBuilder(settings).build(chunks, reset=True)


@pytest.fixture
def service(settings: Settings, bundle: IndexBundle) -> RagService:
    return RagService(settings, bundle)
