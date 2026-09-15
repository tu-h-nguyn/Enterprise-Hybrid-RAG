"""API surface via fastapi.testclient.

``AppState`` is injected directly rather than relying on the lifespan hook, so
the tests bind to the hermetic tmp_path index from conftest instead of whatever
happens to be in data/processed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.deps import AppState
from app.config.settings import Settings
from app.main import create_app
from app.services.document_service import DocumentService
from app.services.rag_service import RagService


@pytest.fixture
def client(settings: Settings, service: RagService) -> TestClient:
    app = create_app()
    app.state.app_state = AppState(settings=settings, rag=service,
                                   documents=DocumentService(settings))
    return TestClient(app)


def test_health_reports_index_readiness_and_backends(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["index_ready"] is True
    assert body["n_chunks"] > 0
    assert body["reranker"] == "lexical"
    assert body["llm_provider"].startswith("extractive")


def test_query_returns_answer_citations_and_metadata(client: TestClient) -> None:
    response = client.post("/query", json={
        "question": "How many days of paid annual leave are employees entitled to?",
        "top_k": 5, "retrieval_method": "hybrid", "rerank": True})

    assert response.status_code == 200
    body = response.json()
    assert "22 days" in body["answer"]
    assert body["citations"]
    assert body["retrieved_chunks"]

    citation = body["citations"][0]
    assert citation["source"].endswith(".md")
    assert citation["chunk_id"]

    metadata = body["metadata"]
    assert metadata["retrieval_method"] == "hybrid"
    assert metadata["reranked"] is True
    assert metadata["no_answer"] is False
    assert metadata["latency_ms"] > 0
    assert metadata["n_final_contexts"] > 0


@pytest.mark.parametrize("method", ["dense", "sparse", "hybrid"])
def test_query_honours_the_retrieval_method_override(client: TestClient, method: str) -> None:
    response = client.post("/query", json={"question": "What is the mileage rate per kilometre?",
                                           "retrieval_method": method})
    assert response.status_code == 200
    assert response.json()["metadata"]["retrieval_method"] == method


def test_query_can_suppress_the_chunk_list(client: TestClient) -> None:
    response = client.post("/query", json={"question": "What is the mileage rate?",
                                           "include_chunks": False})
    assert response.status_code == 200
    assert response.json()["retrieved_chunks"] == []


def test_unanswerable_question_returns_a_refusal_not_a_guess(client: TestClient) -> None:
    response = client.post("/query", json={
        "question": "What is the vesting schedule for employee stock options?"})

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["no_answer"] is True
    assert body["citations"] == []
    assert "could not find enough information" in body["answer"].lower()


def test_query_validates_its_input(client: TestClient) -> None:
    assert client.post("/query", json={"question": "hi"}).status_code == 422       # min_length
    assert client.post("/query", json={}).status_code == 422                       # required
    assert client.post("/query", json={"question": "a valid question here",
                                       "top_k": 99}).status_code == 422            # le=20


def test_documents_listing_reports_the_indexed_corpus(client: TestClient) -> None:
    response = client.get("/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["n_documents"] == 4
    assert {d["source"] for d in body["documents"]} == {
        "leave_policy.md", "security_policy.md", "expense_policy.md", "incident_runbook.md"}


def test_upload_rejects_an_unsupported_file_type(client: TestClient) -> None:
    response = client.post("/documents/upload",
                           files={"file": ("data.xlsx", b"binary", "application/octet-stream")})
    assert response.status_code == 422


def test_upload_then_index_makes_a_new_document_searchable(client: TestClient) -> None:
    payload = ("# Parking Policy\n\n"
               "Parking spaces are allocated by lottery each quarter at every office.\n"
               "Accessible parking spaces are reserved and sit outside the lottery.\n"
               "A cycle to work allowance of 300 euro per year is available to all staff.\n")

    upload = client.post("/documents/upload",
                         files={"file": ("parking_policy.md", payload.encode(), "text/markdown")})
    assert upload.status_code == 201
    assert upload.json()["indexed"] is False

    index = client.post("/documents/index", json={"rebuild": True})
    assert index.status_code == 200
    assert index.json()["n_documents"] == 5

    answer = client.post("/query", json={"question": "How are parking spaces allocated?"})
    assert answer.status_code == 200
    assert "lottery" in answer.json()["answer"].lower()


def test_delete_removes_a_source_file(client: TestClient) -> None:
    assert client.delete("/documents/expense_policy.md").status_code == 200
    assert client.delete("/documents/not_a_real_file.md").status_code == 404


def test_root_and_docs_are_reachable(client: TestClient) -> None:
    assert client.get("/").status_code == 200
    assert client.get("/openapi.json").status_code == 200
