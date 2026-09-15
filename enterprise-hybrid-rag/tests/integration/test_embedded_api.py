"""The single-process mode the public demo runs on.

Without this the demo is untested code on the critical path of the first thing
a reader sees. These tests start the **real** ASGI app — lifespan hook included,
which is what actually builds the service on a fresh host — on a real loopback
socket, and talk to it over real HTTP, the same transport the UI uses.

Isolation is done by pointing the settings at ``tmp_path`` through the
environment rather than by injecting an ``AppState``: uvicorn always runs the
lifespan, which rebuilds that state from the settings, so an injected one would
be silently discarded and the test would quietly exercise ``data/processed``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.main import create_app
from embedded_api import (
    EmbeddedApiError,
    _bind_loopback_socket,
    ensure_index,
    start_embedded_api,
)

QUESTION = "How many days of paid annual leave are employees entitled to?"


@pytest.fixture
def hermetic_app(tmp_path: Path, raw_dir: Path,
                 monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    """The real app, with no index yet — what a fresh deployment looks like."""
    processed = tmp_path / "processed"
    for key, value in {
        "DATA_DIR": tmp_path,
        "RAW_DIR": raw_dir,
        "PROCESSED_DIR": processed,
        "EVALUATION_DIR": tmp_path / "evaluation",
        "CHROMA_DIR": processed / "chroma",
        "BM25_DIR": processed / "bm25",
        "VECTOR_BACKEND": "numpy",
        "EMBEDDING_BACKEND": "tfidf_svd",
        "RERANKER_BACKEND": "lexical",
        "LLM_PROVIDER": "extractive",
        "CHUNK_SIZE_TOKENS": "200",
        "CHUNK_OVERLAP_TOKENS": "40",
    }.items():
        monkeypatch.setenv(key, str(value))

    get_settings.cache_clear()
    try:
        yield create_app()
    finally:
        get_settings.cache_clear()


# ------------------------------------------------------------------- socket
def test_the_port_is_bound_before_it_is_handed_over() -> None:
    """Picking a free port and binding it later is a race; this does not."""
    sock = _bind_loopback_socket()
    try:
        host, port = sock.getsockname()
        assert host == "127.0.0.1"
        assert port > 0
        with pytest.raises(OSError):      # already taken, because we still hold it
            _bind_loopback_socket().bind(("127.0.0.1", port))
    finally:
        sock.close()


# ---------------------------------------------------------------- first boot
def test_a_fresh_deployment_serves_http_builds_an_index_and_answers(
        hermetic_app: FastAPI) -> None:
    """The demo's whole first-visit path, end to end, over a socket.

    ``data/processed`` is not committed — only ``data/raw`` is — so a fresh
    deployment starts with nothing indexed. A visitor must not have to notice
    that, let alone click a button to fix it.
    """
    base_url = start_embedded_api(hermetic_app, timeout_s=60)
    assert base_url.startswith("http://127.0.0.1:")
    assert httpx.get(f"{base_url}/health", timeout=30).json()["index_ready"] is False

    health = ensure_index(base_url)

    assert health["status"] == "ok"
    assert health["index_ready"] is True
    assert health["n_chunks"] > 0

    answer = httpx.post(f"{base_url}/query", timeout=60, json={
        "question": QUESTION, "top_k": 5,
        "retrieval_method": "hybrid", "rerank": True}).json()

    assert "22 days" in answer["answer"]
    assert answer["citations"]


def test_ensure_index_is_idempotent(hermetic_app: FastAPI) -> None:
    """Every rerun of the Streamlit script must not rebuild the corpus."""
    base_url = start_embedded_api(hermetic_app, timeout_s=60)
    first = ensure_index(base_url)

    second = ensure_index(base_url)

    assert second["index_ready"] is True
    assert second["n_chunks"] == first["n_chunks"]


def test_a_dead_host_is_reported_not_swallowed() -> None:
    with pytest.raises(EmbeddedApiError, match="Could not read /health"):
        ensure_index("http://127.0.0.1:1")


# ------------------------------------------------------------------- the UI
def test_the_ui_talks_to_the_same_endpoints_either_way(hermetic_app: FastAPI) -> None:
    """Embedded mode changes the transport, not the contract.

    Every path the UI calls must exist on the app that gets embedded, or the
    demo would diverge from the deployment the rest of the README describes.
    """
    ui_source = (Path(__file__).resolve().parents[2]
                 / "frontend" / "streamlit_app.py").read_text(encoding="utf-8")

    with TestClient(hermetic_app) as client:
        for path in ("/health", "/documents"):
            assert f'"{path}"' in ui_source
            assert client.get(path).status_code == 200
        for path in ("/query", "/documents/index", "/documents/upload"):
            assert f'"{path}"' in ui_source
            assert client.post(path).status_code != 404
