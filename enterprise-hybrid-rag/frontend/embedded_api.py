"""Run the API inside the Streamlit process, for single-process hosts.

Streamlit Community Cloud and Hugging Face Spaces run **one** process: there is
nowhere to put a separate `uvicorn` for the UI to call. Without this module the
project cannot be demonstrated at a URL, only described — which is the one thing
a reader will not do.

The alternative was to let the UI import ``RagService`` and call it directly.
That was rejected: the UI would then hold a second, parallel path into
retrieval, and the thing a visitor sees would no longer be the thing the API
serves. Instead the real ASGI app is started on a loopback socket in a daemon
thread, and the UI keeps talking HTTP to the same endpoints it always did. The
transport changes; the contract does not.

Port selection binds ``127.0.0.1:0`` and hands the already-listening socket to
uvicorn, rather than picking a free port and hoping it is still free a moment
later.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Generous, because a cold start may download ~90 MB of model weights before
#: the app will answer /health.
STARTUP_TIMEOUT_S = 180.0
#: Ingesting 22 documents and building both indexes, on a small shared host.
INDEX_TIMEOUT_S = 900.0


class EmbeddedApiError(RuntimeError):
    """The in-process API did not come up, or could not build an index."""


def _bind_loopback_socket() -> socket.socket:
    """A listening socket on an OS-assigned port, handed straight to uvicorn."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    return sock


def start_embedded_api(app: FastAPI | None = None,
                       timeout_s: float = STARTUP_TIMEOUT_S) -> str:
    """Start the real FastAPI app on loopback and return its base URL.

    Blocks until ``/health`` answers, so the caller never races the server. The
    thread is a daemon: when Streamlit exits, so does the API.

    ``app`` defaults to the application this project serves; it is a parameter
    only so a test can start a hermetic one instead of whatever happens to be
    in ``data/processed``.
    """
    import uvicorn

    if app is None:
        from app.main import app as default_app
        app = default_app

    sock = _bind_loopback_socket()
    port = int(sock.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]},
                             name="embedded-api", daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise EmbeddedApiError(
                "The embedded API thread exited during startup. Run "
                "`uvicorn app.main:app` directly to see the traceback.")
        try:
            response = httpx.get(f"{base_url}/health", timeout=5.0)
            if response.status_code == 200:
                return base_url
        except httpx.HTTPError:
            pass
        time.sleep(0.25)

    raise EmbeddedApiError(
        f"The embedded API did not answer /health within {timeout_s:.0f}s.")


def ensure_index(base_url: str, timeout_s: float = INDEX_TIMEOUT_S) -> dict:
    """Build the index if the host has none, and return the resulting health.

    ``data/processed`` is not committed — only ``data/raw`` is — so a fresh
    deployment starts with no index at all. A visitor should not have to notice
    that, let alone click a button to fix it, so the first caller pays for the
    build and everyone after it finds the index ready.
    """
    health = _health(base_url)
    if health.get("index_ready"):
        return health

    try:
        response = httpx.post(f"{base_url}/documents/index", json={"rebuild": True},
                              timeout=timeout_s)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise EmbeddedApiError(f"Could not build the index: {exc}") from exc

    health = _health(base_url)
    if not health.get("index_ready"):
        raise EmbeddedApiError(
            "The index build reported success but /health still says no index. "
            f"Health: {json.dumps(health)[:300]}")
    return health


def _health(base_url: str) -> dict:
    try:
        response = httpx.get(f"{base_url}/health", timeout=30.0)
        response.raise_for_status()
        return dict(response.json())
    except httpx.HTTPError as exc:
        raise EmbeddedApiError(f"Could not read /health from {base_url}: {exc}") from exc
