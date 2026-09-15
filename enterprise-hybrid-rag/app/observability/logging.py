"""Structured logging and per-stage timing (Phase 18).

Two rules drive this module:

* Logs are JSON by default so they are queryable in any log stack.
* Chunk text is **never** logged unless ``LOG_CHUNK_TEXT=true``. Indexed
  documents are usually confidential; a debug log that silently copies contract
  clauses into a log aggregator is an incident, not a nuisance.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output
                         else logging.Formatter("%(levelname)s %(name)s :: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def redact_chunk(text: str, allowed: bool) -> str:
    """Return chunk text only when explicitly permitted by configuration."""
    return text if allowed else f"<{len(text)} chars redacted>"


class Stopwatch:
    """Accumulates wall-clock time per pipeline stage.

    Used as ``with watch("retrieval"): ...``. Repeated use of the same label
    accumulates rather than overwrites, so a stage invoked twice in one query
    reports its total cost.
    """

    def __init__(self) -> None:
        self._stages: dict[str, float] = {}
        self._started = time.perf_counter()

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self._stages[stage] = round(self._stages.get(stage, 0.0) + elapsed, 2)

    @property
    def stages(self) -> dict[str, float]:
        return dict(self._stages)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000.0, 2)
