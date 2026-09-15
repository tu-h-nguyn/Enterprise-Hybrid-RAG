"""The two network-backed LLM providers, against a real local HTTP server.

These are the code paths that run in production — with OpenAI, with Anthropic,
or with a local Ollama server — and they were previously the only substantial
part of the codebase with no test at all, because exercising them appeared to
need a network and a key. It does not: a socket on localhost speaking the same
protocol exercises the same code.

What is asserted is the part that actually breaks when a provider is swapped:
the URL each provider posts to, the headers it authenticates with, the body it
sends, how it reads the reply, and what it does when the reply is an error or
is malformed. The Anthropic case matters twice over, because the claim in the
module docstring — that the abstraction is not OpenAI-shaped by accident — is
only true if a genuinely different endpoint, header set and body shape work.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from app.generation.llm import AnthropicLLM, ExtractiveLLM, LLMError, LLMRequest, OpenAICompatibleLLM


class _Recorder:
    """Captures what the client sent, and decides what to send back."""

    def __init__(self) -> None:
        self.path: str = ""
        self.headers: dict[str, str] = {}
        self.body: dict[str, Any] = {}
        self.status: int = 200
        self.response: Any = {}


def _make_server(recorder: _Recorder) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        # http.server dictates this method name; it is not project style.
        def do_POST(self) -> None:
            recorder.path = self.path
            recorder.headers = {k.lower(): v for k, v in self.headers.items()}
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                recorder.body = json.loads(raw)
            except json.JSONDecodeError:
                recorder.body = {}
            payload = (recorder.response if isinstance(recorder.response, (str, bytes))
                       else json.dumps(recorder.response)).encode()
            self.send_response(recorder.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            return  # keep pytest output clean

    return HTTPServer(("127.0.0.1", 0), Handler)


@pytest.fixture
def server() -> Iterator[tuple[str, _Recorder]]:
    recorder = _Recorder()
    httpd = _make_server(recorder)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}/v1", recorder
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _request() -> LLMRequest:
    return LLMRequest(system="You are precise.", prompt="How many days of leave?",
                      question="How many days of leave?", context="[Source 1]\n22 days.",
                      temperature=0.0, max_tokens=256)


# ------------------------------------------------------------- OpenAI-shaped
def test_openai_posts_to_chat_completions_with_bearer_auth(server) -> None:
    base_url, rec = server
    rec.response = {"choices": [{"message": {"content": "22 days [Source 1]."}}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 9}}

    llm = OpenAICompatibleLLM("gpt-4o-mini", base_url, "sk-test", timeout_s=10)
    response = llm.generate(_request())

    assert rec.path == "/v1/chat/completions"
    assert rec.headers["authorization"] == "Bearer sk-test"
    assert response.text == "22 days [Source 1]."
    assert response.provider == "openai_compatible"
    assert response.model == "gpt-4o-mini"
    assert response.usage == {"prompt_tokens": 40, "completion_tokens": 9}
    assert response.latency_ms > 0


def test_openai_sends_system_and_user_as_separate_messages(server) -> None:
    """The grounding rules live in the system message; folding them into the
    user turn would quietly change how strongly the model follows them."""
    base_url, rec = server
    rec.response = {"choices": [{"message": {"content": "ok"}}]}

    OpenAICompatibleLLM("m", base_url, "k", timeout_s=10).generate(_request())

    assert rec.body["model"] == "m"
    assert rec.body["temperature"] == 0.0
    assert rec.body["max_tokens"] == 256
    assert [m["role"] for m in rec.body["messages"]] == ["system", "user"]
    assert rec.body["messages"][0]["content"] == "You are precise."
    assert "How many days of leave?" in rec.body["messages"][1]["content"]


def test_openai_turns_an_http_error_into_llm_error(server) -> None:
    base_url, rec = server
    rec.status = 429
    rec.response = {"error": {"message": "rate limited"}}

    llm = OpenAICompatibleLLM("m", base_url, "k", timeout_s=10)
    with pytest.raises(LLMError) as excinfo:
        llm.generate(_request())
    assert "429" in str(excinfo.value)


def test_openai_rejects_an_unexpected_response_shape(server) -> None:
    """A 200 carrying the wrong JSON must fail loudly, not return an empty
    answer that would read downstream as a model with nothing to say."""
    base_url, rec = server
    rec.response = {"unexpected": True}

    llm = OpenAICompatibleLLM("m", base_url, "k", timeout_s=10)
    with pytest.raises(LLMError, match="Unexpected LLM response shape"):
        llm.generate(_request())


def test_openai_tolerates_a_null_content(server) -> None:
    base_url, rec = server
    rec.response = {"choices": [{"message": {"content": None}}]}
    assert OpenAICompatibleLLM("m", base_url, "k", timeout_s=10).generate(_request()).text == ""


def test_openai_refuses_to_start_without_a_key() -> None:
    """Better to fail at construction than to send unauthenticated requests to
    something that may well be a paid endpoint."""
    with pytest.raises(LLMError, match="No API key"):
        OpenAICompatibleLLM("m", "http://localhost:1", "")


def test_openai_strips_a_trailing_slash_from_the_base_url(server) -> None:
    base_url, rec = server
    rec.response = {"choices": [{"message": {"content": "ok"}}]}

    OpenAICompatibleLLM("m", base_url + "/", "k", timeout_s=10).generate(_request())

    assert rec.path == "/v1/chat/completions"  # not //chat/completions


# ----------------------------------------------------------------- Anthropic
def test_anthropic_uses_a_different_endpoint_headers_and_body(server) -> None:
    """The point of carrying a second provider: if the abstraction had been
    OpenAI-shaped by accident, this test could not pass."""
    base_url, rec = server
    rec.response = {"content": [{"type": "text", "text": "22 days [Source 1]."}],
                    "usage": {"input_tokens": 51, "output_tokens": 7}}

    llm = AnthropicLLM("claude-sonnet-4-5", base_url, "sk-ant-test", timeout_s=10)
    response = llm.generate(_request())

    assert rec.path == "/v1/messages"                      # not /chat/completions
    assert rec.headers["x-api-key"] == "sk-ant-test"       # not Authorization
    assert rec.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in rec.headers
    assert rec.body["system"] == "You are precise."        # top level, not a message
    assert [m["role"] for m in rec.body["messages"]] == ["user"]

    assert response.text == "22 days [Source 1]."
    assert response.provider == "anthropic"
    assert response.usage == {"prompt_tokens": 51, "completion_tokens": 7}


def test_anthropic_concatenates_text_blocks_and_ignores_other_types(server) -> None:
    base_url, rec = server
    rec.response = {"content": [{"type": "text", "text": "22 days "},
                                {"type": "thinking", "text": "<ignored>"},
                                {"type": "text", "text": "[Source 1]."}]}

    text = AnthropicLLM("m", base_url, "k", timeout_s=10).generate(_request()).text

    assert text == "22 days [Source 1]."


def test_anthropic_turns_an_http_error_into_llm_error(server) -> None:
    base_url, rec = server
    rec.status = 500
    rec.response = {"error": "boom"}

    with pytest.raises(LLMError) as excinfo:
        AnthropicLLM("m", base_url, "k", timeout_s=10).generate(_request())
    assert "500" in str(excinfo.value)


def test_anthropic_handles_a_reply_with_no_content(server) -> None:
    base_url, rec = server
    rec.response = {"usage": {"input_tokens": 3, "output_tokens": 0}}

    response = AnthropicLLM("m", base_url, "k", timeout_s=10).generate(_request())

    assert response.text == ""
    assert response.usage == {"prompt_tokens": 3, "completion_tokens": 0}


def test_anthropic_refuses_to_start_without_a_key() -> None:
    with pytest.raises(LLMError, match="No API key"):
        AnthropicLLM("m", "http://localhost:1", "")


# ------------------------------------------------------------------ shared
def test_an_unreachable_server_becomes_llm_error() -> None:
    """Port 1 is reserved and refuses connections, so this exercises the
    transport-error branch rather than an HTTP status."""
    llm = OpenAICompatibleLLM("m", "http://127.0.0.1:1/v1", "k", timeout_s=2)
    with pytest.raises(LLMError, match="LLM request failed"):
        llm.generate(_request())


def test_every_provider_describes_itself_for_the_evaluation_manifest() -> None:
    """Evaluation artefacts record describe(); a provider that misreports here
    would let one backend's numbers be published under another's name."""
    assert OpenAICompatibleLLM("gpt-4o-mini", "http://x", "k").describe() == {
        "provider": "openai_compatible", "model": "gpt-4o-mini"}
    assert AnthropicLLM("claude-sonnet-4-5", "http://x", "k").describe() == {
        "provider": "anthropic", "model": "claude-sonnet-4-5"}
    assert ExtractiveLLM().describe() == {
        "provider": "extractive", "model": "extractive-offline"}
