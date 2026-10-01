"""Stdlib tests for the OpenCode Console provider.

A throwaway stdlib HTTP server stands in for the Console endpoint; the client
is real, not mocked. These are hermetic mechanics checks (bearer header, the
skipped ollama probe, the preserved error contracts, the model-list parse) —
only a real Console call proves the hosted integration. No external network.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_agent.codegen import CodegenError, DegenerationError
from semif_agent.console import (
    OPENCODE_BASE_URL,
    ConsoleCodegenClient,
    ConsoleLLMClient,
    OpenCodeConsoleClient,
    OpenCodeError,
)
from semif_agent.llm import LLMClient, LLMError
from semif_agent.provider import ProviderError


def _sse_frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class _FakeConsole(BaseHTTPRequestHandler):
    """Records every request (method, path, lowercased headers). Answers GET
    /models with an OpenAI list and POST with a one-delta SSE reply."""

    reply: str = "{}"
    models: list = ["kimi-k2.7-code", "glm-5.3"]
    calls: list = []

    def _record(self) -> None:
        type(self).calls.append(
            (self.command, self.path, {k.lower(): v for k, v in self.headers.items()})
        )

    def do_GET(self) -> None:
        self._record()
        body = json.dumps(
            {"object": "list", "data": [{"id": m} for m in self.models]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self._record()
        body = (
            _sse_frame({"choices": [{"delta": {"content": self.reply}}]})
            + "data: [DONE]\n\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


def _server(reply: str = "{}", models: list | None = None):
    attrs = {"reply": reply, "calls": []}
    if models is not None:
        attrs["models"] = models
    handler = type("Handler", (_FakeConsole,), attrs)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_console_llm_pins_defaults_and_error_contract():
    client = ConsoleLLMClient(api_key="k", model="m")
    assert client.base_url == OPENCODE_BASE_URL
    assert client.label == "opencode:llm"
    assert client.error_class is LLMError
    assert client.degeneration_error_class is LLMError
    assert client.query_context is False
    # Console rejects reasoning_effort → forced off despite LLMClient's default.
    assert client.disable_thinking is False


def test_console_llm_never_sends_reasoning_effort_even_if_requested():
    client = ConsoleLLMClient(api_key="k", model="m", disable_thinking=True)
    assert client.disable_thinking is False


def test_console_codegen_pins_defaults_and_error_contract():
    client = ConsoleCodegenClient(api_key="k", model="m")
    assert client.base_url == OPENCODE_BASE_URL
    assert client.label == "opencode:codegen"
    assert client.error_class is CodegenError
    assert client.degeneration_error_class is DegenerationError
    assert client.query_context is False
    assert client.presence_penalty == 1.5  # CodegenClient sampler default preserved


def test_console_error_is_provider_error():
    assert issubclass(OpenCodeError, ProviderError)


def test_chat_sends_bearer_and_skips_ollama_probe():
    httpd, base = _server(reply='{"ok": true}')
    try:
        client = ConsoleLLMClient(base_url=base, api_key="secret", model="m", timeout=10)
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == '{"ok": true}'
        calls = httpd.RequestHandlerClass.calls
        # query_context=False → the only request is the chat completion.
        assert [c[1] for c in calls] == ["/chat/completions"]
        assert calls[0][2]["authorization"] == "Bearer secret"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_plain_llm_without_key_sends_no_auth_header():
    httpd, base = _server(reply="{}")
    try:
        LLMClient(base_url=base, model="m", timeout=10).chat(
            [{"role": "user", "content": "hi"}]
        )
        chat = [
            c for c in httpd.RequestHandlerClass.calls if c[1].endswith("/chat/completions")
        ]
        assert chat and "authorization" not in chat[0][2]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_extra_headers_are_merged():
    httpd, base = _server(reply="{}")
    try:
        client = ConsoleLLMClient(
            base_url=base,
            api_key="k",
            extra_headers={"X-Route": "edge"},
            model="m",
            timeout=10,
        )
        client.chat([{"role": "user", "content": "hi"}])
        headers = httpd.RequestHandlerClass.calls[0][2]
        assert headers["x-route"] == "edge"
        assert headers["authorization"] == "Bearer k"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_console_sends_default_user_agent():
    """The Console gateway 403s urllib's default Python-urllib UA."""
    httpd, base = _server(reply="{}")
    try:
        ConsoleLLMClient(base_url=base, model="m", timeout=10).chat(
            [{"role": "user", "content": "hi"}]
        )
        ua = httpd.RequestHandlerClass.calls[0][2].get("user-agent")
        assert ua and "semif-agent" in ua
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_console_user_agent_can_be_overridden():
    httpd, base = _server(reply="{}")
    try:
        ConsoleLLMClient(
            base_url=base, model="m", timeout=10, user_agent="my-agent/9"
        ).chat([{"role": "user", "content": "hi"}])
        assert httpd.RequestHandlerClass.calls[0][2]["user-agent"] == "my-agent/9"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_list_models_parses_catalog():
    httpd, base = _server(models=["kimi-k2.7-code", "glm-5.3"])
    try:
        client = OpenCodeConsoleClient(base_url=base, api_key="k", model="m", timeout=10)
        assert client.list_models(url=f"{base}/models") == ["kimi-k2.7-code", "glm-5.3"]
        assert httpd.RequestHandlerClass.calls[0][2]["authorization"] == "Bearer k"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_provider_endpoint_selects_console_and_resolves_key(monkeypatch):
    from semif_agent import cli

    monkeypatch.setenv("MY_CONSOLE_KEY", "envkey")
    console = cli._provider_endpoint(
        {"provider": "opencode", "api_key_env": "MY_CONSOLE_KEY"},
        "http://localhost:11434/v1",
    )
    assert console["base_url"] == OPENCODE_BASE_URL
    assert console["api_key"] == "envkey"

    ollama = cli._provider_endpoint({"provider": "ollama"}, "http://localhost:11434/v1")
    assert ollama["base_url"] == "http://localhost:11434/v1"
    assert ollama["api_key"] == ""

    literal = cli._provider_endpoint({"provider": "opencode", "api_key": "lit"}, "x")
    assert literal["api_key"] == "lit"

    explicit = cli._provider_endpoint(
        {"provider": "opencode", "base_url": "https://example.test/v1"}, "x"
    )
    assert explicit["base_url"] == "https://example.test/v1"
