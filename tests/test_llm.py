"""Pure-stdlib tests for the `llm` authoring provider.

A throwaway stdlib HTTP server stands in for the OpenAI-compatible endpoint —
the LLMClient itself is real, not mocked.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from semif_agent.decisions import Request
from semif_agent.llm import LLMClient, LLMError
from semif_agent.skills import (
    CategoryDraft,
    SkillDraft,
    build_skills,
    build_tree,
    generate_category,
    generate_skill,
)


def _sse_frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class _FakeOpenAI(BaseHTTPRequestHandler):
    """Single-shot SSE server: the reply arrives as one content delta. The
    `/api/show` window query is answered too but never recorded."""

    reply: str = ""
    received: list = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        if self.path.rstrip("/").endswith("/api/show"):
            body = json.dumps({"parameters": {"num_ctx": 32768}}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        type(self).received.append(json.loads(raw))
        body = (
            _sse_frame({"choices": [{"delta": {"content": self.reply}}]})
            + "data: [DONE]\n\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        pass


def _fake_server(reply: str) -> tuple[ThreadingHTTPServer, str]:
    handler = type("Handler", (_FakeOpenAI,), {"reply": reply, "received": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


def test_parse_json_extracts_object_from_prose():
    parsed = LLMClient._parse_json('Sure, here: {"a": 1, "b": "two"} — done.')
    assert parsed == {"a": 1, "b": "two"}


def test_parse_json_handles_bare_object():
    assert LLMClient._parse_json('{"ok": true}') == {"ok": True}


def test_chat_returns_streamed_content():
    httpd, base = _fake_server('{"title": "track_live", "description": "Follow it."}')
    try:
        client = LLMClient(base_url=base, model="test", timeout=10)
        out = client.chat([{"role": "user", "content": "name this"}])
        assert out == '{"title": "track_live", "description": "Follow it."}'
        assert httpd.RequestHandlerClass.received[0]["model"] == "test"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_generate_category_via_provider():
    httpd, base = _fake_server(
        '{"title": "Track Delivery", "description": "Track parcels end to end."}'
    )
    try:
        client = LLMClient(base_url=base, model="test", timeout=10)
        draft = generate_category(
            client, Request("where is my parcel?"), build_tree(build_skills({"skills": {}}))
        )
        assert isinstance(draft, CategoryDraft)
        assert draft.name == "track_delivery"
        assert draft.description == "Track parcels end to end."
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_generate_skill_via_provider():
    httpd, base = _fake_server(
        '{"title": "track.live", "description": "Follow a package in real time."}'
    )
    try:
        client = LLMClient(base_url=base, model="test", timeout=10)
        draft = generate_skill(
            client,
            Request("track my drone delivery"),
            "tracking",
            build_tree(build_skills({"skills": {}})),
        )
        assert isinstance(draft, SkillDraft)
        assert draft.name == "track.live"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_unreachable_endpoint_raises_llm_error():
    client = LLMClient(base_url="http://127.0.0.1:1/v1", model="test", timeout=2)
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}])
