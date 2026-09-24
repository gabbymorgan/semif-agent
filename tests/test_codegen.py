"""Pure-stdlib tests for skill code-body generation.

Prompt building, draft parsing/validation, body persistence + import, and
tree hot-merge all run without SemIf or a real LLM. The only network usage is a
throwaway stdlib HTTP server that stands in for an OpenAI-compatible endpoint —
the CodegenClient itself is real, not mocked.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from semif_agent.codegen import (
    CodegenClient,
    CodegenError,
    build_skill_body_prompt,
    generate_skill_body,
    parse_skill_body,
    read_skill_contract,
)
from semif_agent.decisions import Request
from semif_agent.skills import (
    SkillBodyStore,
    SkillDraft,
    build_skills,
    build_tree,
    load_skill_module,
    materialize_skill,
    merge_skill_bodies,
    merge_registry,
)

GOOD_BODY = """\
from semif_agent.decisions import DecisionRequest, Option
from semif_agent.skills import ActionResult, Prediction

def predict(ctx, request):
    return Prediction(text="ok", decisions=[])

def act(ctx, request, prediction):
    return ActionResult(action_log="probe ran", new_state=request.text)
"""


def test_read_skill_contract_loads_contract():
    text = read_skill_contract()
    assert "predict" in text and "act" in text
    assert "data/skills" in text


def test_build_skill_body_prompt_includes_contract_request_and_draft():
    tree = build_tree(build_skills({"skills": {}}))
    draft = SkillDraft(name="probe", description="Probe the service.")
    messages = build_skill_body_prompt(
        Request("check if the service is up"), "tracking", draft, tree, "THE CONTRACT"
    )
    assert messages[0]["role"] == "system"
    assert "THE CONTRACT" in messages[0]["content"]
    joined = messages[1]["content"]
    assert "check if the service is up" in joined
    assert "probe" in joined
    assert "tracking.check" in joined


@pytest.mark.parametrize(
    "raw",
    [
        GOOD_BODY,
        "```python\n" + GOOD_BODY + "\n```",
        json.dumps({"code": GOOD_BODY}),
        "Here you go:\n```python\n" + GOOD_BODY + "\n```\nHope that helps.",
        'Sure: ' + json.dumps({"code": GOOD_BODY}) + ' (that was it)',
    ],
)
def test_parse_skill_body_accepts_forms(raw):
    code = parse_skill_body(raw)
    assert "def predict" in code and "def act" in code


def test_parse_skill_body_rejects_empty():
    with pytest.raises(ValueError):
        parse_skill_body("")


def test_parse_skill_body_rejects_invalid_python():
    with pytest.raises(ValueError):
        parse_skill_body("def predict(:\n  pass")


def test_parse_skill_body_rejects_missing_functions():
    with pytest.raises(ValueError):
        parse_skill_body("def predict(ctx, request):\n    return None")


def test_parse_skill_body_rejects_missing_act():
    with pytest.raises(ValueError):
        parse_skill_body("def predict(ctx, request):\n    return None\nx = 1")


def test_body_store_roundtrip(tmp_path):
    store = SkillBodyStore(str(tmp_path / "skills"))
    assert store.list_bodies() == []
    store.write("tracking", "probe", GOOD_BODY)
    assert store.list_bodies() == [("tracking", "probe")]
    target = store.body_path("tracking", "probe")
    assert target.is_file()
    assert "def predict" in target.read_text()


def test_load_skill_module_exposes_predict_act(tmp_path):
    store = SkillBodyStore(str(tmp_path / "skills"))
    store.write("tracking", "probe", GOOD_BODY)
    module = load_skill_module("tracking", "probe", store.path)
    assert callable(module.predict) and callable(module.act)


def test_materialize_skill_builds_runnable_skill(tmp_path):
    store = SkillBodyStore(str(tmp_path / "skills"))
    draft = SkillDraft(name="probe", description="Probe the service.", code=GOOD_BODY)
    skill = materialize_skill(draft, "tracking", store)
    assert skill.name == "probe"
    assert skill.category == "tracking"
    assert callable(skill.predict) and callable(skill.act)


def test_materialize_skill_requires_code(tmp_path):
    store = SkillBodyStore(str(tmp_path / "skills"))
    draft = SkillDraft(name="probe", description="Probe the service.")
    with pytest.raises(ValueError):
        materialize_skill(draft, "tracking", store)


def test_materialize_skill_rejects_import_failure(tmp_path):
    store = SkillBodyStore(str(tmp_path / "skills"))
    bad = "def predict(ctx, request):\n    return None\n"
    draft = SkillDraft(name="probe", description="Probe.", code=bad)
    with pytest.raises(ValueError):
        materialize_skill(draft, "tracking", store)


def test_merge_skill_bodies_upgrades_stub(tmp_path):
    store = SkillBodyStore(str(tmp_path / "skills"))
    store.write("tracking", "probe", GOOD_BODY)
    tree = build_tree(build_skills({"skills": {}}))
    registry = {"tracking": {"description": "", "skills": [{"name": "probe", "description": "Probe."}]}}
    merge_registry(tree, registry)
    upgraded = merge_skill_bodies(tree, store, registry)
    assert upgraded == 1
    skill = next(s for s in tree["tracking"] if s.name == "probe")
    assert callable(skill.predict) and callable(skill.act)
    assert skill.description == "Probe."


def test_merge_skill_bodies_creates_missing_category(tmp_path):
    store = SkillBodyStore(str(tmp_path / "skills"))
    store.write("brand_new", "ping", GOOD_BODY)
    tree = build_tree(build_skills({"skills": {}}))
    upgraded = merge_skill_bodies(tree, store, {})
    assert upgraded == 1
    assert tree["brand_new"][0].name == "ping"


class _FakeOpenAI(BaseHTTPRequestHandler):
    reply: str = GOOD_BODY
    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        type(self).received.append(json.loads(raw))
        body = json.dumps({"choices": [{"message": {"content": self.reply}}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _fake_server(reply: str) -> tuple[ThreadingHTTPServer, str]:
    handler = type("Handler", (_FakeOpenAI,), {"reply": reply, "received": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


def test_generate_skill_body_end_to_end(tmp_path):
    httpd, base = _fake_server(GOOD_BODY)
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        tree = build_tree(build_skills({"skills": {}}))
        draft = SkillDraft(name="probe", description="Probe the service.")
        code = generate_skill_body(client, Request("is the service up?"), "tracking", draft, tree)
        assert "def predict" in code and "def act" in code
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_generate_skill_body_retries_then_fails(tmp_path):
    httpd, base = _fake_server("this is not python at all")
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        tree = build_tree(build_skills({"skills": {}}))
        draft = SkillDraft(name="probe", description="Probe the service.")
        with pytest.raises(ValueError):
            generate_skill_body(client, Request("is the service up?"), "tracking", draft, tree)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_codegen_client_unreachable_raises(tmp_path):
    client = CodegenClient(base_url="http://127.0.0.1:1/v1", model="test", timeout=2)
    tree = build_tree(build_skills({"skills": {}}))
    draft = SkillDraft(name="probe", description="Probe the service.")
    with pytest.raises(CodegenError):
        generate_skill_body(client, Request("is the service up?"), "tracking", draft, tree)


class _SilentOpenAI(BaseHTTPRequestHandler):
    """Accepts the request but never replies; the client must time out."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        time.sleep(5)

    def log_message(self, format, *args):
        pass


class _StalledStreamOpenAI(BaseHTTPRequestHandler):
    """Sends response headers then holds the connection with zero body bytes;
    the streaming client must hit its idle/stall watchdog."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.flush()
        time.sleep(5)

    def log_message(self, format, *args):
        pass


def test_chat_timeout_raises_codegen_error():
    handler = type("Handler", (_SilentOpenAI,), {})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = CodegenClient(
            base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1",
            model="test",
            timeout=0.5,
        )
        with pytest.raises(CodegenError, match="timed out"):
            client.chat([{"role": "user", "content": "hi"}])
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_chat_stream_stall_fails_fast():
    """A silent stream must fail at idle_timeout, not the total budget."""
    handler = type("Handler", (_StalledStreamOpenAI,), {})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = CodegenClient(
            base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1",
            model="test",
            timeout=10,
            stream=True,
            idle_warn=1.0,
            idle_timeout=2.0,
        )
        started = time.monotonic()
        with pytest.raises(CodegenError, match="stalled"):
            client.chat([{"role": "user", "content": "hi"}])
        elapsed = time.monotonic() - started
        assert elapsed < 8.0, f"stall should fail fast, took {elapsed:.1f}s"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_chat_stream_stall_warns_before_failing(capsys):
    handler = type("Handler", (_StalledStreamOpenAI,), {})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = CodegenClient(
            base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1",
            model="test",
            timeout=10,
            stream=True,
            idle_warn=1.0,
            idle_timeout=3.0,
        )
        with pytest.raises(CodegenError, match="stalled"):
            client.chat([{"role": "user", "content": "hi"}])
        captured = capsys.readouterr().out
        assert "no tokens for" in captured
        assert "will fail after" in captured
    finally:
        httpd.shutdown()
        httpd.server_close()


def _sse_frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class _RunawayStreamOpenAI(BaseHTTPRequestHandler):
    """Streams content frames forever, never sending `[DONE]`.

    Stands in for a degenerated generation that keeps producing tokens
    without finishing; the client must trip its total wall-clock budget
    even while bytes are still flowing. Drops out quietly when the client
    aborts the connection.
    """

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.flush()
        try:
            while True:
                frame = _sse_frame(
                    {"choices": [{"delta": {"content": "x"}}]}
                )
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format, *args):
        pass


def test_chat_stream_total_budget_fires_while_streaming():
    """A continuously-streaming runaway must be cut short by the total
    budget, not loop forever. idle_timeout (0.5s) < timeout (1.0s) makes the
    assertion meaningful: had the stream gone silent, "stalled" would have
    fired first — matching "total budget" proves the check ran while tokens
    were still flowing."""
    handler = type("Handler", (_RunawayStreamOpenAI,), {})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = CodegenClient(
            base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1",
            model="test",
            timeout=1.0,
            stream=True,
            idle_warn=0.25,
            idle_timeout=0.5,
        )
        started = time.monotonic()
        with pytest.raises(CodegenError, match="total budget"):
            client.chat([{"role": "user", "content": "hi"}])
        elapsed = time.monotonic() - started
        assert 0.9 <= elapsed < 5.0, f"budget should fire ~1s in, took {elapsed:.1f}s"
    finally:
        httpd.shutdown()
        httpd.server_close()


class _ForeverFrames:
    """Infinite iterable of valid content SSE lines (no `[DONE]`)."""

    def __iter__(self):
        return self

    def __next__(self):
        time.sleep(0.05)
        return _sse_frame({"choices": [{"delta": {"content": "x"}}]}).encode("utf-8")


def test_read_stream_blocking_enforces_total_budget():
    """The blocking fallback must enforce the total budget too — its
    per-read socket timeout bounds individual reads, not the whole stream."""
    client = CodegenClient(
        base_url="http://unused", model="test", timeout=0.5, stream=True
    )
    started = time.monotonic()
    with pytest.raises(CodegenError, match="total budget"):
        client._read_stream_blocking(_ForeverFrames())
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"blocking fallback should fail fast, took {elapsed:.1f}s"


class _StreamingOpenAI(BaseHTTPRequestHandler):
    """Replies with an OpenAI-compatible SSE token stream (COT then content).

    Reasoning field name matches the backend: ollama emits `reasoning`,
    DeepSeek/vllm-style `reasoning_content`. Defaults to ollama's.
    """

    reasoning: str = "thinking about the body..."
    reasoning_key: str = "reasoning"
    content: str = GOOD_BODY
    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        type(self).received.append(json.loads(raw))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        reasoning = type(self).reasoning
        reasoning_key = type(self).reasoning_key
        content = type(self).content
        step = max(len(reasoning) // 4, 1)
        for i in range(0, len(reasoning), step):
            frame = _sse_frame(
                {"choices": [{"delta": {reasoning_key: reasoning[i : i + step]}}]}
            )
            self.wfile.write(frame.encode("utf-8"))
        step = max(len(content) // 4, 1)
        for i in range(0, len(content), step):
            frame = _sse_frame(
                {"choices": [{"delta": {"content": content[i : i + step]}}]}
            )
            self.wfile.write(frame.encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, format, *args):
        pass


def _streaming_server(reasoning_key: str = "reasoning") -> tuple[ThreadingHTTPServer, str]:
    handler = type(
        "Handler", (_StreamingOpenAI,), {"reasoning_key": reasoning_key, "received": []}
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


def test_chat_stream_accumulates_full_content():
    httpd, base = _streaming_server()
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10, stream=True)
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == GOOD_BODY, "streamed deltas must reassemble the full body"
        body = httpd.RequestHandlerClass.received[0]
        assert body["stream"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize("reasoning_key", ["reasoning", "reasoning_content"])
def test_chat_stream_verbose_echoes_tokens(reasoning_key, capsys):
    httpd, base = _streaming_server(reasoning_key=reasoning_key)
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10, stream=True)
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == GOOD_BODY
        captured = capsys.readouterr().out
        assert captured == _StreamingOpenAI.reasoning + GOOD_BODY + "\n"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_chat_omits_max_tokens_by_default():
    httpd, base = _fake_server(GOOD_BODY)
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        client.chat([{"role": "user", "content": "hi"}])
        body = httpd.RequestHandlerClass.received[0]
        assert "max_tokens" not in body
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_chat_includes_max_tokens_when_set():
    httpd, base = _fake_server(GOOD_BODY)
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        client.chat([{"role": "user", "content": "hi"}], max_tokens=512)
        body = httpd.RequestHandlerClass.received[0]
        assert body["max_tokens"] == 512
    finally:
        httpd.shutdown()
        httpd.server_close()