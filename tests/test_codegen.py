"""Pure-stdlib tests for skill code-body generation.

Prompt building, draft parsing/validation, body persistence + import, and
tree hot-merge all run without SemIf or a real LLM. The only network usage is a
throwaway stdlib HTTP server that stands in for an OpenAI-compatible endpoint —
the CodegenClient itself is real, not mocked.
"""

import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from semif_agent.codegen import (
    CodegenClient,
    CodegenError,
    DegenerationError,
    _retry_prompt,
    build_elicitation_prompt,
    build_skill_body_prompt,
    generate_data_contract,
    generate_requirements,
    generate_skill_body,
    generate_skill_tests,
    parse_data_contract,
    parse_elicitation,
    parse_skill_body,
    parse_skill_test,
    read_skill_contract,
    read_testgen_contract,
    regenerate_skill_body,
    run_skill_test,
    skill_contract_ref,
)
from semif_agent.decisions import Request
from semif_agent.skills import (
    SkillDraft,
    SkillStore,
    build_skills,
    build_tree,
    load_skill_module,
    materialize_skill,
    merge_registry,
    merge_skill_store,
)

GOOD_BODY = """\
from semif_agent.decisions import DecisionRequest, Option
from semif_agent.skills import ActionResult, Prediction

def predict(ctx, request):
    return Prediction(text="ok", decisions=[])

def act(ctx, request, prediction):
    return ActionResult(action_log="probe ran", new_state=request.text)
"""

GOOD_CONTRACT = {
    "sender_address": "The email address the message is sent from.",
    "tracking_id": "The package tracking number.",
}

GOOD_TEST = """\
import sys
print("ok")
sys.exit(0)
"""


def test_read_skill_contract_loads_contract():
    text = read_skill_contract()
    assert "predict" in text and "act" in text
    assert "data/skills" in text


def test_skill_contract_ref_returns_git_commit_in_repo():
    """In a git checkout the ref is the real short HEAD sha (revivable with
    `git show <ref>:SKILL.md`), and dirty is a bool. Real subprocess, no
    mocking."""
    repo = Path(__file__).resolve().parent.parent
    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True
    )
    info = skill_contract_ref()
    if git.returncode != 0:
        assert info["ref"] is None
        assert info["dirty"] is None
    else:
        assert info["ref"] == git.stdout.strip()
        assert info["dirty"] in (True, False)


def test_skill_contract_ref_degrades_off_repo(tmp_path):
    """Outside a git checkout the pointer degrades to None rather than a
    non-revivable hash; a missing contract still raises like read_skill_contract."""
    contract = tmp_path / "SKILL.md"
    contract.write_text("contract\n")
    info = skill_contract_ref(str(contract))
    assert info == {"ref": None, "dirty": None}
    with pytest.raises(CodegenError):
        skill_contract_ref(str(tmp_path / "missing.md"))


def test_contract_directs_runner_provided_data():
    """SKILL.md must tell the model that data comes from the runner via
    ctx.config — never embedded, fabricated, or asked of the human. Mock-data
    directives belong in TESTGEN.md, not here."""
    text = read_skill_contract()
    assert "Data comes from the runner, never from you" in text
    assert "`ctx.config`" in text
    assert "never embed or fabricate working values" in text
    assert "internal/mock data model" not in text


def test_testgen_contract_owns_mocking():
    """Mocking/testing has its own contract: TESTGEN.md. It must define the
    flat semantic contract shape and forbid structure/type declarations."""
    text = read_testgen_contract()
    assert "contract.json" in text
    assert "single JSON object" in text
    assert "semantic description" in text
    assert "Do NOT put type declarations" in text


def test_testgen_contract_embeds_fixtures_inline():
    """TESTGEN.md must make the test self-contained: fixture data embedded
    inline as Python literals, no external mock_data.json."""
    text = read_testgen_contract()
    assert "mock_data.json" not in text
    assert "inline" in text
    assert "no external files" in text
    assert "import skill" in text
    assert "Worked example" in text
    assert "FIXTURES" in text


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


def test_build_skill_body_prompt_directs_runner_provided_data():
    tree = build_tree(build_skills({"skills": {}}))
    draft = SkillDraft(name="probe", description="Probe the service.")
    messages = build_skill_body_prompt(
        Request("check if the service is up"), "tracking", draft, tree, "THE CONTRACT"
    )
    joined = messages[1]["content"]
    assert "owns no working data" in joined
    assert "`ctx.config`" in joined
    assert "never embed or fabricate working values" in joined
    assert "internal/mock data model" not in joined


def test_build_skill_body_prompt_includes_requirements():
    tree = build_tree(build_skills({"skills": {}}))
    draft = SkillDraft(name="probe", description="Probe the service.")
    messages = build_skill_body_prompt(
        Request("check if the service is up"),
        "tracking",
        draft,
        tree,
        "THE CONTRACT",
        requirements={"Should it actually send, or draft first?": "draft first"},
    )
    joined = messages[1]["content"]
    assert "Requirements gathered from the product owner" in joined
    assert "draft first" in joined


def test_retry_prompt_directs_runner_provided_data():
    messages = _retry_prompt(
        Request("check if the service is up"),
        "tracking",
        SkillDraft(name="probe", description="Probe the service."),
        "THE CONTRACT",
        ValueError("skill body is empty"),
    )
    joined = messages[1]["content"]
    assert "owns no working data" in joined
    assert "`ctx.config`" in joined
    assert "never embed or fabricate working values" in joined
    assert "internal/mock data model" not in joined


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
    store = SkillStore(str(tmp_path / "skills"))
    assert store.list_skills() == []
    store.write_body("tracking", "probe", GOOD_BODY)
    assert store.list_skills() == [("tracking", "probe")]
    target = store.dir("tracking", "probe") / "skill.py"
    assert target.is_file()
    assert "def predict" in target.read_text()


def test_store_roundtrip_all_deliverables(tmp_path):
    store = SkillStore(str(tmp_path / "skills"))
    store.write_body("tracking", "probe", GOOD_BODY)
    store.write_contract("tracking", "probe", GOOD_CONTRACT)
    store.write_test("tracking", "probe", GOOD_TEST)
    store.write_config("tracking", "probe", {"sender_address": "agent@example.com"})
    directory = store.dir("tracking", "probe")
    assert sorted(p.name for p in directory.iterdir()) == [
        "config.json",
        "contract.json",
        "skill.py",
        "skill.test.py",
    ]
    assert store.read_contract("tracking", "probe") == GOOD_CONTRACT
    assert store.read_config("tracking", "probe") == {"sender_address": "agent@example.com"}
    assert store.read_category_config("tracking") == {}


def test_store_ignores_legacy_single_file_layout(tmp_path):
    """Clean switch: a skill written as <category>/<name>.py is NOT read."""
    legacy = tmp_path / "skills" / "tracking"
    legacy.mkdir(parents=True)
    (legacy / "probe.py").write_text(GOOD_BODY)
    store = SkillStore(str(tmp_path / "skills"))
    assert store.list_skills() == []


def test_load_skill_module_exposes_predict_act(tmp_path):
    store = SkillStore(str(tmp_path / "skills"))
    store.write_body("tracking", "probe", GOOD_BODY)
    module = load_skill_module("tracking", "probe", store.path)
    assert callable(module.predict) and callable(module.act)


def test_materialize_skill_builds_runnable_skill(tmp_path):
    store = SkillStore(str(tmp_path / "skills"))
    store.write_contract("tracking", "probe", GOOD_CONTRACT)
    store.write_config("tracking", "probe", {"sender_address": "agent@example.com"})
    draft = SkillDraft(name="probe", description="Probe the service.", code=GOOD_BODY)
    skill = materialize_skill(draft, "tracking", store)
    assert skill.name == "probe"
    assert skill.category == "tracking"
    assert callable(skill.predict) and callable(skill.act)
    assert skill.contract == GOOD_CONTRACT
    assert skill.config == {"sender_address": "agent@example.com"}


def test_materialize_skill_requires_code(tmp_path):
    store = SkillStore(str(tmp_path / "skills"))
    draft = SkillDraft(name="probe", description="Probe the service.")
    with pytest.raises(ValueError):
        materialize_skill(draft, "tracking", store)


def test_materialize_skill_rejects_import_failure(tmp_path):
    store = SkillStore(str(tmp_path / "skills"))
    bad = "def predict(ctx, request):\n    return None\n"
    draft = SkillDraft(name="probe", description="Probe.", code=bad)
    with pytest.raises(ValueError):
        materialize_skill(draft, "tracking", store)


def test_merge_skill_store_upgrades_stub(tmp_path):
    store = SkillStore(str(tmp_path / "skills"))
    store.write_body("tracking", "probe", GOOD_BODY)
    store.write_contract("tracking", "probe", GOOD_CONTRACT)
    store.write_config("tracking", "probe", {"sender_address": "agent@example.com"})
    tree = build_tree(build_skills({"skills": {}}))
    registry = {"tracking": {"description": "", "skills": [{"name": "probe", "description": "Probe."}]}}
    merge_registry(tree, registry)
    upgraded = merge_skill_store(tree, store, registry)
    assert upgraded == 1
    skill = next(s for s in tree["tracking"] if s.name == "probe")
    assert callable(skill.predict) and callable(skill.act)
    assert skill.description == "Probe."
    assert skill.contract == GOOD_CONTRACT
    assert skill.config == {"sender_address": "agent@example.com"}


def test_merge_skill_store_creates_missing_category(tmp_path):
    store = SkillStore(str(tmp_path / "skills"))
    store.write_body("brand_new", "ping", GOOD_BODY)
    tree = build_tree(build_skills({"skills": {}}))
    upgraded = merge_skill_store(tree, store, {})
    assert upgraded == 1
    assert tree["brand_new"][0].name == "ping"


def _respond_show(handler, body: dict | None = None) -> bool:
    """Serve the ollama `/api/show` reply; returns True if handled."""
    if not handler.path.endswith("/api/show"):
        return False
    encoded = json.dumps(
        body if body is not None else {"parameters": {"num_ctx": 4242}}
    ).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)
    return True


class _FakeOpenAI(BaseHTTPRequestHandler):
    """Single-shot SSE server: the reply arrives as one content delta. The
    `/api/show` window query is answered too (default num_ctx 4242) but never
    recorded in `received`, so chat-payload assertions stay unambiguous."""

    reply: str = GOOD_BODY
    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        if _respond_show(self):
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


def test_sent_payload_system_message_contains_contract_phrase():
    """The real SKILL.md contract must actually reach the model: the recorded
    HTTP payload's system message carries a SKILL.md phrase (no mocking — the
    fake server records the real request body)."""
    httpd, base = _fake_server(GOOD_BODY)
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        tree = build_tree(build_skills({"skills": {}}))
        draft = SkillDraft(name="probe", description="Probe the service.")
        generate_skill_body(client, Request("is the service up?"), "tracking", draft, tree)
        sent = httpd.RequestHandlerClass.received[0]
        system = sent["messages"][0]["content"]
        assert "data/skills" in system
        assert read_skill_contract() in system
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
        raw = self.rfile.read(length).decode("utf-8")
        if _respond_show(self):
            return
        time.sleep(5)

    def log_message(self, format, *args):
        pass


class _StalledStreamOpenAI(BaseHTTPRequestHandler):
    """Sends response headers then holds the connection with zero body bytes;
    the streaming client must hit its idle/stall watchdog."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        if _respond_show(self):
            return
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
        raw = self.rfile.read(length).decode("utf-8")
        if _respond_show(self):
            return
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


class _CapTripOpenAI(BaseHTTPRequestHandler):
    """Streams forever but reports an empty `/api/show` (no num_ctx), so the
    client falls back to its `context_window` config — deterministic budget
    tests at tiny windows."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        if _respond_show(self, {}):
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.flush()
        try:
            while True:
                frame = _sse_frame({"choices": [{"delta": {"content": "x"}}]})
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format, *args):
        pass


def _cap_trip_server() -> tuple[ThreadingHTTPServer, str]:
    handler = type("Handler", (_CapTripOpenAI,), {})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


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
        base_url="http://127.0.0.1:1/v1", model="test", timeout=0.5, stream=True
    )
    budget = client._compute_budget([{"role": "user", "content": "hi"}])
    started = time.monotonic()
    with pytest.raises(CodegenError, match="total budget"):
        client._read_stream_blocking(_ForeverFrames(), budget)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"blocking fallback should fail fast, took {elapsed:.1f}s"


def test_output_cap_trips_at_fraction_of_window():
    """max_output in (0,1) caps at a fraction of the detected window; the cap
    must abort a runaway stream in real time."""
    httpd, base = _cap_trip_server()
    try:
        client = CodegenClient(
            base_url=base,
            model="test",
            timeout=10,
            stream=True,
            context_window=100,
            max_output=0.25,
            idle_timeout=30,
        )
        started = time.monotonic()
        with pytest.raises(CodegenError, match="token budget"):
            client.chat([{"role": "user", "content": "hi"}])
        assert time.monotonic() - started < 8.0
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_output_cap_trips_at_absolute_tokens():
    """max_output >= 1 is an absolute token cap, independent of the window."""
    httpd, base = _cap_trip_server()
    try:
        client = CodegenClient(
            base_url=base,
            model="test",
            timeout=10,
            stream=True,
            context_window=100,
            max_output=8,
            idle_timeout=30,
        )
        with pytest.raises(CodegenError, match="token budget"):
            client.chat([{"role": "user", "content": "hi"}])
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_budget_warns_near_fill_threshold(capsys):
    """The one-shot fill warning fires (before the cap aborts) when the total
    fill estimate crosses warn_point."""
    httpd, base = _cap_trip_server()
    try:
        client = CodegenClient(
            base_url=base,
            model="test",
            timeout=10,
            stream=True,
            context_window=100,
            max_output=0.5,
            warn_fill_ratio=0.2,
            idle_timeout=30,
        )
        with pytest.raises(CodegenError, match="token budget"):
            client.chat([{"role": "user", "content": "hi"}])
        captured = capsys.readouterr().out
        assert "total fill" in captured
        assert "near the" in captured
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_budget_start_line_printed(capsys):
    httpd, base = _streaming_server()
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10, stream=True)
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == GOOD_BODY
        captured = capsys.readouterr().out
        assert captured.startswith("[codegen] context window")
        assert "output cap" in captured
        assert "peak total fill" in captured
    finally:
        httpd.shutdown()
        httpd.server_close()


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
        if _respond_show(self):
            return
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
        assert body["stream_options"] == {"include_usage": True}
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
        assert captured.startswith("[codegen] context window")
        assert captured.endswith(_StreamingOpenAI.reasoning + GOOD_BODY + "\n")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_chat_stream_false_returns_content_without_echo(capsys):
    """`stream: false` must still return the full content (transport always
    streams) but print nothing beyond the request — no token echo."""
    httpd, base = _streaming_server()
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10, stream=False)
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == GOOD_BODY
        assert capsys.readouterr().out == ""
    finally:
        httpd.shutdown()
        httpd.server_close()


class _UsageOpenAI(BaseHTTPRequestHandler):
    """Streams a small content delta, then the exact-token usage chunk
    (`choices: []` — OpenAI's shape when include_usage is set) before
    [DONE]. The `/api/show` window query is answered (default num_ctx 4242)
    and never recorded."""

    usage: dict = {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
    content: str = "ok\n"
    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        if _respond_show(self):
            return
        type(self).received.append(json.loads(raw))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(
            _sse_frame({"choices": [{"delta": {"content": self.content}}]}).encode("utf-8")
        )
        self.wfile.write(
            _sse_frame({"choices": [], "usage": self.usage}).encode("utf-8")
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, format, *args):
        pass


def _usage_server() -> tuple[ThreadingHTTPServer, str]:
    handler = type("Handler", (_UsageOpenAI,), {"received": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


def test_chat_captures_usage_chunk():
    """A usage chunk (empty choices) must be consumed without crashing and
    the real token counts logged after the stream — the include_usage path."""
    httpd, base = _usage_server()
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == "ok\n"
        assert httpd.RequestHandlerClass.received[0]["stream_options"] == {
            "include_usage": True
        }
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize(
    ("smart_limit", "warn_limit", "zone"),
    [
        (250000, 500000, "SMART"),
        (10, 500, "WARN"),
        (10, 14, "DUMB"),
    ],
)
def test_chat_logs_real_usage_with_zone(smart_limit, warn_limit, zone, capsys):
    """Real usage is logged with prompt/completion/total tokens, % of window,
    and the zone by absolute total-token thresholds."""
    httpd, base = _usage_server()
    try:
        client = CodegenClient(
            base_url=base,
            model="test",
            timeout=10,
            smart_limit=smart_limit,
            warn_limit=warn_limit,
        )
        client.chat([{"role": "user", "content": "hi"}])
        captured = capsys.readouterr().out
        assert "prompt 5 · completion 10 · total 15 tokens" in captured
        assert "0% of window" in captured
        assert zone in captured
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_chat_no_usage_chunk_logs_nothing(capsys):
    """Without an include_usage chunk nothing extra is logged — old servers
    keep working and the budget line is the only output."""
    httpd, base = _streaming_server()
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10, stream=True)
        client.chat([{"role": "user", "content": "hi"}])
        captured = capsys.readouterr().out
        assert "usage: prompt" not in captured
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


class _ShowOpenAI(BaseHTTPRequestHandler):
    """Answers `/api/show` with a configurable body and counts the calls."""

    show: dict = {"parameters": {"num_ctx": 4242}}
    show_calls: int = 0
    show_requests: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        type(self).show_calls += 1
        type(self).show_requests.append(json.loads(raw))
        body = json.dumps(type(self).show).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _show_server(show: dict) -> tuple[ThreadingHTTPServer, str]:
    handler = type(
        "Handler",
        (_ShowOpenAI,),
        {"show": show, "show_calls": 0, "show_requests": []},
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


def test_context_window_from_num_ctx():
    httpd, base = _show_server({"parameters": {"num_ctx": 4096}})
    try:
        client = CodegenClient(base_url=base, model="test")
        assert client._context_window() == 4096
        assert httpd.RequestHandlerClass.show_requests[0] == {"model": "test"}
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_context_window_parses_parameters_string():
    """Real ollama serves `parameters` as modelfile text, not a dict."""
    httpd, base = _show_server(
        {"parameters": 'num_ctx 100000\nstop "<|end_of_text|>"\ntemperature 0.0'}
    )
    try:
        client = CodegenClient(base_url=base, model="test")
        assert client._context_window() == 100000
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_context_window_string_parameters_falls_back_to_model_info():
    httpd, base = _show_server(
        {
            "parameters": 'stop "<|end_of_text|>"',
            "model_info": {"qwen35.context_length": 262144},
        }
    )
    try:
        client = CodegenClient(base_url=base, model="test")
        assert client._context_window() == 262144
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_context_window_from_model_info():
    httpd, base = _show_server({"model_info": {"qwen35.context_length": 262144}})
    try:
        client = CodegenClient(base_url=base, model="test")
        assert client._context_window() == 262144
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_context_window_falls_back_to_config():
    httpd, base = _show_server({})
    try:
        client = CodegenClient(base_url=base, model="test", context_window=2048)
        assert client._context_window() == 2048
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_context_window_falls_back_to_default():
    client = CodegenClient(base_url="http://127.0.0.1:1/v1", model="test")
    assert client._context_window() == 100000


def test_context_window_is_cached():
    httpd, base = _show_server({"parameters": {"num_ctx": 4242}})
    try:
        client = CodegenClient(base_url=base, model="test")
        assert client._context_window() == 4242
        assert client._context_window() == 4242
        assert httpd.RequestHandlerClass.show_calls == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_degeneration_check_aborts_stream():
    """A degeneration callback that returns a reason must abort a still-
    streaming generation with CodegenError, before the token budget trips."""
    httpd, base = _cap_trip_server()
    try:
        client = CodegenClient(
            base_url=base,
            model="test",
            timeout=10,
            stream=True,
            context_window=100,
            max_output=0.5,
            idle_timeout=30,
            degeneration_interval=5,
            degeneration_window=20,
            degeneration_min_chars=1,
        )
        calls = []

        def check(recent):
            calls.append(recent)
            return "looping forever"

        started = time.monotonic()
        with pytest.raises(CodegenError, match="degeneration detected"):
            client.chat([{"role": "user", "content": "hi"}], degeneration_check=check)
        assert time.monotonic() - started < 8.0
        assert calls, "degeneration callback must be invoked"
        assert calls[0] == "x", "callback must receive the recent window chars"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_degeneration_check_not_invoked_before_min_chars():
    """The callback must not fire before `min_chars` accumulate — the token
    budget aborts first and the callback stays silent."""
    httpd, base = _cap_trip_server()
    try:
        client = CodegenClient(
            base_url=base,
            model="test",
            timeout=10,
            stream=True,
            context_window=100,
            max_output=0.5,
            idle_timeout=30,
            degeneration_interval=5,
            degeneration_window=20,
            degeneration_min_chars=10**9,
        )
        calls = []

        def check(recent):
            calls.append(recent)
            return "should never fire"

        with pytest.raises(CodegenError, match="token budget"):
            client.chat([{"role": "user", "content": "hi"}], degeneration_check=check)
        assert calls == [], "callback must not run before min_chars is reached"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_degeneration_check_returning_none_continues():
    """A callback that never flags degeneration must not abort — the token
    budget is still the terminator."""
    httpd, base = _cap_trip_server()
    try:
        client = CodegenClient(
            base_url=base,
            model="test",
            timeout=10,
            stream=True,
            context_window=100,
            max_output=0.5,
            idle_timeout=30,
            degeneration_interval=5,
            degeneration_window=20,
            degeneration_min_chars=1,
        )
        calls = []

        def check(recent):
            calls.append(recent)
            return None

        with pytest.raises(CodegenError, match="token budget"):
            client.chat([{"role": "user", "content": "hi"}], degeneration_check=check)
        assert calls, "callback should have been polled across the stream"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_degeneration_disabled_no_callback_no_abort():
    """With no callback (disabled), a stream exceeding min_chars completes
    normally — nothing in the read path assumes a check is present."""
    httpd, base = _streaming_server()
    try:
        client = CodegenClient(
            base_url=base,
            model="test",
            timeout=10,
            degeneration_interval=1,
            degeneration_window=10,
            degeneration_min_chars=1,
        )
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == GOOD_BODY
    finally:
        httpd.shutdown()
        httpd.server_close()


class _SequencedOpenAI(BaseHTTPRequestHandler):
    """Returns one reply per chat request, in order; counts chat calls.

    `/api/show` is answered (default num_ctx 4242) and never counted, so chat
    payload and retry-count assertions stay unambiguous.
    """

    replies: list = []
    received: list = []
    chat_calls: int = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        if _respond_show(self):
            return
        type(self).chat_calls += 1
        type(self).received.append(json.loads(raw))
        index = min(type(self).chat_calls - 1, len(type(self).replies) - 1)
        reply = type(self).replies[index]
        body = (
            _sse_frame({"choices": [{"delta": {"content": reply}}]})
            + "data: [DONE]\n\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _sequenced_server(replies: list) -> tuple[ThreadingHTTPServer, str]:
    handler = type(
        "Handler",
        (_SequencedOpenAI,),
        {"replies": replies, "received": [], "chat_calls": 0},
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


def test_chat_payload_carries_sampler_defaults():
    httpd, base = _fake_server(GOOD_BODY)
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        client.chat([{"role": "user", "content": "hi"}])
        body = httpd.RequestHandlerClass.received[0]
        assert body["temperature"] == 0.7
        assert body["top_p"] == 0.85
        assert body["presence_penalty"] == 1.5
        assert body["frequency_penalty"] == 0.2
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_generate_skill_body_escalates_sampler_on_retry(tmp_path):
    """An invalid first parse must retry with the escalated sampler and a fresh
    corrective prompt (context resets to SMART, no growing conversation)."""
    httpd, base = _sequenced_server(["this is not python at all", GOOD_BODY])
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        tree = build_tree(build_skills({"skills": {}}))
        draft = SkillDraft(name="probe", description="Probe the service.")
        code = generate_skill_body(
            client, Request("is the service up?"), "tracking", draft, tree
        )
        assert "def predict" in code and "def act" in code
        reqs = httpd.RequestHandlerClass.received
        assert len(reqs) == 2, "one retry after the rejected first attempt"
        first, second = reqs
        assert first["temperature"] == 0.7
        assert first["presence_penalty"] == 1.5
        assert second["temperature"] == 0.5
        assert second["top_p"] == 0.85
        assert second["presence_penalty"] == 2.0
        assert second["frequency_penalty"] == 0.3
        retry_user = second["messages"][1]["content"]
        assert "You are looping" in retry_user
        assert "probe" in retry_user
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_generate_skill_body_exhausts_max_attempts(tmp_path):
    httpd, base = _sequenced_server(["this is not python at all"])
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10, max_attempts=3)
        tree = build_tree(build_skills({"skills": {}}))
        draft = SkillDraft(name="probe", description="Probe the service.")
        with pytest.raises(ValueError, match="rejected 3 times"):
            generate_skill_body(
                client, Request("is the service up?"), "tracking", draft, tree
            )
        assert httpd.RequestHandlerClass.chat_calls == 3
    finally:
        httpd.shutdown()
        httpd.server_close()


class _DegeneratingOpenAI(BaseHTTPRequestHandler):
    """Streams forever (degeneration bait) and counts chat requests; answers
    `/api/show` with {} so the client falls back to its `context_window` config."""

    chat_calls: int = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        if _respond_show(self, {}):
            return
        type(self).chat_calls += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.flush()
        try:
            while True:
                frame = _sse_frame({"choices": [{"delta": {"content": "x"}}]})
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format, *args):
        pass


def _degenerating_server() -> tuple[ThreadingHTTPServer, str]:
    handler = type("Handler", (_DegeneratingOpenAI,), {"chat_calls": 0})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


def test_degeneration_error_propagates_without_retry():
    """A degeneration abort is a DegenerationError (CodegenError subclass) and
    must NOT retry — retry is reserved for invalid parses, leaving degeneration
    handling as an explicit choice."""
    assert issubclass(DegenerationError, CodegenError)
    httpd, base = _degenerating_server()
    try:
        client = CodegenClient(
            base_url=base,
            model="test",
            timeout=10,
            stream=True,
            context_window=100000,
            max_output=0.5,
            idle_timeout=30,
            degeneration_interval=1,
            degeneration_window=20,
            degeneration_min_chars=1,
            max_attempts=3,
        )
        tree = build_tree(build_skills({"skills": {}}))
        draft = SkillDraft(name="probe", description="Probe the service.")
        with pytest.raises(DegenerationError, match="degeneration detected"):
            generate_skill_body(
                client,
                Request("is the service up?"),
                "tracking",
                draft,
                tree,
                degeneration_check=lambda recent: "looping",
            )
        assert httpd.RequestHandlerClass.chat_calls == 1, "degeneration must not retry"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_other_codegen_error_does_not_retry():
    """A non-degeneration CodegenError (token budget) must propagate without
    retrying either."""
    httpd, base = _degenerating_server()
    try:
        client = CodegenClient(
            base_url=base,
            model="test",
            timeout=10,
            stream=True,
            context_window=3000,
            max_output=0.01,
            idle_timeout=30,
            max_attempts=3,
        )
        tree = build_tree(build_skills({"skills": {}}))
        draft = SkillDraft(name="probe", description="Probe the service.")
        with pytest.raises(CodegenError, match="token budget"):
            generate_skill_body(
                client, Request("is the service up?"), "tracking", draft, tree
            )
        assert httpd.RequestHandlerClass.chat_calls == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---- elicitation ----

def test_elicitation_prompt_contains_examples_and_antipatterns():
    tree = build_tree(build_skills({"skills": {}}))
    draft = SkillDraft(name="probe", description="Probe the service.")
    messages = build_elicitation_prompt(
        Request("is the service up?"), "tracking", draft, tree, max_questions=3
    )
    system = messages[0]["content"]
    joined = messages[1]["content"]
    assert "requirements only" in system
    assert "should the sender address change?" in system
    assert "product goal" in system
    assert "tracking" in joined
    for example in (
        "just produce a draft you review and approve first",
        "should the skill ask you which one, or automatically pick",
        "report 'not found' as a normal result",
        "a short message back to you, or a file/report saved to disk",
    ):
        assert example in system


def test_parse_elicitation():
    assert parse_elicitation('{"questions": ["A?", "B?"]}') == ["A?", "B?"]
    assert parse_elicitation('Sure:\n{"questions": ["  A?  "]}') == ["A?"]
    assert parse_elicitation('{"questions": []}') == []
    with pytest.raises(ValueError):
        parse_elicitation("not json")
    with pytest.raises(ValueError):
        parse_elicitation('{"questions": "nope"}')


def test_generate_requirements_end_to_end():
    httpd, base = _sequenced_server(['{"questions": ["Draft or send?", "Which one?"]}'])
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        tree = build_tree(build_skills({"skills": {}}))
        draft = SkillDraft(name="probe", description="Probe the service.")
        questions = generate_requirements(
            client, Request("is the service up?"), "tracking", draft, tree, max_questions=3
        )
        assert questions == ["Draft or send?", "Which one?"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_generate_requirements_truncates_to_max():
    httpd, base = _sequenced_server(['{"questions": ["A?", "B?", "C?", "D?"]}'])
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        tree = build_tree(build_skills({"skills": {}}))
        draft = SkillDraft(name="probe", description="Probe the service.")
        questions = generate_requirements(
            client, Request("is the service up?"), "tracking", draft, tree, max_questions=2
        )
        assert questions == ["A?", "B?"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_generate_requirements_degrades_on_parse_failure():
    """Elicitation must never block authoring: a garbage reply degrades to []."""
    httpd, base = _sequenced_server(["this is not json at all"])
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10, max_attempts=2)
        tree = build_tree(build_skills({"skills": {}}))
        draft = SkillDraft(name="probe", description="Probe the service.")
        assert generate_requirements(
            client, Request("is the service up?"), "tracking", draft, tree
        ) == []
        assert httpd.RequestHandlerClass.chat_calls == 2
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---- data contract ----

def test_parse_data_contract_flat_object():
    contract = parse_data_contract(json.dumps(GOOD_CONTRACT))
    assert contract == GOOD_CONTRACT
    contract = parse_data_contract("Here:\n" + json.dumps(GOOD_CONTRACT))
    assert contract == GOOD_CONTRACT
    assert parse_data_contract("{}") == {}


@pytest.mark.parametrize(
    "raw",
    [
        '"just a string"',
        '{"sender": {"type": "string"}}',
        '{"sender": ""}',
        '{"Sender Address": "the sender"}',
        "not json at all",
    ],
)
def test_parse_data_contract_rejects_bad_shape(raw):
    with pytest.raises(ValueError):
        parse_data_contract(raw)


def test_generate_data_contract_end_to_end(tmp_path):
    httpd, base = _sequenced_server([json.dumps(GOOD_CONTRACT)])
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        draft = SkillDraft(name="probe", description="Probe the service.")
        contract = generate_data_contract(
            client, Request("is the service up?"), "tracking", draft, GOOD_BODY
        )
        assert contract == GOOD_CONTRACT
        sent = httpd.RequestHandlerClass.received[0]
        joined = " ".join(m["content"] for m in sent["messages"])
        assert read_testgen_contract() in sent["messages"][0]["content"]
        assert "data contract" in joined
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_generate_data_contract_escalates_on_bad_parse():
    httpd, base = _sequenced_server(["not json", json.dumps(GOOD_CONTRACT)])
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        draft = SkillDraft(name="probe", description="Probe the service.")
        contract = generate_data_contract(
            client, Request("is the service up?"), "tracking", draft, GOOD_BODY
        )
        assert contract == GOOD_CONTRACT
        reqs = httpd.RequestHandlerClass.received
        assert len(reqs) == 2
        assert reqs[1]["temperature"] == 0.5  # escalated sampler
        assert reqs[1]["presence_penalty"] == 2.0
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_generate_data_contract_exhausts_attempts(tmp_path):
    httpd, base = _sequenced_server(["not json"])
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10, max_attempts=2)
        draft = SkillDraft(name="probe", description="Probe the service.")
        with pytest.raises(ValueError, match="rejected 2 times"):
            generate_data_contract(
                client, Request("is the service up?"), "tracking", draft, GOOD_BODY
            )
        assert httpd.RequestHandlerClass.chat_calls == 2
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---- test artifacts ----

def test_parse_skill_test_accepts_forms():
    assert parse_skill_test(GOOD_TEST) == GOOD_TEST.strip()
    fenced = "```python\n" + GOOD_TEST + "```"
    assert parse_skill_test(fenced) == GOOD_TEST.strip()
    wrapped = json.dumps({"code": GOOD_TEST})
    assert parse_skill_test(wrapped) == GOOD_TEST.strip()
    with pytest.raises(ValueError):
        parse_skill_test("def x(:")
    with pytest.raises(ValueError):
        parse_skill_test("")
    with pytest.raises(ValueError):
        parse_skill_test("some prose without code")


def test_generate_skill_tests_shares_contract_context(tmp_path):
    """Decoupled call: the testgen call shares the contract call's base context
    and continues it with the accepted contract as the assistant turn; the reply
    is plain Python (fixtures embedded inline), not a JSON envelope."""
    httpd, base = _sequenced_server([GOOD_TEST])
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        draft = SkillDraft(name="probe", description="Probe the service.")
        test = generate_skill_tests(
            client, Request("is the service up?"), "tracking", draft, GOOD_BODY, GOOD_CONTRACT
        )
        assert test == GOOD_TEST.strip()
        messages = httpd.RequestHandlerClass.received[0]["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]
        assert messages[2]["content"] == json.dumps(GOOD_CONTRACT)
        assert "skill.test.py" in messages[3]["content"]
        assert "valid Python" in messages[3]["content"]
        assert "no JSON" in messages[3]["content"]
        assert "mock_data.json" not in messages[3]["content"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_run_skill_test_verdict(tmp_path):
    skill_dir = tmp_path / "skills" / "tracking" / "probe"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.test.py").write_text(GOOD_TEST)
    passed, _ = run_skill_test(skill_dir)
    assert passed
    (skill_dir / "skill.test.py").write_text("import sys\nprint('boom')\nsys.exit(3)\n")
    passed, output = run_skill_test(skill_dir)
    assert not passed
    assert "boom" in output


def test_run_skill_test_timeout(tmp_path):
    skill_dir = tmp_path / "skills" / "tracking" / "probe"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.test.py").write_text("import time\ntime.sleep(5)\n")
    passed, output = run_skill_test(skill_dir, timeout=0.5)
    assert not passed
    assert "timed out" in output


def test_run_skill_test_missing_script(tmp_path):
    passed, output = run_skill_test(tmp_path)
    assert not passed
    assert "no skill.test.py" in output


def test_regenerate_skill_body_rewrites(tmp_path):
    httpd, base = _sequenced_server([GOOD_BODY])
    try:
        client = CodegenClient(base_url=base, model="test", timeout=10)
        draft = SkillDraft(name="probe", description="Probe the service.")
        code = regenerate_skill_body(
            client, Request("is the service up?"), "tracking", draft, GOOD_BODY,
            "test failed with an error",
        )
        assert "def predict" in code and "def act" in code
        sent = httpd.RequestHandlerClass.received[0]
        joined = " ".join(m["content"] for m in sent["messages"])
        assert "test failed with an error" in joined
        assert "Previous body" in joined
    finally:
        httpd.shutdown()
        httpd.server_close()