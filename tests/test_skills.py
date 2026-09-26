"""Pure-stdlib tests for skill-tree authoring: category and skill prompts, draft
parsing, the category registry, the async codegen-body workflow, and the error
path when the engine is unavailable.

No mocking: the async write test uses a real CodegenClient against a throwaway
stdlib HTTP server (real endpoint, per the repo rule); engine-dependent success
paths are exercised only by the box integration tests against the real decision
model.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from semif_agent.codegen import CodegenClient
from semif_agent.decisions import Request
from semif_agent.engine import EngineConfig, EngineUnavailable, SemIfEngine
from semif_agent.llm import LLMClient
from semif_agent.log import DecisionLog
from semif_agent.scheduler import Scheduler
from semif_agent.skills import (
    ActionResult,
    CategoryDraft,
    CategoryRegistry,
    CreateCategory,
    Prediction,
    Skill,
    SkillDraft,
    SkillStore,
    build_category_prompt,
    build_skill_prompt,
    build_skills,
    build_tree,
    generate_category,
    generate_skill,
    merge_registry,
    navigate,
    parse_category_draft,
    parse_skill_draft,
    resolve_skill_config,
    unresolved_variables,
)
from semif_agent.trace import TraceLog


GOOD_BODY = """\
from semif_agent.decisions import DecisionRequest, Option
from semif_agent.skills import ActionResult, Prediction

def predict(ctx, request):
    return Prediction(text="ok", decisions=[])

def act(ctx, request, prediction):
    return ActionResult(action_log="probe ran", new_state=request.text)
"""

GOOD_TEST = """\
import sys
print("ok")
sys.exit(0)
"""

GOOD_BUNDLE = json.dumps({"test": GOOD_TEST, "mock_data": {}})


def _pipeline_codegen_server(replies: list) -> tuple[ThreadingHTTPServer, str]:
    """Sequenced OpenAI-compatible SSE server for the full authoring pipeline:
    codegen body -> data contract -> test artifacts. Also answers `/api/show`
    so the client's context-window probe succeeds (never counted)."""

    class Handler(BaseHTTPRequestHandler):
        received: list = []
        chat_calls: int = 0

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8")
            if self.path.endswith("/api/show"):
                body = json.dumps({"parameters": {"num_ctx": 4242}}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            type(self).chat_calls += 1
            type(self).received.append(json.loads(raw))
            index = min(type(self).chat_calls - 1, len(replies) - 1)
            reply = replies[index]
            frame = "data: " + json.dumps(
                {"choices": [{"delta": {"content": reply}}]}
            ) + "\n\n"
            body = (frame + "data: [DONE]\n\n").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    handler = type("Handler", (Handler,), {"received": [], "chat_calls": 0})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


def _scheduler(tmp_path, codegen=None):
    return Scheduler(
        engine=SemIfEngine(EngineConfig()),
        llm=LLMClient(base_url="http://localhost:1/v1", model="test"),
        log=DecisionLog(str(tmp_path / "decisions.jsonl")),
        config={
            "skills": {},
            "category_registry": str(tmp_path / "categories.json"),
            "skill_bodies": str(tmp_path / "skills"),
        },
        trace=TraceLog(str(tmp_path / "runs.jsonl")),
        codegen=codegen,
    )


def test_action_result_needs_input_defaults_none():
    assert ActionResult("log", "state").needs_input is None


def test_build_category_prompt_contains_request_and_tree():
    tree = build_tree(build_skills({"skills": {}}))
    messages = build_category_prompt(Request("tracking for my drone delivery"), tree)
    assert messages[0]["role"] == "system"
    assert "category" in messages[0]["content"]
    joined = messages[1]["content"]
    assert "tracking for my drone delivery" in joined
    assert "email: email.compose" in joined


def test_parse_category_draft_plain_json():
    draft = parse_category_draft(
        '{"title": "delivery", "description": "Track and manage package deliveries."}'
    )
    assert draft.name == "delivery"
    assert "package" in draft.description


def test_parse_category_draft_json_in_prose():
    draft = parse_category_draft(
        'Sure! Here you go:\n{"title": "home_automation", "description": "Control '
        'lights, locks, and appliances around the house."}\nHope that helps.'
    )
    assert draft.name == "home_automation"


def test_parse_category_draft_normalizes_title():
    draft = parse_category_draft(
        '{"title": "Home Automation", "description": "Control household devices."}'
    )
    assert draft.name == "home_automation"


def test_parse_category_draft_missing_fields_raises():
    with pytest.raises(ValueError):
        parse_category_draft('{"title": "only_title"}')
    with pytest.raises(ValueError):
        parse_category_draft("not json at all")


def test_parse_category_draft_rejects_unclean_title():
    with pytest.raises(ValueError):
        parse_category_draft('{"title": "ca$h!", "description": "nope"}')


def test_category_registry_roundtrip(tmp_path):
    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    assert registry.read() == {}
    registry.register("delivery", "Track and manage package deliveries.")
    registry.register("delivery", "Track and manage package deliveries, re-registered.")
    loaded = registry.read()
    assert set(loaded) == {"delivery"}
    assert loaded["delivery"]["description"] == (
        "Track and manage package deliveries, re-registered."
    )
    assert loaded["delivery"]["skills"] == []


def test_generate_category_without_engine_raises():
    engine = SemIfEngine(EngineConfig())
    with pytest.raises(EngineUnavailable):
        generate_category(engine, Request("anything"), {})


def test_generate_without_engine_raises():
    engine = SemIfEngine(EngineConfig())
    with pytest.raises(EngineUnavailable):
        engine.generate([{"role": "user", "content": "hi"}])


def test_build_tree_includes_registry_stubs(tmp_path):
    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    registry.register("delivery", "Track and manage package deliveries.")
    tree = build_tree(build_skills({"skills": {}}))
    merge_registry(tree, registry.read())
    assert tree["delivery"] == []


def test_build_skill_prompt_contains_request_category_and_skills():
    tree = build_tree(build_skills({"skills": {}}))
    messages = build_skill_prompt(Request("tracking for my drone delivery"), "tracking", tree)
    assert messages[0]["role"] == "system"
    assert "tracking" in messages[0]["content"]
    joined = messages[1]["content"]
    assert "tracking for my drone delivery" in joined
    assert "tracking.check" in joined


def test_parse_skill_draft_plain_json():
    draft = parse_skill_draft(
        '{"title": "track_live", "description": "Follow a package in real time."}'
    )
    assert draft.name == "track_live"
    assert "real time" in draft.description


def test_parse_skill_draft_json_in_prose():
    draft = parse_skill_draft(
        'Here you go:\n{"title": "resend_email", "description": "Re-send a failed '
        'email draft."}\nHope that helps.'
    )
    assert draft.name == "resend_email"


def test_parse_skill_draft_normalizes_title():
    draft = parse_skill_draft(
        '{"title": "Live Tracking", "description": "Follow a package in real time."}'
    )
    assert draft.name == "live_tracking"


def test_parse_skill_draft_allows_dotted_name():
    draft = parse_skill_draft(
        '{"title": "tracking.status_lookup", "description": "Look up a package status."}'
    )
    assert draft.name == "tracking.status_lookup"


def test_parse_skill_draft_missing_fields_raises():
    with pytest.raises(ValueError):
        parse_skill_draft('{"title": "only_title"}')
    with pytest.raises(ValueError):
        parse_skill_draft("not json at all")


def test_parse_skill_draft_rejects_unclean_title():
    with pytest.raises(ValueError):
        parse_skill_draft('{"title": "ca$h!", "description": "nope"}')


def test_generate_skill_without_engine_raises():
    engine = SemIfEngine(EngineConfig())
    with pytest.raises(EngineUnavailable):
        generate_skill(engine, Request("anything"), "tracking", {})


def test_category_registry_register_skill(tmp_path):
    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    registry.register("delivery", "Track and manage package deliveries.")
    registry.register_skill("delivery", "track_live", "Follow a package in real time.")
    registry.register_skill("delivery", "track_live", "Duplicate, ignored.")
    registry.register_skill("brand_new", "ping", "Probe the service.")
    loaded = registry.read()
    assert loaded["delivery"]["skills"] == [
        {
            "name": "track_live",
            "description": "Follow a package in real time.",
            "request_text": "",
        }
    ]
    assert loaded["brand_new"] == {
        "description": "",
        "skills": [{"name": "ping", "description": "Probe the service.", "request_text": ""}],
    }


def test_category_registry_register_skill_keeps_request_text(tmp_path):
    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    registry.register_skill(
        "delivery",
        "track_live",
        "Follow a package in real time.",
        request_text="track my drone delivery in real time",
    )
    loaded = registry.read()
    assert loaded["delivery"]["skills"][0]["request_text"] == (
        "track my drone delivery in real time"
    )


def test_merge_registry_loads_categories_and_skills(tmp_path):
    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    registry.register_skill("delivery", "track_live", "Follow a package in real time.")
    tree = build_tree(build_skills({"skills": {}}))
    merge_registry(tree, registry.read())
    names = [s.name for s in tree["delivery"]]
    assert names == ["track_live"]
    assert tree["delivery"][0].category == "delivery"


def test_navigate_empty_tree_short_circuits(tmp_path):
    """An empty tree goes straight to CreateCategory without a SemIf call."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    result = navigate(None, log, trace, Request("anything"), {})
    assert isinstance(result, CreateCategory)
    assert log.read() == []
    assert any(e["kind"] == "create_category" for e in trace.read())


def test_dispatch_create_category_without_engine_returns_error(tmp_path):
    """An empty tree short-circuits to CreateCategory; without an engine the
    category authoring fails gracefully instead of leaving the scheduler wedged."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    scheduler = Scheduler(
        engine=SemIfEngine(EngineConfig()),
        llm=LLMClient(base_url="http://localhost:1/v1", model="test"),
        log=log,
        config={
            "skills": {},
            "category_registry": str(tmp_path / "categories.json"),
            "skill_bodies": str(tmp_path / "skills"),
        },
        trace=trace,
    )
    scheduler.tree = {}
    result = scheduler._dispatch(Request("anything"))
    assert result.kind == "error"
    assert "create_category failed" in result.summary


def test_skill_status_reflects_writing_and_noop():
    """A fresh Skill is a stub; marking it writing shows `writing`; a real body
    shows `ready`."""
    stub = Skill(name="probe", category="tracking", description="Probe.")
    assert stub.is_noop()
    assert stub.status == "stub"
    stub.writing = True
    assert stub.status == "writing"
    stub.writing = False

    def predict(ctx, request):
        return None

    def act(ctx, request, prediction):
        return None

    ready = Skill(name="probe", category="tracking", description="Probe.", predict=predict, act=act)
    assert not ready.is_noop()
    assert ready.status == "ready"


def test_restart_skill_without_codegen_returns_error(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.tree["tracking"] = [
        Skill(name="track_live", category="tracking", description="Follow a package.")
    ]
    status, detail = scheduler.restart_skill("tracking", "track_live")
    assert status == "error"
    assert "no codegen" in detail


def test_restart_skill_unknown_leaf_returns_error(tmp_path):
    scheduler = _scheduler(tmp_path)
    status, detail = scheduler.restart_skill("tracking", "missing")
    assert status == "error"
    assert "no skill" in detail


def test_dispatch_skill_awaiting_body_does_not_reauthor(tmp_path):
    """A re-dispatched request whose skill body is still pending must report the
    pending write instead of authoring a second skill."""
    scheduler = _scheduler(tmp_path)
    request = Request("track my package")
    request.meta["awaiting_skill_body"] = ["tracking", "track_live"]
    result = scheduler._dispatch_skill(request, "tracking", 0.5)
    assert result.kind == "create_skill"
    assert "track_live" in result.summary
    assert "restart" in result.summary


def test_run_skill_guards_stub_and_writing_leaves(tmp_path):
    """Running a writing leaf or a body-less stub must not silently no-op."""
    scheduler = _scheduler(tmp_path)
    writing = Skill(name="w", category="tracking", description="writing")
    writing.writing = True
    result = scheduler._run_skill(writing, Request("x"))
    assert result.kind == "error"
    assert "still being written" in result.summary

    stub = Skill(name="s", category="tracking", description="stub")
    result = scheduler._run_skill(stub, Request("x"))
    assert result.kind == "error"
    assert "no body" in result.summary


def test_async_skill_write_materializes_merges_and_requeues(tmp_path):
    """The full async pipeline: _start_skill_write flags the leaf in-progress
    and the single-slot worker runs codegen -> contract -> testgen -> test, then
    materializes the body, hot-merges it into the tree, and re-queues the
    original request for re-dispatch."""
    httpd, base = _pipeline_codegen_server([GOOD_BODY, "{}", GOOD_BUNDLE])
    try:
        scheduler = _scheduler(tmp_path, codegen=CodegenClient(base_url=base, model="test", timeout=10))
        scheduler.tree["tracking"] = [
            Skill(name="track_live", category="tracking", description="Follow a package.")
        ]
        leaf = scheduler.tree["tracking"][0]

        request = Request("track my drone delivery in real time")
        scheduler._start_skill_write(request, "tracking", SkillDraft(name="track_live", description="Follow a package."), 0.5)

        assert leaf.writing is True
        deadline = time.monotonic() + 10
        created = None
        while time.monotonic() < deadline:
            created = next(
                (e for e in scheduler.trace.read() if e["kind"] == "skill_created"),
                None,
            )
            if created:
                break
            time.sleep(0.05)
        assert created is not None, "worker must complete the body write"
        assert created["written"] is True

        upgraded = scheduler.tree["tracking"][0]
        assert upgraded is not leaf
        assert not upgraded.is_noop()
        assert callable(upgraded.predict) and callable(upgraded.act)

        # The worker drains the queue when idle (run_queue), so an empty queue
        # is a valid end state; the requeue trace proves the push happened.
        events = [e["kind"] for e in scheduler.trace.read()]
        assert "skill_requeued" in events
        assert "skill_testing" in events
        assert "skill_ready" in events
        # The full deliverable set landed in the skill folder.
        directory = scheduler.body_store.dir("tracking", "track_live")
        assert sorted(p.name for p in directory.iterdir() if not p.name.startswith("__")) == [
            "contract.json",
            "mock_data.json",
            "skill.py",
            "skill.test.py",
        ]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_async_skill_write_fails_gracefully_on_bad_contract(tmp_path):
    """A contract reply that never parses (after the escalation ladder) leaves
    the leaf a restartable stub — no silent no-op, no wedged worker."""
    httpd, base = _pipeline_codegen_server([GOOD_BODY, "this is not a contract"])
    try:
        scheduler = _scheduler(tmp_path, codegen=CodegenClient(base_url=base, model="test", timeout=10, max_attempts=2))
        scheduler.tree["tracking"] = [
            Skill(name="track_live", category="tracking", description="Follow a package.")
        ]
        leaf = scheduler.tree["tracking"][0]

        request = Request("track my drone delivery in real time")
        scheduler._start_skill_write(request, "tracking", SkillDraft(name="track_live", description="Follow a package."), 0.5)

        deadline = time.monotonic() + 10
        failed = None
        while time.monotonic() < deadline:
            failed = next(
                (e for e in scheduler.trace.read() if e["kind"] == "skill_write_failed"),
                None,
            )
            if failed:
                break
            time.sleep(0.05)
        assert failed is not None, "worker must surface the failure"
        assert not leaf.writing
        assert leaf.is_noop(), "leaf must stay a restartable stub"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_skill_pre_predict_contract_pause_and_resume(tmp_path):
    """A contract variable the runner cannot satisfy pauses BEFORE predict;
    the answer is recorded (engine unavailable -> ask-again, per-fire), then the
    run continues. A second pause comes from the skill's own needs_input."""
    scheduler = _scheduler(tmp_path)
    scheduler.tree["tracking"] = []
    store = scheduler.body_store
    store.write_contract("tracking", "track_live", {"token": "The tracking token."})

    seen = []

    def predict(ctx, request):
        return Prediction(text="", decisions=[])

    def act(ctx, request, prediction):
        if request.user_input:
            return ActionResult(action_log="done", new_state="done")
        return ActionResult(
            action_log="need confirmation", new_state=request.text, needs_input="Confirm?"
        )

    skill = Skill(
        name="track_live",
        category="tracking",
        description="Follow a package.",
        predict=predict,
        act=act,
        contract={"token": "The tracking token."},
    )

    result = scheduler._run_skill(skill, Request("track my package"))
    assert result.kind == "needs_input"
    assert "`token`" in result.summary
    assert scheduler.pending is not None
    assert scheduler.pending.pre_predict is True
    assert scheduler.pending.prediction is None

    status, detail = scheduler.answer("AB123")
    assert status == "needs_input"
    assert scheduler.pending is not None
    assert scheduler.pending.pre_predict is False
    assert scheduler.pending.request.meta["config_answers"]["token"] == "AB123"
    assert "Confirm?" in scheduler.pending.question

    status, detail = scheduler.answer("yes")
    assert status == "ran"
    assert scheduler.pending is None


def test_unresolved_variables_respects_tiered_config(tmp_path):
    """Contract vars are satisfied by global config, category config, skill
    config, or per-fire answers — in that order of precedence."""
    store = SkillStore(str(tmp_path / "skills"))
    store.write_contract("tracking", "probe", {"sender_address": "the sender", "receiver": "the receiver", "tag": "a tag"})
    store.write_config("tracking", "probe", {"sender_address": "skill-value"})
    store.write_category_config("tracking", {"receiver": "category-value"})
    skill = Skill(name="probe", category="tracking", description="Probe.",
                  contract={"sender_address": "the sender", "receiver": "the receiver", "tag": "a tag"},
                  config=store.read_config("tracking", "probe"))
    global_config = {"tag": "global-value"}
    assert unresolved_variables(store, skill, global_config) == []
    merged = resolve_skill_config(store, skill, global_config)
    assert merged["sender_address"] == "skill-value"
    assert merged["receiver"] == "category-value"
    assert merged["tag"] == "global-value"

    skill2 = Skill(name="probe", category="tracking", description="Probe.",
                   contract={"sender_address": "the sender", "receiver": "the receiver", "tag": "a tag"})
    assert unresolved_variables(store, skill2, {}) == ["sender_address", "tag"]
    assert unresolved_variables(store, skill2, {}, answered={"tag": "per-fire"}) == ["sender_address"]

def test_elicit_requirements_asks_and_records(tmp_path):
    """Opt-in elicitation asks the product owner refinement questions before
    the body is written and rides the answers on the draft into the body prompt."""
    httpd, base = _pipeline_codegen_server(
        ['{"questions": ["Draft or send?"]}', GOOD_BODY, "{}", GOOD_BUNDLE]
    )
    try:
        scheduler = _scheduler(tmp_path, codegen=CodegenClient(base_url=base, model="test", timeout=10))
        scheduler.elicitation_enabled = True
        answers = []
        scheduler.asker = lambda q: (answers.append(q), "draft first")[1]
        draft = SkillDraft(name="track_live", description="Follow a package.")
        scheduler._elicit_requirements(Request("track my package"), "tracking", draft)
        assert answers == ["Draft or send?"]
        assert draft.requirements == {"Draft or send?": "draft first"}
        assert any(e["kind"] == "requirements" for e in scheduler.trace.read())
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_elicit_requirements_disabled_skips(tmp_path):
    httpd, base = _pipeline_codegen_server([GOOD_BODY, "{}", GOOD_BUNDLE])
    try:
        scheduler = _scheduler(tmp_path, codegen=CodegenClient(base_url=base, model="test", timeout=10))
        scheduler.elicitation_enabled = False
        scheduler.asker = lambda q: "answer"
        draft = SkillDraft(name="track_live", description="Follow a package.")
        scheduler._elicit_requirements(Request("track my package"), "tracking", draft)
        assert draft.requirements == {}
        assert httpd.RequestHandlerClass.chat_calls == 0
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_elicit_requirements_asker_none_stops(tmp_path):
    """An asker that stops answering (None) halts collection without error."""
    httpd, base = _pipeline_codegen_server(
        ['{"questions": ["A?", "B?"]}', GOOD_BODY, "{}", GOOD_BUNDLE]
    )
    try:
        scheduler = _scheduler(tmp_path, codegen=CodegenClient(base_url=base, model="test", timeout=10))
        scheduler.elicitation_enabled = True
        asked = []
        scheduler.asker = lambda q: (asked.append(q), None)[1]
        draft = SkillDraft(name="track_live", description="Follow a package.")
        scheduler._elicit_requirements(Request("track my package"), "tracking", draft)
        assert asked == ["A?"]
        assert draft.requirements == {}
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_async_skill_write_regen_ladder_on_failing_test(tmp_path):
    """A failing auto-run test triggers the regen decision (regen_test by
    default when no factory), regenerating the test until it passes."""
    failing_bundle = json.dumps({"test": "import sys\nsys.exit(1)", "mock_data": {}})
    httpd, base = _pipeline_codegen_server(
        [GOOD_BODY, "{}", failing_bundle, GOOD_BUNDLE]
    )
    try:
        scheduler = _scheduler(tmp_path, codegen=CodegenClient(base_url=base, model="test", timeout=10))
        scheduler.tree["tracking"] = [
            Skill(name="track_live", category="tracking", description="Follow a package.")
        ]
        scheduler.regen_decision_factory = lambda run_id: (lambda reason: "regen_test")

        request = Request("track my drone delivery in real time")
        scheduler._start_skill_write(
            request, "tracking", SkillDraft(name="track_live", description="Follow a package."), 0.5
        )

        deadline = time.monotonic() + 10
        created = None
        while time.monotonic() < deadline:
            created = next(
                (e for e in scheduler.trace.read() if e["kind"] == "skill_created"),
                None,
            )
            if created:
                break
            time.sleep(0.05)
        assert created is not None and created["written"] is True

        testings = [e for e in scheduler.trace.read() if e["kind"] == "skill_testing"]
        assert [t["passed"] for t in testings] == [False, True], (
            "first attempt must fail, the regen must pass"
        )
        assert any(e["kind"] == "skill_ready" for e in scheduler.trace.read())
    finally:
        httpd.shutdown()
        httpd.server_close()
