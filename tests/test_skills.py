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
from pathlib import Path

import pytest

from semif_agent.cli import REPO_ROOT, load_config
from semif_agent.codegen import CodegenClient
from semif_agent.decisions import Request
from semif_agent.engine import EngineConfig, EngineUnavailable, SemIfEngine
from semif_agent.llm import LLMClient
from semif_agent.log import DecisionLog
from semif_agent.skill import RunResult
from semif_agent.scheduler import (
    DispatchResult,
    PendingQuestion,
    RepairOffer,
    Scheduler,
    SkillWrite,
    build_gate_decision,
)
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
    confirm_skill_fit,
    generate_category,
    generate_skill,
    merge_registry,
    merge_seed_store,
    merge_skill_store,
    navigate,
    parse_category_draft,
    parse_skill_draft,
    resolve_skill_config,
    unresolved_variables,
)
from semif_agent.trace import TraceLog

from tests.conftest import ScriptedEngine


GOOD_BODY = """\
from semif_agent.decisions import DecisionRequest, Option
from semif_agent.skills import ActionResult, Prediction

INTEGRATION = {
    "service": "probe_service",
    "transport": "compute",
    "config_vars": [],
}

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


def _pipeline_codegen_server(replies: list) -> tuple[ThreadingHTTPServer, str]:
    """Sequenced OpenAI-compatible SSE server for the full authoring pipeline:
    codegen body -> data contract -> test. Also answers `/api/show`
    so the client's context-window probe succeeds (never counted)."""

    class Handler(BaseHTTPRequestHandler):
        received: list = []
        chat_calls: int = 0

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8")
            if self.path.endswith("/api/show"):
                body = json.dumps({"parameters": {"num_ctx": 32768}}).encode("utf-8")
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


def _scheduler(tmp_path, codegen=None, choices=None, default="success"):
    """A scheduler driven by the deterministic scripted engine (conftest) so
    scheduling/authoring mechanics are testable without a GGUF. Assessment is a
    SemIf decision now, so a run needs the engine to answer `assess:*`."""
    return Scheduler(
        engine=ScriptedEngine(choices=choices, default=default),
        llm=LLMClient(base_url="http://localhost:1/v1", model="test"),
        log=DecisionLog(str(tmp_path / "decisions.jsonl")),
        config={
            "skills": {},
            "category_registry": str(tmp_path / "categories.json"),
            "skill_bodies": str(tmp_path / "skills"),
            "skill_seeds": str(tmp_path / "seeds"),
        },
        trace=TraceLog(str(tmp_path / "runs.jsonl")),
        codegen=codegen,
    )


def test_action_result_needs_input_defaults_none():
    assert ActionResult("log", "state").needs_input is None


def test_run_summary_is_deterministic_and_surfaces_action_log(tmp_path):
    """The summary is computed, not generated: same inputs, same string.

    It names the category and skill, the ok/failed flag from the SemIf
    assessment, and the skill's real action log. No LLM prose is involved.
    """
    scheduler = _scheduler(tmp_path)

    def predict(ctx, request):
        return Prediction(text="", decisions=[])

    def act(ctx, request, prediction):
        return ActionResult(
            action_log="Next event on 'personal': Team sync — 2026-09-27 10:00 CEST",
            new_state="next event reported",
        )

    skill = Skill(
        name="next_event",
        category="calendar",
        description="Report the next event.",
        predict=predict,
        act=act,
    )

    result = scheduler._run_skill(skill, Request("what is my next event?"))
    assert result.kind == "ran"

    expected = (
        "calendar.next_event: ok — Next event on 'personal': Team sync — 2026-09-27 10:00 CEST"
    )
    assessed = [e for e in scheduler.trace.read() if e["kind"] == "assessed"]
    assert assessed, "the run must be traced"
    assert assessed[-1]["summary"] == expected
    assert assessed[-1]["assessment_summary"] == expected


def test_run_summary_failure_uses_new_state_when_no_action_log(tmp_path):
    """A failed run with an empty action log falls back to the new state."""
    scheduler = _scheduler(
        tmp_path, choices={"achieve the user's goal": "failure",
                           "complete, or should it run again": "complete"}
    )

    def predict(ctx, request):
        return Prediction(text="", decisions=[])

    def act(ctx, request, prediction):
        return ActionResult(action_log="", new_state="service unreachable")

    skill = Skill(name="probe", category="tracking", description="Probe.",
                  predict=predict, act=act)
    scheduler._run_skill(skill, Request("probe it"))
    assessed = [e for e in scheduler.trace.read() if e["kind"] == "assessed"]
    assert assessed[-1]["success"] is False
    assert assessed[-1]["summary"] == "tracking.probe: failed — service unreachable"



def test_build_category_prompt_contains_request_and_tree():
    tree = build_tree(build_skills({"skills": {}}))
    messages = build_category_prompt(Request("tracking for my drone delivery"), tree)
    assert messages[0]["role"] == "system"
    assert "category" in messages[0]["content"]
    joined = messages[1]["content"]
    assert "tracking for my drone delivery" in joined
    assert "response: response.reject" in joined


def test_build_skills_is_internal_behaviors_only():
    """Real integrations are generated or seeded, never fabricated here: a
    hardcoded fake shadows the authoring path for a real skill (navigation
    routes to it). Only service-free behaviors stay built-in."""
    assert [skill.name for skill in build_skills({"skills": {}})] == ["response.reject"]


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
    tree["tracking"] = [
        Skill(
            name="tracking.check",
            category="tracking",
            description="Check the delivery status of a package.",
        )
    ]
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


def _write_seed(seed_store, category, name, description=""):
    directory = seed_store.dir(category, name)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "skill.py").write_text(GOOD_BODY)
    (directory / "contract.json").write_text(json.dumps({"probe_url": "Where to probe."}))
    if description:
        (directory / "manifest.json").write_text(json.dumps({"description": description}))


def test_merge_seed_store_loads_skill_with_runtime_config(tmp_path):
    """A seed is runnable from the committed folder; recorded config lives in
    the runtime store, never in the seed folder."""
    seed_store = SkillStore(str(tmp_path / "seeds"))
    config_store = SkillStore(str(tmp_path / "skills"))
    _write_seed(seed_store, "calendar", "probe", description="Probe a service.")
    config_store.write_config("calendar", "probe", {"probe_url": "https://example.test"})

    tree = build_tree(build_skills({"skills": {}}))
    loaded = merge_seed_store(tree, seed_store, config_store)
    assert loaded == 1
    skill = tree["calendar"][0]
    assert skill.status == "ready"
    assert skill.description == "Probe a service."
    assert skill.contract == {"probe_url": "Where to probe."}
    assert skill.config == {"probe_url": "https://example.test"}
    assert skill.integration_source == "declared"


def test_merge_seed_store_falls_back_to_name_and_body_store_wins(tmp_path):
    seed_store = SkillStore(str(tmp_path / "seeds"))
    body_store = SkillStore(str(tmp_path / "skills"))
    _write_seed(seed_store, "calendar", "probe")

    tree = build_tree(build_skills({"skills": {}}))
    merge_seed_store(tree, seed_store, body_store)
    assert tree["calendar"][0].description == "probe"

    body_store.write_body("calendar", "probe", GOOD_BODY)
    merge_skill_store(tree, body_store, {})
    assert tree["calendar"][0].description == "probe", "the generated body replaces the seed"
    assert len(tree["calendar"]) == 1


def test_scheduler_loads_committed_seed_skills(tmp_path):
    """The real seed package ships in the tree at startup with its manifest
    description and declared integration."""
    seed_root = Path(__file__).resolve().parent.parent / "seeds"
    scheduler = Scheduler(
        engine=SemIfEngine(EngineConfig()),
        llm=LLMClient(base_url="http://localhost:1/v1", model="test"),
        log=DecisionLog(str(tmp_path / "decisions.jsonl")),
        config={
            "category_registry": str(tmp_path / "categories.json"),
            "skill_bodies": str(tmp_path / "skills"),
            "skill_seeds": str(seed_root),
        },
        trace=TraceLog(str(tmp_path / "runs.jsonl")),
    )
    calendar = scheduler.tree["calendar"]
    assert [s.name for s in calendar] == ["next_event"]
    assert calendar[0].description.startswith("Report the next")
    assert calendar[0].status == "ready"
    assert calendar[0].integration["transport"] == "caldav"


def test_load_config_anchors_runtime_paths_to_checkout(tmp_path):
    """Path-valued config keys resolve against the checkout, not the cwd, so a
    fresh clone finds its committed seeds from anywhere; absolute paths pass
    through untouched."""
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "skill_seeds": "seeds",
                "skill_bodies": "data/skills",
                "log": "/absolute/decisions.jsonl",
            }
        )
    )
    config = load_config(str(config_file))
    assert config["skill_seeds"] == str(REPO_ROOT / "seeds")
    assert config["skill_bodies"] == str(REPO_ROOT / "data/skills")
    assert config["category_registry"] == str(REPO_ROOT / "data/categories.json")
    assert config["log"] == "/absolute/decisions.jsonl"
    assert config["trace"] == str(REPO_ROOT / "data/runs.jsonl")


def test_committed_seeds_load_from_a_foreign_cwd(tmp_path, monkeypatch):
    """A fresh checkout's seeds come from the anchored `seeds/` path even when
    the process runs from an unrelated directory (the zero-skills regression:
    relative paths resolved against cwd, so the tree was only response.reject
    and navigation fell into the codegen loop)."""
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "skill_seeds": "seeds",
                "skill_bodies": str(tmp_path / "skills"),
                "category_registry": str(tmp_path / "categories.json"),
                "log": str(tmp_path / "decisions.jsonl"),
                "trace": str(tmp_path / "runs.jsonl"),
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    config = load_config(str(config_file))
    scheduler = Scheduler(
        engine=SemIfEngine(EngineConfig()),
        llm=LLMClient(base_url="http://localhost:1/v1", model="test"),
        log=DecisionLog(config["log"]),
        config=config,
        trace=TraceLog(config["trace"]),
    )
    assert [s.name for s in scheduler.tree["simplex"]] == [
        "connect_link",
        "next_message",
    ]
    assert [s.name for s in scheduler.tree["calendar"]] == ["next_event"]


def test_navigate_empty_tree_short_circuits(tmp_path):
    """An empty tree goes straight to CreateCategory without a SemIf call."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    result = navigate(None, log, trace, Request("anything"), {})
    assert isinstance(result, CreateCategory)
    assert log.read() == []
    assert any(e["kind"] == "create_category" for e in trace.read())


def test_gate_decision_carries_expectation_and_capabilities():
    """The gate must know the user expects handling and what the agent can
    actually do; without that context, lookups are read as small talk and
    dropped (the nextcloud calendar regression)."""
    tree = build_tree(build_skills({"skills": {}}))
    tree["calendar"] = [
        Skill(name="next_event", category="calendar", description="Report the next event.")
    ]
    decision = build_gate_decision(
        Request("tell me the next event in my nextcloud calendar"), tree
    )
    assert "tell me the next event in my nextcloud calendar" in decision.state
    assert "perform a task" in decision.state
    assert "look something up" in decision.state
    assert "calendar: next_event" in decision.state
    assert decision.question == "Should the agent handle this input?"
    assert [o.id for o in decision.options] == ["yes", "no"]
    assert "performing an action" in decision.options[0].description
    assert "greeting" in decision.options[1].description


def test_gate_decision_accepts_imperative_tasks_with_no_matching_skill():
    """A request to do something the tree has no skill for is still actionable —
    the agent learns it. Regression: 'send a simplex message to pepper saying
    hi' scored 0.135 and was dropped as 'no actionable request' because the gate
    context only vouched for lookups."""
    tree = build_tree(build_skills({"skills": {}}))
    tree["simplex"] = [
        Skill(name="next_message", category="simplex", description="Read the next SimpleX message."),
    ]
    decision = build_gate_decision(Request("send a simplex message to pepper saying hi"), tree)
    assert "even if no skill matches yet" in decision.state


def test_gate_decision_empty_tree_still_builds():
    decision = build_gate_decision(Request("hello"), {})
    assert "(none yet)" in decision.state
    assert decision.options[0].id == "yes"


def test_navigate_leaf_offers_create_for_unmatched_action(tmp_path):
    """The create_skill fallback must describe the *trigger* (no skill performs
    the action), not the mechanism ('suggest creating a new skill'): the
    decision model only sees option descriptions, and the generic wording lost
    0.03-0.06 to a read skill on send requests."""

    class Recording:
        def __init__(self):
            self.request = None

        def call(self, request):
            from semif_agent.decisions import DecisionResult

            self.request = request
            ids = [o.id for o in request.options]
            return DecisionResult(request=request, option_ids=ids, probabilities=[1.0] * len(ids))

    engine = Recording()
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    tree = {"simplex": [Skill(name="next_message", category="simplex", description="Read the next SimpleX message.")]}
    navigate(engine, log, trace, Request("send a simplex message"), tree)
    create = next(o for o in engine.request.options if o.id == "create_skill")
    assert "No existing skill performs this action" in create.description
    assert "action" in engine.request.question


def test_confirm_skill_fit_returns_true_when_action_matches(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = ScriptedEngine(choices={"same action": "same"})
    skill = Skill(name="next_message", category="simplex", description="Read the next SimpleX message.")
    assert confirm_skill_fit(engine, log, trace, Request("read the next simplex message"), skill) is True
    row = log.read()[-1]
    assert row["extra"]["phase"] == "navigate:intent"
    assert row["selected"] == "same"


def test_confirm_skill_fit_returns_false_on_mismatch(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = ScriptedEngine(choices={"same action": "different"})
    skill = Skill(name="next_message", category="simplex", description="Read the next SimpleX message.")
    assert confirm_skill_fit(engine, log, trace, Request("send a simplex message"), skill) is False
    assert any(e["kind"] == "intent_guard" and e["fits"] is False for e in trace.read())


def test_dispatch_intent_mismatch_authorizes_new_skill(tmp_path):
    """A routed skill whose action doesn't match the request must not run: the
    guard sends dispatch to create_skill in the same category instead."""
    scheduler = _scheduler(
        tmp_path,
        choices={"same action": "different"},
    )
    scheduler.tree["simplex"] = [
        Skill(name="next_message", category="simplex", description="Read the next SimpleX message."),
    ]
    scheduler._create_skill = lambda request, category: DispatchResult(
        kind="create_skill", summary="authoring requested", skill="send_message"
    )
    request = Request("send a simplex message to pepper")
    result = scheduler._dispatch(request)
    assert result.kind == "create_skill"
    phases = [r["extra"].get("phase") for r in scheduler.log.read()]
    assert "navigate:intent" in phases


def test_dispatch_intent_match_runs_skill(tmp_path):
    scheduler = _scheduler(tmp_path, choices={"same action": "same"})

    def predict(ctx, request):
        return Prediction(text="")

    def act(ctx, request, prediction):
        return ActionResult(action_log="read the next message", new_state="read")

    scheduler.tree.pop("response", None)
    scheduler.tree["simplex"] = [
        Skill(
            name="next_message",
            category="simplex",
            description="Read the next SimpleX message.",
            predict=predict,
            act=act,
        ),
    ]
    request = Request("read the next simplex message")
    result = scheduler._dispatch(request)
    assert result.kind == "ran"
    assert result.skill == "next_message"



def test_dispatch_create_category_without_engine_is_fatal(tmp_path):
    """An empty tree short-circuits to CreateCategory; without an engine the
    category authoring is fatal — the app is marked and the caller stops."""
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
            "skill_seeds": str(tmp_path / "seeds"),
        },
        trace=trace,
    )
    scheduler.tree = {}
    with pytest.raises(EngineUnavailable):
        scheduler._dispatch(Request("anything"))
    assert scheduler.fatal is not None
    assert "not available" in scheduler.fatal


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
    httpd, base = _pipeline_codegen_server([GOOD_BODY, "{}", GOOD_TEST])
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
    scheduler = _scheduler(tmp_path, choices={"ask again each time": "ask_again"})
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
        ['{"questions": ["Draft or send?"]}', GOOD_BODY, "{}", GOOD_TEST]
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
    httpd, base = _pipeline_codegen_server([GOOD_BODY, "{}", GOOD_TEST])
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
        ['{"questions": ["A?", "B?"]}', GOOD_BODY, "{}", GOOD_TEST]
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
    failing_test = "import sys\nsys.exit(1)"
    httpd, base = _pipeline_codegen_server(
        [GOOD_BODY, "{}", failing_test, GOOD_TEST]
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


# ---- deferred questions ----

def test_pending_questions_roundtrip(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.questions.append(
        PendingQuestion(
            id="q1", run_id="r1", category="tracking", skill="probe",
            question="Which service?",
        )
    )
    pending = scheduler.pending_questions()
    assert [q["id"] for q in pending] == ["q1"]
    status, _ = scheduler.answer_question("q1", "Nextcloud")
    assert status == "ok"
    assert scheduler.pending_questions() == []
    assert scheduler.questions == []
    assert any(e["kind"] == "question_answered" for e in scheduler.trace.read())


def test_answer_question_empty_skips(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.questions.append(
        PendingQuestion(
            id="q2", run_id="r1", category="tracking", skill="probe", question="Q?"
        )
    )
    status, detail = scheduler.answer_question("q2", "")
    assert status == "ok"
    assert "skipped" in detail
    assert scheduler.pending_questions() == []


def test_answer_question_unknown_id_errors(tmp_path):
    scheduler = _scheduler(tmp_path)
    status, detail = scheduler.answer_question("nope", "x")
    assert status == "error"
    assert "no pending question" in detail


def test_post_questions_waits_for_answers(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.elicitation_wait = 5.0
    job = SkillWrite(
        request=Request("track it"),
        category="tracking",
        draft=SkillDraft(name="probe", description="Probe."),
        weight=0.5,
    )
    result: dict = {}
    thread = threading.Thread(
        target=lambda: result.update(scheduler._post_questions(job, ["Q1?", "Q2?"]))
    )
    thread.start()
    deadline = time.monotonic() + 5
    pending = []
    while time.monotonic() < deadline:
        pending = scheduler.pending_questions()
        if len(pending) == 2:
            break
        time.sleep(0.02)
    assert len(pending) == 2
    scheduler.answer_question(pending[0]["id"], "A1")
    scheduler.answer_question(pending[1]["id"], "")
    thread.join(timeout=5)
    assert result == {"Q1?": "A1"}
    assert scheduler.pending_questions() == []
    kinds = [e["kind"] for e in scheduler.trace.read()]
    assert "questions_asked" in kinds


def test_post_questions_times_out(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.elicitation_wait = 0.1
    job = SkillWrite(
        request=Request("track it"),
        category="tracking",
        draft=SkillDraft(name="probe", description="Probe."),
        weight=0.5,
    )
    assert scheduler._post_questions(job, ["Q?"]) == {}
    assert scheduler.pending_questions() == []
    assert any(e["kind"] == "questions_timeout" for e in scheduler.trace.read())


# ---- requirements persistence ----

def test_registry_stores_requirements_for_restart(tmp_path):
    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    registry.register_skill(
        "tracking", "probe", "Probe.", request_text="track it",
        requirements={"Which service?": "Nextcloud"},
    )
    row = registry.read()["tracking"]["skills"][0]
    assert row["requirements"] == {"Which service?": "Nextcloud"}
    scheduler = _scheduler(tmp_path)
    draft = scheduler._stub_draft("tracking", "probe")
    assert draft.requirements == {"Which service?": "Nextcloud"}


# ---- repair loop ----

def _repair_offer(offer_id="r1", selected="repair_skill", skill="tracking.check"):
    return RepairOffer(
        id=offer_id,
        run_id="run1",
        category="tracking",
        skill=skill,
        selected=selected,
        reason="boom",
        request_text="track my package",
        failure="connection refused",
    )


def _install_tracking_stub(scheduler):
    """A repair rewrite needs the leaf to exist in the tree (a real skill that
    failed), so a runnable body can replace it."""
    scheduler.tree["tracking"] = [
        Skill(
            name="tracking.check",
            category="tracking",
            description="Check the delivery status of a package.",
        )
    ]


def test_resolve_repair_no_repair_declines(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.repairs.append(_repair_offer())
    assert [r["id"] for r in scheduler.pending_repairs()] == ["r1"]
    status, _ = scheduler.resolve_repair("r1", "no_repair")
    assert status == "ok"
    assert scheduler.pending_repairs() == []
    assert any(e["kind"] == "repair_declined" for e in scheduler.trace.read())


def test_resolve_repair_retry_requeues(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.repairs.append(_repair_offer(selected="retry"))
    status, detail = scheduler.resolve_repair("r1")
    assert status == "running"
    assert "retrying" in detail
    assert any(e["kind"] == "repair_retry" for e in scheduler.trace.read())


def test_resolve_repair_ask_user_question_then_answer_starts_repair(tmp_path):
    scheduler = _scheduler(
        tmp_path,
        codegen=CodegenClient(base_url="http://127.0.0.1:1/v1", model="test", timeout=2),
    )
    _install_tracking_stub(scheduler)
    scheduler.repairs.append(_repair_offer(selected="ask_user"))
    status, _ = scheduler.resolve_repair("r1")
    assert status == "needs_input"
    pending = scheduler.pending_questions()
    assert len(pending) == 1
    assert pending[0]["kind"] == "repair"
    status, _ = scheduler.answer_question(pending[0]["id"], "use the app password")
    assert status == "running"
    writing = [
        e for e in scheduler.trace.read() if e["kind"] == "skill_writing"
    ]
    assert writing and writing[-1]["repair"] is True
    assert any(e["kind"] == "repair_executed" for e in scheduler.trace.read())


def test_repair_budget_is_bounded(tmp_path):
    scheduler = _scheduler(
        tmp_path,
        codegen=CodegenClient(base_url="http://127.0.0.1:1/v1", model="test", timeout=2),
    )
    _install_tracking_stub(scheduler)
    scheduler.repairs.append(_repair_offer("r1"))
    assert scheduler.resolve_repair("r1")[0] == "running"
    scheduler.repairs.append(_repair_offer("r2"))
    status, detail = scheduler.resolve_repair("r2")
    assert status == "error"
    assert "budget" in detail


def test_resolve_repair_unknown_offer_errors(tmp_path):
    scheduler = _scheduler(tmp_path)
    status, detail = scheduler.resolve_repair("missing")
    assert status == "error"
    assert "no repair offer" in detail


def test_exhausted_reentry_offers_repair(tmp_path):
    """A failure whose retry is refused because the reentry budget is spent must
    still surface a repair offer. Before, `updated_request` stayed truthy so the
    offer was skipped and the failure vanished after the retry ladder."""
    scheduler = _scheduler(tmp_path)
    _install_tracking_stub(scheduler)
    skill = scheduler.tree["tracking"][0]
    skill.predict = lambda ctx, request: Prediction(text="")
    skill.act = lambda ctx, request, prediction: ActionResult(
        action_log="connection refused", new_state="unchanged"
    )
    request = Request("track my package")
    request.reentries = scheduler.max_reentries
    outcome = RunResult(
        skill="tracking.check",
        success=False,
        summary="failed",
        action_log="connection refused",
        new_state="unchanged",
        updated_request="track my package",
    )
    scheduler._finish_run(skill, request, outcome)
    assert not any(e["kind"] == "requeued" for e in scheduler.trace.read())
    assert any(e["kind"] == "repair_offered" for e in scheduler.trace.read())
