"""Pure-stdlib tests for skill-tree authoring: category and skill prompts, draft
parsing, the category registry, the async codegen-body workflow, and the error
path when the engine is unavailable.

No mocking: the async write test uses a real CodegenClient against a throwaway
stdlib HTTP server (real endpoint, per the repo rule); engine-dependent success
paths are exercised only by the integration tests against the real decision
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
from semif_agent.engine import EngineConfig, SemIfEngine
from semif_agent.llm import LLMClient, LLMError
from semif_agent.log import DecisionLog
from semif_agent.skill import RunResult
from semif_agent.scheduler import (
    DraftAuthor,
    PendingQuestion,
    RepairOffer,
    Scheduler,
    SkillWrite,
)
from semif_agent.skills import (
    ActionResult,
    CategoryDraft,
    CategoryRegistry,
    CreateCategory,
    CreateSkill,
    Skill,
    SkillDraft,
    SkillStore,
    build_category_prompt,
    build_skill_prompt,
    build_skills,
    build_tree,
    confirm_category_fit,
    confirm_non_action,
    confirm_skill_fit,
    category_descriptions,
    CATEGORY_DESCRIPTIONS,
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
from semif_agent.skills import ActionResult

INTEGRATION = {
    "service": "probe_service",
    "transport": "compute",
    "config_vars": [],
}

CONTRACT = {}

def act(ctx, request):
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

    def act(ctx, request):
        return ActionResult(
            action_log="Next event on 'personal': Team sync — 2026-09-27 10:00 CEST",
            new_state="next event reported",
        )

    skill = Skill(
        name="next_event",
        category="calendar",
        description="Report the next event.",
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

    def act(ctx, request):
        return ActionResult(action_log="", new_state="service unreachable")

    skill = Skill(name="probe", category="tracking", description="Probe.",
                  act=act)
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
    assert "response: response.acknowledge" in joined


def test_build_skills_are_canned_responses_only():
    """Real integrations are generated or seeded, never fabricated here: a
    hardcoded fake shadows the authoring path for a real skill (navigation
    routes to it). The only built-ins are the closed `response` canned tree."""
    skills = build_skills({"skills": {}})
    names = [skill.name for skill in skills]
    assert names and all(name.startswith("response.") for name in names)
    assert "response.clarify" in names
    assert all(skill.category == "response" for skill in skills)
    assert not any(skill.is_noop() for skill in skills)


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


def test_generate_category_without_endpoint_raises():
    client = LLMClient(base_url="http://127.0.0.1:1/v1", model="test", timeout=2)
    with pytest.raises(LLMError):
        generate_category(client, Request("anything"), {})


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


def test_generate_skill_without_endpoint_raises():
    client = LLMClient(base_url="http://127.0.0.1:1/v1", model="test", timeout=2)
    with pytest.raises(LLMError):
        generate_skill(client, Request("anything"), "tracking", {})


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


def test_category_descriptions_never_bare_and_seeded(tmp_path):
    """Every category must carry a usable description at the softmax — a bare
    name carries no signal. Built-ins come from CATEGORY_DESCRIPTIONS; an
    authored category carries its registry description; a legacy/empty entry
    falls back to the built-in or empty string without crashing."""
    tree = build_tree(build_skills({"skills": {}}))
    assert tree["response"][0].category_description  # built-in seed

    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    registry.register("travel", "Booking flights, hotels, and trips.")
    registry.register_skill("travel", "book_flight", "Book a flight.")
    merge_registry(tree, registry.read())
    assert tree["travel"][0].category_description == "Booking flights, hotels, and trips."

    descriptions = category_descriptions(tree)
    assert descriptions["response"] == CATEGORY_DESCRIPTIONS["response"]
    assert descriptions["travel"] == "Booking flights, hotels, and trips."
    # an empty bucket (no skill to carry the description) still resolves to text
    tree["orphan"] = []
    assert category_descriptions(tree)["orphan"] == ""


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
    assert [s.name for s in calendar] == ["create_event", "next_event"]
    next_event = next(s for s in calendar if s.name == "next_event")
    assert next_event.description.startswith("Report the next")
    assert next_event.status == "ready"
    assert next_event.integration["transport"] == "caldav"
    create_event = next(s for s in calendar if s.name == "create_event")
    assert create_event.description.startswith("Create a calendar event")
    assert create_event.integration["transport"] == "caldav"


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
    assert [s.name for s in scheduler.tree["calendar"]] == [
        "create_event",
        "next_event",
    ]


def test_navigate_empty_tree_short_circuits(tmp_path):
    """An empty tree goes straight to CreateCategory without a SemIf call."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    result = navigate(None, log, trace, Request("anything"), {})
    assert isinstance(result, CreateCategory)
    assert log.read() == []
    assert any(e["kind"] == "create_category" for e in trace.read())


def test_navigate_canned_category_never_offers_create(tmp_path):
    """The `response` tree is closed: its leaf choice has no create_skill option
    and never authors, regardless of how weak the match is. The actionability
    guard is what routes a non-request input into it."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    tree = build_tree(build_skills({"skills": {}}))

    engine = ScriptedEngine(choices={"non-request input": "non_action"})
    result = navigate(engine, log, trace, Request("hello there"), tree)
    assert isinstance(result, Skill)
    assert result.category == "response"
    leaf = next(c for c in engine.calls if "canned response" in c.question)
    assert all(o.id != "create_skill" for o in leaf.options)
    assert "create_skill" not in [o.id for o in leaf.options]
    assert result.name in [o.id for o in leaf.options]


def test_navigate_canned_category_points_at_the_catchall(tmp_path):
    """The catchall is offered last and named in the question when nothing
    specific fits."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    tree = build_tree(build_skills({"skills": {}}))
    engine = _ProbEngine(
        {
            "non-request input": {"non_action": 1.0},
            "canned response": {"response.clarify": 1.0},
        }
    )
    result = navigate(engine, log, trace, Request("do the thing"), tree)
    assert result.name == "response.clarify"
    leaf = engine.calls[-1]
    assert "response.clarify" in leaf.question
    assert leaf.options[-1].id == "response.clarify"
    assert "navigate:response" in [r["extra"]["phase"] for r in log.read()]


def test_canned_response_runs_without_assessment(tmp_path):
    """A canned reply is a fixed line: no side effect to assess, no repair loop.
    The runner returns success without an assess:outcome decision."""
    from semif_agent.skill import SkillRunner
    from semif_agent.skills import ActionContext

    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    skill = next(s for s in build_skills({"skills": {}}) if s.name == "response.greeting")
    runner = SkillRunner(ActionContext(engine=_ProbEngine({}), config={}), log)
    outcome = runner.run(skill, Request("hello"))
    assert outcome.success is True
    assert outcome.new_state == "Hello! What can I help you with?"
    assert outcome.error is None
    assert log.read() == []



def test_navigate_leaf_picks_among_existing_skills_only(tmp_path):
    """Two-stage leaf: create_skill is NOT an option in the leaf softmax. The
    leaf decision offers only the existing skills, and navigation returns the
    best-matching one; the intent guard in dispatch owns reuse-vs-create."""

    class Recording:
        def __init__(self):
            self.request = None

        def call(self, request):
            from semif_agent.decisions import DecisionResult

            self.request = request
            ids = [o.id for o in request.options]
            if "non_action" in ids:
                # the top-level actionability guard: this input is a request
                probs = [1.0 if o == "action" else 0.0 for o in ids]
            else:
                probs = [1.0] * len(ids)
            return DecisionResult(request=request, option_ids=ids, probabilities=probs)

    engine = Recording()
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    tree = {
        "simplex": [
            Skill(name="next_message", category="simplex", description="Read the next SimpleX message."),
            Skill(name="connect_link", category="simplex", description="Show the contact link."),
        ]
    }
    result = navigate(engine, log, trace, Request("send a simplex message"), tree)
    ids = [o.id for o in engine.request.options]
    assert ids == ["next_message", "connect_link"]
    assert "create_skill" not in ids
    assert getattr(result, "name", None) in ids


class _ProbEngine:
    """Returns caller-specified probabilities, keyed by a substring of the
    decision question, so a test can force a specific navigation outcome."""

    def __init__(self, by_question: dict[str, dict[str, float]]):
        self.by_question = by_question
        self.calls = []

    def call(self, request):
        from semif_agent.decisions import DecisionResult

        self.calls.append(request)
        ids = [o.id for o in request.options]
        probs_map = {}
        for key, value in self.by_question.items():
            if key in request.question:
                probs_map = value
                break
        if not probs_map:
            probs_map = {ids[0]: 1.0}
        probs_map = {k: v for k, v in probs_map.items() if k in ids}
        if not probs_map:
            probs_map = {ids[0]: 1.0}
        return DecisionResult(
            request=request,
            option_ids=ids,
            probabilities=[float(probs_map.get(i, 0.0)) for i in ids],
        )


def _leaf_tree():
    return {
        "simplex": [
            Skill(
                name="next_message",
                category="simplex",
                description="Read the next SimpleX message.",
            ),
            Skill(
                name="connect_link",
                category="simplex",
                description="Show the contact link.",
            ),
        ]
    }


def test_navigate_leaf_returns_best_existing_skill(tmp_path):
    """The leaf softmax picks the highest-probability existing skill; no create
    branch competes with it (that was the dilution bug)."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = _ProbEngine(
        {
            "top-level category": {"simplex": 1.0},
            "Does the scope": {"covers": 1.0},
            "skill performs": {"next_message": 0.45, "connect_link": 0.55},
        }
    )
    result = navigate(
        engine, log, trace, Request("send a simplex message"), _leaf_tree()
    )
    assert getattr(result, "name", None) == "connect_link"
    assert not isinstance(result, CreateSkill)


def test_navigate_single_skill_category_skips_the_softmax(tmp_path):
    """A one-option softmax is meaningless; the sole skill is returned directly
    and the guard decides reuse-vs-create."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = _ProbEngine(
        {"top-level category": {"simplex": 1.0}, "Does the scope": {"covers": 1.0}}
    )
    tree = {
        "simplex": [
            Skill(name="next_message", category="simplex", description="Read the next SimpleX message.")
        ]
    }
    result = navigate(engine, log, trace, Request("read the next message"), tree)
    assert getattr(result, "name", None) == "next_message"
    # only the category decision was made; the leaf softmax was skipped
    assert all("Which simplex skill" not in c.question for c in engine.calls)


def test_navigate_empty_category_short_circuits_to_create(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = _ProbEngine({"top-level category": {"simplex": 1.0}})
    result = navigate(engine, log, trace, Request("anything"), {"simplex": []})
    assert isinstance(result, CreateSkill)
    assert result.category == "simplex"
    assert any(e["kind"] == "skill_needed" for e in trace.read())


def test_navigate_category_options_carry_descriptions(tmp_path):
    """The category softmax must offer each category WITH its description — bare
    names carry no signal (the misroute bug)."""
    engine = _ProbEngine({"top-level category": {"simplex": 1.0}})
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    tree = {
        "simplex": [
            Skill(
                name="next_message",
                category="simplex",
                description="Read the next SimpleX message.",
                category_description="SimpleX messaging: read/send messages.",
            )
        ]
    }
    navigate(engine, log, trace, Request("read the next message"), tree)
    cat_decision = next(c for c in engine.calls if "top-level category" in c.question)
    by_id = {o.id: o.description for o in cat_decision.options}
    assert by_id["simplex"] == "SimpleX messaging: read/send messages."
    assert "new top-level category is needed" in by_id["create_category"]


def test_navigate_category_rejected_scope_authors_category(tmp_path):
    """The confirm guard is the category create door: a softmax winner whose
    scope the guard rejects authors a new category. The winner is below
    `softmax_bypass_tau`, so the guard actually runs."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    tree = _leaf_tree()
    engine = _ProbEngine(
        {
            "top-level category": {"simplex": 0.45},
            "Does the scope": {"none": 1.0},
        }
    )
    result = navigate(engine, log, trace, Request("book a flight"), tree)
    assert isinstance(result, CreateCategory)
    assert any(e["kind"] == "category_scope_rejected" for e in trace.read())
    assert any(e["kind"] == "create_category" for e in trace.read())


def test_navigate_category_confirmed_scope_descends(tmp_path):
    """A below-threshold softmax winner the guard confirms is descended into."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = _ProbEngine(
        {
            "top-level category": {"simplex": 0.45},
            "Does the scope": {"covers": 1.0},
            "skill performs": {"next_message": 1.0, "connect_link": 0.0},
        }
    )
    result = navigate(engine, log, trace, Request("read a message"), _leaf_tree())
    assert getattr(result, "name", None) == "next_message"
    assert not isinstance(result, CreateCategory)
    assert any(e["kind"] == "category_scope" and e["fits"] for e in trace.read())


def test_navigate_confident_category_winner_skips_scope_confirm(tmp_path):
    """A category softmax winner at/above `softmax_bypass_tau` is decisive: the
    scope confirm is not called (no `navigate:category_scope` decision)."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = _ProbEngine(
        {
            "top-level category": {"simplex": 0.9},
            "skill performs": {"next_message": 1.0, "connect_link": 0.0},
        }
    )
    result = navigate(engine, log, trace, Request("read a message"), _leaf_tree())
    assert getattr(result, "name", None) == "next_message"
    assert any(e["kind"] == "category_scope_bypassed" for e in trace.read())
    assert not any(e["kind"] == "category_scope" for e in trace.read())
    assert all("Does the scope" not in c.question for c in engine.calls)
    phases = [r["extra"].get("phase") for r in log.read()]
    assert "navigate:category_scope" not in phases


def test_confirm_category_fit_threshold(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = ScriptedEngine(choices={"Does the scope": "covers"})
    assert (
        confirm_category_fit(
            engine, log, trace, Request("check my calendar"), "calendar", "Calendars.", tau=0.5
        )
        is True
    )
    assert log.read()[-1]["extra"]["phase"] == "navigate:category_scope"

    engine = ScriptedEngine(choices={"Does the scope": "none"})
    assert (
        confirm_category_fit(
            engine, log, trace, Request("order pizza"), "calendar", "Calendars.", tau=0.5
        )
        is False
    )


def test_confirm_non_action_threshold(tmp_path):
    """The actionability guard returns True for non-request input and False for a
    request, logging phase `navigate:actionability`."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = ScriptedEngine(choices={"non-request input": "non_action"})
    assert (
        confirm_non_action(engine, log, trace, Request("hello there"), tau=0.5) is True
    )
    assert log.read()[-1]["extra"]["phase"] == "navigate:actionability"

    engine = ScriptedEngine(choices={"non-request input": "action"})
    assert (
        confirm_non_action(engine, log, trace, Request("what is 255 * 12?"), tau=0.5)
        is False
    )
    assert any(
        e["kind"] == "actionability" and e["non_action"] is False for e in trace.read()
    )


def test_navigate_actionability_sends_a_request_to_create(tmp_path):
    """A request the softmax dumps in the catchall must author, not get a canned
    reply: the actionability guard is the create door ("what is 255 * 12?")."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    tree = build_tree(build_skills({"skills": {}}))
    engine = _ProbEngine(
        {
            "top-level category": {"response": 1.0},
            "non-request input": {"action": 1.0},
        }
    )
    result = navigate(engine, log, trace, Request("what is 255 * 12?"), tree)
    assert isinstance(result, CreateCategory)
    assert any(
        e["kind"] == "actionability" and e["non_action"] is False for e in trace.read()
    )
    assert any(e["kind"] == "create_category" for e in trace.read())


def test_navigate_actionability_keeps_a_statement_in_response(tmp_path):
    """The mirror: a statement the reworded scope pushes to create_category must
    still get a canned reply ("1+1=2"), not author a category."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    tree = build_tree(build_skills({"skills": {}}))
    engine = _ProbEngine(
        {
            "top-level category": {"create_category": 1.0},
            "non-request input": {"non_action": 1.0},
        }
    )
    result = navigate(engine, log, trace, Request("1+1=2"), tree)
    assert isinstance(result, Skill)
    assert result.category == "response"
    assert any(
        e["kind"] == "actionability" and e["non_action"] is True for e in trace.read()
    )


def test_navigate_actionability_gate_precedes_category_selection(tmp_path):
    """The actionability guard runs at the top: a non-request never reaches the
    category softmax, so it cannot be routed into a real category even when the
    softmax would have picked one."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    tree = build_tree(build_skills({"skills": {}}))
    tree["simplex"] = [
        Skill(
            name="next_message",
            category="simplex",
            description="Read the next SimpleX message.",
        )
    ]
    engine = _ProbEngine(
        {
            "non-request input": {"non_action": 1.0},
            # If the category softmax ran it would route to simplex, not response.
            "top-level category": {"simplex": 1.0},
            "canned response": {"response.greeting": 1.0},
        }
    )
    result = navigate(engine, log, trace, Request("the sky is blue"), tree)
    assert getattr(result, "category", None) == "response"
    assert all(
        "top-level category" not in c.question for c in engine.calls
    ), "the category softmax must not run for non-request input"
    phases = [r["extra"]["phase"] for r in log.read()]
    assert "navigate:actionability" in phases
    assert "navigate:category" not in phases
    assert phases.index("navigate:actionability") < phases.index("navigate:response")


def test_navigate_category_softmax_excludes_the_canned_response_tree(tmp_path):
    """Once actionability is decided at the top, `response` is never offered in
    the category softmax — a request cannot be routed to a canned reply — and
    actionability is logged before category selection."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    tree = build_tree(build_skills({"skills": {}}))
    tree["simplex"] = [
        Skill(
            name="next_message",
            category="simplex",
            description="Read the next SimpleX message.",
        )
    ]
    engine = _ProbEngine(
        {
            "top-level category": {"simplex": 1.0},
            "Does the scope": {"covers": 1.0},
        }
    )
    navigate(engine, log, trace, Request("read the next simplex message"), tree)
    cat_decision = next(
        c for c in engine.calls if "top-level category" in c.question
    )
    ids = [o.id for o in cat_decision.options]
    assert "response" not in ids
    assert "simplex" in ids and "create_category" in ids
    phases = [r["extra"]["phase"] for r in log.read()]
    assert phases.index("navigate:actionability") < phases.index("navigate:category")


def test_dispatch_action_tau_controls_catchall_routing(tmp_path):
    """navigation.action_tau is the catchall create door, separate from the other
    taus: a borderline non-action verdict flips on it."""
    def build(action_tau: float):
        scheduler = Scheduler(
            engine=_ProbEngine(
                {
                    "top-level category": {"response": 1.0},
                    "non-request input": {"non_action": 0.6, "action": 0.4},
                }
            ),
            llm=LLMClient(base_url="http://localhost:1/v1", model="test"),
            log=DecisionLog(str(tmp_path / "decisions.jsonl")),
            config={
                "skills": {},
                "navigation": {"action_tau": action_tau},
                "category_registry": str(tmp_path / "categories.json"),
                "skill_bodies": str(tmp_path / "skills"),
                "skill_seeds": str(tmp_path / "seeds"),
            },
            trace=TraceLog(str(tmp_path / "runs.jsonl")),
        )
        scheduler.tree = build_tree(build_skills({"skills": {}}))
        scheduler._queue_draft = lambda *a, **k: None
        return scheduler

    low = build(0.5)
    assert low.action_tau == 0.5
    ran = low._dispatch(Request("hmm"))
    assert ran.kind == "ran", "non_action 0.6 clears action_tau 0.5: canned reply"

    high = build(0.7)
    assert high.action_tau == 0.7
    created = high._dispatch(Request("hmm"))
    assert created.kind == "create_category", "non_action 0.6 fails action_tau 0.7: author"



def test_dispatch_intent_tau_controls_reuse_vs_create(tmp_path):
    """The leaf reuse-vs-create knob is navigation.intent_tau, separate from the
    top-level tau: a borderline guard verdict flips on it while assessment is
    untouched."""
    def build(intent_tau: float):
        engine = _ProbEngine(
            {
                "top-level category": {"simplex": 1.0},
                "Does the scope": {"covers": 1.0},
                # Below softmax_bypass_tau (0.5), so the intent guard runs.
                "skill performs": {"next_message": 0.45, "connect_link": 0.3},
                "same action": {"same": 0.8, "different": 0.2},
            }
        )
        scheduler = Scheduler(
            engine=engine,
            llm=LLMClient(base_url="http://localhost:1/v1", model="test"),
            log=DecisionLog(str(tmp_path / "decisions.jsonl")),
            config={
                "skills": {},
                "navigation": {"intent_tau": intent_tau},
                "category_registry": str(tmp_path / "categories.json"),
                "skill_bodies": str(tmp_path / "skills"),
                "skill_seeds": str(tmp_path / "seeds"),
            },
            trace=TraceLog(str(tmp_path / "runs.jsonl")),
        )

        def act(ctx, request):
            return ActionResult(action_log="read fixture", new_state="read")

        scheduler.tree = {
            "simplex": [
                Skill(name="next_message", category="simplex", description="Read the next SimpleX message.", act=act),
                Skill(name="connect_link", category="simplex", description="Show the contact link."),
            ]
        }
        queued: list = []
        scheduler._queue_draft = lambda request, category, kind: queued.append(
            (category, kind)
        )
        return scheduler, queued

    low, queued_low = build(0.5)
    assert low.intent_tau == 0.5
    ran = low._dispatch(Request("read the next simplex message"))
    assert ran.kind == "ran", "guard same=0.8 clears intent_tau=0.5: reuse"
    assert not queued_low

    high, queued_high = build(0.95)
    created = high._dispatch(Request("read the next simplex message"))
    assert created.kind == "create_skill", "guard same=0.8 fails intent_tau=0.95: author"
    assert queued_high == [("simplex", "skill")]


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
        choices={"top-level category": "simplex", "same action": "different"},
    )
    scheduler.tree.pop("response", None)
    scheduler.tree["simplex"] = [
        Skill(name="next_message", category="simplex", description="Read the next SimpleX message."),
    ]
    queued: list = []
    scheduler._queue_draft = lambda request, category, kind: queued.append(
        (category, kind)
    )
    request = Request("send a simplex message to pepper")
    result = scheduler._dispatch(request)
    assert result.kind == "create_skill"
    assert queued == [("simplex", "skill")]
    phases = [r["extra"].get("phase") for r in scheduler.log.read()]
    assert "navigate:intent" in phases


def test_dispatch_intent_match_runs_skill(tmp_path):
    scheduler = _scheduler(
        tmp_path, choices={"top-level category": "simplex", "same action": "same"}
    )

    def act(ctx, request):
        return ActionResult(action_log="read the next message", new_state="read")

    scheduler.tree.pop("response", None)
    scheduler.tree["simplex"] = [
        Skill(
            name="next_message",
            category="simplex",
            description="Read the next SimpleX message.",
                        act=act,
        ),
    ]
    request = Request("read the next simplex message")
    result = scheduler._dispatch(request)
    assert result.kind == "ran"
    assert result.skill == "next_message"


def test_dispatch_confident_leaf_winner_skips_intent_guard(tmp_path):
    """A leaf softmax winner at/above `softmax_bypass_tau` skips the intent
    confirm: the guard would reject here (same action = different), but a
    decisive softmax runs the skill instead."""
    engine = _ProbEngine(
        {
            "top-level category": {"simplex": 1.0},
            "skill performs": {"next_message": 0.9, "connect_link": 0.1},
            "same action": {"different": 1.0},
        }
    )
    scheduler = Scheduler(
        engine=engine,
        llm=LLMClient(base_url="http://localhost:1/v1", model="test"),
        log=DecisionLog(str(tmp_path / "decisions.jsonl")),
        config={
            "skills": {},
            "category_registry": str(tmp_path / "categories.json"),
            "skill_bodies": str(tmp_path / "skills"),
            "skill_seeds": str(tmp_path / "seeds"),
        },
        trace=TraceLog(str(tmp_path / "runs.jsonl")),
    )

    def act(ctx, request):
        return ActionResult(action_log="read fixture", new_state="read")

    scheduler.tree = {
        "simplex": [
            Skill(
                name="next_message",
                category="simplex",
                description="Read the next SimpleX message.",
                act=act,
            ),
            Skill(name="connect_link", category="simplex", description="Show the contact link."),
        ]
    }
    result = scheduler._dispatch(Request("read the next simplex message"))
    assert result.kind == "ran"
    assert result.skill == "next_message"
    assert any(e["kind"] == "intent_bypass" for e in scheduler.trace.read())
    phases = [r["extra"].get("phase") for r in scheduler.log.read()]
    assert "navigate:intent" not in phases



def test_dispatch_create_category_queues_draft_and_is_not_fatal(tmp_path):
    """An empty tree short-circuits to CreateCategory; the `llm` draft is
    queued asynchronously. An unreachable llm endpoint is graceful — traced and
    the request is not re-dispatched — never fatal (only SemIf is fatal)."""
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    scheduler = Scheduler(
        engine=SemIfEngine(EngineConfig()),
        llm=LLMClient(base_url="http://127.0.0.1:1/v1", model="test", timeout=2),
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
    result = scheduler._dispatch(Request("anything"))
    assert result.kind == "create_category"
    assert scheduler.fatal is None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if any(e["kind"] == "draft_failed" for e in trace.read()):
            break
        time.sleep(0.02)
    assert any(e["kind"] == "draft_failed" for e in trace.read())
    assert scheduler.fatal is None


def _answer_approval_when_posted(scheduler, approved, timeout=5.0):
    """Approve/deny the first pending creation proposal, as a front end would."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pending = scheduler.pending_approvals()
        if pending:
            scheduler.answer_approval(pending[0]["id"], approved)
            return True
        time.sleep(0.02)
    return False


def test_creation_approval_off_proceeds(tmp_path):
    """The default (no `creation_approval`) leaves creation unchanged: the gate
    returns immediately and posts nothing."""
    scheduler = _scheduler(tmp_path)
    job = DraftAuthor(
        request=Request("book a trip"), category=None, kind="category"
    )
    assert scheduler._request_approval(job, "category", "travel", "Trips.") is True
    assert scheduler.pending_approvals() == []


def test_creation_approval_approved_proceeds(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.creation_approval = True
    scheduler.defer_questions = True
    job = DraftAuthor(
        request=Request("book a trip"), category=None, kind="category"
    )
    result = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__(
            "ok", scheduler._request_approval(job, "category", "travel", "Trips.")
        )
    )
    thread.start()
    assert _answer_approval_when_posted(scheduler, approved=True)
    thread.join()
    assert result["ok"] is True
    kinds = [e["kind"] for e in scheduler.trace.read()]
    assert "creation_approval_requested" in kinds
    assert "creation_approval_approved" in kinds
    assert scheduler.pending_approvals() == []


def test_creation_approval_denied_aborts(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.creation_approval = True
    scheduler.defer_questions = True
    job = DraftAuthor(
        request=Request("book a trip"), category="travel", kind="skill"
    )
    result = {}
    thread = threading.Thread(
        target=lambda: result.__setitem__(
            "ok", scheduler._request_approval(job, "skill", "book_flight", "Book.")
        )
    )
    thread.start()
    assert _answer_approval_when_posted(scheduler, approved=False)
    thread.join()
    assert result["ok"] is False
    assert any(
        e["kind"] == "creation_approval_denied" for e in scheduler.trace.read()
    )


def test_creation_approval_timeout_denies(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.creation_approval = True
    scheduler.defer_questions = True
    scheduler.elicitation_wait = 0.1
    job = DraftAuthor(
        request=Request("book a trip"), category="travel", kind="skill"
    )
    assert scheduler._request_approval(job, "skill", "book_flight", "Book.") is False
    assert any(
        e["kind"] == "creation_approval_timeout" for e in scheduler.trace.read()
    )


def test_creation_approval_without_front_end_denies(tmp_path):
    """With no deferring front end there is nobody to answer, so creation is
    denied immediately rather than stalling the draft worker."""
    scheduler = _scheduler(tmp_path)
    scheduler.creation_approval = True
    scheduler.defer_questions = False
    job = DraftAuthor(
        request=Request("book a trip"), category="travel", kind="skill"
    )
    assert scheduler._request_approval(job, "skill", "book_flight", "Book.") is False
    assert any(
        e["kind"] == "creation_approval_skipped" for e in scheduler.trace.read()
    )


def test_creation_approval_chain_skips_second_prompt(tmp_path):
    """A skill job chained from an approved category is already covered."""
    scheduler = _scheduler(tmp_path)
    scheduler.creation_approval = True
    scheduler.defer_questions = True
    job = DraftAuthor(
        request=Request("book a trip"),
        category="travel",
        kind="skill",
        approved=True,
    )
    assert scheduler._request_approval(job, "skill", "book_flight", "Book.") is True
    assert scheduler.pending_approvals() == []


def test_author_category_denied_does_not_register(tmp_path, monkeypatch):
    """The category hook: a denied proposal registers nothing and queues no
    chained skill draft."""
    scheduler = _scheduler(tmp_path)
    scheduler.creation_approval = True
    scheduler.defer_questions = True
    monkeypatch.setattr(
        "semif_agent.scheduler.generate_category",
        lambda client, request, tree: CategoryDraft(name="travel", description="Trips."),
    )
    job = DraftAuthor(
        request=Request("book a trip"), category=None, kind="category"
    )
    thread = threading.Thread(target=scheduler._author_category, args=(job,))
    thread.start()
    assert _answer_approval_when_posted(scheduler, approved=False)
    thread.join()
    assert "travel" not in scheduler.tree
    assert not any(e["kind"] == "category_created" for e in scheduler.trace.read())
    assert list(scheduler._drafts) == []


def test_author_skill_denied_does_not_register(tmp_path, monkeypatch):
    scheduler = _scheduler(tmp_path)
    scheduler.creation_approval = True
    scheduler.defer_questions = True
    monkeypatch.setattr(
        "semif_agent.scheduler.generate_skill",
        lambda client, request, category, tree: SkillDraft(
            name="book_flight", description="Book a flight."
        ),
    )
    job = DraftAuthor(
        request=Request("book a flight"), category="travel", kind="skill"
    )
    thread = threading.Thread(target=scheduler._author_skill, args=(job,))
    thread.start()
    assert _answer_approval_when_posted(scheduler, approved=False)
    thread.join()
    assert scheduler.tree.get("travel", []) == []
    assert not any(e["kind"] == "skill_created" for e in scheduler.trace.read())


def test_skill_status_reflects_writing_and_noop():
    """A fresh Skill is a stub; marking it writing shows `writing`; a real body
    shows `ready`."""
    stub = Skill(name="probe", category="tracking", description="Probe.")
    assert stub.is_noop()
    assert stub.status == "stub"
    stub.writing = True
    assert stub.status == "writing"
    stub.writing = False

    def act(ctx, request):
        return None

    ready = Skill(name="probe", category="tracking", description="Probe.", act=act)
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
    result = scheduler._dispatch_skill(request, "tracking")
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
    and the single-slot worker runs codegen (body + CONTRACT) -> testgen -> test,
    then materializes the body, hot-merges it into the tree, and re-queues the
    original request for re-dispatch."""
    httpd, base = _pipeline_codegen_server([GOOD_BODY, GOOD_TEST])
    try:
        scheduler = _scheduler(tmp_path, codegen=CodegenClient(base_url=base, model="test", timeout=10))
        scheduler.tree["tracking"] = [
            Skill(name="track_live", category="tracking", description="Follow a package.")
        ]
        leaf = scheduler.tree["tracking"][0]

        request = Request("track my drone delivery in real time")
        scheduler._start_skill_write(request, "tracking", SkillDraft(name="track_live", description="Follow a package."))

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
        assert callable(upgraded.act)

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
    """A body without a valid CONTRACT (after the escalation ladder) leaves the
    leaf a restartable stub — no silent no-op, no wedged worker."""
    httpd, base = _pipeline_codegen_server(["def act(ctx, request):\n    return None\n"])
    try:
        scheduler = _scheduler(tmp_path, codegen=CodegenClient(base_url=base, model="test", timeout=10, max_attempts=2))
        scheduler.tree["tracking"] = [
            Skill(name="track_live", category="tracking", description="Follow a package.")
        ]
        leaf = scheduler.tree["tracking"][0]

        request = Request("track my drone delivery in real time")
        scheduler._start_skill_write(request, "tracking", SkillDraft(name="track_live", description="Follow a package."))

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


def test_async_skill_write_fails_gracefully_on_unimported_names(tmp_path):
    """A body with a valid CONTRACT but a name it never imports is rejected by
    the parse gate, so the retry ladder runs and the leaf stays a restartable
    stub — never a leaf that only fails at first run.

    Regression: the static gate used to accept a body that returned
    `ActionResult` without importing it; materialize then raised
    `name 'ActionResult' is not defined`.
    """
    body = (
        "INTEGRATION = {'service': 'probe_service', 'transport': 'compute', "
        "'config_vars': ['sender_address']}\n"
        "CONTRACT = {'sender_address': 'The sender.'}\n"
        "def act(ctx, request):\n"
        "    sender = ctx.config['sender_address']\n"
        "    return ActionResult(action_log=sender, new_state=request.text)\n"
    )
    httpd, base = _pipeline_codegen_server([body])
    try:
        scheduler = _scheduler(
            tmp_path,
            codegen=CodegenClient(base_url=base, model="test", timeout=10, max_attempts=2),
        )
        scheduler.tree["tracking"] = [
            Skill(name="track_live", category="tracking", description="Follow a package.")
        ]
        leaf = scheduler.tree["tracking"][0]

        request = Request("track my drone delivery in real time")
        scheduler._start_skill_write(
            request, "tracking", SkillDraft(name="track_live", description="Follow a package.")
        )

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
        assert "never imports" in failed.get("message", ""), failed
        assert not leaf.writing
        assert leaf.is_noop(), "leaf must stay a restartable stub"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_skill_pre_act_contract_pause_and_resume(tmp_path):
    """A contract variable the runner cannot satisfy pauses BEFORE act; the
    answer is recorded (engine unavailable -> ask-again, per-fire), then the run
    continues. A second pause comes from the skill's own needs_input."""
    scheduler = _scheduler(tmp_path, choices={"ask again each time": "ask_again"})
    scheduler.tree["tracking"] = []
    store = scheduler.body_store
    store.write_contract("tracking", "track_live", {"token": "The tracking token."})

    seen = []

    def act(ctx, request):
        if request.user_input:
            return ActionResult(action_log="done", new_state="done")
        return ActionResult(
            action_log="need confirmation", new_state=request.text, needs_input="Confirm?"
        )

    skill = Skill(
        name="track_live",
        category="tracking",
        description="Follow a package.",
        act=act,
        contract={"token": "The tracking token."},
    )

    result = scheduler._run_skill(skill, Request("track my package"))
    assert result.kind == "needs_input"
    assert "`token`" in result.summary
    assert scheduler.pending is not None
    assert scheduler.pending.pre_act is True

    reply = scheduler.answer("AB123")
    assert reply.status == "needs_input"
    assert scheduler.pending is not None
    assert scheduler.pending.pre_act is False
    assert scheduler.pending.request.meta["config_answers"]["token"] == "AB123"
    assert "Confirm?" in scheduler.pending.question

    reply = scheduler.answer("yes")
    assert reply.status == "ran"
    assert scheduler.pending is None


def test_pre_act_skip_leaves_variable_unset_and_does_not_reask(tmp_path):
    """An empty answer skips a contract variable: it is left unset, the run
    proceeds, and the same variable is never re-asked (regression: the old
    no-op skip re-asked the identical question forever)."""
    scheduler = _scheduler(tmp_path)
    scheduler.tree["tracking"] = []
    scheduler.body_store.write_contract(
        "tracking", "track_live", {"token": "The tracking token."}
    )

    seen = {}

    def act(ctx, request):
        seen.update(ctx.config)
        return ActionResult(action_log="done", new_state="done")

    skill = Skill(
        name="track_live",
        category="tracking",
        description="Follow a package.",
        act=act,
        contract={"token": "The tracking token."},
    )

    result = scheduler._run_skill(skill, Request("track my package"))
    assert result.kind == "needs_input"
    assert scheduler.pending is not None and scheduler.pending.pre_act is True

    reply = scheduler.answer("")
    assert reply.status == "ran", reply.text
    assert scheduler.pending is None
    assert "token" not in seen, "a skipped variable must stay unset"


def test_pre_act_skip_advances_to_next_missing_variable(tmp_path):
    """Skipping one contract variable leaves just that one empty and moves on to
    the next missing variable, then runs."""
    scheduler = _scheduler(tmp_path)
    scheduler.tree["tracking"] = []
    scheduler.body_store.write_contract(
        "tracking", "probe", {"alpha": "first", "beta": "second"}
    )

    seen = {}

    def act(ctx, request):
        seen.update(ctx.config)
        return ActionResult(action_log="done", new_state="done")

    skill = Skill(
        name="probe",
        category="tracking",
        description="Probe.",
        act=act,
        contract={"alpha": "first", "beta": "second"},
    )

    result = scheduler._run_skill(skill, Request("probe"))
    assert result.kind == "needs_input"
    assert "`alpha`" in result.summary

    reply = scheduler.answer("")
    assert reply.status == "needs_input"
    assert scheduler.pending is not None
    assert "`beta`" in scheduler.pending.question

    reply = scheduler.answer("")
    assert reply.status == "ran", reply.text
    assert "alpha" not in seen and "beta" not in seen


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

def test_deferred_elicitation_asks_and_records(tmp_path):
    """Opt-in elicitation runs on the codegen worker (deferred): questions are
    posted to the question queue, the worker waits, and the answers ride the
    draft into the body prompt."""
    httpd, base = _pipeline_codegen_server(
        ['{"questions": ["Draft or send?"]}']
    )
    try:
        scheduler = _scheduler(tmp_path, codegen=CodegenClient(base_url=base, model="test", timeout=10))
        scheduler.defer_questions = True
        scheduler.elicitation_enabled = True
        request = Request("track my package")
        draft = SkillDraft(name="track_live", description="Follow a package.")
        job = SkillWrite(request=request, category="tracking", draft=draft)

        def answerer() -> None:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                pending = scheduler.pending_questions()
                if pending:
                    scheduler.answer_question(pending[0]["id"], "draft first")
                    return
                time.sleep(0.02)

        thread = threading.Thread(target=answerer)
        thread.start()
        scheduler._deferred_elicit(job, scheduler.tree)
        thread.join()
        assert draft.requirements == {"Draft or send?": "draft first"}
        assert any(e["kind"] == "requirements" for e in scheduler.trace.read())
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_async_skill_write_regen_ladder_on_failing_test(tmp_path):
    """A failing auto-run test triggers the regen decision (regen_test by
    default when no factory), regenerating the test until it passes."""
    failing_test = "import sys\nsys.exit(1)"
    httpd, base = _pipeline_codegen_server(
        [GOOD_BODY, failing_test, GOOD_TEST]
    )
    try:
        scheduler = _scheduler(tmp_path, codegen=CodegenClient(base_url=base, model="test", timeout=10))
        scheduler.tree["tracking"] = [
            Skill(name="track_live", category="tracking", description="Follow a package.")
        ]
        scheduler.regen_decision_factory = lambda run_id: (lambda reason: "regen_test")

        request = Request("track my drone delivery in real time")
        scheduler._start_skill_write(
            request, "tracking", SkillDraft(name="track_live", description="Follow a package.")
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


def test_post_questions_asks_one_at_a_time(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.elicitation_answer_timeout = 5.0
    job = SkillWrite(
        request=Request("track it"),
        category="tracking",
        draft=SkillDraft(name="probe", description="Probe."),
    )
    result: dict = {}
    thread = threading.Thread(
        target=lambda: result.update(scheduler._post_questions(job, ["Q1?", "Q2?"]))
    )
    thread.start()

    def next_pending(deadline: float) -> list:
        while time.monotonic() < deadline:
            pending = scheduler.pending_questions()
            if pending:
                return pending
            time.sleep(0.02)
        return []

    deadline = time.monotonic() + 5
    first = next_pending(deadline)
    assert [q["question"] for q in first] == ["Q1?"]
    scheduler.answer_question(first[0]["id"], "A1")
    second = next_pending(deadline)
    assert [q["question"] for q in second] == ["Q2?"]
    scheduler.answer_question(second[0]["id"], "")
    thread.join(timeout=5)
    assert result == {"Q1?": "A1"}
    assert scheduler.pending_questions() == []
    kinds = [e["kind"] for e in scheduler.trace.read()]
    assert "questions_asked" in kinds
    # An empty answer is accepted (the question is dismissed), not a timeout.
    assert "questions_timeout" not in kinds


def test_post_questions_timeout_drops_remaining(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.elicitation_answer_timeout = 0.2
    job = SkillWrite(
        request=Request("track it"),
        category="tracking",
        draft=SkillDraft(name="probe", description="Probe."),
    )
    result: dict = {}
    thread = threading.Thread(
        target=lambda: result.update(scheduler._post_questions(job, ["Q1?", "Q2?"]))
    )
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        pending = scheduler.pending_questions()
        if pending:
            scheduler.answer_question(pending[0]["id"], "A1")
            break
        time.sleep(0.02)
    thread.join(timeout=3)
    assert result == {"Q1?": "A1"}
    assert scheduler.pending_questions() == []
    events = [e for e in scheduler.trace.read() if e["kind"] == "questions_timeout"]
    assert events and events[-1]["answered"] == 1 and events[-1]["asked"] == 2


def test_post_questions_times_out(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.elicitation_answer_timeout = 0.1
    job = SkillWrite(
        request=Request("track it"),
        category="tracking",
        draft=SkillDraft(name="probe", description="Probe."),
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
    skill.act = lambda ctx, request: ActionResult(
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
