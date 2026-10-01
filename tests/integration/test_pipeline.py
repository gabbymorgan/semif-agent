"""End-to-end pipeline test. Run ONLY on the box with real SemIf + real LLM.

    python -m pytest tests/integration -q

The decision engine is real (llamacpp GGUF), and the `llm` and `codegen`
providers are real OpenAI-compatible endpoints. If any is unavailable this
fails loudly — no mocking.
"""

import json
import time

import pytest

from semif_agent.cli import build_scheduler, load_config
from semif_agent.codegen import (
    CodegenClient,
    generate_skill_body,
    generate_skill_tests,
    parse_contract,
    parse_integration,
)
from semif_agent.decisions import Request
from semif_agent.dream import dream
from semif_agent.engine import EngineUnavailable
from semif_agent.scheduler import SkillWrite
from semif_agent.skills import (
    ActionResult,
    CategoryDraft,
    CreateSkill,
    Skill,
    SkillDraft,
    SkillStore,
    build_skills,
    build_tree,
    generate_category,
    generate_skill,
    materialize_skill,
    navigate,
)


def require_real(config: dict):
    from semif_agent.engine import SemIfEngine, EngineConfig

    engine = SemIfEngine(
        EngineConfig(
            backend=config.get("engine", {}).get("backend", "llamacpp"),
            source=config.get("engine", {}).get("source", ""),
            revision=config.get("engine", {}).get("revision", ""),
            gguf=config.get("engine", {}).get("gguf", ""),
            context_tokens=int(config.get("engine", {}).get("context_tokens", 4096)),
            threads=config.get("engine", {}).get("threads"),
        )
    )
    try:
        engine._ensure_loaded()
    except EngineUnavailable as exc:
        pytest.fail(f"real engine unavailable: {exc}")


def _isolate_runtime(config: dict, tmp_path) -> None:
    """Point every runtime artifact at tmp_path.

    Without this, persisted authoring output in the box's `data/categories.json`
    and `data/skills/` leaks into the tree and navigation routes to a stale stub
    instead of the injected fixture — an environment-dependent failure.
    """
    config["log"] = str(tmp_path / "decisions.jsonl")
    config["trace"] = str(tmp_path / "runs.jsonl")
    config["category_registry"] = str(tmp_path / "categories.json")
    config["skill_bodies"] = str(tmp_path / "skills")


def _install_tracking_fixture(scheduler):
    """A deterministic runnable leaf for end-to-end runs.

    The pipeline tests need a target whose flow is stable; the hardcoded
    email/tracking examples were removed because they faked their integrations.
    This injects a runnable tracking leaf instead — the decision engine, the
    `llm`/`codegen` providers, and the decision log all remain real.
    """

    def act(ctx, request):
        return ActionResult(
            action_log="tracking fixture: checked the package status",
            new_state="package status checked",
        )

    scheduler.tree["tracking"] = [
        Skill(
            name="tracking.check",
            category="tracking",
            description="Check the delivery status of a package.",
            act=act,
        )
    ]


def test_pipeline_end_to_end(tmp_path):
    config = load_config()
    require_real(config)
    _isolate_runtime(config, tmp_path)
    scheduler, config = build_scheduler(config)
    _install_tracking_fixture(scheduler)

    inputs = [
        "tell me if my package was delivered",
        "what is the delivery status of my parcel",
    ]
    for text in inputs:
        status, detail = scheduler.submit(text)
        print(f"[{status}] {detail}")
        assert status in ("running", "preempted", "queued", "rejected")

    scheduler.run_queue()

    rows = scheduler.log.read()
    assert len(rows) > 0, "expected SemIf decisions to be logged"

    phases = [r.get("extra", {}).get("phase") for r in rows]
    assert "navigate:category" in phases, "navigation decisions must be logged"
    assert "navigate:leaf" in phases, "navigation decisions must be logged"
    for row in rows:
        assert row.get("extra", {}).get("run_id"), "every decision must carry a run_id"

    trace_rows = scheduler.trace.read()
    assert any(r["kind"] == "submit" for r in trace_rows)
    assert any(r["kind"] == "assessed" for r in trace_rows)

    report = dream(scheduler.log)
    assert report.cross_entropy is not None
    print(report.render())

    skills = scheduler.status()
    assert "queue:" in skills


def test_navigation_routes_noise_into_the_response_tree(tmp_path):
    """There is no up-front handle/ignore gate any more: every input is
    dispatched, and chatter that is not a task must land in the closed `response`
    tree (a canned reply), never in an integration category or codegen."""
    config = load_config()
    require_real(config)
    _isolate_runtime(config, tmp_path)
    scheduler, config = build_scheduler(config)

    lookups = [
        "tell me the next event in my nextcloud calendar",
        "what is the next event in my nextcloud calendar?",
        "tell me the next even in my calendar",
    ]
    noise = ["hello there", "thanks!", "the sky is blue", "what's up"]
    for text in lookups:
        result = navigate(
            scheduler.engine, scheduler.log, scheduler.trace, Request(text), scheduler.tree
        )
        print(f"[lookup] {text!r} -> {result!r}")
        assert getattr(result, "category", None) == "calendar", f"lookup misrouted: {text!r}"
    for text in noise:
        result = navigate(
            scheduler.engine, scheduler.log, scheduler.trace, Request(text), scheduler.tree
        )
        print(f"[noise] {text!r} -> {result!r}")
        assert getattr(result, "category", None) == "response", f"noise left the response tree: {text!r}"


def test_navigate_routes_unmatched_action_to_create_skill(tmp_path):
    """A send request must navigate to create_skill, not silently to the read
    skill that merely shares the word 'message'.

    Regression: every send phrasing picked simplex.next_message (P 0.46-0.73)
    while the create_skill fallback scored only 0.007-0.09, because its option
    description named the mechanism ("suggest creating a new skill") instead of
    the trigger. This asserts the navigation outcome only — no codegen write.
    """
    config = load_config()
    require_real(config)
    _isolate_runtime(config, tmp_path)
    scheduler, config = build_scheduler(config)
    scheduler.tree["simplex"] = [
        Skill(
            name="next_message",
            category="simplex",
            description="Read the next unread SimpleX message bridge.",
        ),
        Skill(
            name="connect_link",
            category="simplex",
            description="Show (creating if needed) the SimpleX contact link others use to connect to this agent.",
        ),
    ]

    for text in [
        "send a simplex message to pepper saying hi",
        "send a new simplex message to pepper: hey",
    ]:
        before = len(scheduler.log.read())
        result = navigate(
            scheduler.engine,
            scheduler.log,
            scheduler.trace,
            Request(text),
            scheduler.tree,
        )
        leaf_rows = [
            r for r in scheduler.log.read()[before:]
            if r.get("extra", {}).get("phase") == "navigate:leaf"
        ]
        print(f"[nav] {leaf_rows[-1]['predicted_probs'] if leaf_rows else '-'} :: {text!r} -> {result}")
        assert isinstance(result, CreateSkill), f"send must route to create_skill, got {result!r}"
        assert result.category == "simplex"

    read = navigate(
        scheduler.engine,
        scheduler.log,
        scheduler.trace,
        Request("read the next simplex message"),
        scheduler.tree,
    )
    print(f"[nav control] read -> {read}")
    assert getattr(read, "name", None) == "next_message", "a read request must still route to the read skill"


def test_busy_choice_path(tmp_path):
    config = load_config()
    require_real(config)
    _isolate_runtime(config, tmp_path)
    scheduler, config = build_scheduler(config)
    _install_tracking_fixture(scheduler)

    scheduler.busy("driving on the freeway", skill="driving")
    status, detail = scheduler.submit("tell me if my package was delivered")
    print(f"[{status}] {detail}")
    assert status in ("preempted", "queued")
    scheduler.idle()


def test_engine_scoring_only(tmp_path):
    """The pinned decision model scores (makes decisions); it no longer writes
    free text. Text authoring is the `llm` provider's job."""
    config = load_config()
    require_real(config)
    scheduler, config = build_scheduler(config)
    assert not hasattr(scheduler.engine, "generate")


def test_generate_category(tmp_path):
    """Authoring a category stub through the real `llm` provider."""
    config = load_config()
    require_real(config)
    scheduler, config = build_scheduler(config)
    draft = generate_category(
        scheduler.llm,
        Request("tell me if my package was delivered"),
        scheduler.tree,
    )
    print(f"draft: {draft.name!r} — {draft.description!r}")
    assert isinstance(draft, CategoryDraft)
    assert draft.name and draft.description


def test_generate_skill(tmp_path):
    """Authoring a skill leaf stub through the real `llm` provider."""
    config = load_config()
    require_real(config)
    scheduler, config = build_scheduler(config)
    draft = generate_skill(
        scheduler.llm,
        Request("tell me if my package was delivered"),
        "tracking",
        scheduler.tree,
    )
    print(f"draft: {draft.name!r} — {draft.description!r}")
    assert isinstance(draft, SkillDraft)
    assert draft.name and draft.description


def test_generate_skill_body_codegen(tmp_path):
    """A real OpenAI-compatible model writes a real-integration skill body +
    data contract + hermetic test, and the auto-run test passes.

    The body must perform the action via a stdlib transport and declare
    INTEGRATION; the test is a hermetic mechanics check (loopback for HTTP), not
    proof of the live integration. Slow: uses the big codegen model
    (qwen38-iq3s by default). Run this one in the background and poll —
    long-lived ssh sessions get SIGHUP'd.
    """
    config = load_config()
    require_real(config)
    codegen_cfg = config.get("codegen", {})
    client = CodegenClient(
        base_url=codegen_cfg.get("base_url", "http://localhost:11434/v1"),
        model=codegen_cfg.get("model", "qwen38-iq3s"),
        timeout=float(codegen_cfg.get("timeout", 1200.0)),
        stream=True,
    )
    tree = build_tree(build_skills({"skills": {}}))
    draft = SkillDraft(
        name="check_service",
        description="Check whether an HTTP service is reachable.",
    )
    requirements = {
        "How should it connect?": (
            "HTTP GET the service_url from config (no auth) and report the "
            "status code; treat a non-2xx or unreachable host as a failure."
        ),
        "Should it use a test fixture address?": "No; the endpoint comes from config.",
    }
    request = Request("check whether my home server is reachable")
    code = generate_skill_body(
        client,
        request,
        "tracking",
        draft,
        tree,
        requirements=requirements,
    )
    print(f"generated {len(code)} bytes of skill body")
    integration = parse_integration(code)
    print(f"integration: {integration}")
    assert integration["transport"] in ("http", "caldav")
    assert "urllib.request" in code or "http.client" in code, (
        "a service-backed body must call the service with a stdlib transport"
    )
    assert "127.0.0.1" not in code and "localhost" not in code, (
        "the body must take its endpoint from ctx.config, never a fixture"
    )
    contract = parse_contract(code)
    print(f"data contract: {contract}")
    test = generate_skill_tests(
        client,
        request,
        "tracking",
        draft,
        code,
        contract,
    )
    print(f"generated {len(test)} bytes of skill test")

    store = SkillStore(str(tmp_path / "skills"))
    store.write_body("tracking", draft.name, code)
    store.write_contract("tracking", draft.name, contract)
    store.write_test("tracking", draft.name, test)
    from semif_agent.codegen import run_skill_test

    passed, output = run_skill_test(store.dir("tracking", draft.name), timeout=60)
    print(f"auto-run test: passed={passed}\n{output[:400]}")
    assert passed, "the auto-run test must pass"

    draft.code = code
    skill = materialize_skill(draft, "tracking", store)
    assert callable(skill.act)
    assert skill.contract == contract
    assert skill.integration["transport"] in ("http", "caldav")


def test_fidelity_gate_is_a_real_semif_decision(tmp_path):
    """The fidelity gate is a real logged SemIf decision (`authoring:fidelity`).

    A body with no valid INTEGRATION declaration produces static findings, which
    force `reconsider` regardless of the model's probability; a clean,
    declared body is accepted. Both verdicts are recorded as decision rows.
    """
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    config["trace"] = str(tmp_path / "runs.jsonl")
    scheduler, config = build_scheduler(config)
    # The gate needs codegen configured to run; point it at a dead endpoint so
    # a reconsider can't complete a rewrite (the gate still logs its decision).
    scheduler.codegen = CodegenClient(
        base_url="http://127.0.0.1:1/v1", model="test", timeout=2
    )

    job = SkillWrite(
        request=Request("tell me if my package was delivered"),
        category="tracking",
        draft=SkillDraft(
            name="track_delivery", description="Check a package's delivery status."
        ),
        weight=0.5,
    )
    undeclared = '''\
from semif_agent.skills import ActionResult

CONTRACT = {}

def act(ctx, request):
    return ActionResult(action_log="delivered", new_state="delivered")
'''
    gated = scheduler._fidelity_gate(job, undeclared)
    rows = [r for r in scheduler.log.read() if r.get("extra", {}).get("phase") == "authoring:fidelity"]
    assert rows, "the fidelity gate must be a logged decision row"
    print(f"undeclared reconsider: {rows[-1]['extra']['reconsider']} {rows[-1]['predicted_probs']}")
    assert rows[-1]["extra"]["reconsider"] is True, (
        "an undeclared body must be reconsidered"
    )
    assert gated == undeclared, "no codegen means the gate returns the body unchanged"

    declared = '''
INTEGRATION = {"service": "carrier", "transport": "http", "config_vars": ["tracking_url"]}
CONTRACT = {"tracking_url": "Carrier status endpoint."}
import json
import urllib.request

from semif_agent.skills import ActionResult

def act(ctx, request):
    with urllib.request.urlopen(ctx.config["tracking_url"], timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return ActionResult(
        action_log=f"carrier returned status {payload['status']}",
        new_state=payload["status"],
    )
'''
    scheduler._fidelity_gate(job, declared)
    rows = [r for r in scheduler.log.read() if r.get("extra", {}).get("phase") == "authoring:fidelity"]
    print(f"declared reconsider: {rows[-1]['extra']['reconsider']} {rows[-1]['predicted_probs']}")
    assert rows[-1]["extra"]["findings"] == [], "a consistent declaration has no findings"


def test_create_skill_empty_category_does_not_wedge(tmp_path):
    """A dispatch that lands on an empty category must not leave the scheduler wedged.

    Regression: navigation on a category with no skills produced a single-option
    SemIf decision, which the backend rejects; the exception unwound past the
    current-process reset, so every later request queued forever behind a phantom
    current. The empty category must now short-circuit straight to create_skill.
    """
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    config["trace"] = str(tmp_path / "runs.jsonl")
    scheduler, config = build_scheduler(config)
    scheduler.codegen = None  # wedge regression only; skip the ~7min codegen body write
    scheduler.tree["travel_planning"] = []

    for _ in range(2):
        status, detail = scheduler.submit("look up flights to japan for february")
        print(f"[{status}] {detail}")
        assert status in ("running", "preempted", "queued", "rejected", "error")
        assert scheduler.current is None, "scheduler must never stay wedged after a submit"

    status, detail = scheduler.submit("tell me if my package was delivered")
    print(f"[{status}] {detail}")
    assert scheduler.current is None


def test_create_simplex_send_skill(tmp_path):
    """A send request with no matching leaf must author a new simplex skill.

    The `simplex` category already ships the read (`next_message`) and
    `connect_link` seeds; "send a message" is a genuinely unmatched action, so
    navigation routes it to create_skill. Deterministic chain: create_skill ->
    async codegen body write -> re-dispatch of the original request -> run the
    new leaf. Slow: uses real codegen (~25-45 min) and the worker runs in the
    background, so the test polls the trace. Run in the background.
    """
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    config["trace"] = str(tmp_path / "runs.jsonl")
    config["category_registry"] = str(tmp_path / "categories.json")
    config["skill_bodies"] = str(tmp_path / "skills")
    config["codegen"] = {**config.get("codegen", {}), "stream": True}
    scheduler, config = build_scheduler(config)

    seeded = {s.name for s in scheduler.tree.get("simplex", [])}
    assert {"next_message", "connect_link"} <= seeded, (
        f"the simplex seeds must be loaded, got {seeded}"
    )

    status, detail = scheduler.submit("send a simplex message to pepper saying hi")
    print(f"[{status}] {detail}")
    assert status in ("running", "preempted", "queued", "rejected")
    assert "new skill for simplex" in detail, (
        "a send request must route to create_skill for simplex"
    )
    assert scheduler.current is None, "gate must be free again right after the draft is queued"

    draft_deadline = time.monotonic() + 10 * 60
    while time.monotonic() < draft_deadline:
        rows = scheduler.trace.read()
        if any(e["kind"] == "skill_writing" for e in rows):
            break
        time.sleep(5)
    rows = scheduler.trace.read()
    assert not any(e["kind"] == "draft_failed" for e in rows), (
        "the skill draft must be authored"
    )
    writing = [e for e in rows if e["kind"] == "skill_writing"]
    assert writing and writing[-1]["category"] == "simplex", (
        "the async body write must be launched for simplex"
    )

    deadline = time.monotonic() + 55 * 60
    created = assessed = None
    while time.monotonic() < deadline:
        rows = scheduler.trace.read()
        created = next(
            (
                e
                for e in rows
                if e["kind"] == "skill_created"
                and e.get("category") == "simplex"
                and e.get("written")
                and e.get("skill") not in seeded
            ),
            None,
        )
        if created:
            assessed = next(
                (e for e in rows if e["kind"] == "assessed" and e.get("skill") == created["skill"]),
                None,
            )
            if assessed:
                break
        time.sleep(10)
    assert created is not None, (
        "codegen must produce a runnable simplex send body"
    )
    assert assessed is not None, "the created skill must run after the body lands"
    assert assessed["skill"] == created["skill"], "the created skill must run"


def test_skill_pauses_for_input_and_resumes(tmp_path):
    """A run paused for input keeps `current` busy, then `answer` resumes it.

    Uses the real scheduler (real engine + real LLM assessment on the resumed
    run). The skill itself is injected, not authored, so the flow is
    deterministic: pause -> answer -> resume -> assessed.
    """
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    config["trace"] = str(tmp_path / "runs.jsonl")
    scheduler, config = build_scheduler(config)

    seen = []

    def act(ctx, request):
        if request.user_input:
            seen.append(request.user_input)
            return ActionResult(
                action_log=f"resumed with {request.user_input}",
                new_state=f"done {request.user_input}",
            )
        return ActionResult(
            action_log="need a tracking number",
            new_state=request.text,
            needs_input="What's the tracking number?",
        )

    skill = Skill(
        name="track.manual",
        category="tracking",
        description="Resolve a tracking number with the human.",
        act=act,
    )

    result = scheduler._run_skill(skill, Request("track my package manually"))
    print(f"[{result.kind}] {result.summary}")
    assert result.kind == "needs_input"
    assert scheduler.pending is not None
    assert scheduler.current is not None

    status, detail = scheduler.answer("AB123")
    print(f"[{status}] {detail}")
    assert status == "ran"
    assert seen == ["AB123"]
    assert scheduler.pending is None
    assert scheduler.current is None

    rows = scheduler.trace.read()
    kinds = [e["kind"] for e in rows]
    assert "needs_input" in kinds and "answered" in kinds and "assessed" in kinds