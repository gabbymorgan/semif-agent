"""End-to-end pipeline test. Run ONLY on the box with real SemIf + real LLM.

    python -m pytest tests/integration -q

The decision engine is real (llamacpp GGUF) and self-assessment is a real
local LLM endpoint. If either is unavailable this fails loudly — no mocking.
"""

import json
import time

import pytest

from semif_agent.cli import build_scheduler, load_config
from semif_agent.codegen import (
    CodegenClient,
    generate_data_contract,
    generate_skill_body,
    generate_skill_tests,
    parse_integration,
)
from semif_agent.decisions import Request
from semif_agent.dream import dream
from semif_agent.engine import EngineUnavailable
from semif_agent.skills import (
    ActionResult,
    CategoryDraft,
    Prediction,
    Skill,
    SkillDraft,
    SkillStore,
    build_skills,
    build_tree,
    generate_category,
    generate_skill,
    materialize_skill,
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


def _install_tracking_fixture(scheduler):
    """A deterministic runnable leaf for end-to-end runs.

    The pipeline tests need a target whose flow is stable; the hardcoded
    email/tracking examples were removed because they faked their integrations.
    This injects a runnable tracking leaf instead — the decision engine, the
    self-assessment LLM, and the decision log all remain real.
    """

    def predict(ctx, request):
        return Prediction(text="", decisions=[])

    def act(ctx, request, prediction):
        return ActionResult(
            action_log="tracking fixture: checked the package status",
            new_state="package status checked",
        )

    scheduler.tree["tracking"] = [
        Skill(
            name="tracking.check",
            category="tracking",
            description="Check the delivery status of a package.",
            predict=predict,
            act=act,
        )
    ]


def test_pipeline_end_to_end(tmp_path):
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    config["trace"] = str(tmp_path / "runs.jsonl")
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


def test_gate_accepts_information_requests_and_rejects_noise(tmp_path):
    """Regression: the gate must not drop lookups phrased as questions.

    Before the gate state carried the end-user expectation and the available
    skills, "tell me the next event in my nextcloud calendar" scored 0.599
    (just under tau 0.6) and "what is the next event in my nextcloud calendar?"
    scored 0.191, so both were dropped as "no actionable request". All lookup
    variants must pass now, while chatter stays out.
    """
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    config["trace"] = str(tmp_path / "runs.jsonl")
    scheduler, config = build_scheduler(config)

    lookups = [
        "tell me the next event in my nextcloud calendar",
        "what is the next event in my nextcloud calendar?",
        "what is the next event in my nextcloud calendar Morgan",
        "tell me the next event in my nextcloud calendar Morgan",
        "tell me the next even in my calendar",
        "tell me the next event in my personal calendar",
    ]
    noise = ["hello there", "thanks!", "the sky is blue", "what's up"]
    for text in lookups:
        accepted = scheduler._contains_request(Request(text))
        row = scheduler.log.read()[-1]
        print(f"[lookup] {row['predicted_probs']} accepted={accepted} {text!r}")
        assert accepted, f"gate dropped a lookup: {text!r}"
    for text in noise:
        accepted = scheduler._contains_request(Request(text))
        row = scheduler.log.read()[-1]
        print(f"[noise] {row['predicted_probs']} accepted={accepted} {text!r}")
        assert not accepted, f"gate accepted noise: {text!r}"


def test_busy_choice_path(tmp_path):
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    config["trace"] = str(tmp_path / "runs.jsonl")
    scheduler, config = build_scheduler(config)
    _install_tracking_fixture(scheduler)

    scheduler.busy("driving on the freeway", skill="driving")
    status, detail = scheduler.submit("tell me if my package was delivered")
    print(f"[{status}] {detail}")
    assert status in ("preempted", "queued", "dropped")
    scheduler.idle()


def test_engine_generation_normal_mode(tmp_path):
    """The pinned decision model must also generate text in the normal way."""
    config = load_config()
    require_real(config)
    scheduler, config = build_scheduler(config)
    out = scheduler.engine.generate(
        [
            {"role": "system", "content": "Reply with the single word ok."},
            {"role": "user", "content": "say ok"},
        ],
        max_tokens=16,
    )
    print(f"generation: {out!r}")
    assert isinstance(out, str) and out.strip()


def test_generate_category(tmp_path):
    """Authoring a category stub through the real decision model."""
    config = load_config()
    require_real(config)
    scheduler, config = build_scheduler(config)
    draft = generate_category(
        scheduler.engine,
        Request("tell me if my package was delivered"),
        scheduler.tree,
    )
    print(f"draft: {draft.name!r} — {draft.description!r}")
    assert isinstance(draft, CategoryDraft)
    assert draft.name and draft.description


def test_generate_skill(tmp_path):
    """Authoring a skill leaf stub through the real decision model."""
    config = load_config()
    require_real(config)
    scheduler, config = build_scheduler(config)
    draft = generate_skill(
        scheduler.engine,
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
    contract = generate_data_contract(
        client, request, "tracking", draft, code
    )
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
    assert callable(skill.predict) and callable(skill.act)
    assert skill.contract == contract
    assert skill.integration["transport"] in ("http", "caldav")


def test_review_skill_body_real_llm():
    """The real self-assessment model must flag a simulated body and accept a
    body that really performs the action."""
    from semif_agent.llm import LLMClient

    config = load_config()
    llm_cfg = config.get("llm", {})
    client = LLMClient(
        base_url=llm_cfg.get("base_url", "http://localhost:11434/v1"),
        model=llm_cfg.get("model", "qwen3.5:4b"),
        timeout=120.0,
    )
    toy = '''\
INTEGRATION = {"service": "carrier", "transport": "http", "config_vars": ["tracking_id"]}

def predict(ctx, request):
    return Prediction(text="ok", decisions=[])

def act(ctx, request, prediction):
    return ActionResult(
        action_log=f"Package {ctx.config['tracking_id']} is delivered.",
        new_state="delivered",
    )
'''
    review = client.review_skill_body(
        "tell me if my package was delivered",
        "Check a package's delivery status.",
        toy,
        integration={"service": "carrier", "transport": "http", "config_vars": ["tracking_id"]},
    )
    print(f"toy verdict: {review.performs_real_action} — {review.reason}")
    assert review.performs_real_action is False, (
        "a canned status with no real lookup must be rejected"
    )

    real = '''\
import json
import urllib.request

INTEGRATION = {"service": "carrier", "transport": "http", "config_vars": ["tracking_url"]}

def predict(ctx, request):
    return Prediction(text="ok", decisions=[])

def act(ctx, request, prediction):
    with urllib.request.urlopen(ctx.config["tracking_url"], timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return ActionResult(
        action_log=f"carrier returned status {payload['status']}",
        new_state=payload["status"],
    )
'''
    review = client.review_skill_body(
        "tell me if my package was delivered",
        "Check a package's delivery status.",
        real,
        integration={"service": "carrier", "transport": "http", "config_vars": ["tracking_url"]},
    )
    print(f"real verdict: {review.performs_real_action} — {review.reason}")
    assert review.performs_real_action is True, (
        "a body that really calls the carrier API must be accepted"
    )


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


def test_create_category_chain_runs_new_skill(tmp_path):
    """A request that needs a brand-new category must end with a skill run.

    Deterministic chain: create_category -> create_skill in the new category ->
    async codegen body write -> re-dispatch of the original request -> run that
    skill (the leaf answers the request, not the category stub). Slow: uses real
    codegen (~25-45 min) and the worker runs in the background, so the test
    polls the trace for the completion + assessed events. Run in the background.
    """
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    config["trace"] = str(tmp_path / "runs.jsonl")
    config["category_registry"] = str(tmp_path / "categories.json")
    config["skill_bodies"] = str(tmp_path / "skills")
    config["codegen"] = {**config.get("codegen", {}), "stream": True}
    scheduler, config = build_scheduler(config)
    scheduler.tree = {}

    status, detail = scheduler.submit("track my drone delivery in real time")
    print(f"[{status}] {detail}")
    assert status in ("running", "preempted", "queued", "rejected")
    assert scheduler.current is None, "gate must be free again right after the stub is created"

    kinds = [e["kind"] for e in scheduler.trace.read()]
    assert "category_created" in kinds, "category stub must be authored first"
    assert "skill_writing" in kinds, "the async body write must be launched"

    deadline = time.monotonic() + 55 * 60
    created = assessed = None
    while time.monotonic() < deadline:
        rows = scheduler.trace.read()
        created = next((e for e in rows if e["kind"] == "skill_created"), None)
        if created and created.get("written"):
            assessed = next(
                (e for e in rows if e["kind"] == "assessed" and e.get("skill") == created["skill"]),
                None,
            )
            if assessed:
                break
        time.sleep(10)
    assert created is not None and created.get("written"), (
        "codegen must produce a runnable body"
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

    def predict(ctx, request):
        return Prediction(text="", decisions=[])

    def act(ctx, request, prediction):
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
        predict=predict,
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