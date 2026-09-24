"""End-to-end pipeline test. Run ONLY on the box with real SemIf + real LLM.

    python -m pytest tests/integration -q

The decision engine is real (llamacpp GGUF) and self-assessment is a real
local LLM endpoint. If either is unavailable this fails loudly — no mocking.
"""

import json
from pathlib import Path

import pytest

from semif_agent.cli import build_scheduler, load_config
from semif_agent.codegen import CodegenClient, generate_skill_body
from semif_agent.decisions import Request
from semif_agent.dream import dream
from semif_agent.engine import EngineUnavailable
from semif_agent.skills import (
    CategoryDraft,
    SkillBodyStore,
    SkillDraft,
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


def test_pipeline_end_to_end(tmp_path):
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    config["trace"] = str(tmp_path / "runs.jsonl")
    scheduler, config = build_scheduler(config)

    inputs = [
        "send my girlfriend an email that says I'm going to be late to the party",
        "tell me if my package was delivered",
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

    contacts_path = Path(config.get("skills", {}).get("contacts", "data/contacts.json"))
    assert contacts_path.is_file()


def test_busy_choice_path(tmp_path):
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    config["trace"] = str(tmp_path / "runs.jsonl")
    scheduler, config = build_scheduler(config)

    scheduler.busy("driving on the freeway", skill="driving")
    status, detail = scheduler.submit("send my girlfriend an email that says I'm going to be late")
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
    """A real OpenAI-compatible model writes a runnable skill body.

    Slow: uses the big codegen model (qwen38-iq3s by default). Run this one in
    the background and poll — long-lived ssh sessions get SIGHUP'd.
    """
    config = load_config()
    require_real(config)
    codegen_cfg = config.get("codegen", {})
    client = CodegenClient(
        base_url=codegen_cfg.get("base_url", "http://localhost:11434/v1"),
        model=codegen_cfg.get("model", "qwen38-iq3s"),
        timeout=float(codegen_cfg.get("timeout", 1200.0)),
    )
    tree = build_tree(build_skills({"skills": {}}))
    draft = SkillDraft(
        name="check_service",
        description="Check whether a service is reachable.",
    )
    code = generate_skill_body(
        client,
        Request("is my home server reachable right now?"),
        "tracking",
        draft,
        tree,
    )
    print(f"generated {len(code)} bytes of skill body")
    draft.code = code
    store = SkillBodyStore(str(tmp_path / "skills"))
    skill = materialize_skill(draft, "tracking", store)
    assert callable(skill.predict) and callable(skill.act)


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
    run that skill (the leaf answers the request, not the category stub). Slow:
    uses real codegen (~7 min). Run in the background.
    """
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    config["trace"] = str(tmp_path / "runs.jsonl")
    config["category_registry"] = str(tmp_path / "categories.json")
    config["skill_bodies"] = str(tmp_path / "skills")
    scheduler, config = build_scheduler(config)
    scheduler.tree = {}

    status, detail = scheduler.submit("track my drone delivery in real time")
    print(f"[{status}] {detail}")

    rows = scheduler.trace.read()
    kinds = [e["kind"] for e in rows]
    assert "category_created" in kinds, "category stub must be authored first"
    created = next(e for e in rows if e["kind"] == "skill_created")
    assert created["written"] is True, "codegen must produce a runnable body"
    assessed = next(e for e in rows if e["kind"] == "assessed")
    assert assessed["skill"] == created["skill"], "the created skill must run"