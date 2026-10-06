"""Pure-stdlib tests for the housekeeping meta skills and per-category locks.

These cover the deterministic mechanics: name resolution, registry/store
deletion, tombstones, the lock gate, and the scheduler operations the meta
skills call. The SemIf routing of a natural-language housekeeping request is a
real-engine concern, verified by the integration tests on a provisioned host.
"""

import json
from pathlib import Path

import pytest

from semif_agent.decisions import Request
from semif_agent.llm import LLMClient
from semif_agent.log import DecisionLog
from semif_agent.scheduler import Scheduler, SkillWrite
from semif_agent.skills import (
    ActionContext,
    CategoryRegistry,
    DeletedSkills,
    Skill,
    SkillDraft,
    SkillStore,
    apply_tombstones,
    build_housekeeping_skills,
    parse_meta_command,
    resolve_skill_ref,
    _cancel_build_act,
    _delete_skill_act,
)
from semif_agent.trace import TraceLog

from tests.conftest import ScriptedEngine

SEED_BODY = """\
from semif_agent.skills import ActionResult

INTEGRATION = {"service": "probe", "transport": "compute", "config_vars": []}

CONTRACT = {}


def act(ctx, request):
    return ActionResult(action_log="probe ran", new_state="ok")
"""


def _scheduler(tmp_path, choices=None, default="success"):
    return Scheduler(
        engine=ScriptedEngine(choices=choices, default=default),
        llm=LLMClient(base_url="http://localhost:1/v1", model="test"),
        log=DecisionLog(str(tmp_path / "decisions.jsonl")),
        config={
            "skills": {},
            "category_registry": str(tmp_path / "categories.json"),
            "skill_bodies": str(tmp_path / "skills"),
            "skill_seeds": str(tmp_path / "seeds"),
            "deleted_skills": str(tmp_path / "deleted_skills.json"),
        },
        trace=TraceLog(str(tmp_path / "runs.jsonl")),
    )


def _add_skill(scheduler, category, name, origin="generated"):
    def act(ctx, request):
        return None

    skill = Skill(
        name=name,
        category=category,
        description=f"{name} description",
        act=act,
        origin=origin,
    )
    scheduler.tree.setdefault(category, []).append(skill)
    scheduler.registry.register_skill(category, name, skill.description)
    scheduler.body_store.write_body(
        category, name, "def act(ctx, request):\n    return None\n"
    )
    return skill


def _write_seed(root: Path, category, name):
    directory = root / category / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "skill.py").write_text(SEED_BODY)
    (directory / "contract.json").write_text("{}")


# ---- built-ins ----


def test_build_housekeeping_skills():
    skills = build_housekeeping_skills()
    assert {s.name for s in skills} == {
        "delete_skill",
        "clear_config",
        "regen_skill",
        "cancel_build",
    }
    assert all(s.category == "housekeeping" for s in skills)
    assert all(s.origin == "builtin" for s in skills)
    assert all(not s.is_noop() for s in skills)
    assert all(s.category_description for s in skills)


# ---- deterministic meta-command recognition ----


def test_parse_meta_command_forms():
    assert parse_meta_command("delete skill") == "delete_skill"
    assert parse_meta_command("delete skill calendar.foo") == "delete_skill"
    assert parse_meta_command("remove skill simplex.next_message") == "delete_skill"
    assert parse_meta_command("clear config variables") == "clear_config"
    assert parse_meta_command("clear the config variables for calendar.foo") == "clear_config"
    assert parse_meta_command("clear the settings for simplex.next_message") == "clear_config"
    assert parse_meta_command("reset a skill's settings") == "clear_config"
    assert parse_meta_command("regen skill") == "regen_skill"
    assert parse_meta_command("regenerate the simplex skill") == "regen_skill"
    assert parse_meta_command("rewrite the code for calendar.foo") == "regen_skill"
    assert parse_meta_command("cancel skill build") == "cancel_build"
    assert parse_meta_command("cancel the skill build") == "cancel_build"
    assert parse_meta_command("abort the build") == "cancel_build"
    assert parse_meta_command("stop building the new skill") == "cancel_build"


def test_parse_meta_command_leaves_normal_requests_alone():
    for text in (
        "what is on my calendar",
        "read my next simplex message",
        "hello there",
        "set a timer for 5 minutes",
        "send a message to pepper",
        "what is 255 * 12?",
        "stop building the deck",
        "fix my bike",
    ):
        assert parse_meta_command(text) is None, text


def test_dispatch_routes_meta_command_deterministically(tmp_path):
    scheduler = _scheduler(tmp_path)
    _add_skill(scheduler, "calendar", "foo")

    # clear config runs immediately (no confirmation).
    result = scheduler._dispatch(Request("clear config variables for calendar.foo"))
    assert result.kind == "ran"
    assert result.skill == "clear_config"

    # delete routes to the meta skill and pauses for confirmation.
    delete = scheduler._dispatch(Request("delete skill calendar.foo"))
    assert delete.kind == "needs_input"
    assert delete.skill == "delete_skill"
    assert any(e["kind"] == "meta_command" for e in scheduler.trace.read())
    assert any(s.name == "foo" for s in scheduler.tree["calendar"])


# ---- name resolution ----


def _tree():
    return {
        "calendar": [
            Skill(name="next_event", category="calendar", description="d"),
        ],
        "simplex": [
            Skill(name="next_message", category="simplex", description="d"),
        ],
    }


def test_resolve_skill_ref_dotted_and_bare():
    tree = _tree()
    assert resolve_skill_ref(tree, "delete skill calendar.next_event") == (
        "calendar",
        "next_event",
    )
    assert resolve_skill_ref(tree, "delete skill next_message") == (
        "simplex",
        "next_message",
    )


def test_resolve_skill_ref_none_and_boundary():
    tree = {"time": [Skill(name="time", category="time", description="d")]}
    assert resolve_skill_ref(tree, "delete skill nothing") is None
    assert resolve_skill_ref(tree, "sometimes I wonder") is None


def test_resolve_skill_ref_ambiguous():
    tree = {
        "a": [Skill(name="ping", category="a", description="d")],
        "b": [Skill(name="ping", category="b", description="d")],
    }
    assert resolve_skill_ref(tree, "delete skill ping") == [("a", "ping"), ("b", "ping")]


# ---- registry / store / tombstones ----


def test_registry_unregister_skill(tmp_path):
    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    registry.register_skill("delivery", "track", "Track.")
    registry.unregister_skill("delivery", "track")
    assert registry.read()["delivery"]["skills"] == []


def test_skill_store_delete_and_path_guard(tmp_path):
    store = SkillStore(str(tmp_path / "skills"))
    store.write_body("calendar", "foo", "x = 1\n")
    assert store.delete("calendar", "foo") is True
    assert not store.dir("calendar", "foo").exists()
    with pytest.raises(ValueError):
        store.delete("..", "escape")


def test_deleted_skills_roundtrip(tmp_path):
    deleted = DeletedSkills(str(tmp_path / "deleted.json"))
    assert deleted.read() == set()
    deleted.add("calendar", "foo")
    assert deleted.contains("calendar", "foo")
    deleted.remove("calendar", "foo")
    assert deleted.read() == set()


def test_apply_tombstones():
    tree = {
        "calendar": [Skill(name="foo", category="calendar", description="d")],
        "simplex": [Skill(name="bar", category="simplex", description="d")],
    }
    deleted = DeletedSkills("/nonexistent/deleted.json")
    deleted.read = lambda: {"calendar.foo"}  # type: ignore[assignment]
    removed = apply_tombstones(tree, deleted)
    assert removed == 1
    assert tree["calendar"] == []
    assert [s.name for s in tree["simplex"]] == ["bar"]


def test_delete_seed_is_durable(tmp_path):
    _write_seed(tmp_path / "seeds", "calendar", "probe")
    scheduler = _scheduler(tmp_path)
    assert any(s.name == "probe" for s in scheduler.tree["calendar"])
    assert scheduler.tree["calendar"][0].origin == "seed"

    status, _ = scheduler.delete_skill("calendar", "probe")
    assert status == "ok"

    reloaded = _scheduler(tmp_path)
    assert all(s.name != "probe" for s in reloaded.tree.get("calendar", []))


# ---- locks ----


def test_housekeeping_and_response_are_hard_locked(tmp_path):
    scheduler = _scheduler(tmp_path)
    assert scheduler._category_locked("housekeeping") is True
    assert scheduler._category_locked("response") is True
    status, _ = scheduler.set_category_lock("housekeeping", False)
    assert status == "error"


def test_category_lock_config_roundtrip(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.tree["notes"] = []
    assert scheduler._category_locked("notes") is False

    status, _ = scheduler.set_category_lock("notes", True)
    assert status == "ok"
    assert scheduler._category_locked("notes") is True
    config = scheduler.body_store.read_category_config("notes")
    assert config["locks"]["new_skill"] is True

    scheduler.set_category_lock("notes", False)
    assert scheduler._category_locked("notes") is False
    assert any(
        e["kind"] == "category_lock_set" for e in scheduler.trace.read()
    )


def test_dispatch_skill_blocked_when_locked(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.tree["notes"] = []
    scheduler.set_category_lock("notes", True)
    result = scheduler._dispatch_skill(Request("make a note"), "notes")
    assert result.kind == "error"
    assert "locked" in result.summary
    assert any(e["kind"] == "skill_create_blocked" for e in scheduler.trace.read())


def test_restart_skill_blocked_when_locked(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.tree["notes"] = [
        Skill(name="stub", category="notes", description="d")
    ]
    scheduler.set_category_lock("notes", True)
    status, _ = scheduler.restart_skill("notes", "stub")
    assert status == "error"


# ---- operations ----


def test_delete_skill_removes_tree_registry_and_folder(tmp_path):
    scheduler = _scheduler(tmp_path)
    _add_skill(scheduler, "calendar", "foo")
    status, detail = scheduler.delete_skill("calendar", "foo")
    assert status == "ok"
    assert all(s.name != "foo" for s in scheduler.tree["calendar"])
    assert scheduler.body_store.read_contract("calendar", "foo") == {}
    assert not scheduler.body_store.dir("calendar", "foo").exists()
    assert all(
        s.get("name") != "foo"
        for s in scheduler.registry.read()["calendar"]["skills"]
    )
    assert scheduler.deleted.contains("calendar", "foo")
    assert any(e["kind"] == "skill_deleted" for e in scheduler.trace.read())


def test_delete_refuses_builtin(tmp_path):
    scheduler = _scheduler(tmp_path)
    status, _ = scheduler.delete_skill("housekeeping", "delete_skill")
    assert status == "error"
    status, _ = scheduler.delete_skill("response", "response.clarify")
    assert status == "error"


def test_clear_skill_config(tmp_path):
    scheduler = _scheduler(tmp_path)
    skill = _add_skill(scheduler, "calendar", "foo")
    skill.config = {"api_key": "x"}
    scheduler.body_store.write_config("calendar", "foo", {"api_key": "x"})

    status, detail = scheduler.clear_skill_config("calendar", "foo")
    assert status == "ok"
    assert "api_key" in detail
    assert scheduler.body_store.read_config("calendar", "foo") == {}
    assert skill.config == {}
    assert any(e["kind"] == "config_cleared" for e in scheduler.trace.read())


def test_regen_skill_rewrites_existing_body(tmp_path, monkeypatch):
    scheduler = _scheduler(tmp_path)
    scheduler.codegen = object()
    _add_skill(scheduler, "calendar", "foo")
    captured = {}

    def fake_start(request, category, draft, repair_evidence=None, reason_kind="run_failure"):
        captured.update(
            category=category,
            repair_evidence=repair_evidence,
            reason_kind=reason_kind,
        )

    monkeypatch.setattr(scheduler, "_start_skill_write", fake_start)
    status, _ = scheduler.regen_skill("calendar", "foo", "it crashes")
    assert status == "running"
    assert captured["reason_kind"] == "manual"
    assert captured["repair_evidence"]["guidance"] == "it crashes"
    assert any(
        e["kind"] == "skill_regen_started" for e in scheduler.trace.read()
    )


def test_regen_skill_stub_has_no_evidence(tmp_path, monkeypatch):
    scheduler = _scheduler(tmp_path)
    scheduler.codegen = object()
    scheduler.tree["calendar"] = [
        Skill(name="stub", category="calendar", description="d")
    ]
    captured = {}

    def fake_start(request, category, draft, repair_evidence=None, reason_kind="run_failure"):
        captured.update(repair_evidence=repair_evidence, reason_kind=reason_kind)

    monkeypatch.setattr(scheduler, "_start_skill_write", fake_start)
    status, _ = scheduler.regen_skill("calendar", "stub", "make it work")
    assert status == "running"
    assert captured["repair_evidence"] is None


def test_cancel_skill_build_discards_and_deletes(tmp_path):
    scheduler = _scheduler(tmp_path)
    skill = _add_skill(scheduler, "calendar", "foo")
    skill.writing = True
    scheduler._writes.append(
        SkillWrite(
            request=Request("build foo"),
            category="calendar",
            draft=SkillDraft(name="foo", description="d"),
        )
    )

    status, detail = scheduler.cancel_skill_build("calendar", "foo")
    assert status == "ok"
    assert "cancelled" in detail
    assert ("calendar", "foo") in scheduler._cancelled
    assert not any(w.draft.name == "foo" for w in scheduler._writes)
    assert all(s.name != "foo" for s in scheduler.tree["calendar"])
    assert not scheduler.body_store.dir("calendar", "foo").exists()
    assert any(
        e["kind"] == "skill_build_cancelled" for e in scheduler.trace.read()
    )


def test_consume_cancelled_discards_queued_write(tmp_path):
    scheduler = _scheduler(tmp_path)
    job = SkillWrite(
        request=Request("build foo"),
        category="calendar",
        draft=SkillDraft(name="foo", description="d"),
    )
    scheduler._cancelled.add(("calendar", "foo"))
    assert scheduler._consume_cancelled(job) is True
    assert ("calendar", "foo") not in scheduler._cancelled
    assert any(
        e["kind"] == "skill_write_cancelled" for e in scheduler.trace.read()
    )


# ---- meta-skill act flows ----


def test_delete_skill_act_confirms_then_deletes(tmp_path):
    scheduler = _scheduler(tmp_path)
    _add_skill(scheduler, "calendar", "foo")
    ctx = ActionContext(engine=None, config={}, admin=scheduler)
    request = Request("delete skill calendar.foo")

    first = _delete_skill_act(ctx, request)
    assert first.needs_input and "confirm" in first.needs_input.lower()
    assert any(s.name == "foo" for s in scheduler.tree["calendar"])

    request.user_input = "yes"
    second = _delete_skill_act(ctx, request)
    assert second.needs_input is None
    assert all(s.name != "foo" for s in scheduler.tree["calendar"])


def test_delete_skill_act_asks_for_target_when_absent(tmp_path):
    """A name-free request ("delete skill") asks for the target as an input
    variable, then confirms, then deletes."""
    scheduler = _scheduler(tmp_path)
    _add_skill(scheduler, "calendar", "foo")
    ctx = ActionContext(engine=None, config={}, admin=scheduler)
    request = Request("delete skill")

    first = _delete_skill_act(ctx, request)
    assert first.needs_input and "which skill" in first.needs_input.lower()

    request.user_input = "calendar.foo"
    second = _delete_skill_act(ctx, request)
    assert second.needs_input and "confirm" in second.needs_input.lower()

    request.user_input = "yes"
    third = _delete_skill_act(ctx, request)
    assert third.needs_input is None
    assert all(s.name != "foo" for s in scheduler.tree["calendar"])


def test_delete_skill_act_declined_keeps_skill(tmp_path):
    scheduler = _scheduler(tmp_path)
    _add_skill(scheduler, "calendar", "foo")
    ctx = ActionContext(engine=None, config={}, admin=scheduler)
    request = Request("delete skill calendar.foo")
    _delete_skill_act(ctx, request)
    request.user_input = "no"
    result = _delete_skill_act(ctx, request)
    assert result.needs_input is None
    assert "cancelled" in result.action_log
    assert any(s.name == "foo" for s in scheduler.tree["calendar"])


def test_cancel_build_act_targets_writing_skill(tmp_path):
    scheduler = _scheduler(tmp_path)
    skill = _add_skill(scheduler, "calendar", "foo")
    skill.writing = True
    ctx = ActionContext(engine=None, config={}, admin=scheduler)

    request = Request("cancel the skill build")
    first = _cancel_build_act(ctx, request)
    assert first.needs_input and "cancel" in first.needs_input.lower()
    request.user_input = "yes"
    second = _cancel_build_act(ctx, request)
    assert second.needs_input is None
    assert all(s.name != "foo" for s in scheduler.tree["calendar"])


def test_cancel_build_act_reports_when_nothing_writing(tmp_path):
    scheduler = _scheduler(tmp_path)
    _add_skill(scheduler, "calendar", "foo")
    ctx = ActionContext(engine=None, config={}, admin=scheduler)
    result = _cancel_build_act(ctx, Request("cancel the skill build"))
    assert result.needs_input is None
    assert "no skill build" in result.action_log
