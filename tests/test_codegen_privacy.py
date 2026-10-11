"""Codegen privacy invariant: the (primarily-cloud) codegen endpoint is shown
ONLY a de-identified spec.

The user's request, question answers, and run output must never reach a codegen
prompt (body, retry, regen, or testgen). These tests assert that structurally
(the builders take no `Request`) and behaviourally (no captured message contains
a sentinel that only ever lived in the request/run data).
"""

from __future__ import annotations

import inspect

from semif_agent.codegen import (
    _regen_body_prompt,
    _retry_prompt,
    build_skill_body_prompt,
    build_testgen_base_prompt,
    generate_skill_body,
    generate_skill_tests,
    regenerate_skill_body,
)
from semif_agent.decisions import Request
from semif_agent.llm import LLMClient
from semif_agent.log import DecisionLog
from semif_agent.scheduler import Scheduler
from semif_agent.skill import RunResult
from semif_agent.skills import ActionResult, CategoryRegistry, Skill, SkillDraft
from semif_agent.trace import TraceLog

from tests.conftest import ScriptedEngine

#: Values that ONLY appear in the user's request / run output. Any of these in a
#: codegen message would be a leak.
SENTINELS = [
    "Dana",
    "dana@corp.example",
    "Project Falcon",
    "Oct 15",
    "hunter2",
    "4111 1111 1111 1111",
]

GOOD_BODY = """\
import urllib.request
from semif_agent.skills import ActionResult

INTEGRATION = {"service": "probe_service", "transport": "http", "config_vars": ["probe_url"]}
CONTRACT = {"probe_url": "The service base URL."}


def act(ctx, request):
    url = ctx.config["probe_url"]
    with urllib.request.urlopen(f"{url}/status", timeout=15) as resp:
        body = resp.read().decode("utf-8")
    return ActionResult(action_log=f"GET {url}/status", new_state=body)
"""

GOOD_TEST = "import sys\nprint('ok')\n"


class _RecordingCodegen:
    """Minimal codegen client double that captures every outbound message list."""

    stream = False
    max_attempts = 1

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages, max_tokens=None, degeneration_check=None, timeout=None):
        self.calls.append(messages)
        return self.reply

    def all_text(self) -> str:
        return "\n".join(
            message.get("content", "")
            for messages in self.calls
            for message in messages
        )


def _draft() -> SkillDraft:
    return SkillDraft(
        name="check_status",
        description="Check the status of a generic HTTP service.",
        goal="request a service status endpoint and report the response body",
        integration={"service": "probe_service", "transport": "http"},
    )


def _assert_clean(text: str) -> None:
    for sentinel in SENTINELS:
        assert sentinel not in text, f"codegen message leaked {sentinel!r}"


def test_prompt_builders_take_no_request():
    """Structural guard: every codegen prompt builder works from the spec only."""
    for func in (
        build_skill_body_prompt,
        _retry_prompt,
        _regen_body_prompt,
        build_testgen_base_prompt,
        generate_skill_body,
        regenerate_skill_body,
        generate_skill_tests,
    ):
        params = inspect.signature(func).parameters
        assert "request" not in params, f"{func.__name__} must not take a Request"
        assert "requirements" not in params, (
            f"{func.__name__} must not take elicitation answers"
        )


def test_body_and_retry_prompts_carry_only_the_deidentified_goal():
    tree = {"tracking": []}
    draft = _draft()
    body = build_skill_body_prompt("tracking", draft, tree, "THE CONTRACT")
    retry = _retry_prompt("tracking", draft, "THE CONTRACT", ValueError("bad"))
    for messages in (body, retry):
        _assert_clean("\n".join(m.get("content", "") for m in messages))
    assert draft.goal in body[1]["content"]
    assert draft.goal in retry[1]["content"]


def test_generate_skill_body_never_sees_run_or_request_data():
    client = _RecordingCodegen(GOOD_BODY)
    code = generate_skill_body(client, "tracking", _draft(), {"tracking": []})
    assert "def act" in code
    _assert_clean(client.all_text())


def test_regenerate_skill_body_evidence_is_deidentified():
    client = _RecordingCodegen(GOOD_BODY)
    evidence = {
        "failure_kind": "a real run against the live service failed",
        "goal": _draft().goal,
        "previous_body": GOOD_BODY,
    }
    regenerate_skill_body(client, "tracking", _draft(), GOOD_BODY, evidence)
    _assert_clean(client.all_text())


def test_generate_skill_tests_never_sees_the_request():
    client = _RecordingCodegen(GOOD_TEST)
    tests = generate_skill_tests(
        client, "tracking", _draft(), GOOD_BODY, {"probe_url": "URL"}, contract_md="TESTGEN"
    )
    assert tests.strip()
    _assert_clean(client.all_text())


def test_registry_never_stores_the_request_or_answers(tmp_path):
    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    registry.register_skill(
        "tracking",
        "check_status",
        "Check a service.",
        goal="request a service status endpoint and report the response body",
    )
    row = registry.read()["tracking"]["skills"][0]
    assert row["goal"] == "request a service status endpoint and report the response body"
    assert "request_text" not in row
    assert "requirements" not in row
    _assert_clean(str(row))


def _scheduler(tmp_path) -> Scheduler:
    return Scheduler(
        engine=ScriptedEngine(default="retry"),
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


def test_repair_evidence_is_deidentified(tmp_path):
    """The bundle handed to codegen on repair carries no request or run output."""
    scheduler = _scheduler(tmp_path)

    def act(ctx, request):
        return ActionResult(action_log="ran", new_state="ran")

    skill = Skill(
        name="check_status",
        category="tracking",
        description="Check a service.",
        act=act,
    )
    request = Request(
        "email Dana at dana@corp.example about Project Falcon on Oct 15; card 4111 1111 1111 1111"
    )
    outcome = RunResult(
        skill="check_status",
        success=False,
        summary="failed to reach dana@corp.example",
        action_log="POST https://svc returned 401 for hunter2",
        new_state="Dana, Project Falcon, Oct 15",
        outcome="failed",
        error="connection refused for Project Falcon",
    )
    scheduler._propose_repair(skill, request, outcome)
    offer = scheduler.repairs[-1]
    evidence = offer.evidence
    # Only the de-identified fields are present.
    assert "request" not in evidence
    assert "action_log" not in evidence
    assert "new_state" not in evidence
    assert "summary" not in evidence
    assert "error" not in evidence
    assert evidence["failure_kind"]
    _assert_clean(repr(evidence))

