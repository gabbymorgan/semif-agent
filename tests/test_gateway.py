"""Pure-stdlib tests for the messenger gateway (SimpleX command adapter + router).

The gateway is the agent's command surface only. These tests exercise its own
routing, authorization, and batching; parsing/accept/address live in
`tests/test_simplex_ws.py`. The decision engine is never loaded and the LLM
endpoint is unreachable, so scheduler calls degrade to errors — which is fine.
The real `websockets` transport against a live `simplex-chat` daemon is a jarvis
integration concern, not a dev-box unit test.
"""

import asyncio
import json

from semif_agent.cli import _gateway_address_callback
from semif_agent.decisions import Request
from semif_agent.engine import EngineConfig, SemIfEngine
from semif_agent.gateway.base import InboundMessage
from semif_agent.gateway.service import GatewayService
from semif_agent.gateway.simplex import SimplexAdapter
from semif_agent.llm import LLMClient
from semif_agent.log import DecisionLog
from semif_agent.scheduler import PendingApproval, Scheduler
from semif_agent.skills import ActionResult, Skill
from semif_agent.trace import TraceLog

from tests.conftest import ScriptedEngine


def build_scheduler(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = ScriptedEngine(default="success")
    llm = LLMClient(base_url="http://localhost:1/v1", model="test")
    return Scheduler(
        engine=engine,
        llm=llm,
        log=log,
        config={"skills": {}, "skill_seeds": str(tmp_path / "seeds")},
        trace=trace,
    )


def drain_outbound(service) -> list[str]:
    texts = []
    while not service.outbound.empty():
        item = service.outbound.get_nowait()
        if item is not None:
            texts.append(item.text)
    return texts


# ---- adapter: authorization ----

def test_default_deny_and_allowlist():
    assert SimplexAdapter({}).is_authorized("4", "alice") is False
    assert SimplexAdapter({"allowed_users": ["4"]}).is_authorized("4", "alice") is True
    assert SimplexAdapter({"allowed_users": ["alice"]}).is_authorized("4", "alice") is True
    assert SimplexAdapter({"allowed_users": ["bob"]}).is_authorized("4", "alice") is False
    assert SimplexAdapter({"allow_all_users": True}).is_authorized("7", None) is True


# ---- adapter: requirements ----

def test_check_requirements_needs_url():
    ok, hint = SimplexAdapter({"ws_url": ""}).check_requirements()
    assert ok is False
    assert "ws_url" in hint


# ---- startup contact-link announcement ----

class _FakeDaemon:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    async def address(self, timeout=20.0):
        self.calls += 1
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeAdapter:
    def __init__(self, daemon):
        self.daemon = daemon


def test_gateway_address_callback_prints_link_once(capsys):
    daemon = _FakeDaemon(
        {"short_link": "simplex:/abc", "full_link": "simplex://full"}
    )
    announce = _gateway_address_callback(_FakeAdapter(daemon))
    asyncio.run(announce())
    asyncio.run(announce())  # a reconnect must not reprint
    out = capsys.readouterr()
    assert "simplex:/abc" in out.out
    assert "simplex://full" in out.out
    assert daemon.calls == 1


def test_gateway_address_callback_failure_is_not_fatal(capsys):
    daemon = _FakeDaemon(RuntimeError("not connected"))
    announce = _gateway_address_callback(_FakeAdapter(daemon))
    asyncio.run(announce())  # must not raise
    asyncio.run(announce())  # retried on the next connect
    captured = capsys.readouterr()
    assert "could not read contact link" in captured.err
    assert daemon.calls == 2


def test_gateway_address_callback_no_link_warns(capsys):
    daemon = _FakeDaemon({"short_link": "", "full_link": ""})
    announce = _gateway_address_callback(_FakeAdapter(daemon))
    asyncio.run(announce())
    assert "no contact link available" in capsys.readouterr().err


# ---- adapter: send command ----

def test_send_command_is_structured_not_shortcut():
    adapter = SimplexAdapter({})
    command = adapter.send_command("4", "hello")
    assert command.startswith("/_send @4 json ")
    payload = json.loads(command.split(" json ", 1)[1])
    assert payload[0]["msgContent"]["type"] == "text"
    assert payload[0]["msgContent"]["text"] == "hello"
    assert command != "/_send @4 hello"


# ---- adapter: batching ----

def test_batching_concatenates_rapid_messages():
    adapter = SimplexAdapter({"allow_all_users": True, "text_batch_delay": 0.05})
    seen = []

    async def go():
        adapter._schedule_flush(
            InboundMessage(text="one", chat_id="4", contact_id="4"), seen.append
        )
        adapter._schedule_flush(
            InboundMessage(text="two", chat_id="4", contact_id="4"), seen.append
        )
        await asyncio.sleep(0.2)

    asyncio.run(go())
    assert len(seen) == 1
    assert "one" in seen[0].text and "two" in seen[0].text


# ---- service: routing ----

def need_input_skill(seen):
    def act(ctx, request):
        if request.user_input:
            seen.append(request.user_input)
            return ActionResult(action_log=f"got {request.user_input}", new_state="done")
        return ActionResult(
            action_log="ask", new_state=request.text, needs_input="Tracking number?"
        )

    return Skill(
        name="track.manual",
        category="tracking",
        description="Resolve a tracking number with the human.",
                act=act,
    )


class FakeAdapter:
    name = "simplex"


def build_service(scheduler):
    return GatewayService(scheduler, FakeAdapter(), config={})


def test_service_submit_error_is_replied(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    service.handle_inbound(InboundMessage(text="hi", chat_id="4", contact_id="4"))
    outbound = drain_outbound(service)
    assert outbound, "expected a reply"


def test_service_routes_pending_answer_to_same_chat(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    seen = []
    scheduler._run_skill(
        need_input_skill(seen), Request("track it", source="simplex:4")
    )
    assert scheduler.pending is not None

    service.handle_inbound(InboundMessage(text="AB123", chat_id="4", contact_id="4"))
    assert seen == ["AB123"]
    assert scheduler.pending is None
    assert any("AB123" in text for text in drain_outbound(service))


def test_service_refuses_other_chat_during_pending(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    scheduler._run_skill(need_input_skill([]), Request("track it", source="simplex:4"))
    question = scheduler.pending.question

    service.handle_inbound(InboundMessage(text="hello", chat_id="7", contact_id="7"))
    assert scheduler.pending is not None
    assert scheduler.pending.question == question
    assert any("another conversation" in text for text in drain_outbound(service))


def test_service_requeue_hook_carries_ownership(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    service._owners["parent"] = "4"
    service.on_request_requeued("parent", "child")
    assert service._owner_of("child") == "4"


def test_service_drain_routes_queued_run_to_owner(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    request = Request("queued work", source="simplex:4")
    scheduler.queue.push(request, 0.5)
    service._owners[request.id] = "4"

    service.drain()
    outbound = drain_outbound(service)
    assert outbound, "the queued run's outcome must be routed back to its owner"


def test_service_answered_question_routes_back(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    # Post a deferred question as the codegen worker would.
    from semif_agent.scheduler import PendingQuestion
    import uuid
    scheduler.questions.append(
        PendingQuestion(
            id=uuid.uuid4().hex[:12],
            run_id="r1",
            category="c",
            skill="s",
            question="Which account?",
        )
    )
    service._owners["r1"] = "4"
    service.surface()
    assert any("Which account?" in text for text in drain_outbound(service))

    service.handle_inbound(InboundMessage(text="work", chat_id="4", contact_id="4"))
    assert not scheduler.pending_questions()


def test_service_surfaces_and_resolves_approval(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    scheduler.approvals.append(
        PendingApproval(
            id="a1",
            run_id="r1",
            kind="skill",
            category="tracking",
            skill="track_live",
            description="Follow a package.",
        )
    )
    service._owners["r1"] = "4"
    service.surface()
    assert any(
        "Approve creating new skill tracking.track_live" in text
        for text in drain_outbound(service)
    )

    service.handle_inbound(InboundMessage(text="yes", chat_id="4", contact_id="4"))
    assert scheduler.pending_approvals() == []
    assert any("approved" in text for text in drain_outbound(service))
