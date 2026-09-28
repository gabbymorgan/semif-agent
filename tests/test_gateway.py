"""Pure-stdlib tests for the messenger gateway (SimpleX adapter + router).

The decision engine is never loaded and the LLM endpoint is unreachable, so
scheduler calls degrade to errors — which is fine: these tests exercise the
gateway's own routing, authorization, parsing, and batching. The real
`websockets` transport against a live `simplex-chat` daemon is a jarvis
integration concern, not a dev-box unit test.
"""

import asyncio
import json

from semif_agent.decisions import DecisionRequest, DecisionResult, Option, Request
from semif_agent.engine import EngineConfig, SemIfEngine
from semif_agent.gateway.base import InboundMessage
from semif_agent.gateway.service import GatewayService
from semif_agent.gateway.simplex import SimplexAdapter
from semif_agent.llm import LLMClient
from semif_agent.log import DecisionLog
from semif_agent.scheduler import Scheduler
from semif_agent.skills import ActionResult, Prediction, Skill
from semif_agent.trace import TraceLog


def build_scheduler(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = SemIfEngine(EngineConfig())
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


# ---- adapter: parsing ----

def direct_item(text="hello", contact_id="4", display="alice", direction="directRcv", kind="text"):
    # v7 wire shape: the Contact's `displayName` is null and the peer name lives
    # in `profile.displayName`; `localDisplayName` is auto-suffixed on collision.
    # Received text is tagged `rcvMsgContent` with the message under `msgContent`.
    return {
        "chatInfo": {
            "type": "direct",
            "chatId": contact_id,
            "contact": {
                "contactId": contact_id,
                "localDisplayName": display,
                "displayName": None,
                "profile": {"displayName": display},
            },
        },
        "chatItem": {
            "chatDir": {"type": direction},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": kind, "text": text},
            },
        },
    }


def test_parse_authorized_direct_text():
    adapter = SimplexAdapter({"allowed_users": ["4"]})
    message = adapter._parse_chat_item(direct_item())
    assert message is not None
    assert message.text == "hello"
    assert message.chat_id == "4"
    assert message.contact_id == "4"
    assert message.display_name == "alice"


def test_parse_filters_echo_group_and_non_text():
    adapter = SimplexAdapter({"allow_all_users": True})
    assert adapter._parse_chat_item(direct_item(direction="directSnd")) is None
    assert adapter._parse_chat_item(direct_item(kind="image")) is None
    group = direct_item()
    group["chatInfo"]["type"] = "group"
    assert adapter._parse_chat_item(group) is None


def test_parse_filters_unauthorized_contact():
    adapter = SimplexAdapter({"allowed_users": ["9"]})
    assert adapter._parse_chat_item(direct_item(contact_id="4", display="alice")) is None


def test_parse_real_v7_payload_with_suffixed_local_name():
    # Captured live from simplex-chat v7 (`/_get chat @3`): the text is nested
    # under content.msgContent; Contact.displayName is null, the peer name is in
    # profile.displayName, and a collision-suffixed localDisplayName must not
    # defeat the allowlist.
    adapter = SimplexAdapter({"allowed_users": ["pepper"]})
    item = {
        "chatInfo": {
            "type": "direct",
            "chatId": 6,
            "contact": {
                "contactId": 6,
                "localDisplayName": "pepper_1",
                "displayName": None,
                "profile": {"displayName": "pepper"},
            },
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {
                    "type": "text",
                    "text": "what is the next thing on my calendar?",
                },
            },
        },
    }
    message = adapter._parse_chat_item(item)
    assert message is not None
    assert message.text == "what is the next thing on my calendar?"
    assert message.contact_id == "6"
    assert message.display_name == "pepper"


def test_parse_accepts_aeson_nested_content_shape():
    # The daemon's DB/Aeson encoding nests under the constructor key instead of
    # tagging with `type`; the parser tolerates both.
    adapter = SimplexAdapter({"allow_all_users": True})
    item = direct_item()
    item["chatItem"]["content"] = {
        "rcvMsgContent": {"msgContent": {"type": "text", "text": "hi"}}
    }
    message = adapter._parse_chat_item(item)
    assert message is not None
    assert message.text == "hi"


# ---- adapter: send command ----

def test_send_command_is_structured_not_shortcut():
    adapter = SimplexAdapter({})
    command = adapter.send_command("4", "hello")
    assert command.startswith("/_send @4 json ")
    payload = json.loads(command.split(" json ", 1)[1])
    assert payload[0]["msgContent"]["type"] == "text"
    assert payload[0]["msgContent"]["text"] == "hello"
    assert command != "/_send @4 hello"


# ---- adapter: contact request acceptance ----

class _FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data)["cmd"])


def _contact_request_event(req):
    return json.dumps(
        {"resp": {"type": "receivedContactRequest", "contactRequest": req}}
    )


def test_accept_uses_v7_contact_request_id():
    # v7 exposes contactRequestId (and contactId_); /_accept takes the request id.
    adapter = SimplexAdapter({"auto_accept": True})
    ws = _FakeWS()
    asyncio.run(adapter._consume(_contact_request_event({"contactRequestId": 7, "contactId_": 9}), lambda m: None, ws))
    assert ws.sent == ["/_accept 7"]


def test_accept_falls_back_to_contact_id_for_legacy_events():
    adapter = SimplexAdapter({"auto_accept": True})
    ws = _FakeWS()
    asyncio.run(adapter._consume(_contact_request_event({"contactId": 4}), lambda m: None, ws))
    assert ws.sent == ["/_accept 4"]


def test_accept_disabled_sends_nothing():
    adapter = SimplexAdapter({"auto_accept": False})
    ws = _FakeWS()
    asyncio.run(adapter._consume(_contact_request_event({"contactRequestId": 7, "contactId_": 9}), lambda m: None, ws))
    assert ws.sent == []


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
    def predict(ctx, request):
        return Prediction(text="", decisions=[])

    def act(ctx, request, prediction):
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
        predict=predict,
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
    assert "unavailable" in outbound[-1]


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
    assert any("queued work" in text or "unavailable" in text or "failed" in text for text in outbound)


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


def test_service_observer_sees_authorized_inbound(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    seen = []
    service.observer = seen.append
    service.handle_inbound(InboundMessage(text="hi", chat_id="4", contact_id="4"))
    assert [m.text for m in seen] == ["hi"]


def test_service_pull_mode_buffers_without_dispatch(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = GatewayService(scheduler, FakeAdapter(), config={"inbox": {"dispatch": False}})
    seen = []
    service.observer = seen.append
    service.handle_inbound(InboundMessage(text="hi", chat_id="4", contact_id="4"))
    assert [m.text for m in seen] == ["hi"], "pull mode must still buffer for a read skill"
    assert drain_outbound(service) == [], "pull mode must not reply or dispatch"
    assert scheduler.current is None
    assert scheduler.pending is None


# ---- adapter: correlated request/response + contact address ----


class FakeWS:
    """A fake simplex-chat socket that answers correlated commands."""

    def __init__(self, adapter, responses):
        self.adapter = adapter
        self.responses = list(responses)
        self.sent = []

    async def send(self, raw):
        message = json.loads(raw)
        self.sent.append(message)
        response = self.responses.pop(0)
        await self.adapter._consume(
            json.dumps({"corrId": message["corrId"], "resp": response}), None, self
        )


def test_correlated_response_resolves_pending_future():
    adapter = SimplexAdapter({})

    async def scenario():
        adapter._loop = asyncio.get_running_loop()
        adapter._ws = FakeWS(adapter, [])
        future = adapter._loop.create_future()
        adapter._pending["req-9"] = future
        await adapter._consume(
            json.dumps({"corrId": "req-9", "resp": {"type": "ok", "value": 1}}), None, adapter._ws
        )
        assert "req-9" not in adapter._pending, "a resolved request must be dropped"
        return future.result()

    assert asyncio.run(scenario()) == {"type": "ok", "value": 1}


def test_address_returns_existing_link():
    adapter = SimplexAdapter({"user_id": 3})

    async def scenario():
        adapter._loop = asyncio.get_running_loop()
        ws = FakeWS(
            adapter,
            [
                {
                    "type": "userContactLink",
                    "user": {"userId": 3},
                    "contactLink": {
                        "userContactLinkId": 1,
                        "connLinkContact": {
                            "connShortLink": "simplex:/a",
                            "connFullLink": "https://a",
                        },
                    },
                }
            ],
        )
        adapter._ws = ws
        return await adapter.address(timeout=5), ws

    result, ws = asyncio.run(scenario())
    assert result == {"short_link": "simplex:/a", "full_link": "https://a", "created": False}
    assert [m["cmd"] for m in ws.sent] == ["/_show_address 3"]


def test_address_creates_when_missing():
    adapter = SimplexAdapter({"user_id": 1})

    async def scenario():
        adapter._loop = asyncio.get_running_loop()
        ws = FakeWS(
            adapter,
            [
                {"type": "userContactLink", "contactLink": None},
                {
                    "type": "userContactLinkCreated",
                    "connLinkContact": {"connShortLink": "simplex:/new", "connFullLink": ""},
                },
            ],
        )
        adapter._ws = ws
        return await adapter.address(timeout=5), ws

    result, ws = asyncio.run(scenario())
    assert result == {"short_link": "simplex:/new", "full_link": "", "created": True}
    assert [m["cmd"] for m in ws.sent] == ["/_show_address 1", "/_address 1"]


def test_request_address_without_connection_raises():
    import pytest

    adapter = SimplexAdapter({})
    with pytest.raises(RuntimeError):
        adapter.request_address(timeout=1)
