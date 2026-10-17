"""Pure-stdlib tests for the messenger gateway (SimpleX command adapter + router).

The gateway is the agent's command surface only. These tests exercise its own
routing, authorization, and batching; parsing/accept/address live in
`tests/test_simplex_ws.py`. The decision engine is never loaded and the LLM
endpoint is unreachable, so scheduler calls degrade to errors — which is fine.
The real `websockets` transport against a live `simplex-chat` daemon is a
live-daemon integration concern, not a unit test.
"""

import asyncio
import json
import queue
import time

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
        config={"skills": {}, "skill_packs": str(tmp_path / "seeds")},
        trace=trace,
    )


def drain_outbound(service) -> list[str]:
    texts = []
    for platform in service.platforms:
        outbound = service.outbound[platform]
        while not outbound.empty():
            item = outbound.get_nowait()
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


def test_service_input_ready_tracks_scheduler(tmp_path):
    # The voice gateway holds the wake word on this: not ready while a run is
    # executing, ready when idle or paused for the user's answer.
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    assert service.input_ready() is True          # idle
    scheduler.busy("working")
    assert service.input_ready() is False         # a run is executing
    scheduler.pending = object()                  # paused for the answer
    assert service.input_ready() is True
    scheduler.pending = None
    scheduler.current = None
    assert service.input_ready() is True


def test_service_wires_input_ready_into_voice_adapter(tmp_path):
    # A constructed service hands its `input_ready` to an adapter that exposes
    # `set_input_ready` (the voice gateway).
    class ReadyAdapter:
        name = "voice"

        def __init__(self):
            self.ready = None

        def set_input_ready(self, fn):
            self.ready = fn

    scheduler = build_scheduler(tmp_path)
    adapter = ReadyAdapter()
    service = GatewayService(scheduler, adapter, config={})
    assert adapter.ready == service.input_ready


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


def test_service_survives_answer_failure(tmp_path):
    # A resume that raises must come back as a reply, never propagate into the
    # front end (the voice adapter calls handle_inbound synchronously).
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    scheduler._run_skill(need_input_skill([]), Request("track it", source="simplex:4"))
    assert scheduler.pending is not None

    def boom(*_args, **_kwargs):
        raise RuntimeError("kaboom")

    scheduler._answer = boom
    service.handle_inbound(InboundMessage(text="AB123", chat_id="4", contact_id="4"))
    assert any("kaboom" in text for text in drain_outbound(service))


def test_answer_engine_unavailable_is_fatal(tmp_path):
    from semif_agent.engine import EngineUnavailable

    scheduler = build_scheduler(tmp_path)
    scheduler._run_skill(need_input_skill([]), Request("track it", source="simplex:4"))

    def boom(*_args, **_kwargs):
        raise EngineUnavailable("engine down")

    scheduler._answer = boom
    reply = scheduler.answer("AB123")
    assert reply.status == "fatal"
    assert scheduler.fatal is not None


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
    service._owners["parent"] = ("simplex", "4")
    service.on_request_requeued("parent", "child")
    assert service._owner_of("child") == ("simplex", "4")


def test_service_drain_routes_queued_run_to_owner(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    request = Request("queued work", source="simplex:4")
    scheduler.queue.push(request)
    service._owners[request.id] = ("simplex", "4")

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
    service._owners["r1"] = ("simplex", "4")
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
    service._owners["r1"] = ("simplex", "4")
    service.surface()
    assert any(
        "Approve creating new skill tracking.track_live" in text
        for text in drain_outbound(service)
    )

    service.handle_inbound(InboundMessage(text="yes", chat_id="4", contact_id="4"))
    assert scheduler.pending_approvals() == []
    assert any("approved" in text for text in drain_outbound(service))


# ---- service: spoken action vocabulary (punctuation, phrases) ----

def test_normalize_action_strips_punctuation_and_phrases():
    from semif_agent.gateway.service import _normalize_action

    assert _normalize_action("Approve.") == "approve"
    assert _normalize_action("Yes.") == "yes"
    assert _normalize_action("  no!  ") == "no"
    assert _normalize_action("yes please") == "yes"
    assert _normalize_action("approve it") == "approve"
    assert _normalize_action("Retry.") == "retry"
    assert _normalize_action("never mind") == "never mind"
    assert _normalize_action("") == ""
    assert _normalize_action("what is 255 times 12?") == "what"


def test_service_approval_accepts_spoken_punctuation(tmp_path):
    # STT emits "Approve." / "Yes." with a trailing period; the exact-map lookup
    # used to miss it and fall through to submit_request, so the approval timed
    # out and nothing was ever created (the voice authoring failure).
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
    service._owners["r1"] = ("simplex", "4")
    service.surface()
    drain_outbound(service)

    service.handle_inbound(InboundMessage(text="Approve.", chat_id="4", contact_id="4"))
    assert scheduler.pending_approvals() == []
    assert any("approved" in text for text in drain_outbound(service))
    assert not any(
        e["kind"] == "submit" and e.get("text") == "Approve."
        for e in scheduler.trace.read()
    ), "a spoken approval must not be dispatched as a new request"


def test_service_approval_accepts_spoken_no(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    scheduler.approvals.append(
        PendingApproval(
            id="a1",
            run_id="r1",
            kind="category",
            category="text_analysis",
            skill="",
            description="Analyze text.",
        )
    )
    service._owners["r1"] = ("simplex", "4")
    service.surface()
    drain_outbound(service)

    service.handle_inbound(InboundMessage(text="No thanks.", chat_id="4", contact_id="4"))
    assert scheduler.pending_approvals() == []
    assert any("denied" in text for text in drain_outbound(service))


def test_service_repair_accepts_spoken_punctuation(tmp_path):
    from semif_agent.scheduler import RepairOffer

    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    scheduler.repairs.append(
        RepairOffer(
            id="r1",
            run_id="run1",
            category="tracking",
            skill="track_live",
            selected="retry",
            reason="failed",
            request_text="track it",
            failure="boom",
        )
    )
    service._owners["run1"] = ("simplex", "4")
    service.surface()
    drain_outbound(service)

    service.handle_inbound(InboundMessage(text="retry.", chat_id="4", contact_id="4"))
    offer = next(r for r in scheduler.repairs if r.id == "r1")
    assert offer.status == "done"
    assert not any(
        e["kind"] == "submit" and e.get("text") == "retry."
        for e in scheduler.trace.read()
    ), "a spoken repair action must not be dispatched as a new request"


# ---- service: background-drain outbox routing ----

def test_service_drain_routes_background_outbox(tmp_path):
    # A background drain (the codegen worker's post-authoring re-run, a repair
    # retry) buffers its reply; the front end's drain must still route it.
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)
    request = Request("queued work", source="simplex:4")
    scheduler.queue.push(request)
    service._owners[request.id] = ("simplex", "4")

    scheduler.run_queue_buffered()  # the off-gate completion
    assert len(scheduler._outbox) == 1

    service.drain()
    assert drain_outbound(service), "the buffered background reply must reach the owner"
    assert list(scheduler._outbox) == []



# ---- service: result-only front end (voice) ----

class FakeResultOnlyAdapter:
    name = "voice"
    result_only = True


def build_result_only_service(scheduler, config=None):
    return GatewayService(scheduler, FakeResultOnlyAdapter(), config=config or {})


def test_result_detail_unwraps_scheduler_summary():
    from semif_agent.gateway.service import _result_detail

    assert _result_detail("time.now: ok — 2026-10-06 12:00") == "2026-10-06 12:00"
    assert _result_detail("time.now: failed — boom") == "boom"
    assert _result_detail("a plain reply") == "a plain reply"


def test_result_only_speaks_detail_and_drops_status(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_result_only_service(scheduler)
    service._reply("voice", "c", "time.now: ok — 12:00", kind="result")
    service._reply("voice", "c", "queued", kind="status")
    service._reply("voice", "c", "Which account?", kind="question")
    assert drain_outbound(service) == ["12:00", "Which account?"]


def test_plain_adapter_keeps_full_scheduler_output(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_service(scheduler)  # FakeAdapter (simplex), not result_only
    service._reply("simplex", "4", "time.now: ok — 12:00", kind="result")
    service._reply("simplex", "4", "queued", kind="status")
    assert drain_outbound(service) == [
        "time.now: ok — 12:00",
        "queued",
    ]


def test_config_can_turn_result_only_off(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_result_only_service(
        scheduler, config={"voice": {"result_only": False}}
    )
    service._reply("voice", "c", "time.now: ok — 12:00", kind="result")
    service._reply("voice", "c", "queued", kind="status")
    assert drain_outbound(service) == ["time.now: ok — 12:00", "queued"]


# ---- LXMF adapter ----

def test_lxmf_default_deny_and_allowlist():
    from semif_agent.gateway.lxmf import LxmfAdapter

    assert LxmfAdapter({}).is_authorized("a" * 32, "alice") is False
    assert LxmfAdapter({"allowed_users": ["a" * 32]}).is_authorized("a" * 32, "alice") is True
    assert LxmfAdapter({"allowed_users": ["alice"]}).is_authorized("a" * 32, "alice") is True
    assert LxmfAdapter({"allowed_users": ["bob"]}).is_authorized("a" * 32, "alice") is False
    assert LxmfAdapter({"allow_all_users": True}).is_authorized("a" * 32, None) is True


def test_lxmf_batching_concatenates_rapid_messages():
    from semif_agent.gateway.lxmf import LxmfAdapter

    adapter = LxmfAdapter({"allow_all_users": True, "text_batch_delay": 0.05})
    seen: list = []
    adapter._on_inbound = seen.append
    for text in ("one", "two"):
        adapter._on_item(
            {"source_hash": "ab" * 16, "content": text, "display_name": None}
        )
    time.sleep(0.2)
    assert len(seen) == 1
    assert "one" in seen[0].text and "two" in seen[0].text


def test_lxmf_denies_unknown_sender_without_dispatch():
    from semif_agent.gateway.lxmf import LxmfAdapter

    adapter = LxmfAdapter({})
    seen: list = []
    adapter._on_inbound = seen.append
    adapter._on_item({"source_hash": "cd" * 16, "content": "hi"})
    time.sleep(0.1)
    assert seen == []


def test_lxmf_adapter_is_result_only():
    from semif_agent.gateway.lxmf import LxmfAdapter

    # The LXMF gateway replies with the skill's result line only (drop the
    # scheduler's bookkeeping and the `<skill>: ok —` wrapper), overridable via
    # `gateway.lxmf.result_only: false`.
    assert LxmfAdapter({}).result_only is True


def test_lxmf_service_sends_result_only(tmp_path):
    from semif_agent.gateway.lxmf import LxmfAdapter

    scheduler = build_scheduler(tmp_path)
    service = GatewayService(scheduler, LxmfAdapter({"allow_all_users": True}))
    service._reply("lxmf", "c", "time.now: ok — 12:00", kind="result")
    service._reply("lxmf", "c", "queued", kind="status")
    service._reply("lxmf", "c", "Which account?", kind="question")
    assert drain_outbound(service) == ["12:00", "Which account?"]


def test_lxmf_config_can_turn_result_only_off(tmp_path):
    from semif_agent.gateway.lxmf import LxmfAdapter

    scheduler = build_scheduler(tmp_path)
    service = GatewayService(
        scheduler,
        LxmfAdapter({"allow_all_users": True}),
        config={"lxmf": {"result_only": False}},
    )
    service._reply("lxmf", "c", "time.now: ok — 12:00", kind="result")
    service._reply("lxmf", "c", "queued", kind="status")
    assert drain_outbound(service) == ["time.now: ok — 12:00", "queued"]


# ---- LXMF voice (Sideband push-to-talk) messages ----

OPUS_AUDIO = {
    "codec": 0x10,
    "codec_name": "opus_ogg",
    "data": b"OggS-payload",
    "kind": "opus_ogg",
    "codec2_rate": None,
}
CODEC2_AUDIO = {
    "codec": 0x08,
    "codec_name": "codec2_2400",
    "data": b"\x01\x02\x03\x04\x05\x06",
    "kind": "codec2",
    "codec2_rate": 2400,
}


class FakeTranscriber:
    """A transcriber double: records the raw payloads, returns fixed text."""

    def __init__(self, text="set a timer for five minutes", error=None):
        self.text = text
        self.error = error
        self.calls: list = []

    def transcribe_bytes(self, data):
        self.calls.append(data)
        if self.error is not None:
            raise self.error
        return self.text


def _wait(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_lxmf_transcribes_voice_message_into_a_command():
    from semif_agent.gateway.lxmf import LxmfAdapter

    transcriber = FakeTranscriber("what time is it")
    adapter = LxmfAdapter(
        {"allow_all_users": True, "voice": {"enabled": True}}, transcriber=transcriber
    )
    seen: list = []
    adapter._on_inbound = seen.append
    adapter._on_item(
        {
            "source_hash": "ab" * 16,
            "content": "",
            "display_name": None,
            "audio": dict(OPUS_AUDIO),
        }
    )
    assert _wait(lambda: seen)
    assert seen[0].text == "what time is it"
    assert seen[0].chat_id == "ab" * 16
    assert transcriber.calls == [b"OggS-payload"]


def test_lxmf_voice_disabled_ignores_audio():
    from semif_agent.gateway.lxmf import LxmfAdapter

    # A transcriber is present, but `voice.enabled` is not — nothing runs.
    adapter = LxmfAdapter({"allow_all_users": True}, transcriber=FakeTranscriber("hi"))
    seen: list = []
    adapter._on_inbound = seen.append
    adapter._on_item(
        {"source_hash": "ab" * 16, "content": "", "audio": dict(OPUS_AUDIO)}
    )
    time.sleep(0.1)
    assert seen == []


def test_lxmf_voice_unsupported_codec_notifies_instead_of_dropping():
    from semif_agent.gateway.lxmf import LxmfAdapter, VOICE_UNSUPPORTED_MESSAGE

    adapter = LxmfAdapter(
        {"allow_all_users": True, "voice": {"enabled": True}},
        transcriber=FakeTranscriber(),
    )
    adapter._outbound = queue.Queue()
    adapter._on_item(
        {
            "source_hash": "ab" * 16,
            "content": "",
            "audio": {"codec": 0x01, "codec_name": "codec2_450pwb", "data": b"x",
                      "kind": "unknown", "codec2_rate": None},
        }
    )
    message = adapter._outbound.get(timeout=2)
    assert message.chat_id == "ab" * 16
    assert message.text == VOICE_UNSUPPORTED_MESSAGE


def test_lxmf_transcribes_codec2_voice_message(monkeypatch):
    import semif_agent.gateway.lxmf as lxmf
    from semif_agent.gateway.lxmf import LxmfAdapter

    monkeypatch.setattr(lxmf, "check_codec2_requirements", lambda: (True, None))
    decoded = []

    def fake_decode(data, rate):
        decoded.append((data, rate))
        return b"\x00\x00" * 320  # 20 ms at 8 kHz

    monkeypatch.setattr(lxmf, "decode_codec2", fake_decode)
    transcriber = FakeTranscriber("turn on the kitchen light")
    adapter = LxmfAdapter(
        {"allow_all_users": True, "voice": {"enabled": True}}, transcriber=transcriber
    )
    seen: list = []
    adapter._on_inbound = seen.append
    adapter._on_item(
        {"source_hash": "ab" * 16, "content": "", "audio": dict(CODEC2_AUDIO)}
    )
    assert _wait(lambda: seen)
    assert seen[0].text == "turn on the kitchen light"
    assert decoded == [(CODEC2_AUDIO["data"], 2400)]
    # The Codec2 PCM is wrapped in a WAV before transcription.
    assert transcriber.calls and transcriber.calls[0][:4] == b"RIFF"


def test_lxmf_codec2_unavailable_notifies(monkeypatch):
    import semif_agent.gateway.lxmf as lxmf
    from semif_agent.gateway.lxmf import LxmfAdapter, VOICE_CODEC2_MESSAGE

    monkeypatch.setattr(lxmf, "check_codec2_requirements", lambda: (False, "no pycodec2"))
    adapter = LxmfAdapter(
        {"allow_all_users": True, "voice": {"enabled": True}},
        transcriber=FakeTranscriber(),
    )
    adapter._outbound = queue.Queue()
    adapter._on_item(
        {"source_hash": "ab" * 16, "content": "", "audio": dict(CODEC2_AUDIO)}
    )
    message = adapter._outbound.get(timeout=2)
    assert message.text == VOICE_CODEC2_MESSAGE


def test_lxmf_transcribe_failure_notifies():
    from semif_agent.gateway.lxmf import LxmfAdapter, VOICE_FAILED_MESSAGE

    adapter = LxmfAdapter(
        {"allow_all_users": True, "voice": {"enabled": True}},
        transcriber=FakeTranscriber(error=RuntimeError("boom")),
    )
    adapter._outbound = queue.Queue()
    seen: list = []
    adapter._on_inbound = seen.append
    adapter._on_item(
        {"source_hash": "ab" * 16, "content": "", "audio": dict(OPUS_AUDIO)}
    )
    message = adapter._outbound.get(timeout=2)
    assert message.text == VOICE_FAILED_MESSAGE
    assert seen == []


def test_lxmf_voice_keeps_caption_when_transcription_fails():
    from semif_agent.gateway.lxmf import LxmfAdapter

    adapter = LxmfAdapter(
        {"allow_all_users": True, "voice": {"enabled": True}},
        transcriber=FakeTranscriber(error=RuntimeError("boom")),
    )
    adapter._outbound = queue.Queue()
    seen: list = []
    adapter._on_inbound = seen.append
    adapter._on_item(
        {
            "source_hash": "ab" * 16,
            "content": "please create a task",
            "audio": dict(OPUS_AUDIO),
        }
    )
    assert _wait(lambda: seen)
    assert seen[0].text == "please create a task"
    assert adapter._outbound.empty()


# ---- LXMF voice replies (Piper -> Opus/OGG) ----

class FakeSynthesizer:
    """A synthesizer double: records texts, returns fixed PCM/rate."""

    def __init__(self, pcm=b"\x00\x00" * 100, sample_rate=22050, error=None):
        self.pcm = pcm
        self.sample_rate = sample_rate
        self.error = error
        self.calls: list = []

    def synthesize(self, text):
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return self.pcm, self.sample_rate


def _outbound(chat, text):
    from semif_agent.gateway.base import OutboundMessage

    return OutboundMessage(chat_id=chat, text=text)


def test_lxmf_voice_message_marks_chat_and_replies_with_voice(monkeypatch):
    import semif_agent.gateway.lxmf as lxmf
    from semif_agent.gateway.lxmf import LxmfAdapter

    monkeypatch.setattr(lxmf, "pcm_to_opus_ogg", lambda pcm, rate, bitrate=0: b"OggS")
    synth = FakeSynthesizer()
    adapter = LxmfAdapter(
        {"allow_all_users": True, "voice": {"enabled": True}},
        transcriber=FakeTranscriber("what time is it"),
        synthesizer=synth,
    )
    # A voice message marks the chat for voice replies.
    adapter._on_item(
        {"source_hash": "ab" * 16, "content": "", "audio": dict(OPUS_AUDIO)}
    )
    assert "ab" * 16 in adapter._voice_chats
    audio = adapter._voice_reply(_outbound("ab" * 16, "It is 12:00."))
    assert audio == b"OggS"
    assert synth.calls == ["It is 12:00."]


def test_lxmf_text_chat_replies_with_text():
    from semif_agent.gateway.lxmf import LxmfAdapter

    synth = FakeSynthesizer()
    adapter = LxmfAdapter(
        {"allow_all_users": True, "voice": {"enabled": True}},
        transcriber=FakeTranscriber("hi"),
        synthesizer=synth,
    )
    # No voice message from this chat: a text reply.
    assert adapter._voice_reply(_outbound("ab" * 16, "hello")) is None
    assert synth.calls == []


def test_lxmf_text_message_reverts_chat_to_text_replies():
    from semif_agent.gateway.lxmf import LxmfAdapter

    synth = FakeSynthesizer()
    adapter = LxmfAdapter(
        {"allow_all_users": True, "voice": {"enabled": True}},
        transcriber=FakeTranscriber("hi"),
        synthesizer=synth,
    )
    adapter._on_item(
        {"source_hash": "ab" * 16, "content": "", "audio": dict(OPUS_AUDIO)}
    )
    assert "ab" * 16 in adapter._voice_chats
    adapter._on_item({"source_hash": "ab" * 16, "content": "now in text"})
    assert "ab" * 16 not in adapter._voice_chats


def test_lxmf_reply_with_voice_disabled_sends_text():
    from semif_agent.gateway.lxmf import LxmfAdapter

    synth = FakeSynthesizer()
    adapter = LxmfAdapter(
        {"allow_all_users": True, "voice": {"enabled": True, "reply_with_voice": False}},
        transcriber=FakeTranscriber("hi"),
        synthesizer=synth,
    )
    adapter._on_item(
        {"source_hash": "ab" * 16, "content": "", "audio": dict(OPUS_AUDIO)}
    )
    assert "ab" * 16 not in adapter._voice_chats
    assert adapter._voice_reply(_outbound("ab" * 16, "hello")) is None


def test_lxmf_voice_reply_falls_back_to_text_on_synth_failure():
    from semif_agent.gateway.lxmf import LxmfAdapter

    synth = FakeSynthesizer(error=RuntimeError("piper boom"))
    adapter = LxmfAdapter(
        {"allow_all_users": True, "voice": {"enabled": True}},
        transcriber=FakeTranscriber("hi"),
        synthesizer=synth,
    )
    adapter._on_item(
        {"source_hash": "ab" * 16, "content": "", "audio": dict(OPUS_AUDIO)}
    )
    assert adapter._voice_reply(_outbound("ab" * 16, "hello")) is None


def test_lxmf_voice_reply_bounds_length(monkeypatch):
    import semif_agent.gateway.lxmf as lxmf
    from semif_agent.gateway.lxmf import LxmfAdapter

    monkeypatch.setattr(lxmf, "pcm_to_opus_ogg", lambda pcm, rate, bitrate=0: b"OggS")
    synth = FakeSynthesizer()
    adapter = LxmfAdapter(
        {"allow_all_users": True, "voice": {"enabled": True, "max_speak_chars": 10}},
        transcriber=FakeTranscriber("hi"),
        synthesizer=synth,
    )
    adapter._on_item(
        {"source_hash": "ab" * 16, "content": "", "audio": dict(OPUS_AUDIO)}
    )
    adapter._voice_reply(_outbound("ab" * 16, "one two three four five six"))
    assert len(synth.calls[0]) <= 10


# ---- service: multiple platforms ----

class FakeLxmfAdapter:
    name = "lxmf"


def build_multi_service(scheduler):
    return GatewayService(
        scheduler, [FakeAdapter(), FakeLxmfAdapter()], config={}
    )


def test_service_refuses_other_platform_during_pending(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_multi_service(scheduler)
    scheduler._run_skill(need_input_skill([]), Request("track it", source="simplex:4"))
    question = scheduler.pending.question

    service.handle_inbound(
        InboundMessage(text="hello", chat_id="abc", contact_id="abc"), platform="lxmf"
    )
    assert scheduler.pending is not None
    assert scheduler.pending.question == question
    assert any("another conversation" in text for text in drain_outbound(service))


def test_service_reply_routes_to_owning_platform(tmp_path):
    scheduler = build_scheduler(tmp_path)
    service = build_multi_service(scheduler)
    service.handle_inbound(
        InboundMessage(text="hi", chat_id="abc", contact_id="abc"), platform="lxmf"
    )
    assert not service.outbound["lxmf"].empty()
    assert service.outbound["simplex"].empty()
