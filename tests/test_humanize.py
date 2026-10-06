"""Pure-stdlib tests for the gateway's optional LLM result cleanup.

The `Humanizer` is a thin, always-falls-back rephraser over an injected client;
the gateway wires it to the scheduler's `llm` client and only for result lines.
No engine and no real LLM endpoint are involved — a fake client records the
call and returns a canned reply.
"""

from semif_agent.gateway.humanize import Humanizer, _clean_reply
from semif_agent.gateway.service import GatewayService


class FakeClient:
    def __init__(self, reply="", error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def chat(self, messages, max_tokens=None, timeout=None, **kwargs):
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens, "timeout": timeout}
        )
        if self.error is not None:
            raise self.error
        return self.reply


class FakeAdapter:
    name = "simplex"


class FakeScheduler:
    def __init__(self, llm):
        self.llm = llm


# ---- Humanizer: rewriting + fallback ----

def test_humanize_rewrites_and_passes_timeout():
    client = FakeClient(reply="Your event was added to the calendar.")
    h = Humanizer(client, timeout=7.0, max_tokens=64)
    assert h.humanize("PUT 201 created") == "Your event was added to the calendar."
    call = client.calls[0]
    assert call["timeout"] == 7.0
    assert call["max_tokens"] == 64
    assert "PUT 201 created" in call["messages"][1]["content"]


def test_humanize_collapses_whitespace_and_strips_quotes():
    client = FakeClient(reply='"Done.\n  All good."')
    assert Humanizer(client).humanize("ok") == "Done. All good."


def test_humanize_falls_back_on_client_error():
    client = FakeClient(error=RuntimeError("unreachable"))
    assert Humanizer(client).humanize("raw result") == "raw result"


def test_humanize_falls_back_on_empty_reply():
    client = FakeClient(reply="   ")
    assert Humanizer(client).humanize("raw result") == "raw result"


def test_humanize_falls_back_on_refusal():
    client = FakeClient(reply="I cannot rewrite that.")
    assert Humanizer(client).humanize("raw result") == "raw result"


def test_humanize_falls_back_on_runaway_length():
    client = FakeClient(reply="x" * 1000)
    assert Humanizer(client, max_chars=100).humanize("raw") == "raw"


def test_humanize_disabled_is_passthrough_without_calling():
    client = FakeClient(reply="should not be used")
    h = Humanizer(client, enabled=False)
    assert h.humanize("raw result") == "raw result"
    assert client.calls == []


def test_humanize_no_client_is_disabled():
    h = Humanizer(None)
    assert h.enabled is False
    assert h.humanize("raw result") == "raw result"


def test_clean_reply_handles_empty():
    assert _clean_reply("") == ""
    assert _clean_reply(None) == ""


# ---- gateway wiring ----

def build_service(llm, config):
    return GatewayService(FakeScheduler(llm), FakeAdapter(), config=config)


def test_gateway_humanizes_only_result_lines():
    llm = FakeClient(reply="It is 12:00.")
    service = build_service(llm, {"humanize": {"enabled": True}, "simplex": {}})
    service._reply("simplex", "c", "time.now: ok — 12:00", kind="result")
    service._reply("simplex", "c", "queued", kind="status")
    service._reply("simplex", "c", "Which account?", kind="question")
    assert [service.outbound["simplex"].get_nowait().text for _ in range(3)] == [
        "It is 12:00.",
        "queued",
        "Which account?",
    ]


def test_gateway_humanize_off_by_default():
    llm = FakeClient(reply="should not be used")
    service = build_service(llm, {"simplex": {}})
    service._reply("simplex", "c", "time.now: ok — 12:00", kind="result")
    assert service.outbound["simplex"].get_nowait().text == "time.now: ok — 12:00"
    assert llm.calls == []


def test_gateway_per_platform_override_enables():
    llm = FakeClient(reply="It is 12:00.")
    service = build_service(
        llm, {"humanize": {"enabled": False}, "simplex": {"humanize": True}}
    )
    service._reply("simplex", "c", "time.now: ok — 12:00", kind="result")
    assert service.outbound["simplex"].get_nowait().text == "It is 12:00."


def test_gateway_per_platform_override_disables():
    llm = FakeClient(reply="should not be used")
    service = build_service(
        llm, {"humanize": {"enabled": True}, "simplex": {"humanize": False}}
    )
    service._reply("simplex", "c", "time.now: ok — 12:00", kind="result")
    assert service.outbound["simplex"].get_nowait().text == "time.now: ok — 12:00"
    assert llm.calls == []


def test_gateway_humanize_keeps_wrapper_when_result_only_off():
    # Humanize unwraps the scheduler wrapper, like result_only, but only for
    # the result line — status chatter still flows on a plain chat platform.
    llm = FakeClient(reply="Done.")
    service = build_service(llm, {"humanize": {"enabled": True}, "simplex": {}})
    service._reply("simplex", "c", "track.manual: ok — sent", kind="result")
    assert service.outbound["simplex"].get_nowait().text == "Done."
