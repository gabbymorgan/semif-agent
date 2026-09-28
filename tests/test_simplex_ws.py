"""Pure-stdlib tests for the neutral SimpleX daemon protocol layer.

`semif_agent.simplex_ws` is shared by the command gateway and the forwarding
bridge, so its parsing/accept/address behavior is tested here, independent of
either front end. The real `websockets` transport against a live daemon is a
jarvis integration concern.
"""

import asyncio
import json

import pytest

from semif_agent.simplex_ws import (
    SimplexDaemon,
    accept_command,
    contact_link,
    parse_direct_text_item,
    send_text_command,
)


# ---- parsing ----

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


def test_parse_direct_text():
    message = parse_direct_text_item(direct_item())
    assert message is not None
    assert message["text"] == "hello"
    assert message["contact_id"] == "4"
    assert message["display_name"] == "alice"


def test_parse_filters_echo_group_and_non_text():
    assert parse_direct_text_item(direct_item(direction="directSnd")) is None
    assert parse_direct_text_item(direct_item(kind="image")) is None
    group = direct_item()
    group["chatInfo"]["type"] = "group"
    assert parse_direct_text_item(group) is None


def test_parse_is_policy_free():
    # No allowlist lives here: the bridge buffers everyone; the gateway filters.
    assert parse_direct_text_item(direct_item(contact_id="999", display="ghost")) is not None


def test_parse_real_v7_payload_with_suffixed_local_name():
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
    message = parse_direct_text_item(item)
    assert message is not None
    assert message["text"] == "what is the next thing on my calendar?"
    assert message["contact_id"] == "6"
    assert message["display_name"] == "pepper"


def test_parse_accepts_aeson_nested_content_shape():
    item = direct_item()
    item["chatItem"]["content"] = {
        "rcvMsgContent": {"msgContent": {"type": "text", "text": "hi"}}
    }
    message = parse_direct_text_item(item)
    assert message is not None
    assert message["text"] == "hi"


# ---- send / accept commands ----

def test_send_command_is_structured_not_shortcut():
    command = send_text_command("4", "hello")
    assert command.startswith("/_send @4 json ")
    payload = json.loads(command.split(" json ", 1)[1])
    assert payload[0]["msgContent"]["type"] == "text"
    assert payload[0]["msgContent"]["text"] == "hello"
    assert command != "/_send @4 hello"


def test_accept_command_uses_v7_request_id():
    assert accept_command({"contactRequest": {"contactRequestId": 7, "contactId_": 9}}) == "/_accept 7"
    assert accept_command({"contactRequest": {"contactId": 4}}) == "/_accept 4"
    assert accept_command({}) is None


def test_contact_link_reads_both_shapes():
    nested = {"contactLink": {"connLinkContact": {"connShortLink": "simplex:/a"}}}
    flat = {"connLinkContact": {"connFullLink": "https://a"}}
    direct = {"connShortLink": "simplex:/b"}
    assert contact_link(nested)["connShortLink"] == "simplex:/a"
    assert contact_link(flat)["connFullLink"] == "https://a"
    assert contact_link(direct)["connShortLink"] == "simplex:/b"
    assert contact_link({}) == {}


# ---- daemon: accept + round-trips ----

class FakeWS:
    """A fake simplex-chat socket that answers correlated commands."""

    def __init__(self, daemon, responses):
        self.daemon = daemon
        self.responses = list(responses)
        self.sent = []

    async def send(self, raw):
        message = json.loads(raw)
        self.sent.append(message)
        if self.responses:
            response = self.responses.pop(0)
            await self.daemon._consume(
                json.dumps({"corrId": message["corrId"], "resp": response})
            )


def _contact_request_event(req):
    return json.dumps(
        {"resp": {"type": "receivedContactRequest", "contactRequest": req}}
    )


def test_accept_uses_v7_contact_request_id():
    daemon = SimplexDaemon("ws://x", auto_accept=True)
    ws = FakeWS(daemon, [])
    daemon._ws = ws
    asyncio.run(daemon._consume(_contact_request_event({"contactRequestId": 7, "contactId_": 9})))
    assert [m["cmd"] for m in ws.sent] == ["/_accept 7"]


def test_accept_falls_back_to_contact_id_for_legacy_events():
    daemon = SimplexDaemon("ws://x", auto_accept=True)
    ws = FakeWS(daemon, [])
    daemon._ws = ws
    asyncio.run(daemon._consume(_contact_request_event({"contactId": 4})))
    assert [m["cmd"] for m in ws.sent] == ["/_accept 4"]


def test_accept_disabled_sends_nothing():
    daemon = SimplexDaemon("ws://x", auto_accept=False)
    ws = FakeWS(daemon, [])
    daemon._ws = ws
    asyncio.run(daemon._consume(_contact_request_event({"contactRequestId": 7})))
    assert ws.sent == []


def test_new_chat_items_reach_the_handler():
    daemon = SimplexDaemon("ws://x")
    seen = []
    daemon._on_message = seen.append
    event = json.dumps(
        {"resp": {"type": "newChatItems", "chatItems": [direct_item(text="hi")]}}
    )
    asyncio.run(daemon._consume(event))
    assert [m["text"] for m in seen] == ["hi"]


def test_correlated_response_resolves_pending_future():
    daemon = SimplexDaemon("ws://x")

    async def scenario():
        daemon._loop = asyncio.get_running_loop()
        daemon._ws = FakeWS(daemon, [])
        future = daemon._loop.create_future()
        daemon._pending["req-9"] = future
        await daemon._consume(
            json.dumps({"corrId": "req-9", "resp": {"type": "ok", "value": 1}})
        )
        assert "req-9" not in daemon._pending, "a resolved request must be dropped"
        return future.result()

    assert asyncio.run(scenario()) == {"type": "ok", "value": 1}


def test_address_returns_existing_link():
    daemon = SimplexDaemon("ws://x", user_id=3)

    async def scenario():
        daemon._loop = asyncio.get_running_loop()
        ws = FakeWS(
            daemon,
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
        daemon._ws = ws
        return await daemon.address(timeout=5), ws

    result, ws = asyncio.run(scenario())
    assert result == {"short_link": "simplex:/a", "full_link": "https://a", "created": False}
    assert [m["cmd"] for m in ws.sent] == ["/_show_address 3"]


def test_address_creates_when_missing():
    daemon = SimplexDaemon("ws://x", user_id=1)

    async def scenario():
        daemon._loop = asyncio.get_running_loop()
        ws = FakeWS(
            daemon,
            [
                {"type": "userContactLink", "contactLink": None},
                {
                    "type": "userContactLinkCreated",
                    "connLinkContact": {"connShortLink": "simplex:/new", "connFullLink": ""},
                },
            ],
        )
        daemon._ws = ws
        return await daemon.address(timeout=5), ws

    result, ws = asyncio.run(scenario())
    assert result == {"short_link": "simplex:/new", "full_link": "", "created": True}
    assert [m["cmd"] for m in ws.sent] == ["/_show_address 1", "/_address 1"]


def test_request_address_without_connection_raises():
    daemon = SimplexDaemon("ws://x")
    with pytest.raises(RuntimeError):
        daemon.request_address(timeout=1)
