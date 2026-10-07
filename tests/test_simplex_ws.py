"""Pure-stdlib tests for the neutral SimpleX daemon protocol layer.

`semif_agent.simplex_ws` is shared by the command gateway and the forwarding
bridge, so its parsing/accept/address behavior is tested here, independent of
either front end. The real `websockets` transport against a live daemon is a
live-daemon integration concern.
"""

import asyncio
import json

import pytest

from semif_agent.simplex_ws import (
    SimplexDaemon,
    accept_command,
    contact_entry,
    contact_link,
    parse_chat,
    parse_chat_item,
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


def test_contact_entry_reads_v7_shape():
    assert contact_entry(
        {"contactId": 4, "displayName": None, "profile": {"displayName": "Alice"}}
    ) == {"id": "4", "display_name": "Alice", "local_name": "Alice"}
    assert contact_entry({"contactId": 7, "localDisplayName": "bob_1"}) == {
        "id": "7",
        "display_name": "bob_1",
        "local_name": "bob_1",
    }
    assert contact_entry({"chatId": 9, "profile": {"displayName": "Carol"}}) == {
        "id": "9",
        "display_name": "Carol",
        "local_name": "Carol",
    }
    assert contact_entry({"profile": {"displayName": "no id"}}) is None


def test_contact_entry_keeps_unique_local_name_and_health():
    # Two peers share the profile name "pepper"; the daemon's local name is
    # unique, and the connection health tells a live peer from a stale one.
    healthy = contact_entry(
        {
            "contactId": 4,
            "localDisplayName": "pepper_1",
            "profile": {"displayName": "pepper"},
            "contactStatus": "active",
            "activeConn": {"connStatus": {"type": "ready"}, "authErrCounter": 0},
        }
    )
    assert healthy == {
        "id": "4",
        "display_name": "pepper",
        "local_name": "pepper_1",
        "connected": True,
        "auth_errors": 0,
    }
    stale = contact_entry(
        {
            "contactId": 3,
            "localDisplayName": "pepper",
            "profile": {"displayName": "pepper"},
            "contactStatus": "active",
            "activeConn": {"connStatus": {"type": "ready"}, "authErrCounter": 2},
        }
    )
    assert stale["local_name"] == "pepper"
    assert stale["connected"] is True
    assert stale["auth_errors"] == 2


def test_contact_entry_marks_a_non_ready_connection_disconnected():
    entry = contact_entry(
        {
            "contactId": 4,
            "profile": {"displayName": "Alice"},
            "contactStatus": "active",
            "activeConn": {"connStatus": {"type": "connecting"}},
        }
    )
    assert entry["connected"] is False
    assert "auth_errors" not in entry


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


# ---- daemon: contacts + on_connected hook ----

def test_contacts_parses_contacts_list():
    daemon = SimplexDaemon("ws://x", user_id=1)

    async def scenario():
        daemon._loop = asyncio.get_running_loop()
        ws = FakeWS(
            daemon,
            [
                {"type": "activeUser", "user": {"userId": 3}},
                {
                    "type": "contactsList",
                    "contacts": [
                        {"contactId": 4, "profile": {"displayName": "Alice"}},
                        {"contactId": 7, "localDisplayName": "bob_1"},
                        {"profile": {"displayName": "no id"}},
                    ],
                },
            ],
        )
        daemon._ws = ws
        return await daemon.contacts(timeout=5), ws

    result, ws = asyncio.run(scenario())
    assert result == [
        {"id": "4", "display_name": "Alice", "local_name": "Alice"},
        {"id": "7", "display_name": "bob_1", "local_name": "bob_1"},
    ]
    assert [m["cmd"] for m in ws.sent] == ["/user", "/_contacts 3"]


def test_contacts_falls_back_to_configured_user_id():
    daemon = SimplexDaemon("ws://x", user_id=1)

    async def scenario():
        daemon._loop = asyncio.get_running_loop()
        ws = FakeWS(
            daemon,
            [
                {"type": "chatCmdError", "chatError": {"type": "error"}},
                {"type": "contactsList", "contacts": []},
            ],
        )
        daemon._ws = ws
        return await daemon.contacts(timeout=5), ws

    result, ws = asyncio.run(scenario())
    assert result == []
    assert [m["cmd"] for m in ws.sent] == ["/user", "/_contacts 1"]


def test_active_user_is_cached_across_calls():
    daemon = SimplexDaemon("ws://x", user_id=1)

    async def scenario():
        daemon._loop = asyncio.get_running_loop()
        ws = FakeWS(
            daemon,
            [
                {"type": "activeUser", "user": {"userId": 3}},
                {"type": "contactsList", "contacts": []},
                {"type": "contactsList", "contacts": []},
            ],
        )
        daemon._ws = ws
        await daemon.contacts(timeout=5)
        await daemon.contacts(timeout=5)
        return ws

    ws = asyncio.run(scenario())
    assert [m["cmd"] for m in ws.sent] == ["/user", "/_contacts 3", "/_contacts 3"]


def test_contacts_returns_empty_on_error_response():
    daemon = SimplexDaemon("ws://x", user_id=1)

    async def scenario():
        daemon._loop = asyncio.get_running_loop()
        ws = FakeWS(
            daemon,
            [
                {"type": "activeUser", "user": {"userId": 1}},
                {"type": "chatCmdError", "chatError": {"type": "error"}},
            ],
        )
        daemon._ws = ws
        return await daemon.contacts(timeout=5)

    assert asyncio.run(scenario()) == []


def test_request_contacts_without_connection_raises():
    daemon = SimplexDaemon("ws://x")
    with pytest.raises(RuntimeError):
        daemon.request_contacts(timeout=1)


def test_on_connected_hook_runs_and_failures_are_swallowed():
    calls = []

    async def hook():
        calls.append(1)

    asyncio.run(SimplexDaemon("ws://x", on_connected=hook)._notify_connected())
    assert calls == [1]

    async def bad():
        raise ValueError("nope")

    # A hook failure must not propagate (it would tear down the connection).
    asyncio.run(SimplexDaemon("ws://x", on_connected=bad)._notify_connected())
    asyncio.run(SimplexDaemon("ws://x")._notify_connected())


def test_on_connected_hook_may_roundtrip_while_reading():
    """The hook runs concurrently with the read loop, not blocking it.

    Regression: `on_connected` was awaited inline before the read loop started,
    so a hook that issued a correlated command (the gateway's startup contact
    link) could never receive its own response and always timed out.
    """
    daemon = SimplexDaemon("ws://x", user_id=1)
    seen = []

    async def hook():
        link = await daemon.address(timeout=2)
        seen.append(link)

    daemon.on_connected = hook
    daemon._loop = asyncio.new_event_loop()

    async def scenario():
        daemon._loop = asyncio.get_running_loop()
        ws = FakeWS(
            daemon,
            [
                {
                    "type": "userContactLink",
                    "contactLink": {
                        "connLinkContact": {"connShortLink": "simplex:/a", "connFullLink": ""}
                    },
                }
            ],
        )
        daemon._ws = ws
        hook_task = asyncio.create_task(daemon._notify_connected())
        # The read loop is what resolves the correlated response.
        for message in list(ws.sent):
            pass
        await hook_task
        return ws

    ws = asyncio.run(scenario())
    assert seen == [{"short_link": "simplex:/a", "full_link": "", "created": False}]
    assert [m["cmd"] for m in ws.sent] == ["/_show_address 1"]


# ---- chat previews / history (true unread) ----

def chat_item(item_id=10, text="hello", status="rcvNew", direction="directRcv", shape="api"):
    meta = {"itemId": item_id, "itemTs": "2026-01-01T00:00:00Z", "itemStatus": {"type": status}}
    if shape == "api":
        content = {"type": "rcvMsgContent", "msgContent": {"type": "text", "text": text}}
    else:
        content = {"rcvMsgContent": {"msgContent": {"type": "text", "text": text}}}
    return {"chatDir": {"type": direction}, "meta": meta, "content": content}


def api_chat(contact_id="4", display="Alice", unread_count=2, min_unread="10",
             unread_chat=True, items=None):
    return {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": contact_id, "profile": {"displayName": display}},
        },
        "chatStats": {
            "unreadCount": unread_count,
            "unreadMentions": 0,
            "reportsCount": 0,
            "minUnreadItemId": min_unread,
            "unreadChat": unread_chat,
        },
        "chatItems": items if items is not None else [chat_item()],
    }


def test_parse_chat_item_reads_status_and_text():
    parsed = parse_chat_item(chat_item(item_id=12, text="hi", status="rcvNew"))
    assert parsed == {
        "item_id": "12",
        "text": "hi",
        "status": "rcvNew",
        "unread": True,
        "direction": "directRcv",
        "sent_at": "2026-01-01T00:00:00Z",
    }
    read = parse_chat_item(chat_item(item_id=11, text="old", status="rcvRead"))
    assert read["unread"] is False


def test_parse_chat_item_accepts_aeson_shape_and_drops_non_text():
    assert parse_chat_item(chat_item(shape="aeson"))["text"] == "hello"
    assert parse_chat_item(chat_item(text="")) is None
    file_item = chat_item()
    file_item["content"] = {"type": "rcvFile", "file": {}}
    assert parse_chat_item(file_item) is None
    assert parse_chat_item({"meta": {}, "content": {}}) is None


def test_parse_chat_reads_stats_and_skips_non_direct():
    parsed = parse_chat(api_chat(contact_id="7", display="Bob", unread_count=3, min_unread="42"))
    assert parsed["contact_id"] == "7"
    assert parsed["display_name"] == "Bob"
    assert parsed["unread_count"] == 3
    assert parsed["min_unread_item_id"] == "42"
    assert parsed["unread"] is True
    assert [m["item_id"] for m in parsed["messages"]] == ["10"]

    group = api_chat()
    group["chatInfo"] = {"type": "group", "groupInfo": {}}
    assert parse_chat(group) is None


def test_parse_chat_unread_false_when_no_unread():
    parsed = parse_chat(api_chat(unread_count=0, min_unread=None, unread_chat=False))
    assert parsed["unread"] is False
    assert parsed["min_unread_item_id"] == ""


def test_chats_sends_unread_filter_and_parses_previews():
    daemon = SimplexDaemon("ws://x", user_id=1)

    async def scenario():
        daemon._loop = asyncio.get_running_loop()
        ws = FakeWS(
            daemon,
            [
                {"type": "activeUser", "user": {"userId": 3}},
                {"type": "apiChats", "chats": [api_chat()]},
            ],
        )
        daemon._ws = ws
        return await daemon.chats(unread_only=True, count=5, timeout=5), ws

    result, ws = asyncio.run(scenario())
    assert [m["cmd"] for m in ws.sent] == [
        "/user",
        '/_get chats 3 count=5 {"type": "filters", "favorite": false, "unread": true}',
    ]
    assert result[0]["contact_id"] == "4"
    assert result[0]["unread_count"] == 2


def test_chats_returns_empty_on_error_response():
    daemon = SimplexDaemon("ws://x", user_id=1)

    async def scenario():
        daemon._loop = asyncio.get_running_loop()
        ws = FakeWS(
            daemon,
            [
                {"type": "activeUser", "user": {"userId": 1}},
                {"type": "chatCmdError", "chatError": {"type": "error"}},
            ],
        )
        daemon._ws = ws
        return await daemon.chats(timeout=5)

    assert asyncio.run(scenario()) == []


def test_chat_history_uses_chat_ref_and_parses_items():
    daemon = SimplexDaemon("ws://x")

    async def scenario():
        daemon._loop = asyncio.get_running_loop()
        ws = FakeWS(
            daemon,
            [
                {
                    "type": "apiChat",
                    "chat": api_chat(
                        items=[
                            chat_item(item_id=10, text="first", status="rcvRead"),
                            chat_item(item_id=11, text="second", status="rcvNew"),
                        ]
                    ),
                }
            ],
        )
        daemon._ws = ws
        return await daemon.chat_history("@7", count=9, timeout=5), ws

    result, ws = asyncio.run(scenario())
    assert [m["cmd"] for m in ws.sent] == ["/_get chat @7 count=9"]
    assert [(m["item_id"], m["unread"]) for m in result] == [("10", False), ("11", True)]


def test_chat_history_returns_empty_on_error_response():
    daemon = SimplexDaemon("ws://x")

    async def scenario():
        daemon._loop = asyncio.get_running_loop()
        ws = FakeWS(daemon, [{"type": "chatCmdError", "chatError": {"type": "error"}}])
        daemon._ws = ws
        return await daemon.chat_history("4", timeout=5)

    assert asyncio.run(scenario()) == []


def test_request_chats_without_connection_raises():
    daemon = SimplexDaemon("ws://x")
    with pytest.raises(RuntimeError):
        daemon.request_chats(timeout=1)


def test_request_chat_history_without_connection_raises():
    daemon = SimplexDaemon("ws://x")
    with pytest.raises(RuntimeError):
        daemon.request_chat_history("4", timeout=1)
