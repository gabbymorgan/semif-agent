"""Pure-stdlib tests for the gateway's local messaging bridge.

The bridge is exercised over a real loopback HTTP server (no mocking of its
transport); the decision engine and LLM are never involved. These prove the
mechanics a skill depends on: peek/pop of buffered inbound messages, recipient
resolution, outbound routing onto the adapter queue, and token/body validation.
"""

import json
import queue
import urllib.error
import urllib.request

import pytest

from semif_agent.gateway.base import InboundMessage, OutboundMessage
from semif_agent.gateway.bridge import MessagingBridge


def start_bridge(**config):
    outbound = queue.Queue()
    bridge = MessagingBridge(outbound, config=config)
    port = bridge.start()
    return bridge, outbound, f"http://127.0.0.1:{port}"


def inbound(text, contact_id="4", display_name="Alice"):
    return InboundMessage(
        text=text, chat_id=contact_id, contact_id=contact_id, display_name=display_name
    )


def get(url, token=None):
    headers = {"X-Semif-Token": token} if token else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def post(url, payload, token=None):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Semif-Token"] = token
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_health_and_contacts():
    bridge, _, base = start_bridge()
    try:
        assert get(f"{base}/health") == {"ok": True, "platform": "simplex"}
        bridge.record_inbound(inbound("hi", contact_id="4", display_name="Alice"))
        assert get(f"{base}/contacts") == {
            "contacts": [{"id": "4", "display_name": "Alice"}]
        }
    finally:
        bridge.stop()


def test_inbox_peek_then_pop_consumes_oldest():
    bridge, _, base = start_bridge()
    try:
        bridge.record_inbound(inbound("first", contact_id="4", display_name="Alice"))
        bridge.record_inbound(inbound("second", contact_id="7", display_name="Bob"))
        peeked = get(f"{base}/inbox")["messages"]
        assert [m["text"] for m in peeked] == ["first", "second"]
        assert get(f"{base}/inbox")["messages"] == peeked, "peek must not consume"

        popped = get(f"{base}/inbox/next")["message"]
        assert popped["text"] == "first"
        assert popped["contact_id"] == "4"
        remaining = get(f"{base}/inbox")["messages"]
        assert [m["text"] for m in remaining] == ["second"]
    finally:
        bridge.stop()


def test_inbox_next_filters_by_contact():
    bridge, _, base = start_bridge()
    try:
        bridge.record_inbound(inbound("first", contact_id="4", display_name="Alice"))
        bridge.record_inbound(inbound("second", contact_id="7", display_name="Bob"))
        popped = get(f"{base}/inbox/next?contact=7")["message"]
        assert popped["text"] == "second"
        assert get(f"{base}/inbox/next?contact=7")["message"] is None
        assert [m["text"] for m in get(f"{base}/inbox")["messages"]] == ["first"]
    finally:
        bridge.stop()


def test_inbox_is_bounded():
    bridge, _, base = start_bridge(max_inbox=2)
    try:
        for text in ("one", "two", "three"):
            bridge.record_inbound(inbound(text))
        messages = get(f"{base}/inbox")["messages"]
        assert [m["text"] for m in messages] == ["two", "three"], "oldest must be dropped"
    finally:
        bridge.stop()


def test_send_routes_to_outbound_queue():
    bridge, outbound, base = start_bridge()
    try:
        result = post(f"{base}/send", {"recipient": "7", "text": "on my way"})
        assert result == {"ok": True, "contact_id": "7"}
        message = outbound.get_nowait()
        assert isinstance(message, OutboundMessage)
        assert message.chat_id == "7"
        assert message.text == "on my way"
    finally:
        bridge.stop()


def test_send_resolves_known_display_name():
    bridge, outbound, base = start_bridge()
    try:
        bridge.record_inbound(inbound("hi", contact_id="4", display_name="Alice"))
        result = post(f"{base}/send", {"recipient": "Alice", "text": "hello"})
        assert result["contact_id"] == "4"
        assert outbound.get_nowait().chat_id == "4"
    finally:
        bridge.stop()


def test_send_rejects_unknown_recipient():
    bridge, outbound, base = start_bridge()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(f"{base}/send", {"recipient": "ghost", "text": "hello"})
        assert exc.value.code == 400
        assert outbound.empty(), "a rejected send must not enqueue anything"
    finally:
        bridge.stop()


def test_send_validates_body():
    bridge, outbound, base = start_bridge()
    try:
        for payload in ({"recipient": "", "text": "hi"}, {"recipient": "4", "text": " "}):
            with pytest.raises(urllib.error.HTTPError) as exc:
                post(f"{base}/send", payload)
            assert exc.value.code == 400
        assert outbound.empty()
    finally:
        bridge.stop()


def test_token_is_required_when_configured():
    bridge, _, base = start_bridge(token="sekret")
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(f"{base}/inbox")
        assert exc.value.code == 401
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(f"{base}/send", {"recipient": "4", "text": "hi"})
        assert exc.value.code == 401
        assert get(f"{base}/inbox", token="sekret") == {"messages": []}
    finally:
        bridge.stop()
