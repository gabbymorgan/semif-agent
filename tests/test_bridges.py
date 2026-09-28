"""Pure-stdlib tests for the standalone bridge services.

The SimpleX bridge is exercised over a real loopback HTTP server (no mocking of
its HTTP transport) with a fake daemon standing in for the `simplex-chat`
connection — the decision engine and LLM are never involved. These prove the
mechanics a skill depends on: peek/pop of buffered inbound messages, recipient
resolution, outbound routing, address lookup, and token/body validation.
"""

import json
import urllib.error
import urllib.request

import pytest

from semif_agent.bridges.registry import describe_bridges, known_infos
from semif_agent.bridges.simplex import SimplexBridge


class FakeDaemon:
    """Stands in for the bridge's own simplex-chat connection."""

    def __init__(self, link=None, error=None):
        self.link = link
        self.error = error
        self.sent = []
        self.closed = False
        self.on_message = None

    def check_requirements(self):
        return True, None

    def run(self, on_message):
        self.on_message = on_message

    def enqueue(self, chat_id, text=""):
        if chat_id is not None:
            self.sent.append((chat_id, text))

    def request_address(self, timeout=20.0):
        if self.error is not None:
            raise self.error
        return self.link

    def close(self):
        self.closed = True


def start_bridge(daemon=None, **config):
    daemon = daemon or FakeDaemon()
    bridge = SimplexBridge(config, daemon=daemon)
    port = bridge.start()
    return bridge, daemon, f"http://127.0.0.1:{port}"


def inbound(text, contact_id="4", display_name="Alice"):
    return {"text": text, "contact_id": contact_id, "display_name": display_name}


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
        bridge._on_message(inbound("hi", contact_id="4", display_name="Alice"))
        assert get(f"{base}/contacts") == {
            "contacts": [{"id": "4", "display_name": "Alice"}]
        }
    finally:
        bridge.stop()


def test_inbox_peek_then_pop_consumes_oldest():
    bridge, _, base = start_bridge()
    try:
        bridge._on_message(inbound("first", contact_id="4", display_name="Alice"))
        bridge._on_message(inbound("second", contact_id="7", display_name="Bob"))
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
        bridge._on_message(inbound("first", contact_id="4", display_name="Alice"))
        bridge._on_message(inbound("second", contact_id="7", display_name="Bob"))
        popped = get(f"{base}/inbox/next?contact=7")["message"]
        assert popped["text"] == "second"
        assert get(f"{base}/inbox/next?contact=7")["message"] is None
        assert [m["text"] for m in get(f"{base}/inbox")["messages"]] == ["first"]
    finally:
        bridge.stop()


def test_inbox_buffers_any_sender():
    # No allowlist: a brand-new contact who reached the bot via its invite link
    # must be visible, which is the whole point of the forwarding bridge.
    bridge, _, base = start_bridge()
    try:
        bridge._on_message(inbound("who are you?", contact_id="99", display_name="stranger"))
        messages = get(f"{base}/inbox")["messages"]
        assert [m["text"] for m in messages] == ["who are you?"]
    finally:
        bridge.stop()


def test_inbox_is_bounded():
    bridge, _, base = start_bridge(max_inbox=2)
    try:
        for text in ("one", "two", "three"):
            bridge._on_message(inbound(text))
        messages = get(f"{base}/inbox")["messages"]
        assert [m["text"] for m in messages] == ["two", "three"], "oldest must be dropped"
    finally:
        bridge.stop()


def test_send_routes_to_daemon():
    bridge, daemon, base = start_bridge()
    try:
        result = post(f"{base}/send", {"recipient": "7", "text": "on my way"})
        assert result == {"ok": True, "contact_id": "7"}
        assert daemon.sent == [("7", "on my way")]
    finally:
        bridge.stop()


def test_send_resolves_known_display_name():
    bridge, daemon, base = start_bridge()
    try:
        bridge._on_message(inbound("hi", contact_id="4", display_name="Alice"))
        result = post(f"{base}/send", {"recipient": "Alice", "text": "hello"})
        assert result["contact_id"] == "4"
        assert daemon.sent == [("4", "hello")]
    finally:
        bridge.stop()


def test_send_rejects_unknown_recipient():
    bridge, daemon, base = start_bridge()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(f"{base}/send", {"recipient": "ghost", "text": "hello"})
        assert exc.value.code == 400
        assert daemon.sent == [], "a rejected send must not enqueue anything"
    finally:
        bridge.stop()


def test_send_validates_body():
    bridge, daemon, base = start_bridge()
    try:
        for payload in ({"recipient": "", "text": "hi"}, {"recipient": "4", "text": " "}):
            with pytest.raises(urllib.error.HTTPError) as exc:
                post(f"{base}/send", payload)
            assert exc.value.code == 400
        assert daemon.sent == []
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


def test_address_returns_daemon_link():
    link = {"short_link": "simplex:/abc", "full_link": "https://x", "created": False}
    bridge, _, base = start_bridge(daemon=FakeDaemon(link=link))
    try:
        assert get(f"{base}/address") == link
    finally:
        bridge.stop()


def test_address_without_connection_is_unavailable():
    bridge, _, base = start_bridge(
        daemon=FakeDaemon(error=RuntimeError("not connected"))
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(f"{base}/address")
        assert exc.value.code == 503
    finally:
        bridge.stop()


def test_address_lookup_failure_is_reported():
    bridge, _, base = start_bridge(daemon=FakeDaemon(error=TimeoutError("slow")))
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(f"{base}/address")
        assert exc.value.code == 502
    finally:
        bridge.stop()


def test_stop_closes_the_daemon():
    bridge, daemon, _ = start_bridge()
    bridge.stop()
    assert daemon.closed is True


# ---- catalog / codegen surface ----

def test_known_infos_include_simplex():
    infos = {info.name: info for info in known_infos()}
    assert "simplex" in infos
    assert infos["simplex"].url_config_var == "simplex_bridge_url"


def test_describe_bridges_names_the_service_and_config_var():
    text = describe_bridges()
    assert "simplex" in text
    assert "simplex_bridge_url" in text
    assert "/inbox/next" in text
    assert "never speak" in text
