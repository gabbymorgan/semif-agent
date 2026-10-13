"""Hermetic mechanics test for simplex.next_message.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the standalone SimpleX forwarding bridge. This proves the body
peeks the inbox, resolves the conversation through a SemIf sub-decision (a
strong winner is used, otherwise the configured default, otherwise the weak
winner), pops the real `next` message, and fails honestly when the bridge is
unreachable. It does NOT prove the live integration — only a real run against
the bridge and a real contact does.
"""

import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_agent.decisions import DecisionResult, Request
from semif_agent.skills import ActionContext

import skill


class FakeEngine:
    """Test-only engine.

    By default it picks the requested option id (else the first) with full
    confidence. Pass `probs` (a mapping of option id -> probability) to script an
    arbitrary distribution, e.g. a near-tie that must fall back to the default.
    """

    def __init__(self, pick=None, probs=None):
        self.pick = pick
        self.probs = probs
        self.decisions = []

    def call(self, decision):
        self.decisions.append(decision)
        option_ids = [option.id for option in decision.options]
        if self.probs is not None:
            probabilities = [float(self.probs.get(option_id, 0.0)) for option_id in option_ids]
        else:
            chosen = self.pick if self.pick in option_ids else option_ids[0]
            probabilities = [
                1.0 if option_id == chosen else 0.0 for option_id in option_ids
            ]
        return DecisionResult(
            request=decision, option_ids=option_ids, probabilities=probabilities
        )


class BridgeHandler(BaseHTTPRequestHandler):
    messages = []
    unread_chats = []
    history = {}
    requests = []
    tokens = []
    read_calls = []
    read_ok = True

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.requests.append(self.path)
        self.tokens.append(self.headers.get("X-Semif-Token"))
        url = urllib.parse.urlparse(self.path)
        if url.path == "/inbox":
            self._send({"messages": list(self.messages)})
        elif url.path == "/inbox/next":
            contact = (urllib.parse.parse_qs(url.query).get("contact") or [None])[0]
            found = None
            for index, message in enumerate(self.messages):
                if contact is None or message["contact_id"] == str(contact):
                    found = self.messages.pop(index)
                    break
            self._send({"message": found})
        elif url.path == "/unread":
            self._send({"chats": list(self.unread_chats)})
        elif url.path == "/history":
            contact = (urllib.parse.parse_qs(url.query).get("contact") or [None])[0]
            self._send(
                {"contact_id": contact, "messages": list(self.history.get(str(contact), []))}
            )
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if url.path != "/read":
            self._send({"error": "not found"}, 404)
            return
        self.read_calls.append(payload)
        if self.read_ok:
            self._send(
                {
                    "ok": True,
                    "contact_id": payload.get("contact"),
                    "item_ids": payload.get("item_ids"),
                    "status": "read",
                }
            )
        else:
            self._send({"ok": False, "error": "the daemon rejected the read"})

    def log_message(self, *args):
        pass


def start_server(messages, unread_chats=None, history=None, read_ok=True):
    handler = type(
        "Handler",
        (BridgeHandler,),
        {
            "messages": list(messages),
            "unread_chats": list(unread_chats or []),
            "history": dict(history or {}),
            "requests": [],
            "tokens": [],
            "read_calls": [],
            "read_ok": read_ok,
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler, f"http://127.0.0.1:{server.server_address[1]}"


def message(identifier, contact, name, text):
    return {
        "id": identifier,
        "item_id": str(identifier),
        "contact_id": contact,
        "display_name": name,
        "text": text,
        "received_at": 1.0,
    }


def config(url, default="", token=""):
    return {
        "simplex_bridge_url": url,
        "simplex_default_contact": default,
        "simplex_bridge_token": token,
    }


def test_multiple_senders_uses_semif_decision():
    server, handler, url = start_server(
        [
            message("m1", "4", "Alice", "hello there"),
            message("m2", "7", "Bob", "shipment arrived"),
        ]
    )
    try:
        engine = FakeEngine(pick="7")
        ctx = ActionContext(engine=engine, config=config(url))
        request = Request("read my next simplex message")
        action = skill.act(ctx, request)
        assert len(action.decisions) == 1, "act must log the conversation choice"
        assert len(engine.decisions) == 1
        assert "Bob" in action.action_log, action.action_log
        assert "shipment arrived" in action.action_log, action.action_log
        assert action.new_state, "act must set a new state"

        assert handler.requests[0] == "/inbox", handler.requests
        assert handler.requests[1] == "/inbox/next?contact=7", handler.requests
        assert [m["id"] for m in handler.messages] == ["m1"], "the popped message must be consumed"
        assert handler.read_calls == [{"contact": "7", "item_ids": ["m2"]}], (
            "the reported item must be marked read, and only it"
        )
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_strong_winner_overrides_default_contact():
    server, handler, url = start_server(
        [
            message("m1", "4", "Alice", "hello there"),
            message("m2", "7", "Bob", "shipment arrived"),
        ]
    )
    try:
        engine = FakeEngine(probs={"4": 0.1, "7": 0.9})
        ctx = ActionContext(engine=engine, config=config(url, default="Alice"))
        request = Request("read my next simplex message from Bob")
        action = skill.act(ctx, request)
        assert len(engine.decisions) == 1, "the conversation is chosen with one SemIf decision"
        assert handler.requests[1] == "/inbox/next?contact=7", handler.requests
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_weak_winner_falls_back_to_default_contact():
    server, handler, url = start_server(
        [
            message("m1", "4", "Alice", "hello there"),
            message("m2", "7", "Bob", "shipment arrived"),
        ]
    )
    try:
        # Bob edges out Alice but stays below the strong-winner threshold, so the
        # configured default ("Alice") is used instead.
        engine = FakeEngine(probs={"4": 0.45, "7": 0.5})
        ctx = ActionContext(engine=engine, config=config(url, default="Alice"))
        request = Request("read my next simplex message")
        action = skill.act(ctx, request)
        assert len(engine.decisions) == 1
        assert handler.requests[1] == "/inbox/next?contact=4", handler.requests
        assert "hello there" in action.action_log, action.action_log
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_weak_winner_without_default_stands():
    server, handler, url = start_server(
        [
            message("m1", "4", "Alice", "hello there"),
            message("m2", "7", "Bob", "shipment arrived"),
        ]
    )
    try:
        engine = FakeEngine(probs={"4": 0.45, "7": 0.5})
        ctx = ActionContext(engine=engine, config=config(url))
        request = Request("read my next simplex message")
        action = skill.act(ctx, request)
        assert len(engine.decisions) == 1
        assert handler.requests[1] == "/inbox/next?contact=7", handler.requests
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_single_sender_needs_no_decision():
    server, handler, url = start_server([message("m1", "4", "Alice", "hi")])
    try:
        engine = FakeEngine()
        ctx = ActionContext(engine=engine, config=config(url))
        request = Request("read my next simplex message")
        action = skill.act(ctx, request)
        assert engine.decisions == []
        assert "hi" in action.action_log, action.action_log
        assert handler.read_calls == [{"contact": "4", "item_ids": ["m1"]}]
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_empty_inbox_is_reported_honestly():
    server, handler, url = start_server([])
    try:
        engine = FakeEngine()
        ctx = ActionContext(engine=engine, config=config(url))
        request = Request("read my next simplex message")
        action = skill.act(ctx, request)
        assert "no unread" in action.action_log, action.action_log
        # An empty live buffer falls back to the daemon's persistent unread.
        assert handler.requests == ["/inbox", "/unread"], handler.requests
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_empty_live_inbox_falls_back_to_persistent_unread():
    server, handler, url = start_server(
        [],
        unread_chats=[
            {
                "contact_id": "3",
                "display_name": "pepper",
                "unread_count": 2,
                "min_unread_item_id": "8",
                "unread": True,
                "messages": [],
            }
        ],
        history={
            "3": [
                {"item_id": "8", "text": "hello", "status": "rcvNew", "unread": True},
                {"item_id": "9", "text": "hey yo", "status": "rcvNew", "unread": True},
            ]
        },
    )
    try:
        engine = FakeEngine()
        ctx = ActionContext(engine=engine, config=config(url))
        request = Request("read my next simplex message")
        action = skill.act(ctx, request)
        assert engine.decisions == [], "a single unread chat needs no decision"
        assert "hello" in action.action_log, action.action_log
        assert "pepper" in action.action_log, action.action_log
        assert handler.requests == ["/inbox", "/unread", "/history?contact=3&count=50"], (
            handler.requests
        )
        assert handler.read_calls == [{"contact": "3", "item_ids": ["8"]}]
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_persistent_unread_prefers_a_healthy_conversation():
    server, handler, url = start_server(
        [],
        unread_chats=[
            {
                "contact_id": "3",
                "display_name": "pepper",
                "unread_count": 1,
                "unread": True,
                "connected": True,
                "auth_errors": 2,
                "messages": [],
            },
            {
                "contact_id": "4",
                "display_name": "pepper",
                "unread_count": 1,
                "unread": True,
                "connected": True,
                "auth_errors": 0,
                "messages": [],
            },
        ],
        history={
            "3": [{"item_id": "8", "text": "stale hello", "status": "rcvNew", "unread": True}],
            "4": [{"item_id": "20", "text": "fresh hello", "status": "rcvNew", "unread": True}],
        },
    )
    try:
        engine = FakeEngine()
        ctx = ActionContext(engine=engine, config=config(url))
        action = skill.act(ctx, Request("read my next simplex message"))
        assert engine.decisions == [], "one healthy chat must win without a decision"
        assert "fresh hello" in action.action_log, action.action_log
        assert handler.requests == ["/inbox", "/unread", "/history?contact=4&count=50"], (
            handler.requests
        )
        assert handler.read_calls == [{"contact": "4", "item_ids": ["20"]}]
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_unreachable_bridge_fails_honestly():
    engine = FakeEngine()
    ctx = ActionContext(engine=engine, config=config("http://127.0.0.1:1"))
    request = Request("read my next simplex message")
    action = skill.act(ctx, request)
    assert action.action_log and "simplex.next_message" not in action.action_log, action.action_log
    assert action.new_state == request.text, "a failed read must not fake a result"
    print(action.action_log)


def test_auth_token_is_sent_when_configured():
    server, handler, url = start_server([message("m1", "4", "Alice", "hi")])
    try:
        engine = FakeEngine()
        ctx = ActionContext(engine=engine, config=config(url, token="sekret"))
        request = Request("read my next simplex message")
        action = skill.act(ctx, request)
        assert "hi" in action.action_log, action.action_log
        assert set(handler.tokens) == {"sekret"}, handler.tokens
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_failed_read_marking_is_surfaced_honestly():
    """A message is still reported when consuming it fails, with an honest note
    instead of a fabricated failure of the read."""
    server, handler, url = start_server(
        [message("m1", "4", "Alice", "hi")], read_ok=False
    )
    try:
        ctx = ActionContext(engine=FakeEngine(), config=config(url))
        action = skill.act(ctx, Request("read my next simplex message"))
        assert "hi" in action.action_log, action.action_log
        assert "could not mark it read" in action.action_log, action.action_log
        assert handler.read_calls == [{"contact": "4", "item_ids": ["m1"]}]
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def main():
    test_multiple_senders_uses_semif_decision()
    test_strong_winner_overrides_default_contact()
    test_weak_winner_falls_back_to_default_contact()
    test_weak_winner_without_default_stands()
    test_single_sender_needs_no_decision()
    test_empty_inbox_is_reported_honestly()
    test_empty_live_inbox_falls_back_to_persistent_unread()
    test_persistent_unread_prefers_a_healthy_conversation()
    test_unreachable_bridge_fails_honestly()
    test_auth_token_is_sent_when_configured()
    test_failed_read_marking_is_surfaced_honestly()
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
