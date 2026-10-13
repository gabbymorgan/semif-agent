"""Hermetic mechanics test for outlook.send_message.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the local LLM bridge (`POST /chat`), the Outlook bridge's
contact lookup, and `POST /mail/send`. This proves the body composes a draft,
resolves the recipient, asks for confirmation BEFORE sending, sends for real on
`yes`, cancels otherwise, and fails honestly when the LLM bridge is down. It does
NOT prove the live integration — only a real run against the user's Microsoft
account does.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from semif_agent.decisions import DecisionResult, Request
from semif_agent.skills import ActionContext

import skill


class FakeEngine:
    """Picks `pick` (else the first option) with full confidence."""

    def __init__(self, pick=None):
        self.pick = pick
        self.decisions = []

    def call(self, decision):
        self.decisions.append(decision)
        ids = [option.id for option in decision.options]
        chosen = self.pick if self.pick in ids else ids[0]
        probs = [1.0 if option_id == chosen else 0.0 for option_id in ids]
        return DecisionResult(request=decision, option_ids=ids, probabilities=probs)


class BridgeHandler(BaseHTTPRequestHandler):
    draft = {"recipient": "Sam", "subject": "Lunch", "body": "Tomorrow?"}
    contacts = [{"id": "c1", "display_name": "Sam", "emails": ["sam@example.com"]}]
    llm_status = 200
    sends = []
    queries = []

    def _read(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def do_POST(self):
        body = self._read()
        if self.path == "/chat":
            if self.llm_status != 200:
                self._send({"error": "boom"}, self.llm_status)
                return
            self._send({"text": json.dumps(self.draft)}, 200)
            return
        if self.path == "/mail/send":
            self.sends.append(json.loads(body.decode("utf-8")))
            self._send({"ok": True, "to": self.sends[-1].get("to")}, 200)
            return
        self._send({"error": "not found"}, 404)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/contacts":
            self.queries.append(parse_qs(parsed.query).get("query", [""])[0])
            self._send({"contacts": self.contacts}, 200)
            return
        self._send({"error": "not found"}, 404)

    def _send(self, payload, status):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_server(**overrides):
    attrs = {
        "draft": {"recipient": "Sam", "subject": "Lunch", "body": "Tomorrow?"},
        "contacts": [{"id": "c1", "display_name": "Sam", "emails": ["sam@example.com"]}],
        "llm_status": 200,
        "sends": [],
        "queries": [],
    }
    attrs.update(overrides)
    handler = type("Handler", (BridgeHandler,), attrs)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler, f"http://127.0.0.1:{server.server_address[1]}"


def context(url, engine):
    return ActionContext(
        engine=engine,
        config={
            "outlook_bridge_url": url,
            "outlook_bridge_token": "",
            "llm_bridge_url": url,
            "llm_bridge_token": "",
        },
    )


def test_confirms_then_sends():
    server, handler, url = start_server()
    try:
        ctx = context(url, FakeEngine())
        request = Request("email Sam that lunch is tomorrow")
        action = skill.act(ctx, request)
        assert action.needs_input and "send" in action.needs_input.lower(), action.needs_input
        assert handler.sends == [], "nothing may be sent before confirmation"
        request.user_input = "yes"
        action = skill.act(ctx, request)
        assert action.needs_input is None, action.needs_input
        assert handler.sends[0]["to"] == ["sam@example.com"], handler.sends
        assert handler.sends[0]["subject"] == "Lunch", handler.sends
        assert "sent" in action.action_log.lower(), action.action_log
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_cancel_does_not_send():
    server, handler, url = start_server()
    try:
        ctx = context(url, FakeEngine())
        request = Request("email Sam that lunch is tomorrow")
        skill.act(ctx, request)
        request.user_input = "no"
        action = skill.act(ctx, request)
        assert handler.sends == [], "a cancelled message must not be sent"
        assert "cancel" in action.action_log.lower(), action.action_log
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_multiple_contacts_use_semif_choice():
    contacts = [
        {"id": "c1", "display_name": "Sam A", "emails": ["sam.a@example.com"]},
        {"id": "c2", "display_name": "Sam B", "emails": ["sam.b@example.com"]},
    ]
    server, handler, url = start_server(contacts=contacts)
    try:
        engine = FakeEngine(pick="sam.b@example.com")
        ctx = context(url, engine)
        request = Request("email Sam that lunch is tomorrow")
        action = skill.act(ctx, request)
        assert len(engine.decisions) == 1, engine.decisions
        assert "sam.b@example.com" in action.needs_input, action.needs_input
        request.user_input = "yes"
        action = skill.act(ctx, request)
        assert handler.sends[0]["to"] == ["sam.b@example.com"], handler.sends
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_explicit_address_is_used():
    server, handler, url = start_server(
        draft={"recipient": "bob@example.com", "subject": "Late", "body": "I'll be late"}
    )
    try:
        ctx = context(url, FakeEngine(pick="sam@example.com"))
        request = Request("email bob@example.com that I'll be late")
        skill.act(ctx, request)
        request.user_input = "yes"
        skill.act(ctx, request)
        assert handler.sends[0]["to"] == ["bob@example.com"], handler.sends
        # No contact lookup is needed for an explicit address.
        assert handler.queries == [], handler.queries
        print(handler.sends[0])
    finally:
        server.shutdown()
        server.server_close()


def test_llm_bridge_failure_is_reported_honestly():
    server, handler, url = start_server(llm_status=502)
    try:
        ctx = context(url, FakeEngine())
        request = Request("email Sam that lunch is tomorrow")
        action = skill.act(ctx, request)
        assert "502" in action.action_log, action.action_log
        assert handler.sends == []
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def main():
    test_confirms_then_sends()
    test_cancel_does_not_send()
    test_multiple_contacts_use_semif_choice()
    test_explicit_address_is_used()
    test_llm_bridge_failure_is_reported_honestly()
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
