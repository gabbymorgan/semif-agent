"""Hermetic mechanics test for simplex.next_message.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the standalone SimpleX forwarding bridge. This proves the body peeks the
inbox, resolves the conversation through a SemIf sub-decision (or a configured
default), pops the real `next` message, and fails honestly when the bridge is
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
    """Test-only engine: picks the requested option id, else the first."""

    def __init__(self, pick=None):
        self.pick = pick
        self.decisions = []

    def call(self, decision):
        self.decisions.append(decision)
        option_ids = [option.id for option in decision.options]
        chosen = self.pick if self.pick in option_ids else option_ids[0]
        probabilities = [
            1.0 if option_id == chosen else 0.0 for option_id in option_ids
        ]
        return DecisionResult(
            request=decision, option_ids=option_ids, probabilities=probabilities
        )


class BridgeHandler(BaseHTTPRequestHandler):
    messages = []
    requests = []
    tokens = []

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
        else:
            self._send({"error": "not found"}, 404)

    def log_message(self, *args):
        pass


def start_server(messages):
    handler = type(
        "Handler",
        (BridgeHandler,),
        {"messages": list(messages), "requests": [], "tokens": []},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler, f"http://127.0.0.1:{server.server_address[1]}"


def message(identifier, contact, name, text):
    return {
        "id": identifier,
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
        prediction = skill.predict(ctx, request)
        assert prediction.text == "contact: 7", prediction.text
        assert len(prediction.decisions) == 1, "predict must log the conversation choice"
        assert len(engine.decisions) == 1

        action = skill.act(ctx, request, prediction)
        assert "Bob" in action.action_log, action.action_log
        assert "shipment arrived" in action.action_log, action.action_log
        assert action.new_state, "act must set a new state"

        assert handler.requests[0] == "/inbox", handler.requests
        assert handler.requests[1] == "/inbox/next?contact=7", handler.requests
        assert [m["id"] for m in handler.messages] == ["m1"], "the popped message must be consumed"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_default_contact_skips_decision():
    server, handler, url = start_server(
        [
            message("m1", "4", "Alice", "hello there"),
            message("m2", "7", "Bob", "shipment arrived"),
        ]
    )
    try:
        engine = FakeEngine()
        ctx = ActionContext(engine=engine, config=config(url, default="Alice"))
        request = Request("read my next simplex message")
        prediction = skill.predict(ctx, request)
        assert prediction.text == "contact: 4", prediction.text
        assert engine.decisions == [], "a configured default needs no SemIf decision"

        action = skill.act(ctx, request, prediction)
        assert "Alice" in action.action_log, action.action_log
        assert handler.requests[1] == "/inbox/next?contact=4", handler.requests
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
        prediction = skill.predict(ctx, request)
        assert prediction.text == "contact: 4", prediction.text
        assert engine.decisions == []
        action = skill.act(ctx, request, prediction)
        assert "hi" in action.action_log, action.action_log
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
        prediction = skill.predict(ctx, request)
        assert prediction.text == "settled: no unread messages", prediction.text
        action = skill.act(ctx, request, prediction)
        assert "no unread" in action.action_log, action.action_log
        assert handler.requests == ["/inbox"], "an empty inbox needs no pop"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_unreachable_bridge_fails_honestly():
    engine = FakeEngine()
    ctx = ActionContext(engine=engine, config=config("http://127.0.0.1:1"))
    request = Request("read my next simplex message")
    prediction = skill.predict(ctx, request)
    assert prediction.text.startswith("bridge error:"), prediction.text
    action = skill.act(ctx, request, prediction)
    assert action.action_log.startswith("bridge error:"), action.action_log
    assert action.new_state == request.text, "a failed read must not fake a result"
    print(action.action_log)


def test_auth_token_is_sent_when_configured():
    server, handler, url = start_server([message("m1", "4", "Alice", "hi")])
    try:
        engine = FakeEngine()
        ctx = ActionContext(engine=engine, config=config(url, token="sekret"))
        request = Request("read my next simplex message")
        prediction = skill.predict(ctx, request)
        action = skill.act(ctx, request, prediction)
        assert "hi" in action.action_log, action.action_log
        assert set(handler.tokens) == {"sekret"}, handler.tokens
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def main():
    test_multiple_senders_uses_semif_decision()
    test_default_contact_skips_decision()
    test_single_sender_needs_no_decision()
    test_empty_inbox_is_reported_honestly()
    test_unreachable_bridge_fails_honestly()
    test_auth_token_is_sent_when_configured()
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
