"""Hermetic mechanics test for outlook.next_message.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the local Outlook bridge's `/mail/messages` route. This proves
the body asks the bridge for the inbox and reports the messages (and fails
honestly when the bridge is unavailable). It does NOT prove the live
integration — only a real run against the user's Microsoft account does.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_agent.decisions import Request
from semif_agent.skills import ActionContext

import skill


class FakeEngine:
    def call(self, decision):  # pragma: no cover - this skill makes no decision
        raise AssertionError("outlook.next_message must not ask the decision engine")


class BridgeHandler(BaseHTTPRequestHandler):
    messages = []
    requests = []
    status = 200

    def do_GET(self):
        self.requests.append(self.path)
        if self.status != 200:
            self._send({"error": "boom"}, self.status)
            return
        if self.path.startswith("/mail/messages"):
            self._send({"folder": "inbox", "messages": self.messages}, 200)
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


def start_server(messages=None, status=200):
    handler = type(
        "Handler",
        (BridgeHandler,),
        {
            "messages": messages
            if messages is not None
            else [
                {
                    "id": "m1",
                    "subject": "Lunch?",
                    "from": "Sam <sam@example.com>",
                    "received": "2026-10-08T12:00:00Z",
                    "is_read": False,
                }
            ],
            "requests": [],
            "status": status,
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler, f"http://127.0.0.1:{server.server_address[1]}"


def run(url, text="check my email"):
    ctx = ActionContext(engine=FakeEngine(), config={"outlook_bridge_url": url, "outlook_bridge_token": ""})
    return skill.act(ctx, Request(text))


def test_reports_recent_messages():
    server, handler, url = start_server()
    try:
        action = run(url)
        assert "Lunch?" in action.action_log, action.action_log
        assert "sam@example.com" in action.action_log, action.action_log
        assert action.new_state == action.action_log
        assert handler.requests, "the bridge must be called"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_empty_inbox_is_reported():
    server, _handler, url = start_server(messages=[])
    try:
        action = run(url)
        assert "empty" in action.action_log.lower(), action.action_log
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_bridge_error_is_reported_honestly():
    server, _handler, url = start_server(status=502)
    try:
        action = run(url)
        assert "502" in action.action_log, action.action_log
        assert action.new_state == "check my email", "a failed read must not fake a result"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_unreachable_bridge_fails_honestly():
    action = run("http://127.0.0.1:1")
    assert "outlook.next_message" not in action.action_log, action.action_log
    assert action.new_state == "check my email"
    print(action.action_log)


def main():
    test_reports_recent_messages()
    test_empty_inbox_is_reported()
    test_bridge_error_is_reported_honestly()
    test_unreachable_bridge_fails_honestly()
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
