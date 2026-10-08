"""Hermetic mechanics test for outlook.list_contacts.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the local Outlook bridge's `/contacts` route and records the
query it received. This proves the body searches the contacts and reports the
matches (and fails honestly when the bridge is unavailable). It does NOT prove
the live integration — only a real run against the user's Microsoft account does.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from semif_agent.decisions import Request
from semif_agent.skills import ActionContext

import skill


class FakeEngine:
    def call(self, decision):  # pragma: no cover - this skill makes no decision
        raise AssertionError("outlook.list_contacts must not ask the decision engine")


class BridgeHandler(BaseHTTPRequestHandler):
    contacts = []
    status = 200
    queries = []

    def do_GET(self):
        if self.status != 200:
            self._send({"error": "boom"}, self.status)
            return
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


def start_server(contacts=None, status=200):
    handler = type(
        "Handler",
        (BridgeHandler,),
        {
            "contacts": contacts
            if contacts is not None
            else [
                {
                    "id": "c1",
                    "display_name": "Bob Smith",
                    "emails": ["bob@example.com"],
                    "phones": ["555-1234"],
                }
            ],
            "status": status,
            "queries": [],
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler, f"http://127.0.0.1:{server.server_address[1]}"


def run(url, text="find Bob's email"):
    ctx = ActionContext(engine=FakeEngine(), config={"outlook_bridge_url": url, "outlook_bridge_token": ""})
    return skill.act(ctx, Request(text))


def test_reports_matching_contacts():
    server, handler, url = start_server()
    try:
        action = run(url)
        assert "Bob Smith" in action.action_log, action.action_log
        assert "bob@example.com" in action.action_log, action.action_log
        assert handler.queries and "Bob" in handler.queries[0], handler.queries
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_no_matches_is_reported():
    server, _handler, url = start_server(contacts=[])
    try:
        action = run(url)
        assert "no matching contacts" in action.action_log.lower(), action.action_log
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_bridge_error_is_reported_honestly():
    server, _handler, url = start_server(status=502)
    try:
        action = run(url)
        assert "502" in action.action_log, action.action_log
        assert action.new_state == "find Bob's email"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def main():
    test_reports_matching_contacts()
    test_no_matches_is_reported()
    test_bridge_error_is_reported_honestly()
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
