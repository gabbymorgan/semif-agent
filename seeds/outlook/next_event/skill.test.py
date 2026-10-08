"""Hermetic mechanics test for outlook.next_event.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the local Outlook bridge's `/calendars/events` route. This
proves the body asks the bridge for upcoming events and reports them (and fails
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
        raise AssertionError("outlook.next_event must not ask the decision engine")


class BridgeHandler(BaseHTTPRequestHandler):
    events = []
    status = 200

    def do_GET(self):
        if self.status != 200:
            self._send({"error": "boom"}, self.status)
            return
        if self.path.startswith("/calendars/events"):
            self._send({"calendar": "", "events": self.events}, 200)
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


def start_server(events=None, status=200):
    handler = type(
        "Handler",
        (BridgeHandler,),
        {
            "events": events
            if events is not None
            else [
                {
                    "id": "e1",
                    "subject": "Standup",
                    "start": "2026-10-08T09:00:00.0000000",
                    "end": "2026-10-08T09:30:00.0000000",
                    "all_day": False,
                    "location": "Room 1",
                }
            ],
            "status": status,
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def run(url, text="what's on my calendar"):
    ctx = ActionContext(engine=FakeEngine(), config={"outlook_bridge_url": url, "outlook_bridge_token": ""})
    return skill.act(ctx, Request(text))


def test_reports_upcoming_events():
    server, url = start_server()
    try:
        action = run(url)
        assert "Standup" in action.action_log, action.action_log
        assert "Room 1" in action.action_log, action.action_log
        assert "2026-10-08 09:00" in action.action_log, action.action_log
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_no_events_is_reported():
    server, url = start_server(events=[])
    try:
        action = run(url)
        assert "no upcoming events" in action.action_log.lower(), action.action_log
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_bridge_error_is_reported_honestly():
    server, url = start_server(status=502)
    try:
        action = run(url)
        assert "502" in action.action_log, action.action_log
        assert action.new_state == "what's on my calendar"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_unreachable_bridge_fails_honestly():
    action = run("http://127.0.0.1:1")
    assert action.new_state == "what's on my calendar"
    print(action.action_log)


def main():
    test_reports_upcoming_events()
    test_no_events_is_reported()
    test_bridge_error_is_reported_honestly()
    test_unreachable_bridge_fails_honestly()
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
