"""Hermetic mechanics test for outlook.create_event.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the local Outlook bridge's `/calendars/events` route and records
the payload. This proves the body derives the title/date/time, asks for a missing
piece and resumes, creates the event with a real POST, and fails honestly when
the bridge errors. It does NOT prove the live integration — only a real run
against the user's Microsoft account does.
"""

import json
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_agent.decisions import Request
from semif_agent.skills import ActionContext

import skill

FIXED_NOW = datetime(2026, 10, 7, 10, 0, tzinfo=timezone.utc)
skill._now = lambda ctx: FIXED_NOW


class FakeEngine:
    def call(self, decision):  # pragma: no cover - this skill makes no decision
        raise AssertionError("outlook.create_event must not ask the decision engine")


class BridgeHandler(BaseHTTPRequestHandler):
    status = 200
    posts = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path != "/calendars/events":
            self._send({"error": "not found"}, 404)
            return
        self.posts.append(json.loads(body.decode("utf-8")))
        if self.status != 200:
            self._send({"error": "boom"}, self.status)
            return
        self._send({"ok": True, "id": "e1", "subject": self.posts[-1].get("subject")}, 201)

    def _send(self, payload, status):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_server(status=200):
    handler = type("Handler", (BridgeHandler,), {"status": status, "posts": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler, f"http://127.0.0.1:{server.server_address[1]}"


def context(url, tz=""):
    config = {"outlook_bridge_url": url, "outlook_bridge_token": ""}
    if tz:
        config["timezone"] = tz
    return ActionContext(engine=FakeEngine(), config=config)


def test_explicit_datetime():
    server, handler, url = start_server()
    try:
        action = skill.act(context(url), Request("add dentist appointment on 2026-11-10 at 2pm"))
        assert action.needs_input is None, action.needs_input
        post = handler.posts[0]
        assert post["subject"] == "dentist appointment", post
        assert post["start"] == "2026-11-10T14:00:00", post
        assert post["end"] == "2026-11-10T15:00:00", post
        assert post["all_day"] is False, post
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_timezone_is_sent():
    server, handler, url = start_server()
    try:
        skill.act(
            context(url, tz="America/Chicago"),
            Request("add lunch on 2026-11-10 at noon"),
        )
        assert handler.posts[0]["timezone"] == "America/Chicago", handler.posts[0]
        print(handler.posts[0])
    finally:
        server.shutdown()
        server.server_close()


def test_all_day_event():
    server, handler, url = start_server()
    try:
        skill.act(context(url), Request("add conference all day on 2026-11-10"))
        post = handler.posts[0]
        assert post["all_day"] is True, post
        assert "end" not in post, post
        assert post["subject"] == "conference", post
        print(post)
    finally:
        server.shutdown()
        server.server_close()


def test_missing_time_asks_then_creates():
    server, handler, url = start_server()
    try:
        ctx = context(url)
        request = Request("add dentist appointment on 2026-11-10")
        action = skill.act(ctx, request)
        assert action.needs_input, "a missing start time must be asked"
        assert handler.posts == []
        request.user_input = "2pm"
        action = skill.act(ctx, request)
        assert action.needs_input is None, action.needs_input
        assert handler.posts[0]["start"] == "2026-11-10T14:00:00", handler.posts[0]
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_bridge_error_is_reported_honestly():
    server, _handler, url = start_server(status=502)
    try:
        request = Request("add dentist appointment on 2026-11-10 at 2pm")
        action = skill.act(context(url), request)
        assert "502" in action.action_log, action.action_log
        assert action.new_state == request.text, "a failed create must not fake a result"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def main():
    test_explicit_datetime()
    test_timezone_is_sent()
    test_all_day_event()
    test_missing_time_asks_then_creates()
    test_bridge_error_is_reported_honestly()
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
