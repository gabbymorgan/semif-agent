"""Hermetic mechanics test for outlook.create_task.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the local Outlook bridge's `/tasks` route and records the
payload. This proves the body derives a title + due date, asks for a missing
title and resumes, creates the task with a real POST, and fails honestly when the
bridge errors. It does NOT prove the live integration — only a real run against
the user's Microsoft account does.
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
        raise AssertionError("outlook.create_task must not ask the decision engine")


class BridgeHandler(BaseHTTPRequestHandler):
    status = 200
    posts = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path != "/tasks":
            self._send({"error": "not found"}, 404)
            return
        self.posts.append(json.loads(body.decode("utf-8")))
        if self.status != 200:
            self._send({"error": "boom"}, self.status)
            return
        self._send({"ok": True, "id": "t1", "title": self.posts[-1].get("title")}, 201)

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


def config(url):
    return {"outlook_bridge_url": url, "outlook_bridge_token": ""}


def test_plain_task():
    server, handler, url = start_server()
    try:
        ctx = ActionContext(engine=FakeEngine(), config=config(url))
        action = skill.act(ctx, Request("add buy milk to my tasks"))
        assert action.needs_input is None, action.needs_input
        assert handler.posts[0]["title"] == "buy milk", handler.posts[0]
        assert "due" not in handler.posts[0], handler.posts[0]
        assert "buy milk" in action.action_log
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_task_with_due_date():
    server, handler, url = start_server()
    try:
        ctx = ActionContext(engine=FakeEngine(), config=config(url))
        action = skill.act(ctx, Request("remind me to call the dentist tomorrow"))
        assert action.needs_input is None, action.needs_input
        assert handler.posts[0]["title"] == "call dentist", handler.posts[0]
        assert handler.posts[0]["due"] == "2026-10-08T00:00:00", handler.posts[0]
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_missing_title_asks_then_creates():
    server, handler, url = start_server()
    try:
        ctx = ActionContext(engine=FakeEngine(), config=config(url))
        request = Request("add a task")
        action = skill.act(ctx, request)
        assert action.needs_input, "a missing title must be asked"
        assert handler.posts == []
        request.user_input = "water the plants"
        action = skill.act(ctx, request)
        assert action.needs_input is None, action.needs_input
        assert handler.posts[0]["title"] == "water the plants", handler.posts[0]
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_bridge_error_is_reported_honestly():
    server, _handler, url = start_server(status=502)
    try:
        ctx = ActionContext(engine=FakeEngine(), config=config(url))
        request = Request("add buy milk to my tasks")
        action = skill.act(ctx, request)
        assert "502" in action.action_log, action.action_log
        assert action.new_state == request.text, "a failed create must not fake a result"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def main():
    test_plain_task()
    test_task_with_due_date()
    test_missing_title_asks_then_creates()
    test_bridge_error_is_reported_honestly()
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
