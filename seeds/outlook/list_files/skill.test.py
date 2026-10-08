"""Hermetic mechanics test for outlook.list_files.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the local Outlook bridge's `/files` route and records the path
it received. This proves the body lists OneDrive and reports the entries (and
fails honestly when the bridge is unavailable). It does NOT prove the live
integration — only a real run against the user's Microsoft account does.
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
        raise AssertionError("outlook.list_files must not ask the decision engine")


class BridgeHandler(BaseHTTPRequestHandler):
    entries = []
    status = 200
    paths = []

    def do_GET(self):
        if self.status != 200:
            self._send({"error": "boom"}, self.status)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/files":
            path = parse_qs(parsed.query).get("path", ["/"])[0]
            self.paths.append(path)
            self._send({"path": path, "entries": self.entries}, 200)
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


def start_server(entries=None, status=200):
    handler = type(
        "Handler",
        (BridgeHandler,),
        {
            "entries": entries
            if entries is not None
            else [
                {"id": "d1", "name": "Docs", "is_folder": True, "size": 0},
                {"id": "f1", "name": "notes.txt", "is_folder": False, "size": 5},
            ],
            "status": status,
            "paths": [],
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler, f"http://127.0.0.1:{server.server_address[1]}"


def run(url, text="list my files"):
    ctx = ActionContext(engine=FakeEngine(), config={"outlook_bridge_url": url, "outlook_bridge_token": ""})
    return skill.act(ctx, Request(text))


def test_lists_root_files():
    server, handler, url = start_server()
    try:
        action = run(url)
        assert "Docs (folder)" in action.action_log, action.action_log
        assert "notes.txt (file, 5 bytes)" in action.action_log, action.action_log
        assert handler.paths and handler.paths[0] == "/", handler.paths
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_named_folder_is_requested():
    server, handler, url = start_server()
    try:
        run(url, "list files in Documents")
        assert handler.paths and handler.paths[0] == "Documents", handler.paths
        print(handler.paths)
    finally:
        server.shutdown()
        server.server_close()


def test_empty_folder_is_reported():
    server, _handler, url = start_server(entries=[])
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
        assert action.new_state == "list my files"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def main():
    test_lists_root_files()
    test_named_folder_is_requested()
    test_empty_folder_is_reported()
    test_bridge_error_is_reported_honestly()
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
