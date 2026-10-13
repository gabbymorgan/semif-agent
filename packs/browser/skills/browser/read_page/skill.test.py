"""Hermetic mechanics test for browser.read_page.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the local browser bridge's `/goto` and `/content` routes. This
proves the body reads the page through the bridge (navigating first only when the
request names a URL) and fails honestly when the bridge is unavailable. It does
NOT prove the live integration — only a real run against a real browser does.
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
        raise AssertionError("browser.read_page must not ask the decision engine")


class BridgeHandler(BaseHTTPRequestHandler):
    body = "First line.\nSecond line."
    status = 200
    requests = []

    def _read(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _send(self, payload, status):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        self.requests.append(("POST", self.path, self._read().decode("utf-8")))
        if self.status != 200:
            self._send({"error": "boom"}, self.status)
            return
        if self.path == "/goto":
            self._send({"url": "https://example.com/", "title": "Example", "status": 200}, 200)
            return
        self._send({"error": "not found"}, 404)

    def do_GET(self):
        self.requests.append(("GET", self.path, ""))
        if self.status != 200:
            self._send({"error": "boom"}, self.status)
            return
        if self.path.startswith("/content"):
            self._send(
                {
                    "url": "https://example.com/",
                    "title": "Example",
                    "format": "text",
                    "content": self.body,
                    "truncated": False,
                },
                200,
            )
            return
        self._send({"error": "not found"}, 404)

    def log_message(self, *args):
        pass


def start_server(status=200):
    handler = type("Handler", (BridgeHandler,), {"status": status, "requests": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler, f"http://127.0.0.1:{server.server_address[1]}"


def run(url, text="read the current page"):
    ctx = ActionContext(engine=FakeEngine(), config={"browser_bridge_url": url, "browser_bridge_token": ""})
    return skill.act(ctx, Request(text))


def test_reads_the_current_page_without_navigating():
    server, handler, url = start_server()
    try:
        action = run(url)
        assert "First line." in action.action_log, action.action_log
        assert "Second line." in action.action_log, action.action_log
        assert action.new_state == action.action_log
        assert not any(path == "/goto" for _method, path, _ in handler.requests), "no URL means no navigation"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_navigates_when_the_request_names_a_url():
    server, handler, url = start_server()
    try:
        action = run(url, "read https://example.com/")
        assert "Example" in action.action_log, action.action_log
        assert any(method == "POST" and path == "/goto" for method, path, _ in handler.requests)
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_bridge_error_is_reported_honestly():
    server, _handler, url = start_server(status=502)
    try:
        action = run(url)
        assert "502" in action.action_log, action.action_log
        assert action.new_state == "read the current page"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def main():
    test_reads_the_current_page_without_navigating()
    test_navigates_when_the_request_names_a_url()
    test_bridge_error_is_reported_honestly()
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
