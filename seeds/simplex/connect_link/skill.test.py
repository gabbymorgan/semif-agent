"""Hermetic mechanics test for simplex.connect_link.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the standalone SimpleX forwarding bridge. This proves the body health-
checks the bridge, fetches the contact link, reports a freshly created link, and
fails honestly when the bridge has no provider or is unreachable. It does NOT
prove the live integration — only a real run against the bridge and a real
simplex-chat daemon does.
"""

import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_agent.decisions import Request
from semif_agent.skills import ActionContext

import skill


class BridgeHandler(BaseHTTPRequestHandler):
    address = {}
    address_status = 200
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
        if url.path == "/health":
            self._send({"ok": True, "platform": "simplex"})
        elif url.path == "/address":
            self._send(self.address, self.address_status)
        else:
            self._send({"error": "not found"}, 404)

    def log_message(self, *args):
        pass


def start_server(address=None, address_status=200):
    handler = type(
        "Handler",
        (BridgeHandler,),
        {
            "address": address or {},
            "address_status": address_status,
            "requests": [],
            "tokens": [],
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler, f"http://127.0.0.1:{server.server_address[1]}"


def config(url, token=""):
    return {"simplex_bridge_url": url, "simplex_bridge_token": token}


def test_reports_existing_link():
    server, handler, url = start_server(
        {"short_link": "simplex:/contact#abc", "full_link": "https://smp/short", "created": False}
    )
    try:
        ctx = ActionContext(engine=None, config=config(url))
        request = Request("give me your simplex link")
        prediction = skill.predict(ctx, request)
        action = skill.act(ctx, request, prediction)
        assert "simplex:/contact#abc" in action.action_log, action.action_log
        assert "https://smp/short" in action.action_log, action.action_log
        assert "current" in action.action_log, action.action_log
        assert handler.requests == ["/health", "/address"], handler.requests
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_reports_freshly_created_link():
    server, handler, url = start_server(
        {"short_link": "simplex:/contact#new", "full_link": "", "created": True}
    )
    try:
        ctx = ActionContext(engine=None, config=config(url))
        request = Request("create a simplex invitation link")
        prediction = skill.predict(ctx, request)
        action = skill.act(ctx, request, prediction)
        assert "simplex:/contact#new" in action.action_log, action.action_log
        assert "created" in action.action_log, action.action_log
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_no_provider_is_reported_honestly():
    server, handler, url = start_server({"error": "address lookup is not available"}, 503)
    try:
        ctx = ActionContext(engine=None, config=config(url))
        request = Request("give me your simplex link")
        prediction = skill.predict(ctx, request)
        action = skill.act(ctx, request, prediction)
        assert "503" in action.action_log, action.action_log
        assert action.new_state == request.text, "a failed lookup must not fake a result"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_empty_link_is_reported_honestly():
    server, handler, url = start_server({"short_link": "", "full_link": "", "created": False})
    try:
        ctx = ActionContext(engine=None, config=config(url))
        request = Request("give me your simplex link")
        prediction = skill.predict(ctx, request)
        action = skill.act(ctx, request, prediction)
        assert "no SimpleX" in action.action_log, action.action_log
        assert action.new_state == request.text
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_unreachable_bridge_fails_honestly():
    ctx = ActionContext(engine=None, config=config("http://127.0.0.1:1"))
    request = Request("give me your simplex link")
    prediction = skill.predict(ctx, request)
    assert prediction.text.startswith("bridge error:"), prediction.text
    action = skill.act(ctx, request, prediction)
    assert action.action_log.startswith("bridge error:"), action.action_log
    assert action.new_state == request.text
    print(action.action_log)


def test_auth_token_is_sent_when_configured():
    server, handler, url = start_server(
        {"short_link": "simplex:/contact#abc", "full_link": "", "created": False}
    )
    try:
        ctx = ActionContext(engine=None, config=config(url, token="sekret"))
        request = Request("give me your simplex link")
        prediction = skill.predict(ctx, request)
        action = skill.act(ctx, request, prediction)
        assert "simplex:/contact#abc" in action.action_log, action.action_log
        assert set(handler.tokens) == {"sekret"}, handler.tokens
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def main():
    test_reports_existing_link()
    test_reports_freshly_created_link()
    test_no_provider_is_reported_honestly()
    test_empty_link_is_reported_honestly()
    test_unreachable_bridge_fails_honestly()
    test_auth_token_is_sent_when_configured()
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
