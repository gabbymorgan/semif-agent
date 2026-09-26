"""Pure-stdlib tests for the self-assessment LLM client.

The only network usage is a throwaway stdlib HTTP server standing in for an
OpenAI-compatible endpoint; the client itself is real, not mocked.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_agent.llm import LLMClient


def _chat_server(content: str):
    class Handler(BaseHTTPRequestHandler):
        received: list = []

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            type(self).received.append(
                json.loads(self.rfile.read(length).decode("utf-8"))
            )
            body = json.dumps(
                {"choices": [{"message": {"content": content}}]}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    handler = type("Handler", (Handler,), {"received": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


def _dead_url() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}/v1"


def test_review_skill_body_parses_fake_verdict():
    httpd, base = _chat_server(
        '{"performs_real_action": false, "reason": "returns a canned string"}'
    )
    try:
        client = LLMClient(base_url=base, model="test", timeout=10)
        review = client.review_skill_body(
            "send the email",
            "Send an email through the user's mail account.",
            "def act(ctx, request, prediction):\n    return 'sent'",
            requirements={"Which service?": "SMTP"},
            integration={"service": "mail", "transport": "smtp"},
        )
        assert review.performs_real_action is False
        assert "canned" in review.reason
        sent = httpd.RequestHandlerClass.received[0]["messages"]
        joined = " ".join(message["content"] for message in sent)
        assert "send the email" in joined
        assert "SMTP" in joined
        assert "smtp" in joined
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_review_skill_body_accepts_real_body():
    httpd, base = _chat_server(
        '{"performs_real_action": true, "reason": "calls urllib"}'
    )
    try:
        client = LLMClient(base_url=base, model="test", timeout=10)
        review = client.review_skill_body("check the service", "Check it.", "code")
        assert review.performs_real_action is True
        assert review.reason == "calls urllib"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_review_skill_body_degrades_when_unreachable():
    client = LLMClient(base_url=_dead_url(), model="test", timeout=2)
    review = client.review_skill_body("x", "y", "z")
    assert review.performs_real_action is True
    assert "review failed" in review.reason
