"""Hermetic mechanics test for simplex.send_message.

Stands up loopback HTTP servers that imitate the simplex and llm bridges,
points the fixture config at them, and drives the body's real `act` so the
request construction, recipient/message resolution and error paths are
exercised without leaving the machine.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import skill
from semif_agent.skills import ActionContext


# ---------------------------------------------------------------------------
# inline fixtures
# ---------------------------------------------------------------------------

CONTACTS = [
    {"id": "c-pepper", "display_name": "pepper"},
    {"id": "c-salt", "display_name": "salt"},
]

SIMPLEX_TOKEN = "s3cret-simplex-token"
LLM_TOKEN = "s3cret-llm-token"
LLM_REPLY = "a friendly hello"


class FakeRequest:
    """Minimal stand-in for the agent's Request object (test-only)."""

    def __init__(self, text, user_input="", run_ledger=None):
        self.text = text
        self.user_input = user_input
        self.run_ledger = list(run_ledger or [])


class FakeResult:
    """Stand-in for a decision result: exposes `selected` and `prob(id)`."""

    def __init__(self, selected, confidence):
        self.selected = selected
        self._confidence = confidence

    def prob(self, option_id):
        return float(self._confidence) if option_id == self.selected else 0.0


class FakeEngine:
    """Test-only decision engine."""

    def __init__(self, selected=None, confidence=1.0):
        self.selected = selected
        self.confidence = confidence
        self.calls = []

    def call(self, decision):
        self.calls.append(decision)
        selected = self.selected
        if selected is None:
            options = list(decision.options)
            selected = options[0].id if options else None
        return FakeResult(selected, self.confidence)


class _BaseHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return None


class SimplexHandler(_BaseHandler):
    contacts = []
    send_error = False
    seen = []

    def do_GET(self):
        SimplexHandler.seen.append(
            {
                "method": "GET",
                "path": self.path,
                "token": self.headers.get("X-Semif-Token"),
            }
        )
        if self.path == "/contacts":
            self._send_json(200, {"contacts": SimplexHandler.contacts})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        body = self._read_body()
        SimplexHandler.seen.append(
            {
                "method": "POST",
                "path": self.path,
                "token": self.headers.get("X-Semif-Token"),
                "body": body,
            }
        )
        if self.path != "/send":
            self._send_json(404, {"error": "not found"})
            return
        if SimplexHandler.send_error:
            self._send_json(400, {"error": "unknown recipient"})
            return
        recipient = (body or {}).get("recipient", "")
        self._send_json(200, {"ok": True, "contact_id": recipient})


class LlmHandler(_BaseHandler):
    reply = LLM_REPLY
    seen = []

    def do_POST(self):
        body = self._read_body()
        LlmHandler.seen.append(
            {
                "path": self.path,
                "token": self.headers.get("X-Semif-Token"),
                "body": body,
            }
        )
        if self.path == "/chat":
            self._send_json(200, {"text": LlmHandler.reply})
        else:
            self._send_json(404, {"error": "not found"})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def start_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def base_url(server):
    return f"http://127.0.0.1:{server.server_address[1]}"


def reset(contacts=None, send_error=False):
    SimplexHandler.contacts = [dict(c) for c in (contacts if contacts is not None else CONTACTS)]
    SimplexHandler.send_error = send_error
    SimplexHandler.seen = []
    LlmHandler.seen = []
    LlmHandler.reply = LLM_REPLY


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_happy_path():
    """A named contact plus an explicit message: GET /contacts then POST /send."""
    reset()
    server = start_server(SimplexHandler)
    try:
        ctx = ActionContext(
            engine=FakeEngine(),
            config={
                "simplex_bridge_url": base_url(server),
                "simplex_bridge_token": SIMPLEX_TOKEN,
            },
        )
        result = skill.act(
            ctx, FakeRequest("send a message to pepper on simplex saying hey")
        )

        assert result.action_log, "act must log what it did"
        assert not result.needs_input, result.action_log
        assert "sent" in result.action_log.lower(), result.action_log

        gets = [r for r in SimplexHandler.seen if r["method"] == "GET"]
        posts = [r for r in SimplexHandler.seen if r["method"] == "POST"]
        assert any(r["path"] == "/contacts" for r in gets), SimplexHandler.seen
        assert len(posts) == 1, SimplexHandler.seen
        assert posts[0]["path"] == "/send", SimplexHandler.seen
        assert posts[0]["body"] == {"recipient": "c-pepper", "text": "hey"}, posts[0]

        for seen in SimplexHandler.seen:
            assert seen["token"] == SIMPLEX_TOKEN, seen

        state = json.loads(result.new_state)
        assert state["sent"] is True, state
        assert state["contact_id"] == "c-pepper", state
        assert state["text"] == "hey", state
    finally:
        server.shutdown()
        server.server_close()


def test_ambiguous_contact_uses_engine():
    """Two named contacts: the body must ask the engine and honour the pick."""
    reset()
    server = start_server(SimplexHandler)
    try:
        engine = FakeEngine(selected="c-salt", confidence=0.9)
        ctx = ActionContext(
            engine=engine,
            config={"simplex_bridge_url": base_url(server)},
        )
        result = skill.act(
            ctx, FakeRequest("send a message to pepper and salt saying hi")
        )

        assert not result.needs_input, result.action_log
        assert len(engine.calls) == 1, engine.calls
        option_ids = [o.id for o in engine.calls[0].options]
        assert set(option_ids) == {"c-pepper", "c-salt"}, option_ids

        posts = [r for r in SimplexHandler.seen if r["method"] == "POST"]
        assert len(posts) == 1, SimplexHandler.seen
        assert posts[0]["body"] == {"recipient": "c-salt", "text": "hi"}, posts[0]
    finally:
        server.shutdown()
        server.server_close()


def test_message_written_by_llm_bridge():
    """No literal text: the body must ask the llm bridge, then send its reply."""
    reset()
    simplex = start_server(SimplexHandler)
    llm = start_server(LlmHandler)
    try:
        ctx = ActionContext(
            engine=FakeEngine(),
            config={
                "simplex_bridge_url": base_url(simplex),
                "simplex_bridge_token": SIMPLEX_TOKEN,
                "llm_bridge_url": base_url(llm),
                "llm_bridge_token": LLM_TOKEN,
            },
        )
        result = skill.act(ctx, FakeRequest("tell pepper something nice"))

        assert not result.needs_input, result.action_log
        assert len(LlmHandler.seen) == 1, LlmHandler.seen
        chat = LlmHandler.seen[0]
        assert chat["path"] == "/chat", chat
        assert chat["token"] == LLM_TOKEN, chat
        assert chat["body"]["messages"], chat

        posts = [r for r in SimplexHandler.seen if r["method"] == "POST"]
        assert len(posts) == 1, SimplexHandler.seen
        assert posts[0]["body"] == {"recipient": "c-pepper", "text": LLM_REPLY}, posts[0]
    finally:
        simplex.shutdown()
        simplex.server_close()
        llm.shutdown()
        llm.server_close()


def test_chained_result_is_sent():
    """As the later step of a chain, no literal text: send the prior step's result."""
    reset()
    simplex = start_server(SimplexHandler)
    llm = start_server(LlmHandler)
    try:
        ctx = ActionContext(
            engine=FakeEngine(),
            config={
                "simplex_bridge_url": base_url(simplex),
                "simplex_bridge_token": SIMPLEX_TOKEN,
                "llm_bridge_url": base_url(llm),
            },
        )
        request = FakeRequest(
            "send the result of 22 * 10 to pepper on simplex",
            run_ledger=[
                {
                    "query": "send the result of 22 * 10 to pepper on simplex",
                    "skill": "calculator.calculate",
                    "outcome": "Twenty-two times ten is two hundred twenty.",
                }
            ],
        )
        result = skill.act(ctx, request)

        assert not result.needs_input, result.action_log
        # The prior result is used verbatim; the LLM bridge must NOT be asked to
        # recompute it.
        assert LlmHandler.seen == [], LlmHandler.seen
        posts = [r for r in SimplexHandler.seen if r["method"] == "POST"]
        assert len(posts) == 1, SimplexHandler.seen
        assert posts[0]["body"] == {
            "recipient": "c-pepper",
            "text": "Twenty-two times ten is two hundred twenty.",
        }, posts[0]
    finally:
        simplex.shutdown()
        simplex.server_close()
        llm.shutdown()
        llm.server_close()


def test_missing_text_without_llm_asks():
    """A clear recipient but no message text and no llm bridge: ask the user."""
    reset()
    server = start_server(SimplexHandler)
    try:
        ctx = ActionContext(
            engine=FakeEngine(),
            config={"simplex_bridge_url": base_url(server)},
        )
        result = skill.act(ctx, FakeRequest("tell pepper something nice"))

        assert result.needs_input, result.action_log
        assert "say" in result.needs_input.lower(), result.needs_input
        posts = [r for r in SimplexHandler.seen if r["method"] == "POST"]
        assert posts == [], SimplexHandler.seen
    finally:
        server.shutdown()
        server.server_close()


def test_unknown_contact_asks():
    """No matching contact in the bridge list: ask which contact to message."""
    reset(contacts=[])
    server = start_server(SimplexHandler)
    try:
        ctx = ActionContext(
            engine=FakeEngine(),
            config={"simplex_bridge_url": base_url(server)},
        )
        result = skill.act(
            ctx, FakeRequest("send a message to pepper on simplex saying hey")
        )

        assert result.needs_input, result.action_log
        assert "contact" in result.needs_input.lower(), result.needs_input
        posts = [r for r in SimplexHandler.seen if r["method"] == "POST"]
        assert posts == [], SimplexHandler.seen
    finally:
        server.shutdown()
        server.server_close()


def test_send_refused_is_reported():
    """A 400 from POST /send must be surfaced, not silently swallowed."""
    reset(send_error=True)
    server = start_server(SimplexHandler)
    try:
        ctx = ActionContext(
            engine=FakeEngine(),
            config={"simplex_bridge_url": base_url(server)},
        )
        result = skill.act(
            ctx, FakeRequest("send a message to pepper on simplex saying hey")
        )

        assert not result.needs_input, result.action_log
        assert "refused" in result.action_log.lower(), result.action_log
        assert "400" in result.action_log, result.action_log
        posts = [r for r in SimplexHandler.seen if r["method"] == "POST"]
        assert len(posts) == 1, SimplexHandler.seen
    finally:
        server.shutdown()
        server.server_close()


def test_unreachable_bridge_is_reported():
    """A dead bridge must produce a plain failure log, not an exception."""
    reset()
    server = start_server(SimplexHandler)
    url = base_url(server)
    server.shutdown()
    server.server_close()

    ctx = ActionContext(
        engine=FakeEngine(),
        config={"simplex_bridge_url": url},
    )
    result = skill.act(
        ctx, FakeRequest("send a message to pepper on simplex saying hey")
    )
    assert not result.needs_input, result.action_log
    assert "could not reach" in result.action_log.lower(), result.action_log


def main():
    test_happy_path()
    test_ambiguous_contact_uses_engine()
    test_message_written_by_llm_bridge()
    test_chained_result_is_sent()
    test_missing_text_without_llm_asks()
    test_unknown_contact_asks()
    test_send_refused_is_reported()
    test_unreachable_bridge_is_reported()
    print("ok")


if __name__ == "__main__":
    main()
