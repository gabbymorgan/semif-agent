"""Pure-stdlib tests for the remote Winnow decision engine provider.

The engine is exercised over a real loopback HTTP server (the transport is not
mocked): the handler records the request body and returns a canned
``/v1/systemone`` response, so the tests prove the decision-contract mapping and
the strict distribution validation. No model, no GGUF, no external network.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from semif_agent.decisions import DecisionRequest, Option
from semif_agent.engine import EngineUnavailable
from semif_agent.winnow import WinnowConfig, WinnowEngine


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the test output quiet
        pass

    def _read(self) -> dict:
        length = int(self.headers.get("content-length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            status = self.server.state.get("health_status", 200)
            self._send(status, {"status": "ok"} if status == 200 else {"error": "down"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = self._read()
        self.server.state.setdefault("requests", []).append(
            (self.path, body, {k.lower(): v for k, v in self.headers.items()})
        )
        if self.path == "/v1/systemone":
            status = self.server.state.get("systemone_status", 200)
            response = self.server.state.get("systemone_response")
            if response is None:
                response = _default_systemone(body)
            self._send(status, response)
        elif self.path == "/v1/chat/completions":
            self._send(
                200,
                {"choices": [{"message": {"role": "assistant", "content": " hello "}}]},
            )
        else:
            self._send(404, {"error": "not found"})


def _default_systemone(body: dict) -> dict:
    criteria = body["questions"]["decision"]["criteria"]
    keys = list(criteria)
    probs = {
        key: (0.9 if index == 0 else 0.1 / max(len(keys) - 1, 1))
        for index, key in enumerate(keys)
    }
    # Return the keys in REVERSE order to prove the engine aligns by id, not by
    # position in the returned object.
    return {
        "model": "Winnow-12B",
        "answers": {
            "decision": {
                "type": "choice",
                "probabilities": dict(reversed(list(probs.items()))),
                "confidence": 0.9,
                "choice": keys[0],
            }
        },
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }


@pytest.fixture()
def winnow():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield server, base
    server.shutdown()
    server.server_close()


def _engine(base: str, **cfg) -> WinnowEngine:
    return WinnowEngine(
        WinnowConfig(base_url=base, model="Winnow-12B", timeout=5.0, **cfg)
    )


def _decision(*option_ids: str) -> DecisionRequest:
    return DecisionRequest(
        state="the user said hello",
        question="Which category handles this request?",
        options=[Option(oid, f"description for {oid}") for oid in option_ids],
    )


def test_call_maps_decision_to_a_choice_question(winnow):
    server, base = winnow
    engine = _engine(base)
    result = engine.call(_decision("response", "calendar"))

    assert result.option_ids == ["response", "calendar"]
    assert result.prob("response") == pytest.approx(0.9)
    assert result.prob("calendar") == pytest.approx(0.1)
    assert result.selected == "response"
    assert result.extra["input_tokens"] == 42

    path, body, _ = server.state["requests"][-1]
    assert path == "/v1/systemone"
    assert body["model"] == "Winnow-12B"
    assert body["state"] == "the user said hello"
    question = body["questions"]["decision"]
    assert question["type"] == "choice"
    assert question["instructions"] == "Which category handles this request?"
    assert question["criteria"] == {
        "response": "description for response",
        "calendar": "description for calendar",
    }


def test_call_normalizes_an_unnormalized_distribution(winnow):
    server, base = winnow
    server.state["systemone_response"] = {
        "answers": {"decision": {"type": "choice", "probabilities": {"a": 0.4, "b": 0.4}}}
    }
    result = _engine(base).call(_decision("a", "b"))
    assert result.probs == pytest.approx({"a": 0.5, "b": 0.5})


def test_call_rejects_mismatched_probability_keys(winnow):
    server, base = winnow
    server.state["systemone_response"] = {
        "answers": {"decision": {"type": "choice", "probabilities": {"a": 1.0}}}
    }
    with pytest.raises(EngineUnavailable):
        _engine(base).call(_decision("a", "b"))


def test_call_rejects_all_zero_distribution(winnow):
    server, base = winnow
    server.state["systemone_response"] = {
        "answers": {"decision": {"type": "choice", "probabilities": {"a": 0.0, "b": 0.0}}}
    }
    with pytest.raises(EngineUnavailable):
        _engine(base).call(_decision("a", "b"))


def test_call_rejects_http_error(winnow):
    server, base = winnow
    server.state["systemone_status"] = 500
    with pytest.raises(EngineUnavailable):
        _engine(base).call(_decision("a", "b"))


def test_call_rejects_missing_answer(winnow):
    server, base = winnow
    server.state["systemone_response"] = {"answers": {}}
    with pytest.raises(EngineUnavailable):
        _engine(base).call(_decision("a", "b"))


def test_call_sends_bearer_token(winnow):
    server, base = winnow
    _engine(base, api_key="secret").call(_decision("a", "b"))
    _, _, headers = server.state["requests"][-1]
    assert headers.get("authorization") == "Bearer secret"


def test_generate_uses_chat_completions(winnow):
    server, base = winnow
    text = _engine(base).generate([{"role": "user", "content": "hi"}], max_tokens=16)
    assert text == "hello"
    path, body, _ = server.state["requests"][-1]
    assert path == "/v1/chat/completions"
    assert body["max_tokens"] == 16


def test_warm_checks_health_and_marks_loaded(winnow):
    server, base = winnow
    engine = _engine(base)
    assert engine.loaded is False
    engine.warm()
    assert engine.loaded is True

    server.state["health_status"] = 503
    with pytest.raises(EngineUnavailable):
        _engine(base).warm()


def test_unreachable_endpoint_raises_engine_unavailable():
    engine = WinnowEngine(
        WinnowConfig(base_url="http://127.0.0.1:1", model="Winnow-12B", timeout=1.0)
    )
    with pytest.raises(EngineUnavailable):
        engine.call(_decision("a", "b"))
