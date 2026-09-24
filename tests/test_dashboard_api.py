"""Dashboard API tests against a real stdlib HTTP server on an ephemeral port.

The scheduler is constructed with the lazy SemIfEngine (never loaded), so this
runs anywhere without SemIf. Submit degrades to a JSON error, which is the
expected behaviour on the thin dev box; relabel works against a seeded log.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from semif_agent.dashboard import DashboardHandler
from semif_agent.decisions import DecisionRequest, DecisionResult, Option, Request
from semif_agent.engine import EngineConfig, SemIfEngine
from semif_agent.llm import LLMClient
from semif_agent.log import DecisionLog
from semif_agent.scheduler import Scheduler
from semif_agent.skills import ActionResult, Prediction, Skill
from semif_agent.trace import TraceLog


def build_scheduler(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = SemIfEngine(EngineConfig())
    llm = LLMClient(base_url="http://localhost:1/v1", model="test")
    scheduler = Scheduler(
        engine=engine,
        llm=llm,
        log=log,
        config={"skills": {}},
        trace=trace,
    )
    return scheduler


def seed_decision(log, decision_id="abc123"):
    request = DecisionRequest(
        id=decision_id,
        state="some state",
        question="Pick one?",
        options=[Option("a", "A."), Option("b", "B.")],
    )
    result = DecisionResult(
        request=request, option_ids=["a", "b"], probabilities=[0.3, 0.7]
    )
    log.append(request, result, extra={"phase": "gate", "run_id": "run-1"})


class Server:
    def __init__(self, scheduler):
        handler = type("Handler", (DashboardHandler,), {"scheduler": scheduler})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]

    def get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as res:
            return res.status, json.loads(res.read().decode("utf-8"))

    def post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as res:
            return res.status, json.loads(res.read().decode("utf-8"))

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_tree_endpoint(tmp_path):
    server = Server(build_scheduler(tmp_path))
    try:
        status, payload = server.get("/api/tree")
        assert status == 200
        assert "email" in payload["categories"]
        assert any(s["name"] == "email.compose" for s in payload["categories"]["email"])
    finally:
        server.close()


def test_trace_endpoint_empty(tmp_path):
    server = Server(build_scheduler(tmp_path))
    try:
        status, payload = server.get("/api/trace")
        assert status == 200
        assert payload["runs"] == []
    finally:
        server.close()


def test_submit_without_engine_returns_error_json(tmp_path):
    scheduler = build_scheduler(tmp_path)
    server = Server(scheduler)
    try:
        status, payload = server.post("/api/submit", {"text": "do something"})
        assert status == 200
        assert payload["status"] == "error"
        assert "engine" in payload["detail"]
    finally:
        server.close()


def test_relabel_roundtrip(tmp_path):
    scheduler = build_scheduler(tmp_path)
    seed_decision(scheduler.log)
    server = Server(scheduler)
    try:
        status, payload = server.get("/api/trace")
        assert status == 200
        assert payload["runs"]
        assert payload["runs"][0]["decisions"][0]["id"] == "abc123"

        status, payload = server.post("/api/relabel", {"id": "abc123", "outcome": "b"})
        assert status == 200
        assert payload["ok"] is True

        status, payload = server.get("/api/trace")
        row = payload["runs"][0]["decisions"][0]
        assert row["observed_outcome"] == "b"
        assert row["label_source"] == "human"
        assert row["cost"]["weight"] == 3.0

        status, payload = server.post("/api/relabel", {"id": "nope", "outcome": "a"})
        assert payload["ok"] is False
    finally:
        server.close()


def test_submit_trace_event_recorded_even_when_engine_missing(tmp_path):
    scheduler = build_scheduler(tmp_path)
    server = Server(scheduler)
    try:
        server.post("/api/submit", {"text": "hello"})
        status, payload = server.get("/api/trace")
        runs = payload["runs"]
        assert len(runs) == 1
        kinds = [e["kind"] for e in runs[0]["events"]]
        assert "submit" in kinds
    finally:
        server.close()


def test_skill_writing_and_created_events_in_payload(tmp_path):
    scheduler = build_scheduler(tmp_path)
    scheduler.trace.append("submit", "run-9", text="track my package")
    scheduler.trace.append(
        "skill_writing",
        "run-9",
        category="tracking",
        skill="track_live",
        description="Follow a package in real time.",
        model="qwen38-iq3s",
    )
    scheduler.trace.append(
        "skill_created",
        "run-9",
        category="tracking",
        skill="track_live",
        description="Follow a package in real time.",
        body="data/skills/tracking/track_live.py",
        written=True,
    )
    server = Server(scheduler)
    try:
        status, payload = server.get("/api/trace")
        assert status == 200
        run = next(r for r in payload["runs"] if r["run_id"] == "run-9")
        kinds = [e["kind"] for e in run["events"]]
        assert "skill_writing" in kinds and "skill_created" in kinds
        writing = next(e for e in run["events"] if e["kind"] == "skill_writing")
        assert writing["skill"] == "track_live"
        assert "real time" in writing["description"]
        assert writing["model"] == "qwen38-iq3s"
        created = next(e for e in run["events"] if e["kind"] == "skill_created")
        assert created["written"] is True
        assert created["body"] == "data/skills/tracking/track_live.py"
    finally:
        server.close()


def need_input_skill(seen):
    def predict(ctx, request):
        return Prediction(text="", decisions=[])

    def act(ctx, request, prediction):
        if request.user_input:
            seen.append(request.user_input)
            return ActionResult(action_log="ok", new_state=f"done {request.user_input}")
        return ActionResult(
            action_log="ask",
            new_state=request.text,
            needs_input="What's the tracking number?",
        )

    return Skill(
        name="track.manual",
        category="tracking",
        description="Resolve a tracking number with the human.",
        predict=predict,
        act=act,
    )


def test_answer_without_pending_returns_error_json(tmp_path):
    scheduler = build_scheduler(tmp_path)
    server = Server(scheduler)
    try:
        status, payload = server.post("/api/answer", {"text": "hello"})
        assert status == 200
        assert payload["status"] == "error"
        assert "waiting for input" in payload["detail"]
    finally:
        server.close()


def test_status_includes_pending(tmp_path):
    scheduler = build_scheduler(tmp_path)
    scheduler._run_skill(need_input_skill([]), Request("track my package"))
    server = Server(scheduler)
    try:
        status, payload = server.get("/api/status")
        assert status == 200
        assert payload["pending"]["skill"] == "track.manual"
        assert payload["pending"]["question"] == "What's the tracking number?"
    finally:
        server.close()


def test_answer_roundtrip_via_api(tmp_path):
    scheduler = build_scheduler(tmp_path)
    seen = []
    scheduler._run_skill(need_input_skill(seen), Request("track my package"))
    server = Server(scheduler)
    try:
        status, payload = server.post("/api/answer", {"text": "AB123"})
        assert status == 200
        assert payload["status"] == "ran"
        assert seen == ["AB123"]

        status, payload = server.get("/api/status")
        assert payload["pending"] is None
    finally:
        server.close()