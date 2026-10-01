"""Local browser dashboard for the SemIf agent.

A pure-stdlib HTTP server on localhost serving a Redux-DevTools-style
inspector over the agent's decision flow. Reads the decision log
(`decisions.jsonl`) plus the run lifecycle trace (`runs.jsonl`), exposes the
static skill tree, the dream cost report, and the write endpoints: submit a
request, answer a run paused for input, answer a deferred authoring question,
resolve a repair offer, restart a skill write, and relabel a decision (human
override).

The scheduler's engine and LLM are built lazily, so the dashboard runs on the
thin dev box in replay mode (reads logs; submit degrades to a JSON error) and
in live mode on the box with SemIf + a local LLM.

    python -m semif_agent.cli dashboard [--port 8765]
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .dream import dream
from .scheduler import Scheduler
from .skills import build_skills, build_tree

STATIC_DIR = Path(__file__).parent / "static"

JSON_HEADERS = {"Content-Type": "application/json; charset=utf-8"}


def _phase(row: dict) -> str:
    return (row.get("extra") or {}).get("phase", "")


def _enrich_decision(row: dict, costs: dict) -> dict:
    out = dict(row)
    out["phase"] = _phase(row)
    cost = costs.get(row.get("id"))
    if cost is not None:
        out["cost"] = cost
    return out


def build_payload(scheduler: Scheduler) -> dict:
    """Group logged decisions by run, attach lifecycle events and dream costs."""
    events_by_run = scheduler.trace.runs()
    rows = scheduler.log.read()

    costs = {}
    for cost_row in dream(scheduler.log).rows:
        costs[cost_row.decision_id] = {
            "predicted": cost_row.predicted,
            "nll": cost_row.nll,
            "weight": cost_row.weight,
            "correct": cost_row.correct,
        }

    by_run: dict[str, dict] = {}
    for row in rows:
        run_id = (row.get("extra") or {}).get("run_id", "?")
        run = by_run.setdefault(
            run_id, {"run_id": run_id, "events": [], "decisions": [], "first_ts": None}
        )
        run["first_ts"] = row["ts"] if run["first_ts"] is None else min(run["first_ts"], row["ts"])
        run["decisions"].append(_enrich_decision(row, costs))
    for run_id, events in events_by_run.items():
        run = by_run.setdefault(
            run_id, {"run_id": run_id, "events": [], "decisions": [], "first_ts": None}
        )
        run["events"] = events
        event_ts = [e["ts"] for e in events if "ts" in e]
        if event_ts:
            earliest = min(event_ts)
            run["first_ts"] = earliest if run["first_ts"] is None else min(run["first_ts"], earliest)

    runs = sorted(by_run.values(), key=lambda r: (r["first_ts"] is None, r["first_ts"] or 0))
    for run in runs:
        run["decisions"].sort(key=lambda d: d["ts"])
    return {"runs": runs}


def build_dream_report(scheduler: Scheduler) -> dict:
    report = dream(scheduler.log)
    return {
        "rows": len(report.rows),
        "skipped": report.skipped,
        "human_overrides": report.human_overrides,
        "cross_entropy": report.cross_entropy,
        "accuracy": report.accuracy,
        "ece": report.ece,
    }


def build_status(scheduler: Scheduler) -> dict:
    with scheduler._lock:
        current = (
            {"skill": scheduler.current.skill, "request_id": scheduler.current.request.id}
            if scheduler.current
            else None
        )
        pending = (
            {
                "skill": scheduler.pending.skill.name,
                "question": scheduler.pending.question,
                "run_id": scheduler.pending.request.id,
            }
            if scheduler.pending
            else None
        )
        queue = [
            {"id": request.id, "weight": weight, "text": request.text[:80]}
            for weight, request in scheduler.queue.items()
        ]
        tau = scheduler.tau
        queue_max = scheduler.queue.max_size
        questions = scheduler.pending_questions()
        repairs = scheduler.pending_repairs()
    return {
        "current": current,
        "pending": pending,
        "queue": queue,
        "tau": tau,
        "queue_max": queue_max,
        "questions": questions,
        "repairs": repairs,
    }


def build_tree_payload(scheduler: Scheduler) -> dict:
    with scheduler._lock:
        snapshot = {
            category: list(skills) for category, skills in scheduler.tree.items()
        }
    return {
        "categories": {
            category: [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "status": skill.status,
                    "integration": skill.integration or None,
                    "integration_source": skill.integration_source,
                }
                for skill in skills
            ]
            for category, skills in sorted(snapshot.items())
        }
    }


class DashboardHandler(BaseHTTPRequestHandler):
    scheduler: Scheduler
    lock: threading.Lock = threading.Lock()

    server_version = "semif-dashboard/0.1"

    # ---- helpers ----

    def _send(self, code: int, payload: dict, headers: dict | None = None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        for key, value in JSON_HEADERS.items():
            self.send_header(key, value)
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str):
        self._send(code, {"error": message})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _serve_static(self, rel: str):
        if not rel or rel == "/":
            rel = "index.html"
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())):
            self._send_error(403, "forbidden")
            return
        if not target.is_file():
            self._send_error(404, "not found")
            return
        body = target.read_bytes()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get(target.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- routing ----

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/" or path.startswith("/static/"):
            rel = path.removeprefix("/static/").removeprefix("/")
            self._serve_static(rel)
            return
        if path == "/api/trace":
            self._send(200, build_payload(self.scheduler))
            return
        if path == "/api/tree":
            self._send(200, build_tree_payload(self.scheduler))
            return
        if path == "/api/dream":
            self._send(200, build_dream_report(self.scheduler))
            return
        if path == "/api/status":
            self._send(200, build_status(self.scheduler))
            return
        if path == "/api/questions":
            self._send(200, {"questions": self.scheduler.pending_questions()})
            return
        if path == "/api/repairs":
            self._send(200, {"repairs": self.scheduler.pending_repairs()})
            return
        self._send_error(404, "no such endpoint")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/submit":
            with self.lock:
                try:
                    body = self._read_json()
                    status, detail, request_id = self.scheduler.submit_request(
                        str(body.get("text", "")), source=str(body.get("source", "dashboard"))
                    )
                except Exception as exc:
                    self._send_error(500, str(exc))
                    return
            self._send(200, {"status": status, "detail": detail, "request_id": request_id})
            return
        if path == "/api/relabel":
            with self.lock:
                try:
                    body = self._read_json()
                    ok = self.scheduler.log.relabel(str(body.get("id", "")), str(body.get("outcome", "")))
                except Exception as exc:
                    self._send_error(500, str(exc))
                    return
            self._send(200, {"ok": ok})
            return
        if path == "/api/answer":
            with self.lock:
                try:
                    body = self._read_json()
                    status, detail = self.scheduler.answer(str(body.get("text", "")))
                except Exception as exc:
                    self._send_error(500, str(exc))
                    return
            self._send(200, {"status": status, "detail": detail})
            return
        if path == "/api/restart":
            with self.lock:
                try:
                    body = self._read_json()
                    status, detail = self.scheduler.restart_skill(
                        str(body.get("category", "")), str(body.get("skill", ""))
                    )
                except Exception as exc:
                    self._send_error(500, str(exc))
                    return
            self._send(200, {"status": status, "detail": detail})
            return
        if path == "/api/questions":
            with self.lock:
                try:
                    body = self._read_json()
                    status, detail = self.scheduler.answer_question(
                        str(body.get("id", "")), str(body.get("text", ""))
                    )
                except Exception as exc:
                    self._send_error(500, str(exc))
                    return
            self._send(200, {"status": status, "detail": detail})
            return
        if path == "/api/repair":
            with self.lock:
                try:
                    body = self._read_json()
                    action = body.get("action")
                    status, detail = self.scheduler.resolve_repair(
                        str(body.get("id", "")),
                        None if action is None else str(action),
                    )
                except Exception as exc:
                    self._send_error(500, str(exc))
                    return
            self._send(200, {"status": status, "detail": detail})
            return
        self._send_error(404, "no such endpoint")

    def log_message(self, format, *args):
        pass


def serve(scheduler: Scheduler, port: int = 8765, host: str = "127.0.0.1") -> None:
    handler = type("Handler", (DashboardHandler,), {"scheduler": scheduler, "lock": threading.Lock()})
    server = ThreadingHTTPServer((host, port), handler)
    bound = server.server_address[0]
    url = f"http://{bound}:{server.server_address[1]}/" if bound != "0.0.0.0" else _lan_url(port)
    print(f"semif dashboard on {url}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _lan_url(port: int) -> str:
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
    except OSError:
        ip = "127.0.0.1"
    return f"http://{ip}:{port}/"