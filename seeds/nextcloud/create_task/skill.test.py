"""Hermetic mechanics test for nextcloud.create_task.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the Nextcloud bridge (`GET /tasklists`, `POST /tasks`) and the
LLM bridge's `POST /chat`. This proves the body phrases the task through the LLM
bridge, picks a task list (the sole one, or a SemIf choice when there are
several), creates the task through the bridge, and fails honestly when the LLM
bridge or the Nextcloud bridge errors. It does NOT prove the live integration —
only a real run against the user's Nextcloud does.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_agent.decisions import Request
from semif_agent.skills import ActionContext

import skill


def make_handler(routes, seen):
    """Build a request handler that records every request and replays `routes`."""

    class Handler(BaseHTTPRequestHandler):
        def _respond(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else None
            except ValueError:
                body = raw.decode("utf-8", "replace")
            seen.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                }
            )
            key = (self.command, self.path.split("?")[0])
            status, payload = routes.get(key, (404, {"error": "no route"}))
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = _respond
        do_POST = _respond

        def log_message(self, *args):
            pass

    return Handler


def start_server(routes):
    seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(routes, seen))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d" % server.server_address[1]
    return server, seen, url


def stop(server):
    server.shutdown()
    server.server_close()


def closed_port_url():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return "http://127.0.0.1:%d" % port


class FakeResult:
    """Minimal stand-in for a DecisionResult: the body only reads `.selected`."""

    def __init__(self, selected):
        self.selected = selected


class FakeEngine:
    """Test-only decision engine; records the requests the body asked for."""

    def __init__(self, choice=None):
        self.choice = choice
        self.calls = []

    def call(self, decision):
        self.calls.append(decision)
        return FakeResult(self.choice)


def test_happy_path_single_task_list():
    llm_routes = {
        ("POST", "/chat"): (
            200,
            {
                "text": '{"summary": "Eat hot dogs", '
                '"description": "Buy and eat hot dogs"}'
            },
        )
    }
    nc_routes = {
        ("GET", "/tasklists"): (
            200,
            {"tasklists": [{"name": "tasks", "label": "Tasks"}]},
        ),
        ("POST", "/tasks"): (
            200,
            {"ok": True, "uid": "uid-1", "calendar": "tasks"},
        ),
    }
    llm_server, llm_seen, llm_url = start_server(llm_routes)
    nc_server, nc_seen, nc_url = start_server(nc_routes)
    try:
        engine = FakeEngine()
        ctx = ActionContext(
            engine=engine,
            config={
                "nextcloud_bridge_url": nc_url,
                "nextcloud_bridge_token": "nc-secret",
                "llm_bridge_url": llm_url,
                "llm_bridge_token": "llm-secret",
            },
        )
        request = Request("Add a task in next cloud for eating hot dogs.")
        result = skill.act(ctx, request)

        assert not engine.calls, "a single task list must not need a decision"
        assert "Eat hot dogs" in result.action_log, result.action_log
        assert "Tasks" in result.action_log, result.action_log
        assert "uid-1" in result.action_log, result.action_log

        assert len(llm_seen) == 1, llm_seen
        chat = llm_seen[0]
        assert chat["method"] == "POST", chat
        assert chat["path"] == "/chat", chat
        assert chat["headers"].get("x-semif-token") == "llm-secret", chat["headers"]
        assert chat["body"]["messages"][1]["content"] == request.text, chat["body"]
        assert "summary" in chat["body"]["messages"][0]["content"], chat["body"]

        assert [r["path"] for r in nc_seen] == ["/tasklists", "/tasks"], nc_seen
        listing = nc_seen[0]
        assert listing["method"] == "GET", listing
        assert listing["headers"].get("x-semif-token") == "nc-secret", listing["headers"]
        create = nc_seen[1]
        assert create["method"] == "POST", create
        assert create["headers"].get("x-semif-token") == "nc-secret", create["headers"]
        assert create["body"] == {
            "calendar": "tasks",
            "summary": "Eat hot dogs",
            "description": "Buy and eat hot dogs",
        }, create["body"]
        print("happy path:", result.action_log)
    finally:
        stop(llm_server)
        stop(nc_server)


def test_multiple_task_lists_asks_and_omits_token_header():
    llm_routes = {
        ("POST", "/chat"): (200, {"text": '{"summary": "Eat hot dogs"}'})
    }
    nc_routes = {
        ("GET", "/tasklists"): (
            200,
            {
                "tasklists": [
                    {"name": "tasks", "label": "Tasks"},
                    {"name": "work", "label": "Work"},
                ]
            },
        ),
        ("POST", "/tasks"): (200, {"ok": True, "uid": "uid-2", "calendar": "work"}),
    }
    llm_server, llm_seen, llm_url = start_server(llm_routes)
    nc_server, nc_seen, nc_url = start_server(nc_routes)
    try:
        engine = FakeEngine(choice="work")
        ctx = ActionContext(
            engine=engine,
            config={
                "nextcloud_bridge_url": nc_url,
                "nextcloud_bridge_token": "",
                "llm_bridge_url": llm_url,
                "llm_bridge_token": "",
            },
        )
        result = skill.act(
            ctx, Request("Add a task in next cloud for eating hot dogs.")
        )

        assert len(engine.calls) == 1, "two task lists must trigger one decision"
        decision = engine.calls[0]
        assert [o.id for o in decision.options] == ["tasks", "work"], decision.options
        assert result.decisions, "the decision must be recorded on the result"

        assert "x-semif-token" not in llm_seen[0]["headers"], llm_seen[0]["headers"]
        create = nc_seen[-1]
        assert create["path"] == "/tasks", create
        assert create["body"]["calendar"] == "work", create["body"]
        assert create["body"]["description"] == "", create["body"]
        assert "x-semif-token" not in create["headers"], create["headers"]
        assert "Work" in result.action_log, result.action_log
        print("decision path:", result.action_log)
    finally:
        stop(llm_server)
        stop(nc_server)


def test_llm_bridge_failure_leaves_nextcloud_untouched():
    llm_routes = {("POST", "/chat"): (502, {"error": "model down"})}
    nc_routes = {
        ("GET", "/tasklists"): (200, {"tasklists": [{"name": "tasks"}]})
    }
    llm_server, _llm_seen, llm_url = start_server(llm_routes)
    nc_server, nc_seen, nc_url = start_server(nc_routes)
    try:
        ctx = ActionContext(
            engine=FakeEngine(),
            config={
                "nextcloud_bridge_url": nc_url,
                "llm_bridge_url": llm_url,
            },
        )
        result = skill.act(ctx, Request("Add a task in next cloud for hot dogs."))
        assert "could not phrase the task" in result.action_log, result.action_log
        assert "model down" in result.action_log, result.action_log
        assert nc_seen == [], "Nextcloud must not be touched when phrasing fails"
        print("llm failure:", result.action_log)
    finally:
        stop(llm_server)
        stop(nc_server)


def test_llm_reply_without_task_text():
    llm_routes = {("POST", "/chat"): (200, {"text": "I cannot help with that."})}
    nc_routes = {
        ("GET", "/tasklists"): (200, {"tasklists": [{"name": "tasks"}]})
    }
    llm_server, _llm_seen, llm_url = start_server(llm_routes)
    nc_server, nc_seen, nc_url = start_server(nc_routes)
    try:
        ctx = ActionContext(
            engine=FakeEngine(),
            config={"nextcloud_bridge_url": nc_url, "llm_bridge_url": llm_url},
        )
        result = skill.act(ctx, Request("do something vague"))
        assert "no usable task text" in result.action_log, result.action_log
        assert nc_seen == [], "Nextcloud must not be touched without a task title"
        print("unusable llm reply:", result.action_log)
    finally:
        stop(llm_server)
        stop(nc_server)


def test_no_task_lists_available():
    llm_routes = {
        ("POST", "/chat"): (200, {"text": '{"summary": "Eat hot dogs"}'})
    }
    nc_routes = {("GET", "/tasklists"): (200, {"tasklists": []})}
    llm_server, _llm_seen, llm_url = start_server(llm_routes)
    nc_server, nc_seen, nc_url = start_server(nc_routes)
    try:
        ctx = ActionContext(
            engine=FakeEngine(),
            config={"nextcloud_bridge_url": nc_url, "llm_bridge_url": llm_url},
        )
        result = skill.act(ctx, Request("Add a task in next cloud for hot dogs."))
        assert "no Nextcloud task lists" in result.action_log, result.action_log
        assert [r["path"] for r in nc_seen] == ["/tasklists"], nc_seen
        print("no task lists:", result.action_log)
    finally:
        stop(llm_server)
        stop(nc_server)


def test_unreachable_nextcloud_bridge():
    llm_routes = {
        ("POST", "/chat"): (200, {"text": '{"summary": "Eat hot dogs"}'})
    }
    llm_server, _llm_seen, llm_url = start_server(llm_routes)
    try:
        ctx = ActionContext(
            engine=FakeEngine(),
            config={
                "nextcloud_bridge_url": closed_port_url(),
                "llm_bridge_url": llm_url,
            },
        )
        result = skill.act(ctx, Request("Add a task in next cloud for hot dogs."))
        assert "could not list Nextcloud task lists" in result.action_log, result.action_log
        print("unreachable nextcloud:", result.action_log)
    finally:
        stop(llm_server)


def test_task_creation_failure_is_reported():
    llm_routes = {
        ("POST", "/chat"): (200, {"text": '{"summary": "Eat hot dogs"}'})
    }
    nc_routes = {
        ("GET", "/tasklists"): (
            200,
            {"tasklists": [{"name": "tasks", "label": "Tasks"}]},
        ),
        ("POST", "/tasks"): (502, {"error": "CalDAV rejected the VTODO"}),
    }
    llm_server, _llm_seen, llm_url = start_server(llm_routes)
    nc_server, nc_seen, nc_url = start_server(nc_routes)
    try:
        ctx = ActionContext(
            engine=FakeEngine(),
            config={"nextcloud_bridge_url": nc_url, "llm_bridge_url": llm_url},
        )
        result = skill.act(ctx, Request("Add a task in next cloud for hot dogs."))
        assert "failed to add task" in result.action_log, result.action_log
        assert "CalDAV rejected the VTODO" in result.action_log, result.action_log
        assert [r["path"] for r in nc_seen] == ["/tasklists", "/tasks"], nc_seen
        print("creation failure:", result.action_log)
    finally:
        stop(llm_server)
        stop(nc_server)


def main():
    test_happy_path_single_task_list()
    test_multiple_task_lists_asks_and_omits_token_header()
    test_llm_bridge_failure_leaves_nextcloud_untouched()
    test_llm_reply_without_task_text()
    test_no_task_lists_available()
    test_unreachable_nextcloud_bridge()
    test_task_creation_failure_is_reported()
    print("ok")


if __name__ == "__main__":
    main()
