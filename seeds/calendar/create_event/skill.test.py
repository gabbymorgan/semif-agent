"""Hermetic mechanics test for calendar.create_event.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays a Nextcloud CalDAV endpoint (PROPFIND discovery + PUT create)
and the LLM bridge's `POST /chat`. This proves the body extracts the event title
and description through the LLM bridge (caching the call across a `needs_input`
resume), parses the date/time from the request, asks for a missing field and
resumes, resolves the target calendar with a SemIf choice over the owned
calendars (a strong winner is used, otherwise the configured default), writes a
real iCalendar PUT, and fails honestly when the LLM bridge or the server errors.
It does NOT prove the live integration — only a real run against the user's
Nextcloud does.
"""

import json
import sys
import threading
import urllib.parse
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_agent.decisions import DecisionResult, Request
from semif_agent.skills import ActionContext

import skill

FIXED_NOW = datetime(2026, 10, 7, 10, 0, tzinfo=timezone.utc)
skill._now = lambda ctx: FIXED_NOW

DEFAULT_LLM_REPLY = {"title": "event", "description": ""}


def next_weekday(now, weekday):
    return now.date() + timedelta(days=(weekday - now.date().weekday()) % 7)


class FakeEngine:
    """Test-only engine.

    By default it picks the requested option id (else the first) with full
    confidence. Pass `probs` (a mapping of option id -> probability) to script an
    arbitrary distribution, e.g. a near-tie that must fall back to the default.
    """

    def __init__(self, pick=None, probs=None):
        self.pick = pick
        self.probs = probs
        self.decisions = []

    def call(self, decision):
        self.decisions.append(decision)
        option_ids = [option.id for option in decision.options]
        if self.probs is not None:
            probabilities = [float(self.probs.get(option_id, 0.0)) for option_id in option_ids]
        else:
            chosen = self.pick if self.pick in option_ids else option_ids[0]
            probabilities = [
                1.0 if option_id == chosen else 0.0 for option_id in option_ids
            ]
        return DecisionResult(
            request=decision, option_ids=option_ids, probabilities=probabilities
        )


class CalDavHandler(BaseHTTPRequestHandler):
    calendars = []
    put_status = 201
    requests = []
    puts = []
    llm_reply = DEFAULT_LLM_REPLY
    llm_status = 200
    llm_requests = []

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _send(self, payload, status, content_type):
        body = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PROPFIND(self):
        self._read_body()
        self.requests.append(("PROPFIND", self.path))
        responses = []
        for calendar in self.calendars:
            responses.append(
                f'<d:response><d:href>/remote.php/dav/calendars/alice/'
                f'{calendar["name"]}/</d:href>'
                f"<d:propstat><d:prop>"
                f'<d:displayname>{calendar["label"]}</d:displayname>'
                f"<d:resourcetype><d:collection/><c:calendar/></d:resourcetype>"
                f"</d:prop><d:status>HTTP/1.1 200 OK</d:status></d:propstat>"
                f"</d:response>"
            )
        body = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
            + "".join(responses)
            + "</d:multistatus>"
        )
        self._send(body, 207, "application/xml; charset=utf-8")

    def do_PUT(self):
        body = self._read_body()
        self.puts.append(
            {"path": self.path, "body": body, "headers": dict(self.headers)}
        )
        self._send(b"", self.put_status, "text/plain")

    def do_POST(self):
        body = self._read_body()
        if self.path != "/chat":
            self._send(json.dumps({"error": "not found"}), 404, "application/json")
            return
        self.llm_requests.append(body)
        if self.llm_status != 200:
            self._send(json.dumps({"error": "boom"}), self.llm_status, "application/json")
            return
        self._send(
            json.dumps({"text": json.dumps(self.llm_reply)}),
            200,
            "application/json",
        )

    def log_message(self, *args):
        pass


def start_server(calendars=None, put_status=201, llm_reply=None, llm_status=200):
    handler = type(
        "Handler",
        (CalDavHandler,),
        {
            "calendars": calendars
            if calendars is not None
            else [
                {"name": "personal", "label": "Personal"},
                {"name": "work", "label": "Work"},
            ],
            "put_status": put_status,
            "requests": [],
            "puts": [],
            "llm_reply": llm_reply if llm_reply is not None else DEFAULT_LLM_REPLY,
            "llm_status": llm_status,
            "llm_requests": [],
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler, f"http://127.0.0.1:{server.server_address[1]}"


def config(url, default="personal", llm_url=None):
    return {
        "nextcloud_url": url,
        "nextcloud_username": "alice",
        "nextcloud_app_password": "secret",
        "nextcloud_default_calendar": default,
        "llm_bridge_url": url if llm_url is None else llm_url,
        "llm_bridge_token": "",
    }


def run(handler, url, text, default="personal", engine=None, llm_url=None):
    engine = engine or FakeEngine()
    ctx = ActionContext(engine=engine, config=config(url, default, llm_url))
    request = Request(text)
    action = skill.act(ctx, request)
    return action, request, engine


def test_example_creates_event_on_default_calendar():
    server, handler, url = start_server(
        llm_reply={
            "title": "Ashley's dance recital",
            "description": "Dance recital at 5pm.",
        }
    )
    try:
        # The query names no calendar, so the SemIf choice is a near-tie and the
        # configured default ("personal") wins.
        engine = FakeEngine(probs={"personal": 0.4, "work": 0.35})
        action, _request, engine = run(
            handler,
            url,
            "add Ashley's dance recital to the calendar for 5pm Tuesday",
            engine=engine,
        )
        assert action.needs_input is None, action.needs_input
        assert len(engine.decisions) == 1, "the calendar is chosen with one SemIf decision"
        assert engine.decisions[0].state == (
            "add Ashley's dance recital to the calendar for 5pm Tuesday"
        ), engine.decisions[0]
        assert "referred to in the user query" in engine.decisions[0].question
        assert len(handler.puts) == 1, handler.puts
        put = handler.puts[0]
        assert put["path"].startswith("/remote.php/dav/calendars/alice/personal/"), put
        assert put["path"].endswith(".ics"), put
        assert put["headers"].get("If-None-Match") == "*"
        body = put["body"].decode("utf-8")
        assert "SUMMARY:Ashley's dance recital" in body, body
        assert "DESCRIPTION:Dance recital at 5pm." in body, body
        tuesday = next_weekday(FIXED_NOW, 1)
        assert f"DTSTART:{tuesday.strftime('%Y%m%d')}T170000Z" in body, body
        assert f"DTEND:{tuesday.strftime('%Y%m%d')}T180000Z" in body, body
        assert "Ashley's dance recital" in action.action_log, action.action_log
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_missing_time_asks_then_creates():
    server, handler, url = start_server(
        llm_reply={"title": "dentist appointment", "description": ""}
    )
    try:
        action, request, engine = run(
            handler, url, "add dentist appointment on 2026-11-10"
        )
        assert action.needs_input, "a missing start time must be asked"
        assert "time" in action.needs_input.lower(), action.needs_input
        assert handler.puts == []
        assert len(handler.llm_requests) == 1, "the LLM is called once, then cached"
        request.user_input = "2pm"
        ctx = ActionContext(engine=engine, config=config(url))
        action = skill.act(ctx, request)
        assert action.needs_input is None, action.needs_input
        assert len(handler.puts) == 1, handler.puts
        assert len(handler.llm_requests) == 1, "the resume must reuse the cached fields"
        body = handler.puts[0]["body"].decode("utf-8")
        assert "SUMMARY:dentist appointment" in body, body
        assert "DTSTART:20261110T140000Z" in body, body
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_missing_day_asks_then_creates():
    server, handler, url = start_server(
        llm_reply={"title": "dentist appointment", "description": ""}
    )
    try:
        action, request, engine = run(handler, url, "add dentist appointment at 2pm")
        assert action.needs_input, "a missing day must be asked"
        assert "day" in action.needs_input.lower(), action.needs_input
        request.user_input = "2026-11-10"
        ctx = ActionContext(engine=engine, config=config(url))
        action = skill.act(ctx, request)
        assert action.needs_input is None, action.needs_input
        body = handler.puts[0]["body"].decode("utf-8")
        assert "DTSTART:20261110T140000Z" in body, body
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_missing_title_asks():
    server, handler, url = start_server(llm_reply={"title": "", "description": ""})
    try:
        action, _request, _engine = run(handler, url, "schedule on 2026-11-10 at 2pm")
        assert action.needs_input, "a missing title must be asked"
        assert "call the event" in action.needs_input.lower(), action.needs_input
        assert handler.puts == []
        print(action.needs_input)
    finally:
        server.shutdown()
        server.server_close()


def test_named_calendar_is_used():
    server, handler, url = start_server(
        llm_reply={"title": "team sync", "description": ""}
    )
    try:
        engine = FakeEngine(pick="work")
        action, _request, engine = run(
            handler,
            url,
            "add team sync to my work calendar on 2026-11-10 at 9am",
            engine=engine,
        )
        assert action.needs_input is None, action.needs_input
        assert len(engine.decisions) == 1, engine.decisions
        assert [option.id for option in engine.decisions[0].options] == [
            "personal",
            "work",
        ], engine.decisions[0].options
        assert handler.puts[0]["path"].startswith(
            "/remote.php/dav/calendars/alice/work/"
        ), handler.puts[0]
        body = handler.puts[0]["body"].decode("utf-8")
        assert "SUMMARY:team sync" in body, body
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_weak_winner_falls_back_to_default():
    server, handler, url = start_server(llm_reply={"title": "standup", "description": ""})
    try:
        # Work edges out personal but stays below the strong-winner threshold, so
        # the configured default ("personal") is used instead.
        engine = FakeEngine(probs={"personal": 0.45, "work": 0.5})
        action, _request, engine = run(
            handler,
            url,
            "add standup on 2026-11-10 at 9am",
            engine=engine,
        )
        assert action.needs_input is None, action.needs_input
        assert len(engine.decisions) == 1, engine.decisions
        assert handler.puts[0]["path"].startswith(
            "/remote.php/dav/calendars/alice/personal/"
        ), handler.puts[0]
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_unknown_default_uses_semif_decision():
    server, handler, url = start_server(llm_reply={"title": "standup", "description": ""})
    try:
        # The configured default is not among the account's calendars and the
        # SemIf winner is weak: the winner is used anyway (never blocks).
        engine = FakeEngine(probs={"personal": 0.3, "work": 0.4})
        action, _request, engine = run(
            handler,
            url,
            "add standup on 2026-11-10 at 9am",
            default="missing",
            engine=engine,
        )
        assert action.needs_input is None, action.needs_input
        assert len(engine.decisions) == 1, engine.decisions
        assert action.decisions and len(action.decisions) == 1, action.decisions
        assert handler.puts[0]["path"].startswith(
            "/remote.php/dav/calendars/alice/work/"
        ), handler.puts[0]
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_all_day_event():
    server, handler, url = start_server(llm_reply={"title": "conference", "description": ""})
    try:
        action, _request, _engine = run(
            handler, url, "add conference all day on 2026-11-10"
        )
        assert action.needs_input is None, action.needs_input
        body = handler.puts[0]["body"].decode("utf-8")
        assert "DTSTART;VALUE=DATE:20261110" in body, body
        assert "DTEND;VALUE=DATE:20261111" in body, body
        assert "SUMMARY:conference" in body, body
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_llm_bridge_failure_is_reported_honestly():
    server, handler, url = start_server()
    try:
        action, request, _engine = run(
            handler,
            url,
            "add standup on 2026-11-10 at 9am",
            llm_url="http://127.0.0.1:1",
        )
        assert "llm bridge" in action.action_log, action.action_log
        assert action.new_state == request.text, "a failed extraction must not fake a result"
        assert handler.puts == [], "no event is created when the LLM bridge is down"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_llm_bridge_error_status_is_reported_honestly():
    server, handler, url = start_server(llm_status=502)
    try:
        action, request, _engine = run(
            handler, url, "add standup on 2026-11-10 at 9am"
        )
        assert "HTTP 502" in action.action_log, action.action_log
        assert action.new_state == request.text
        assert handler.puts == []
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_server_error_is_reported_honestly():
    server, handler, url = start_server(
        put_status=500, llm_reply={"title": "standup", "description": ""}
    )
    try:
        action, request, _engine = run(
            handler, url, "add standup on 2026-11-10 at 9am"
        )
        assert "500" in action.action_log, action.action_log
        assert action.new_state == request.text, "a failed create must not fake a result"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()


def test_unreachable_server_fails_honestly():
    action, request, _engine = run(
        None, "http://127.0.0.1:1", "add standup on 2026-11-10 at 9am"
    )
    assert action.action_log.startswith("calendar.create_event:"), action.action_log
    assert action.new_state == request.text
    print(action.action_log)


def main():
    test_example_creates_event_on_default_calendar()
    test_missing_time_asks_then_creates()
    test_missing_day_asks_then_creates()
    test_missing_title_asks()
    test_named_calendar_is_used()
    test_weak_winner_falls_back_to_default()
    test_unknown_default_uses_semif_decision()
    test_all_day_event()
    test_llm_bridge_failure_is_reported_honestly()
    test_llm_bridge_error_status_is_reported_honestly()
    test_server_error_is_reported_honestly()
    test_unreachable_server_fails_honestly()
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
