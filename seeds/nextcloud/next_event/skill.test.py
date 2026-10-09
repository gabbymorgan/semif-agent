"""Hermetic mechanics test for nextcloud.next_event.

Run from this folder: `python skill.test.py`. No external network: a loopback
http.server plays the Nextcloud CalDAV endpoint. This proves the body builds
real CalDAV requests, discovers calendars, resolves the calendar through a
SemIf sub-decision (a strong winner is used, otherwise the configured default,
otherwise the weak winner), parses expanded iCalendar events, and fails honestly
when the service errors. It does NOT prove the live integration — only a real
run against the user's Nextcloud does.
"""

import base64
import sys
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_agent.decisions import DecisionResult, Request
from semif_agent.skills import ActionContext

import skill

CALENDARS = [("personal", "Personal"), ("work", "Work")]
ICS = ""


def build_ics():
    """Fixture events relative to now, so the test never rots."""
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=2)
    soon = now + timedelta(hours=2)
    all_day = (now + timedelta(days=2)).date()

    def stamp(value):
        return value.strftime("%Y%m%dT%H%M%SZ")

    return (
        "BEGIN:VCALENDAR\nVERSION:2.0\n"
        "BEGIN:VEVENT\nUID:past@example\n"
        f"DTSTART:{stamp(past)}\nDTEND:{stamp(past + timedelta(hours=1))}\n"
        "SUMMARY:Already happened\nEND:VEVENT\n"
        "BEGIN:VEVENT\nUID:soon@example\n"
        f"DTSTART:{stamp(soon)}\nDTEND:{stamp(soon + timedelta(hours=1))}\n"
        "SUMMARY:Team sync\nLOCATION:Room 3\nEND:VEVENT\n"
        "BEGIN:VEVENT\nUID:allday@example\n"
        f"DTSTART;VALUE=DATE:{all_day.strftime('%Y%m%d')}\n"
        "SUMMARY:Offsite\nEND:VEVENT\n"
        "END:VCALENDAR"
    )


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
    requests = []
    calendar_count = 2
    report_status = 207

    def _record(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.requests.append(
            {
                "command": self.command,
                "path": self.path,
                "depth": self.headers.get("Depth"),
                "authorization": self.headers.get("Authorization"),
                "body": body.decode("utf-8", "replace"),
            }
        )

    def _multistatus(self, payload):
        body = payload.encode("utf-8")
        self.send_response(207)
        self.send_header("Content-Type", "application/xml; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PROPFIND(self):
        self._record()
        responses = "".join(
            f"<d:response><d:href>/remote.php/dav/calendars/testuser/{name}/</d:href>"
            "<d:propstat><d:prop>"
            f"<d:displayname>{label}</d:displayname>"
            "<d:resourcetype><d:collection/><c:calendar/></d:resourcetype>"
            "</d:prop><d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>"
            for name, label in CALENDARS[: self.calendar_count]
        )
        self._multistatus(
            '<?xml version="1.0" encoding="utf-8"?>'
            '<d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
            f"{responses}</d:multistatus>"
        )

    def do_REPORT(self):
        self._record()
        if self.report_status != 207:
            self.send_response(self.report_status)
            self.end_headers()
            return
        self._multistatus(
            '<?xml version="1.0" encoding="utf-8"?>'
            '<d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
            "<d:response>"
            "<d:href>/remote.php/dav/calendars/testuser/work/event.ics</d:href>"
            "<d:propstat><d:prop>"
            f"<c:calendar-data>{ICS}</c:calendar-data>"
            "</d:prop><d:status>HTTP/1.1 200 OK</d:status></d:propstat>"
            "</d:response></d:multistatus>"
        )

    def log_message(self, *args):
        pass


def start_server(**overrides):
    handler = type("Handler", (CalDavHandler,), {"requests": [], **overrides})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, handler, f"http://127.0.0.1:{server.server_address[1]}"


def fixture_config(url, default="personal"):
    return {
        "nextcloud_url": url,
        "nextcloud_username": "testuser",
        "nextcloud_app_password": "app-pw",
        "nextcloud_default_calendar": default,
    }


def main():
    global ICS
    ICS = build_ics()

    # A strong winner is used even though a default is configured.
    server, handler, url = start_server()
    try:
        engine = FakeEngine(pick="work")
        ctx = ActionContext(engine=engine, config=fixture_config(url))
        request = Request("what is the next event on my work calendar?")
        action = skill.act(ctx, request)
        assert len(action.decisions) == 1, "act must log the calendar choice"
        assert len(engine.decisions) == 1
        assert [option.id for option in engine.decisions[0].options] == [
            "personal",
            "work",
        ], engine.decisions[0].options
        assert "Team sync" in action.action_log, action.action_log
        assert action.new_state, "act must set a new state"

        commands = [entry["command"] for entry in handler.requests]
        assert commands == ["PROPFIND", "REPORT"], commands
        report = handler.requests[1]
        assert report["path"] == "/remote.php/dav/calendars/testuser/work/", report["path"]
        assert report["depth"] == "1", report["depth"]
        expected_auth = "Basic " + base64.b64encode(b"testuser:app-pw").decode("ascii")
        assert report["authorization"] == expected_auth, report["authorization"]
        assert "<c:expand" in report["body"], "recurrence expansion must be requested"
        assert "c:time-range" in report["body"], "the query must bound a time range"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()

    # A weak winner falls back to the configured default calendar.
    server, handler, url = start_server()
    try:
        engine = FakeEngine(probs={"personal": 0.45, "work": 0.5})
        ctx = ActionContext(engine=engine, config=fixture_config(url, default="personal"))
        request = Request("what is the next event on my calendar?")
        action = skill.act(ctx, request)
        assert len(engine.decisions) == 1
        assert handler.requests[1]["path"] == "/remote.php/dav/calendars/testuser/personal/", handler.requests[1]["path"]
        assert "Team sync" in action.action_log, action.action_log
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()

    # A weak winner with no configured default stands.
    server, handler, url = start_server()
    try:
        engine = FakeEngine(probs={"personal": 0.45, "work": 0.5})
        ctx = ActionContext(engine=engine, config=fixture_config(url, default=""))
        request = Request("what is the next event on my calendar?")
        action = skill.act(ctx, request)
        assert len(engine.decisions) == 1
        assert handler.requests[1]["path"] == "/remote.php/dav/calendars/testuser/work/", handler.requests[1]["path"]
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()

    # A single calendar needs no SemIf decision.
    server, handler, url = start_server(calendar_count=1)
    try:
        engine = FakeEngine()
        ctx = ActionContext(engine=engine, config=fixture_config(url))
        request = Request("what is the next event on my calendar?")
        action = skill.act(ctx, request)
        assert engine.decisions == [], "a single calendar needs no SemIf decision"
        assert "Team sync" in action.action_log, action.action_log
        assert handler.requests[1]["path"] == "/remote.php/dav/calendars/testuser/personal/"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()

    # A failing REPORT is reported honestly.
    server, handler, url = start_server(report_status=500)
    try:
        engine = FakeEngine(pick="personal")
        ctx = ActionContext(engine=engine, config=fixture_config(url))
        request = Request("what is the next event on my calendar?")
        action = skill.act(ctx, request)
        assert "failed" in action.action_log, action.action_log
        assert action.new_state == request.text, "a failed query must not fake a result"
        print(action.action_log)
    finally:
        server.shutdown()
        server.server_close()

    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
