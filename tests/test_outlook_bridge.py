"""Pure-stdlib tests for the Outlook (Microsoft Graph) bridge.

A loopback `http.server` plays Microsoft's token endpoint and Graph API so the
bridge's own client is exercised over HTTP with no mocking of the transport. The
decision engine and LLM are never involved. A few route tests inject a fake
client to check wiring/validation without a server. Passing these proves
mechanics only; a real run against the user's Microsoft account is the only proof
the integration works.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from semif_agent.bridges.outlook import OutlookBridge
from semif_agent.bridges.outlook_client import (
    OutlookAuthRequired,
    OutlookClient,
    OutlookError,
    load_token,
    run_device_code_flow,
    save_token,
)
from semif_agent.bridges.registry import build_bridge, derived_config_vars, describe_bridges, known_infos


class FakeGraph:
    """A minimal but real Microsoft-shaped HTTP server (token + Graph)."""

    def __init__(self):
        self.calls = []
        self.refresh_calls = 0
        self.device_polls = 0
        self.token_status = 200
        self._server = None
        self._thread = None

    # ---- lifecycle ----

    def start(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _read(self):
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else b""

            def _send(self, status, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _dispatch(self, method):
                body = self._read()
                fake.calls.append((method, self.path, dict(self.headers), body))
                status, payload = fake.handle(method, self.path, body)
                if status == 204:
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if isinstance(payload, bytes):
                    self.send_response(status)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self._send(status, payload)

            def do_GET(self):
                self._dispatch("GET")

            def do_POST(self):
                self._dispatch("POST")

            def do_PUT(self):
                self._dispatch("PUT")

            def do_PATCH(self):
                self._dispatch("PATCH")

            def do_DELETE(self):
                self._dispatch("DELETE")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # ---- routing ----

    def handle(self, method, path, body):
        parsed = urlparse(path)
        route = parsed.path
        query = parse_qs(parsed.query)

        # ---- OAuth token endpoint ----
        if route.endswith("/oauth2/v2.0/devicecode"):
            return 200, {
                "device_code": "dev-code",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://microsoft.com/devicelogin",
                "interval": 1,
                "expires_in": 900,
                "message": "go to https://microsoft.com/devicelogin and enter ABCD-EFGH",
            }
        if route.endswith("/oauth2/v2.0/token"):
            fields = parse_qs(body.decode("utf-8"))
            grant = (fields.get("grant_type") or [""])[0]
            if grant == "refresh_token":
                self.refresh_calls += 1
                return self.token_status, {
                    "access_token": f"refreshed-{self.refresh_calls}",
                    "refresh_token": f"rotated-{self.refresh_calls}",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                    "scope": "Mail.ReadWrite",
                }
            # device-code poll: pending once, then success
            self.device_polls += 1
            if self.device_polls < 2:
                return 400, {"error": "authorization_pending"}
            return 200, {
                "access_token": "device-access",
                "refresh_token": "device-refresh",
                "expires_in": 3600,
                "token_type": "Bearer",
                "scope": "Mail.ReadWrite",
            }

        # ---- Graph ----
        if method == "GET" and route == "/v1.0/me":
            return 200, {
                "id": "user-1",
                "displayName": "Jane Doe",
                "mail": "jane@example.com",
                "userPrincipalName": "jane@example.com",
            }
        if method == "GET" and route == "/v1.0/me/mailFolders":
            return 200, {
                "value": [
                    {"id": "inbox", "displayName": "Inbox", "totalItemCount": 3, "unreadItemCount": 1}
                ]
            }
        if method == "GET" and route == "/v1.0/me/mailFolders/inbox/messages":
            return 200, {
                "value": [
                    {
                        "id": "m1",
                        "subject": "Hello",
                        "from": {"emailAddress": {"name": "Alice", "address": "alice@example.com"}},
                        "toRecipients": [{"emailAddress": {"address": "jane@example.com"}}],
                        "ccRecipients": [],
                        "receivedDateTime": "2026-10-08T09:00:00Z",
                        "isRead": False,
                        "hasAttachments": False,
                        "bodyPreview": "Hi there",
                        "importance": "normal",
                        "webLink": "https://outlook/m1",
                    }
                ]
            }
        if method == "GET" and route == "/v1.0/me/messages/m1":
            return 200, {
                "id": "m1",
                "subject": "Hello",
                "from": {"emailAddress": {"name": "Alice", "address": "alice@example.com"}},
                "toRecipients": [{"emailAddress": {"address": "jane@example.com"}}],
                "ccRecipients": [],
                "receivedDateTime": "2026-10-08T09:00:00Z",
                "isRead": False,
                "hasAttachments": False,
                "body": {"contentType": "text", "content": "Hi there"},
                "bodyPreview": "Hi there",
                "importance": "normal",
                "webLink": "https://outlook/m1",
            }
        if method == "POST" and route == "/v1.0/me/sendMail":
            self.sent = json.loads(body.decode("utf-8"))
            return 202, {}
        if method == "POST" and route == "/v1.0/me/messages":
            payload = json.loads(body.decode("utf-8"))
            return 201, {"id": "draft-1", "subject": payload.get("subject", "")}
        if method == "GET" and route == "/v1.0/me/calendars":
            return 200, {
                "value": [
                    {"id": "cal1", "name": "Calendar", "color": "auto", "isDefaultCalendar": True, "canEdit": True}
                ]
            }
        if method == "GET" and route == "/v1.0/me/calendarView":
            return 200, {
                "value": [
                    {
                        "id": "e1",
                        "subject": "Standup",
                        "start": {"dateTime": "2026-10-08T09:00:00.0000000", "timeZone": "UTC"},
                        "end": {"dateTime": "2026-10-08T09:30:00.0000000", "timeZone": "UTC"},
                        "isAllDay": False,
                        "location": {"displayName": "Room 1"},
                        "bodyPreview": "",
                        "showAs": "busy",
                        "isOnlineMeeting": False,
                        "webLink": "https://outlook/e1",
                        "organizer": {"emailAddress": {"address": "jane@example.com"}},
                    }
                ]
            }
        if method == "POST" and route == "/v1.0/me/events":
            payload = json.loads(body.decode("utf-8"))
            return 201, {
                "id": "new-event",
                "subject": payload.get("subject"),
                "webLink": "https://outlook/new-event",
            }
        if method == "PATCH" and route.startswith("/v1.0/me/events/"):
            payload = json.loads(body.decode("utf-8"))
            return 200, {"id": route.rsplit("/", 1)[-1], "subject": payload.get("subject", "")}
        if method == "DELETE" and route.startswith("/v1.0/me/events/"):
            return 204, {}
        if method == "GET" and route == "/v1.0/me/todo/lists":
            return 200, {
                "value": [
                    {"id": "list1", "displayName": "Tasks", "isOwner": True, "wellknownListName": "defaultList"}
                ]
            }
        if method == "GET" and route == "/v1.0/me/todo/lists/list1/tasks":
            return 200, {
                "value": [
                    {
                        "id": "t1",
                        "title": "Buy milk",
                        "status": "notStarted",
                        "importance": "normal",
                        "dueDateTime": {"dateTime": "2026-10-10T00:00:00", "timeZone": "UTC"},
                        "completedDateTime": {},
                        "body": {"content": ""},
                    }
                ]
            }
        if method == "POST" and route == "/v1.0/me/todo/lists/list1/tasks":
            payload = json.loads(body.decode("utf-8"))
            return 201, {"id": "t2", "title": payload.get("title")}
        if method == "PATCH" and route.startswith("/v1.0/me/todo/lists/list1/tasks/"):
            payload = json.loads(body.decode("utf-8"))
            return 200, {"id": route.rsplit("/", 1)[-1], "status": payload.get("status", "notStarted")}
        if method == "DELETE" and route.startswith("/v1.0/me/todo/lists/list1/tasks/"):
            return 204, {}
        if method == "GET" and route == "/v1.0/me/contacts":
            return 200, {
                "value": [
                    {
                        "id": "c1",
                        "displayName": "Bob Smith",
                        "givenName": "Bob",
                        "surname": "Smith",
                        "emailAddresses": [{"address": "bob@example.com"}],
                        "businessPhones": ["555-1234"],
                        "homePhones": [],
                        "mobilePhone": None,
                        "companyName": "Acme",
                        "jobTitle": "Dev",
                    }
                ]
            }
        if method == "GET" and route == "/v1.0/me/drive/root/children":
            return 200, {
                "value": [
                    {
                        "id": "f1",
                        "name": "notes.txt",
                        "size": 5,
                        "lastModifiedDateTime": "2026-10-08T08:00:00Z",
                        "file": {"mimeType": "text/plain"},
                        "parentReference": {"path": "/drive/root:"},
                        "webUrl": "https://onedrive/f1",
                    }
                ]
            }
        if route.startswith("/v1.0/me/drive/root:/") and route.endswith(":"):
            return 200, {
                "id": "f1",
                "name": "notes.txt",
                "size": 5,
                "file": {"mimeType": "text/plain"},
                "lastModifiedDateTime": "2026-10-08T08:00:00Z",
                "webUrl": "https://onedrive/f1",
            }
        if route.startswith("/v1.0/me/drive/root:/") and route.endswith("/content"):
            if method == "GET":
                return 200, b"hello"
            return 201, {"id": "f2", "size": 5, "webUrl": "https://onedrive/f2"}

        return 404, {"error": {"code": "notFound", "message": f"no fake route for {method} {route}"}}


@pytest.fixture()
def fake():
    server = FakeGraph()
    base = server.start()
    try:
        yield server, base
    finally:
        server.stop()


def make_client(fake, tmp_path, *, token="tok", expires_at=None, refresh="r1", graph_suffix="/v1.0"):
    server, base = fake
    token_path = tmp_path / "token.json"
    if token is not None:
        save_token(
            str(token_path),
            {
                "access_token": token,
                "refresh_token": refresh,
                "expires_at": 9999999999 if expires_at is None else expires_at,
                "token_type": "Bearer",
            },
        )
    return OutlookClient(
        "client-1",
        tenant="common",
        token_path=str(token_path),
        graph=base + graph_suffix,
        login=base,
    )


# ---- OAuth ----

def test_device_code_flow_polls_until_success(fake):
    server, base = fake
    lines = []
    token = run_device_code_flow(
        "client-1", "common", out=lines.append, sleep=lambda _s: None, login=base
    )
    assert token["access_token"] == "device-access"
    assert token["refresh_token"] == "device-refresh"
    assert any("ABCD-EFGH" in line for line in lines)
    assert server.device_polls == 2


def test_token_refresh_on_expiry(fake, tmp_path):
    server, base = fake
    client = make_client(fake, tmp_path, token="stale", expires_at=0)
    user = client.me()
    assert user["display_name"] == "Jane Doe"
    assert server.refresh_calls == 1
    stored = load_token(str(tmp_path / "token.json"))
    assert stored["access_token"] == "refreshed-1"
    assert stored["refresh_token"] == "rotated-1"


def test_missing_token_raises_auth_required(tmp_path):
    client = OutlookClient(
        "client-1", token_path=str(tmp_path / "missing.json"), graph="http://127.0.0.1:1/v1.0"
    )
    with pytest.raises(OutlookAuthRequired):
        client.me()


def test_not_configured_without_client_id(tmp_path):
    client = OutlookClient("", token_path=str(tmp_path / "t.json"))
    assert client.configured is False


# ---- Graph client ----

def test_me_and_mail(fake, tmp_path):
    client = make_client(fake, tmp_path)
    assert client.me()["mail"] == "jane@example.com"
    folders = client.list_mail_folders()
    assert folders[0]["name"] == "Inbox"
    messages = client.list_messages("inbox")
    assert messages[0]["subject"] == "Hello"
    assert messages[0]["from"] == "Alice <alice@example.com>"
    assert messages[0]["to"] == ["jane@example.com"]
    full = client.get_message("m1")
    assert full["body"] == "Hi there"
    assert full["body_type"] == "text"


def test_send_mail(fake, tmp_path):
    server, _base = fake
    client = make_client(fake, tmp_path)
    result = client.send_mail(["sam@example.com"], "Lunch", "Tomorrow?", cc=["bob@example.com"])
    assert result["ok"] is True
    sent = server.sent
    assert sent["message"]["subject"] == "Lunch"
    assert sent["message"]["toRecipients"][0]["emailAddress"]["address"] == "sam@example.com"
    assert sent["message"]["ccRecipients"][0]["emailAddress"]["address"] == "bob@example.com"
    assert sent["saveToSentItems"] is True


def test_create_draft(fake, tmp_path):
    client = make_client(fake, tmp_path)
    result = client.create_draft(["sam@example.com"], "Draft", "body")
    assert result["id"] == "draft-1"
    assert result["subject"] == "Draft"


def test_list_events(fake, tmp_path):
    client = make_client(fake, tmp_path)
    events = client.list_events()
    assert events[0]["subject"] == "Standup"
    assert events[0]["start"] == "2026-10-08T09:00:00.0000000"
    assert events[0]["location"] == "Room 1"
    assert events[0]["organizer"] == "jane@example.com"


def test_create_event_builds_graph_body(fake, tmp_path):
    server, _base = fake
    client = make_client(fake, tmp_path)
    result = client.create_event(
        subject="Dentist",
        start="2026-11-10T14:00:00",
        all_day=False,
        location="Clinic",
        timezone_name="America/Chicago",
    )
    assert result["ok"] is True
    assert result["id"] == "new-event"
    call = next(c for c in server.calls if c[0] == "POST" and c[1] == "/v1.0/me/events")
    body = json.loads(call[3].decode("utf-8"))
    assert body["subject"] == "Dentist"
    assert body["start"] == {"dateTime": "2026-11-10T14:00:00", "timeZone": "America/Chicago"}
    # end defaults to one hour after start
    assert body["end"]["dateTime"] == "2026-11-10T15:00:00"
    assert body["isAllDay"] is False
    assert body["location"] == {"displayName": "Clinic"}


def test_create_event_all_day_defaults_end_next_day(fake, tmp_path):
    server, _base = fake
    client = make_client(fake, tmp_path)
    client.create_event(subject="Conference", start="2026-11-10T00:00:00", all_day=True)
    call = next(c for c in server.calls if c[0] == "POST" and c[1] == "/v1.0/me/events")
    body = json.loads(call[3].decode("utf-8"))
    assert body["isAllDay"] is True
    assert body["end"]["dateTime"].startswith("2026-11-11")


def test_tasks_flow(fake, tmp_path):
    client = make_client(fake, tmp_path)
    lists = client.list_task_lists()
    assert lists[0]["is_default"] is True
    tasks = client.list_tasks()
    assert tasks[0]["title"] == "Buy milk"
    created = client.create_task(title="Call mom", due="2026-10-12T09:00:00")
    assert created["id"] == "t2"
    completed = client.complete_task("t1")
    assert completed["status"] == "completed"


def test_contacts(fake, tmp_path):
    client = make_client(fake, tmp_path)
    contacts = client.list_contacts("bob")
    assert contacts[0]["display_name"] == "Bob Smith"
    assert contacts[0]["emails"] == ["bob@example.com"]
    assert "555-1234" in contacts[0]["phones"]


def test_files_read_and_write(fake, tmp_path):
    client = make_client(fake, tmp_path)
    entries = client.list_files("/")
    assert entries[0]["name"] == "notes.txt"
    read = client.read_file("notes.txt")
    assert read["content"] == "hello"
    assert read["encoding"] == "utf-8"
    written = client.write_file("notes.txt", "hello", overwrite=True)
    assert written["ok"] is True
    assert written["path"] == "notes.txt"


def test_unreachable_graph_is_an_outlook_error(tmp_path):
    client = OutlookClient(
        "client-1", token_path=str(tmp_path / "t.json"), graph="http://127.0.0.1:1/v1.0"
    )
    save_token(str(tmp_path / "t.json"), {"access_token": "x", "refresh_token": "r", "expires_at": 9999999999})
    with pytest.raises(OutlookError):
        client.me()


# ---- bridge routes ----

class StubClient:
    configured = True

    def __init__(self):
        self.calls = []

    def me(self):
        return {"display_name": "Jane"}

    def list_mail_folders(self):
        return [{"id": "inbox", "name": "Inbox"}]

    def list_messages(self, folder="", count=25, search="", unread_only=False):
        self.calls.append(("list_messages", folder, count, search, unread_only))
        return [{"id": "m1", "subject": "Hello"}]

    def get_message(self, message_id):
        return {"id": message_id, "body": "hi"}

    def send_mail(self, to, subject, body, cc=(), content_type="text", save_to_sent=True):
        self.calls.append(("send_mail", to, subject, body, cc))
        return {"ok": True, "to": to}

    def create_event(self, **kwargs):
        self.calls.append(("create_event", kwargs))
        return {"ok": True, "id": "e1"}

    def list_events(self, calendar="", start="", end=""):
        return [{"id": "e1", "subject": "Standup"}]

    def create_task(self, **kwargs):
        self.calls.append(("create_task", kwargs))
        return {"ok": True, "id": "t1"}

    def list_files(self, path="/"):
        return [{"name": "notes.txt"}]

    def write_file(self, path, content, encoding="utf-8", overwrite=True):
        return {"ok": True, "path": path}

    def list_calendars(self):
        return []

    def list_task_lists(self):
        return []

    def list_tasks(self, task_list=""):
        return []

    def list_contacts(self, query="", count=50):
        return []


def test_bridge_routes_wire_to_client():
    stub = StubClient()
    bridge = OutlookBridge({"client_id": "c", "token_path": "/tmp/x"}, client=stub)
    assert bridge.handle_get("/health", {}) == (200, {"ok": True, "platform": "outlook"})
    status, payload = bridge.handle_get("/mail/messages", {"folder": ["inbox"], "count": ["10"]})
    assert status == 200 and payload["messages"][0]["subject"] == "Hello"
    assert stub.calls[-1] == ("list_messages", "inbox", 10, "", False)
    status, payload = bridge.handle_post(
        "/mail/send", {"to": ["sam@example.com"], "subject": "Hi", "body": "there"}
    )
    assert status == 200 and payload["ok"] is True
    assert stub.calls[-1] == ("send_mail", ["sam@example.com"], "Hi", "there", [])
    status, _payload = bridge.handle_post("/calendars/events", {"subject": "Standup", "start": "2026-01-01T09:00:00"})
    assert status == 200
    assert stub.calls[-1][0] == "create_event"


def test_bridge_validation_errors():
    bridge = OutlookBridge({"client_id": "c"}, client=StubClient())
    status, payload = bridge.handle_post("/calendars/events", {"start": "2026-01-01T09:00:00"})
    assert status == 400 and "subject" in payload["error"]
    status, payload = bridge.handle_post("/mail/send", {"subject": "no recipient"})
    assert status == 400 and "to" in payload["error"]
    status, _ = bridge.handle_post("/nope", {})
    assert status == 404


def test_bridge_unconfigured_is_503():
    bridge = OutlookBridge({}, client=StubClient())
    bridge.client = type("C", (), {"configured": False})()
    assert bridge.handle_get("/mail/messages", {})[0] == 503
    assert bridge.handle_post("/mail/send", {"to": ["a@b.c"]})[0] == 503


def test_bridge_auth_required_is_503():
    class Unauthorized:
        configured = True

        def list_messages(self, *args, **kwargs):
            raise OutlookAuthRequired("run scripts/outlook-auth.py")

    bridge = OutlookBridge({"client_id": "c"}, client=Unauthorized())
    status, payload = bridge.handle_get("/mail/messages", {})
    assert status == 503
    assert "outlook-auth" in payload["error"]


def test_bridge_maps_real_failure_to_502():
    class Broken:
        configured = True

        def list_messages(self, *args, **kwargs):
            raise OutlookError("boom")

    bridge = OutlookBridge({"client_id": "c"}, client=Broken())
    assert bridge.handle_get("/mail/messages", {})[0] == 502


def test_check_requirements_is_stdlib_only():
    assert OutlookBridge({}).check_requirements() == (True, None)


# ---- catalog ----

def test_known_infos_include_outlook():
    infos = {info.name: info for info in known_infos()}
    assert "outlook" in infos
    info = infos["outlook"]
    assert info.service == "outlook"
    assert info.url_config_var == "outlook_bridge_url"
    assert info.auth_header == "X-Semif-Token"
    assert info.auth_config_var == "outlook_bridge_token"


def test_describe_bridges_covers_outlook():
    text = describe_bridges()
    assert "outlook" in text
    assert "outlook_bridge_url" in text
    assert "Microsoft To Do" in text


def test_derived_config_vars_for_outlook():
    derived = derived_config_vars(
        {"bridges": {"outlook": {"host": "127.0.0.1", "port": 5231, "token": "o"}}}
    )
    assert derived["outlook_bridge_url"] == "http://127.0.0.1:5231"
    assert derived["outlook_bridge_token"] == "o"


def test_build_bridge_returns_outlook():
    bridge = build_bridge({"bridges": {"outlook": {"client_id": "c"}}}, "outlook")
    assert isinstance(bridge, OutlookBridge)
    assert bridge.client_id == "c"
