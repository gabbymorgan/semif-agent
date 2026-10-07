"""Pure-stdlib tests for the Nextcloud bridge.

A loopback `http.server` plays a real Nextcloud (WebDAV files, CalDAV events and
tasks, CardDAV contacts, OCS Notes) so the bridge's own client is exercised over
HTTP with no mocking of the transport. The decision engine and LLM are never
involved. A few route tests inject a fake client to check wiring/validation
without a server. Passing these proves mechanics only; a real run against the
user's Nextcloud is the only proof the integration works.
"""

import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from semif_agent.bridges.nextcloud import NextcloudBridge
from semif_agent.bridges.nextcloud_client import NextcloudClient, NextcloudError
from semif_agent.bridges.registry import build_bridge, describe_bridges, known_infos

USER = "alice"
PASSWORD = "app-secret"
MODIFIED = "Mon, 01 Jan 2026 12:00:00 GMT"


def _norm(path):
    text = "/" + str(path or "").strip("/")
    return "/" if text == "/" else text


def _parent(path):
    if path == "/":
        return None
    return path.rsplit("/", 1)[0] or "/"


class FakeNextcloud:
    """A minimal but real Nextcloud-shaped HTTP server for the client."""

    def __init__(self):
        self.files = {
            "/": {"type": "dir"},
            "/Documents": {"type": "dir"},
            "/Documents/readme.txt": {"type": "file", "content": b"hello world"},
            "/image.png": {"type": "file", "content": b"\x89PNG\x00\xff"},
        }
        self.calendars = {"personal": {"label": "Personal", "objects": {}}}
        self.addressbooks = {"contacts": {"label": "Contacts", "cards": {}}}
        self.notes = {}
        self._next_note_id = 1
        self.calls = []
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

            def _authorized(self):
                expected = "Basic " + base64.b64encode(
                    f"{USER}:{PASSWORD}".encode()
                ).decode()
                return self.headers.get("Authorization") == expected

            def _send(self, status, body=b"", content_type="application/xml", headers=None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _dispatch(self, method):
                body = self._read()
                fake.calls.append((method, self.path, dict(self.headers), body))
                if not self._authorized():
                    self._send(401, b'{"error":"unauthorized"}', "application/json")
                    return
                status, payload, content_type, headers = fake.handle(
                    method, self.path, body
                )
                self._send(status, payload, content_type, headers)

            def do_GET(self):
                self._dispatch("GET")

            def do_PUT(self):
                self._dispatch("PUT")

            def do_POST(self):
                self._dispatch("POST")

            def do_DELETE(self):
                self._dispatch("DELETE")

            def do_MOVE(self):
                self._dispatch("MOVE")

            def do_COPY(self):
                self._dispatch("COPY")

            def do_MKCOL(self):
                self._dispatch("MKCOL")

            def do_PROPFIND(self):
                self._dispatch("PROPFIND")

            def do_REPORT(self):
                self._dispatch("REPORT")

            def do_SEARCH(self):
                self._dispatch("SEARCH")

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
        url = urlparse(path)
        route = url.path
        query = parse_qs(url.query)
        if route.startswith("/ocs/"):
            return self._ocs(method, route, body)
        if route.startswith("/remote.php/dav/files/"):
            return self._files(method, route, body)
        if route.startswith("/remote.php/dav/calendars/"):
            return self._calendars(method, route, body)
        if route.startswith("/remote.php/dav/addressbooks/"):
            return self._addressbooks(method, route, body)
        if route == "/remote.php/dav/":
            return self._search(body)
        return 404, b"not found", "text/plain", {}

    def _xml(self, responses):
        return (
            '<?xml version="1.0"?>'
            '<d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav" '
            'xmlns:card="urn:ietf:params:xml:ns:carddav" '
            'xmlns:oc="http://owncloud.org/ns" xmlns:nc="http://nextcloud.org/ns">'
            + "".join(responses)
            + "</d:multistatus>"
        ).encode()

    @staticmethod
    def _response(href, props):
        return (
            f"<d:response><d:href>{href}</d:href>"
            f"<d:propstat><d:prop>{props}</d:prop>"
            "<d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>"
        )

    # ---- files ----

    def _file_response(self, dav_path):
        info = self.files.get(dav_path)
        if info is None:
            return None
        is_dir = info["type"] == "dir"
        suffix = dav_path if dav_path == "/" else dav_path + ("/" if is_dir else "")
        href = f"/remote.php/dav/files/{USER}{suffix}"
        name = "root" if dav_path == "/" else dav_path.rsplit("/", 1)[-1]
        size = 0 if is_dir else len(info.get("content", b""))
        resourcetype = "<d:collection/>" if is_dir else ""
        content_type = "httpd/unix-directory" if is_dir else "text/plain"
        props = (
            f"<d:resourcetype>{resourcetype}</d:resourcetype>"
            f"<d:displayname>{name}</d:displayname>"
            f"<d:getcontentlength>{size}</d:getcontentlength>"
            f"<d:getlastmodified>{MODIFIED}</d:getlastmodified>"
            f'<d:getetag>"etag-{name}"</d:getetag>'
            f"<d:getcontenttype>{content_type}</d:getcontenttype>"
        )
        return self._response(href, props)

    def _files(self, method, route, body):
        rest = route[len("/remote.php/dav/files/") :]
        user, _, tail = rest.partition("/")
        dav_path = _norm(tail)
        if method == "PROPFIND":
            depth = None
            # depth is read from the request headers by the handler; emulate via
            # the client always sending one — Depth "1" lists children.
            if dav_path not in self.files:
                return 404, b"not found", "text/plain", {}
            responses = []
            # The Depth header is not threaded here; both stat and list are
            # covered by returning the entry, and list adds children.
            responses.append(self._file_response(dav_path))
            for candidate in self.files:
                if _parent(candidate) == dav_path and candidate != dav_path:
                    responses.append(self._file_response(candidate))
            return 207, self._xml([r for r in responses if r]), "application/xml", {}
        if method == "GET":
            info = self.files.get(dav_path)
            if info is None or info["type"] != "file":
                return 404, b"not found", "text/plain", {}
            return 200, info["content"], "application/octet-stream", {}
        if method == "PUT":
            self.files[dav_path] = {"type": "file", "content": body}
            return 201, b"", "text/plain", {"ETag": '"new"'}
        if method == "MKCOL":
            self.files[dav_path] = {"type": "dir"}
            return 201, b"", "text/plain", {}
        if method == "DELETE":
            self.files.pop(dav_path, None)
            return 204, b"", "text/plain", {}
        if method in ("MOVE", "COPY"):
            return 201, b"", "text/plain", {}
        return 405, b"method not allowed", "text/plain", {}

    def _search(self, body):
        text = body.decode("utf-8", "replace")
        needle = ""
        if "<d:literal>" in text:
            needle = text.split("<d:literal>", 1)[1].split("</d:literal>", 1)[0].strip("%")
        responses = []
        for dav_path, info in self.files.items():
            if info["type"] != "file":
                continue
            if needle and needle.lower() not in dav_path.lower():
                continue
            responses.append(self._file_response(dav_path))
        return 207, self._xml(responses), "application/xml", {}

    # ---- calendars ----

    def _calendar_home(self):
        responses = [
            self._response(
                f"/remote.php/dav/calendars/{USER}/",
                "<d:resourcetype><d:collection/></d:resourcetype>"
                f"<d:displayname>{USER}</d:displayname>",
            )
        ]
        for name, calendar in self.calendars.items():
            responses.append(
                self._response(
                    f"/remote.php/dav/calendars/{USER}/{name}/",
                    "<d:resourcetype><d:collection/><c:calendar/></d:resourcetype>"
                    f"<d:displayname>{calendar['label']}</d:displayname>"
                    "<c:calendar-description>desc</c:calendar-description>"
                    '<nc:calendar-color>#ff0000</nc:calendar-color>'
                    "<c:getctag>ctag-1</c:getctag>",
                )
            )
        return self._xml(responses)

    def _calendars(self, method, route, body):
        rest = route[len("/remote.php/dav/calendars/") :]
        user, _, tail = rest.partition("/")
        tail = _norm(tail)
        if tail == "/":
            if method == "PROPFIND":
                return 207, self._calendar_home(), "application/xml", {}
            return 405, b"method not allowed", "text/plain", {}
        name, _, leaf = tail.lstrip("/").partition("/")
        calendar = self.calendars.get(name)
        if calendar is None:
            return 404, b"not found", "text/plain", {}
        if not leaf:
            if method == "REPORT":
                responses = []
                for uid, ics in calendar["objects"].items():
                    responses.append(
                        self._response(
                            f"/remote.php/dav/calendars/{USER}/{name}/{uid}.ics",
                            f'<d:getetag>"etag-{uid}"</d:getetag>'
                            f"<c:calendar-data>{_xml_escape(ics)}</c:calendar-data>",
                        )
                    )
                return 207, self._xml(responses), "application/xml", {}
            return 405, b"method not allowed", "text/plain", {}
        uid = leaf[:-4] if leaf.endswith(".ics") else leaf
        if method == "PUT":
            calendar["objects"][uid] = body.decode("utf-8")
            return 201, b"", "text/plain", {"ETag": f'"etag-{uid}"'}
        if method == "GET":
            ics = calendar["objects"].get(uid)
            if ics is None:
                return 404, b"not found", "text/plain", {}
            return 200, ics.encode(), "text/calendar", {"ETag": f'"etag-{uid}"'}
        if method == "DELETE":
            calendar["objects"].pop(uid, None)
            return 204, b"", "text/plain", {}
        return 405, b"method not allowed", "text/plain", {}

    # ---- contacts ----

    def _addressbook_home(self):
        responses = [
            self._response(
                f"/remote.php/dav/addressbooks/users/{USER}/",
                "<d:resourcetype><d:collection/></d:resourcetype>"
                f"<d:displayname>{USER}</d:displayname>",
            )
        ]
        for name, book in self.addressbooks.items():
            responses.append(
                self._response(
                    f"/remote.php/dav/addressbooks/users/{USER}/{name}/",
                    "<d:resourcetype><d:collection/><card:addressbook/></d:resourcetype>"
                    f"<d:displayname>{book['label']}</d:displayname>"
                    "<card:addressbook-description>d</card:addressbook-description>",
                )
            )
        return self._xml(responses)

    def _addressbooks(self, method, route, body):
        rest = route[len("/remote.php/dav/addressbooks/") :]
        parts = rest.split("/")
        tail = _norm("/".join(parts[2:]) if len(parts) > 2 else "/")
        if tail == "/":
            if method == "PROPFIND":
                return 207, self._addressbook_home(), "application/xml", {}
            return 405, b"method not allowed", "text/plain", {}
        name, _, leaf = tail.lstrip("/").partition("/")
        book = self.addressbooks.get(name)
        if book is None:
            return 404, b"not found", "text/plain", {}
        if not leaf:
            if method == "REPORT":
                responses = []
                for uid, vcf in book["cards"].items():
                    responses.append(
                        self._response(
                            f"/remote.php/dav/addressbooks/users/{USER}/{name}/{uid}.vcf",
                            f'<d:getetag>"etag-{uid}"</d:getetag>'
                            f"<card:address-data>{_xml_escape(vcf)}</card:address-data>",
                        )
                    )
                return 207, self._xml(responses), "application/xml", {}
            return 405, b"method not allowed", "text/plain", {}
        uid = leaf[:-4] if leaf.endswith(".vcf") else leaf
        if method == "PUT":
            book["cards"][uid] = body.decode("utf-8")
            return 201, b"", "text/plain", {"ETag": f'"etag-{uid}"'}
        if method == "GET":
            vcf = book["cards"].get(uid)
            if vcf is None:
                return 404, b"not found", "text/plain", {}
            return 200, vcf.encode(), "text/vcard", {"ETag": f'"etag-{uid}"'}
        if method == "DELETE":
            book["cards"].pop(uid, None)
            return 204, b"", "text/plain", {}
        return 405, b"method not allowed", "text/plain", {}

    # ---- OCS (notes + info) ----

    def _ocs_json(self, data, status=200):
        return status, json.dumps({"ocs": {"meta": {"statuscode": 200}, "data": data}}).encode(), "application/json", {}

    def _ocs(self, method, route, body):
        if route == "/ocs/v2.php/cloud/user":
            return self._ocs_json(
                {"id": USER, "display-name": "Alice", "email": "alice@example.com"}
            )
        if route == "/ocs/v2.php/cloud/capabilities":
            return self._ocs_json({"version": {"major": 28}})
        prefix = "/ocs/v2.php/apps/notes/api/v1/notes"
        if route == prefix:
            if method == "GET":
                return self._ocs_json(list(self.notes.values()))
            if method == "POST":
                payload = json.loads(body.decode() or "{}")
                note_id = self._next_note_id
                self._next_note_id += 1
                note = {"id": note_id, "title": payload.get("title", ""),
                        "content": payload.get("content", ""),
                        "category": payload.get("category", "")}
                self.notes[note_id] = note
                return self._ocs_json(note)
        if route.startswith(prefix + "/"):
            note_id = int(route.rsplit("/", 1)[-1])
            note = self.notes.get(note_id)
            if note is None:
                return 404, b'{"ocs":{"meta":{"statuscode":404},"data":[]}}', "application/json", {}
            if method == "GET":
                return self._ocs_json(note)
            if method == "PUT":
                payload = json.loads(body.decode() or "{}")
                note.update({k: v for k, v in payload.items() if v is not None})
                return self._ocs_json(note)
            if method == "DELETE":
                self.notes.pop(note_id, None)
                return self._ocs_json([])
        return 404, b"not found", "text/plain", {}


def _xml_escape(text):
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


@pytest.fixture()
def nextcloud():
    fake = FakeNextcloud()
    base = fake.start()
    try:
        yield fake, base
    finally:
        fake.stop()


def make_client(base, **kwargs):
    return NextcloudClient(base, USER, PASSWORD, **kwargs)


# ---- client: files ----


def test_client_sends_basic_auth(nextcloud):
    fake, base = nextcloud
    make_client(base).list_files("/")
    assert fake.calls, "the server must have received a request"
    header = fake.calls[0][2].get("Authorization")
    assert header == "Basic " + base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()


def test_list_files_parses_entries(nextcloud):
    _fake, base = nextcloud
    result = make_client(base).list_files("/")
    assert result["path"] == "/"
    names = {entry["name"]: entry for entry in result["entries"]}
    assert names["Documents"]["type"] == "dir"
    assert names["image.png"]["type"] == "file"
    assert names["image.png"]["size"] == len(b"\x89PNG\x00\xff")
    assert names["image.png"]["etag"] == '"etag-image.png"'
    # Directories sort before files.
    assert result["entries"][0]["type"] == "dir"


def test_list_files_of_a_subfolder(nextcloud):
    _fake, base = nextcloud
    result = make_client(base).list_files("/Documents")
    assert [entry["name"] for entry in result["entries"]] == ["readme.txt"]


def test_stat_returns_one_entry(nextcloud):
    _fake, base = nextcloud
    entry = make_client(base).stat("/Documents/readme.txt")
    assert entry["type"] == "file"
    assert entry["size"] == 11


def test_read_text_file(nextcloud):
    _fake, base = nextcloud
    result = make_client(base).read_file("/Documents/readme.txt")
    assert result["encoding"] == "utf-8"
    assert result["content"] == "hello world"


def test_read_binary_file_is_base64(nextcloud):
    _fake, base = nextcloud
    result = make_client(base).read_file("/image.png")
    assert result["encoding"] == "base64"
    assert base64.b64decode(result["content"]) == b"\x89PNG\x00\xff"


def test_write_file_round_trip(nextcloud):
    fake, base = nextcloud
    client = make_client(base)
    assert client.write_file("/new.txt", "content", overwrite=False)["ok"] is True
    put = [call for call in fake.calls if call[0] == "PUT"][0]
    assert put[2].get("If-None-Match") == "*"
    assert client.read_file("/new.txt")["content"] == "content"


def test_write_file_base64(nextcloud):
    fake, base = nextcloud
    make_client(base).write_file("/bin.dat", base64.b64encode(b"\x00\x01").decode(), encoding="base64")
    assert fake.files["/bin.dat"]["content"] == b"\x00\x01"


def test_write_file_rejects_bad_base64(nextcloud):
    _fake, base = nextcloud
    with pytest.raises(ValueError):
        make_client(base).write_file("/bin.dat", "not base64!", encoding="base64")


def test_mkdir_and_delete(nextcloud):
    fake, base = nextcloud
    client = make_client(base)
    client.mkdir("/NewFolder")
    assert fake.files["/NewFolder"]["type"] == "dir"
    client.delete("/NewFolder")
    assert "/NewFolder" not in fake.files


def test_move_sends_destination_header(nextcloud):
    fake, base = nextcloud
    result = make_client(base).move("/Documents/readme.txt", "/Documents/renamed.txt", overwrite=True)
    assert result["destination"] == "/Documents/renamed.txt"
    move = [call for call in fake.calls if call[0] == "MOVE"][0]
    assert move[2]["Destination"].endswith("/remote.php/dav/files/alice/Documents/renamed.txt")
    assert move[2]["Overwrite"] == "T"


def test_copy_sends_destination_header(nextcloud):
    fake, base = nextcloud
    make_client(base).copy("/Documents/readme.txt", "/Documents/copy.txt")
    copy = [call for call in fake.calls if call[0] == "COPY"][0]
    assert copy[2]["Overwrite"] == "F"


def test_search_returns_matching_files(nextcloud):
    _fake, base = nextcloud
    results = make_client(base).search("readme")
    assert [entry["name"] for entry in results] == ["readme.txt"]


# ---- client: calendars ----


def test_list_calendars(nextcloud):
    _fake, base = nextcloud
    calendars = make_client(base).list_calendars()
    assert calendars == [
        {
            "name": "personal",
            "label": "Personal",
            "description": "desc",
            "color": "#ff0000",
            "ctag": "ctag-1",
        }
    ]


def test_create_and_list_event(nextcloud):
    _fake, base = nextcloud
    client = make_client(base)
    created = client.create_event(
        "personal",
        "uid-1",
        "Lunch with Dana",
        "2026-03-01T12:00:00+00:00",
        end="2026-03-01T13:00:00+00:00",
        location="Cafe",
    )
    assert created["uid"] == "uid-1"
    events = client.list_events("personal")
    assert len(events) == 1
    event = events[0]
    assert event["summary"] == "Lunch with Dana"
    assert event["location"] == "Cafe"
    assert event["start"] == "2026-03-01T12:00:00+00:00"
    assert event["all_day"] is False
    assert event["uid"] == "uid-1"


def test_all_day_event_parses(nextcloud):
    _fake, base = nextcloud
    client = make_client(base)
    client.create_event("personal", "u2", "Holiday", "2026-07-04", end="2026-07-05", all_day=True)
    event = client.list_events("personal")[0]
    assert event["all_day"] is True
    assert event["start"] == "2026-07-04"


def test_create_event_sends_if_none_match(nextcloud):
    fake, base = nextcloud
    make_client(base).create_event("personal", "u3", "X", "2026-03-01T12:00:00Z")
    put = [call for call in fake.calls if call[0] == "PUT"][0]
    assert put[2].get("If-None-Match") == "*"


def test_update_event_preserves_other_properties(nextcloud):
    _fake, base = nextcloud
    client = make_client(base)
    client.create_event(
        "personal", "u4", "Original", "2026-03-01T12:00:00Z",
        end="2026-03-01T13:00:00Z", location="Office",
    )
    client.update_event("personal", "u4", summary="Renamed")
    event = client.list_events("personal")[0]
    assert event["summary"] == "Renamed"
    assert event["location"] == "Office", "unchanged properties must survive an update"
    assert event["start"] == "2026-03-01T12:00:00+00:00"


def test_update_event_uses_if_match(nextcloud):
    fake, base = nextcloud
    client = make_client(base)
    client.create_event("personal", "u5", "X", "2026-03-01T12:00:00Z")
    client.update_event("personal", "u5", summary="Y")
    put = [call for call in fake.calls if call[0] == "PUT"][-1]
    assert put[2].get("If-Match") == '"etag-u5"'


def test_delete_event(nextcloud):
    _fake, base = nextcloud
    client = make_client(base)
    client.create_event("personal", "u6", "X", "2026-03-01T12:00:00Z")
    assert client.delete_object("personal", "u6")["ok"] is True
    assert client.list_events("personal") == []


def test_update_missing_event_is_a_real_failure(nextcloud):
    _fake, base = nextcloud
    with pytest.raises(NextcloudError) as exc:
        make_client(base).update_event("personal", "ghost", summary="X")
    assert exc.value.status == 404


# ---- client: tasks ----


def test_create_and_list_task(nextcloud):
    _fake, base = nextcloud
    client = make_client(base)
    client.create_task("personal", "t1", "Buy milk", due="2026-03-02T09:00:00Z", priority=1)
    tasks = client.list_tasks("personal")
    assert len(tasks) == 1
    task = tasks[0]
    assert task["summary"] == "Buy milk"
    assert task["status"] == "NEEDS-ACTION"
    assert task["priority"] == 1
    assert task["due"] == "2026-03-02T09:00:00+00:00"


def test_complete_task(nextcloud):
    _fake, base = nextcloud
    client = make_client(base)
    client.create_task("personal", "t2", "Do it")
    client.complete_task("personal", "t2")
    task = client.list_tasks("personal")[0]
    assert task["status"] == "COMPLETED"
    assert task["percent_complete"] == 100


# ---- client: contacts ----


def test_list_addressbooks(nextcloud):
    _fake, base = nextcloud
    assert make_client(base).list_addressbooks() == [
        {"name": "contacts", "label": "Contacts", "description": "d"}
    ]


def test_create_and_list_contact(nextcloud):
    _fake, base = nextcloud
    client = make_client(base)
    client.create_contact(
        "contacts", "c1", fn="Dana Lee", emails=["dana@example.com"], phones=["555-1234"], org="Acme"
    )
    contacts = client.list_contacts("contacts")
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact["fn"] == "Dana Lee"
    assert contact["emails"] == ["dana@example.com"]
    assert contact["phones"] == ["555-1234"]
    assert contact["org"] == "Acme"


def test_contact_query_filters(nextcloud):
    _fake, base = nextcloud
    client = make_client(base)
    client.create_contact("contacts", "c2", fn="Dana Lee", emails=["dana@example.com"])
    client.create_contact("contacts", "c3", fn="Sam Ray", emails=["sam@example.com"])
    assert [c["fn"] for c in client.list_contacts("contacts", "dana")] == ["Dana Lee"]
    assert [c["fn"] for c in client.list_contacts("contacts", "nobody")] == []


def test_update_contact_changes_only_named_fields(nextcloud):
    _fake, base = nextcloud
    client = make_client(base)
    client.create_contact("contacts", "c4", fn="Old", emails=["old@example.com"])
    client.update_contact("contacts", "c4", fn="New")
    contact = client.list_contacts("contacts")[0]
    assert contact["fn"] == "New"
    assert contact["emails"] == ["old@example.com"]


def test_delete_contact(nextcloud):
    _fake, base = nextcloud
    client = make_client(base)
    client.create_contact("contacts", "c5", fn="Gone")
    client.delete_contact("contacts", "c5")
    assert client.list_contacts("contacts") == []


# ---- client: notes + info ----


def test_user_info(nextcloud):
    _fake, base = nextcloud
    assert make_client(base).user_info()["display_name"] == "Alice"


def test_capabilities(nextcloud):
    _fake, base = nextcloud
    assert make_client(base).capabilities()["version"]["major"] == 28


def test_notes_crud(nextcloud):
    _fake, base = nextcloud
    client = make_client(base)
    created = client.create_note("Shopping", "milk, eggs", category="Home")
    note_id = created["id"]
    assert client.list_notes()[0]["title"] == "Shopping"
    assert client.get_note(note_id)["content"] == "milk, eggs"
    updated = client.update_note(note_id, content="milk")
    assert updated["content"] == "milk"
    assert client.delete_note(note_id)["ok"] is True
    assert client.list_notes() == []


# ---- client: failure mapping ----


def test_http_error_carries_status(nextcloud):
    _fake, base = nextcloud
    with pytest.raises(NextcloudError) as exc:
        make_client(base).read_file("/missing.txt")
    assert exc.value.status == 404


def test_unconfigured_client_is_not_configured():
    assert NextcloudClient("", "", "").configured is False
    assert NextcloudClient("https://cloud", "alice", "pw").configured is True


# ---- bridge routes (real client over loopback) ----


def bridge_for(base, **config):
    return NextcloudBridge({"url": base, "username": USER, "app_password": PASSWORD, **config})


def test_bridge_health():
    bridge = NextcloudBridge({})
    assert bridge.handle_get("/health", {}) == (200, {"ok": True, "platform": "nextcloud"})


def test_bridge_unconfigured_is_503():
    bridge = NextcloudBridge({})
    status, payload = bridge.handle_get("/files", {})
    assert status == 503
    assert "configured" in payload["error"]


def test_bridge_lists_files(nextcloud):
    _fake, base = nextcloud
    status, payload = bridge_for(base).handle_get("/files", {"path": ["/Documents"]})
    assert status == 200
    assert [entry["name"] for entry in payload["entries"]] == ["readme.txt"]


def test_bridge_create_event(nextcloud):
    _fake, base = nextcloud
    status, payload = bridge_for(base, default_calendar="personal").handle_post(
        "/calendars/events",
        {"summary": "Sync", "start": "2026-03-01T12:00:00Z"},
    )
    assert status == 200
    assert payload["ok"] is True and payload["calendar"] == "personal"
    assert payload["uid"]


def test_bridge_create_event_requires_summary(nextcloud):
    _fake, base = nextcloud
    status, payload = bridge_for(base, default_calendar="personal").handle_post(
        "/calendars/events", {"start": "2026-03-01T12:00:00Z"}
    )
    assert status == 400
    assert "summary" in payload["error"]


def test_bridge_requires_calendar_when_ambiguous(nextcloud):
    fake, base = nextcloud
    fake.calendars["work"] = {"label": "Work", "objects": {}}
    status, payload = bridge_for(base).handle_post(
        "/calendars/events", {"summary": "X", "start": "2026-03-01T12:00:00Z"}
    )
    assert status == 400
    assert "calendar is required" in payload["error"]


def test_bridge_uses_only_calendar_when_unset(nextcloud):
    _fake, base = nextcloud
    status, payload = bridge_for(base).handle_post(
        "/calendars/events", {"summary": "Only", "start": "2026-03-01T12:00:00Z"}
    )
    assert status == 200 and payload["calendar"] == "personal"


def test_bridge_maps_nextcloud_error_to_502(nextcloud):
    _fake, base = nextcloud
    status, payload = bridge_for(base).handle_get("/files/read", {"path": ["/missing.txt"]})
    assert status == 502
    assert "404" in payload["error"]


def test_bridge_notes_create_and_get(nextcloud):
    _fake, base = nextcloud
    bridge = bridge_for(base)
    status, payload = bridge.handle_post("/notes", {"title": "T", "content": "C"})
    assert status == 200
    note_id = payload["note"]["id"]
    status, payload = bridge.handle_get("/notes/get", {"id": [str(note_id)]})
    assert status == 200 and payload["note"]["title"] == "T"


def test_bridge_rejects_bad_types(nextcloud):
    _fake, base = nextcloud
    bridge = bridge_for(base, default_calendar="personal")
    status, payload = bridge.handle_post(
        "/calendars/events", {"summary": "X", "start": "2026-01-01T00:00:00Z", "all_day": "yes"}
    )
    assert status == 400
    status, payload = bridge.handle_post("/files/write", {"path": "/x", "content": "c", "overwrite": 1})
    assert status == 400


def test_bridge_token_guard(nextcloud):
    _fake, base = nextcloud
    bridge = bridge_for(base, token="sekret")
    port = bridge.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/files?path=/", timeout=5)
        assert exc.value.code == 401
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/health", headers={"X-Semif-Token": "sekret"}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.loads(response.read()) == {"ok": True, "platform": "nextcloud"}
    finally:
        bridge.stop()


def test_bridge_falls_back_to_top_level_config():
    bridge = build_bridge(
        {
            "nextcloud_url": "https://cloud.example.com",
            "nextcloud_username": "bob",
            "nextcloud_app_password": "pw",
            "nextcloud_default_calendar": "work",
        },
        "nextcloud",
    )
    assert isinstance(bridge, NextcloudBridge)
    assert bridge.url == "https://cloud.example.com"
    assert bridge.username == "bob"
    assert bridge.app_password == "pw"
    assert bridge.default_calendar == "work"


def test_bridge_block_overrides_top_level():
    bridge = build_bridge(
        {
            "nextcloud_url": "https://fallback.example.com",
            "nextcloud_username": "fallback",
            "bridges": {"nextcloud": {"url": "https://block.example.com", "username": "block"}},
        },
        "nextcloud",
    )
    assert bridge.url == "https://block.example.com"
    assert bridge.username == "block"


# ---- fake-client route tests ----


class FakeClient:
    configured = True

    def __init__(self):
        self.calls = []
        self.calendars = [{"name": "personal", "label": "Personal"}]

    def list_calendars(self):
        return self.calendars

    def create_event(self, calendar, uid, **fields):
        self.calls.append(("create_event", calendar, uid, fields))
        return {"ok": True, "uid": uid, "calendar": calendar, "etag": ""}

    def write_file(self, path, content, encoding, overwrite):
        self.calls.append(("write_file", path, content, encoding, overwrite))
        return {"ok": True, "path": path}


def test_bridge_passes_encoding_and_overwrite_through():
    client = FakeClient()
    bridge = NextcloudBridge({"url": "x", "username": "u", "app_password": "p"}, client=client)
    status, payload = bridge.handle_post(
        "/files/write",
        {"path": "/x", "content": "ZGF0YQ==", "encoding": "base64", "overwrite": False},
    )
    assert status == 200
    assert client.calls == [("write_file", "/x", "ZGF0YQ==", "base64", False)]


def test_bridge_generates_uid_when_absent():
    client = FakeClient()
    bridge = NextcloudBridge({"url": "x", "username": "u", "app_password": "p"}, client=client)
    status, payload = bridge.handle_post(
        "/calendars/events", {"summary": "X", "start": "2026-01-01T00:00:00Z"}
    )
    assert status == 200
    assert payload["uid"]
    assert client.calls[0][1] == "personal"


# ---- catalog / codegen surface ----


def test_known_infos_include_nextcloud():
    infos = {info.name: info for info in known_infos()}
    assert "nextcloud" in infos
    info = infos["nextcloud"]
    assert info.url_config_var == "nextcloud_bridge_url"
    assert info.auth_header == "X-Semif-Token"
    assert info.auth_config_var == "nextcloud_bridge_token"


def test_describe_bridges_renders_nextcloud():
    text = describe_bridges()
    assert "service: nextcloud" in text
    assert "nextcloud_bridge_url" in text
    assert "/calendars/events" in text
    assert "/files/read" in text
    assert "/notes" in text


def test_nextcloud_bridge_check_requirements_is_stdlib():
    ok, hint = NextcloudBridge({}).check_requirements()
    assert ok is True and hint is None
