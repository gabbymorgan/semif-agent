"""Nextcloud bridge: files, calendars, tasks, contacts, and notes over HTTP.

Nextcloud speaks several native protocols (WebDAV for files, CalDAV for events
and tasks, CardDAV for contacts, and OCS for the Notes app and server info). A
codegen-authored skill body must never speak those directly. This bridge owns the
connection — base URL, username, and app password — and exposes one uniform,
token-guarded localhost JSON API, so a body only reads `nextcloud_bridge_url` and
`nextcloud_bridge_token` from `ctx.config` and calls a documented route.

There is no service daemon: Nextcloud is plain HTTPS, so the bridge is stdlib-only
(`urllib` + `xml.etree`) and starts instantly. The connection comes from
`bridges.nextcloud` (url/username/app_password/default_calendar/default_addressbook),
falling back to the top-level `nextcloud_url`/`nextcloud_username`/
`nextcloud_app_password`/`nextcloud_default_calendar` values the nextcloud seeds
already use, so the same account is configured in one place.

The actual DAV/OCS work lives in `nextcloud_client.py`; the routes here validate
input and map a real failure (`NextcloudError`) to a 502 rather than a fabricated
success.
"""

from __future__ import annotations

import uuid

from .base import BridgeInfo, BridgeService
from .nextcloud_client import DEFAULT_TIMEOUT, NextcloudClient, NextcloudError

#: Nextcloud Deck backs each board with a VTODO-only calendar named like this.
_DECK_PREFIX = "app-generated--deck--"

_COMPONENT_KINDS = {
    "VTODO": "task list",
    "VEVENT": "event calendar",
    "VJOURNAL": "journal",
}


def _supports(calendar: dict, component: str) -> bool:
    """Whether a calendar accepts `component`; unknown (empty) means allow."""
    components = calendar.get("components") or []
    return not components or component in components


def _is_deck_board(calendar: dict) -> bool:
    return str(calendar.get("name") or "").startswith(_DECK_PREFIX)


def _component_kind(component: str) -> str:
    return _COMPONENT_KINDS.get(component, "calendar of that kind")


def _join_components(calendar: dict) -> str:
    components = calendar.get("components") or []
    return ", ".join(components) if components else "an unknown component set"


def _task_list_hint(calendars: list[dict]) -> str:
    names = [
        c["name"]
        for c in calendars
        if _supports(c, "VTODO") and not _is_deck_board(c)
    ]
    if names:
        return "task lists: " + ", ".join(names)
    return "no VTODO-capable task lists are available"


class NextcloudBridge(BridgeService):
    name = "nextcloud"
    INFO = BridgeInfo(
        name="nextcloud",
        service="nextcloud",
        description=(
            "Read and write the user's Nextcloud: files (WebDAV), calendar events "
            "and tasks (CalDAV), contacts (CardDAV), and Notes (the Notes app)."
        ),
        url_config_var="nextcloud_bridge_url",
        endpoints=(
            "GET /health -> 200 {\"ok\": true, \"platform\": \"nextcloud\"} — the "
            "bridge is up",
            "GET /user -> 200 {\"user\": {id, display_name, email, language, "
            "locale}} — the account (OCS)",
            "GET /capabilities -> 200 {Nextcloud server capabilities} (OCS)",
            "GET /files?path=<dir> -> 200 {\"path\", \"entries\": [{name, path, "
            "type (file|dir), size, modified, etag, content_type}]} — list a "
            "folder (default /)",
            "GET /files/stat?path=<path> -> 200 {one entry, same fields as an "
            "entries item}",
            "GET /files/read?path=<path> -> 200 {\"path\", \"content\", "
            "\"encoding\" (utf-8|base64), \"size\"} — file contents; binary is "
            "base64",
            "GET /files/search?query=<text>&path=<dir> -> 200 {\"results\": "
            "[entry]} — filename search under a folder",
            "POST /files/write {\"path\", \"content\", \"encoding\"?: "
            "utf-8|base64, \"overwrite\"?: bool} -> 200 {\"ok\": true, \"path\"}",
            "POST /files/mkdir {\"path\"} -> 200 {\"ok\": true, \"path\"}",
            "POST /files/delete {\"path\"} -> 200 {\"ok\": true, \"path\"}",
            "POST /files/move {\"path\", \"destination\", \"overwrite\"?: bool} "
            "-> 200 {\"ok\": true, \"path\", \"destination\"}",
            "POST /files/copy {\"path\", \"destination\", \"overwrite\"?: bool} "
            "-> 200 {\"ok\": true, \"path\", \"destination\"}",
            "GET /calendars -> 200 {\"calendars\": [{name, label, description, "
            "color, ctag, components}]} — components lists the CalDAV component "
            "types each collection accepts (VEVENT, VTODO, VJOURNAL)",
            "GET /tasklists -> 200 {\"tasklists\": [{name, label, description, "
            "color, ctag, components}]} — the user's task lists (calendars that "
            "accept VTODO), excluding Nextcloud Deck boards; use this to pick a "
            "target for POST /tasks",
            "GET /calendars/events?calendar=<name>&start=<iso>&end=<iso> -> 200 "
            "{\"calendar\", \"events\": [{uid, summary, start, end, all_day, "
            "location, description, rrule, status, href, etag}]} — start/end "
            "optional (ISO 8601); without them every event is returned",
            "POST /calendars/events {\"calendar\"?, \"uid\"?, \"summary\", "
            "\"start\", \"end\"?, \"all_day\"?, \"location\"?, \"description\"?, "
            "\"rrule\"?, \"overwrite\"?} -> 200 {\"ok\": true, \"uid\", "
            "\"calendar\", \"etag\"}",
            "POST /calendars/events/update {\"calendar\"?, \"uid\", \"summary\"?, "
            "\"start\"?, \"end\"?, \"all_day\"?, \"location\"?, \"description\"?, "
            "\"rrule\"?} -> 200 {\"ok\": true, \"uid\", \"calendar\", \"etag\"} — "
            "only the named fields change",
            "POST /calendars/events/delete {\"calendar\"?, \"uid\"} -> 200 "
            "{\"ok\": true, \"uid\", \"calendar\"}",
            "GET /tasks?calendar=<name> -> 200 {\"calendar\", \"tasks\": [{uid, "
            "summary, due, status, percent_complete, priority, description, href, "
            "etag}]} — CalDAV VTODO; calendar must be a VTODO-capable task list "
            "(see GET /tasklists), defaulting to the configured task calendar",
            "POST /tasks {\"calendar\"?, \"uid\"?, \"summary\", \"due\"?, "
            "\"priority\"?, \"description\"?} -> 200 {\"ok\": true, \"uid\", "
            "\"calendar\", \"etag\"} — calendar must be a VTODO-capable task list "
            "(see GET /tasklists); an event-only calendar is rejected with 400",
            "POST /tasks/update {\"calendar\"?, \"uid\", \"summary\"?, \"due\"?, "
            "\"priority\"?, \"description\"?, \"status\"?, \"percent_complete\"?} "
            "-> 200 {ok, uid}",
            "POST /tasks/complete {\"calendar\"?, \"uid\"} -> 200 {ok, uid} — "
            "marks the task COMPLETED",
            "POST /tasks/delete {\"calendar\"?, \"uid\"} -> 200 {ok}",
            "GET /addressbooks -> 200 {\"addressbooks\": [{name, label, "
            "description}]}",
            "GET /contacts?addressbook=<name>&query=<text> -> 200 "
            "{\"addressbook\", \"contacts\": [{uid, fn, n, emails, phones, org, "
            "title, note, url, href, etag}]} — query filters by name/org/email/"
            "phone",
            "POST /contacts {\"addressbook\"?, \"uid\"?, \"fn\", \"emails\"?, "
            "\"phones\"?, \"org\"?, \"title\"?, \"note\"?, \"url\"?, \"n\"?} -> "
            "200 {ok, uid}",
            "POST /contacts/update {\"addressbook\"?, \"uid\", ...fields} -> 200 "
            "{ok, uid}",
            "POST /contacts/delete {\"addressbook\"?, \"uid\"} -> 200 {ok}",
            "GET /notes -> 200 {\"notes\": [{id, title, category, modified, ...}]} "
            "— Nextcloud Notes app (metadata only)",
            "GET /notes/get?id=<id> -> 200 {\"note\": {id, title, content, "
            "category, ...}}",
            "POST /notes {\"title\", \"content\"?, \"category\"?} -> 200 "
            "{\"note\": {...}}",
            "POST /notes/update {\"id\", \"title\"?, \"content\"?, "
            "\"category\"?} -> 200 {\"note\": {...}}",
            "POST /notes/delete {\"id\"} -> 200 {\"ok\": true, \"id\"}",
            "any request -> 401 {\"error\": \"unauthorized\"} when the token is "
            "configured and the auth header is missing or wrong",
            "503 {\"error\": \"...\"} when the bridge has no Nextcloud connection "
            "configured; 502 {\"error\": \"...\"} on a real Nextcloud failure "
            "(transport, HTTP error, or bad reply)",
        ),
        config_vars=("nextcloud_bridge_url", "nextcloud_bridge_token"),
        config_var_docs=(
            (
                "nextcloud_bridge_url",
                "Base URL of the local Nextcloud bridge (e.g. "
                "http://127.0.0.1:5230); never hardcode it in the body.",
            ),
            (
                "nextcloud_bridge_token",
                "Shared secret for the bridge, if one is configured; sent as the "
                "X-Semif-Token header. Leave blank when the bridge requires no auth.",
            ),
        ),
        auth_header="X-Semif-Token",
        auth_config_var="nextcloud_bridge_token",
    )

    def __init__(self, config: dict | None = None, trace=None, fallback: dict | None = None, client=None):
        super().__init__(config, trace)
        cfg = config or {}
        fb = fallback or {}

        def pick(key: str, default=""):
            value = cfg.get(key)
            if value in (None, ""):
                value = fb.get(key)
            return default if value in (None, "") else value

        self.url = str(pick("url")).strip()
        self.username = str(pick("username")).strip()
        self.app_password = str(pick("app_password"))
        self.default_calendar = str(pick("default_calendar")).strip()
        self.default_task_calendar = str(pick("default_task_calendar")).strip()
        self.default_addressbook = str(pick("default_addressbook")).strip()
        self.verify_tls = bool(cfg.get("verify_tls", fb.get("verify_tls", True)))
        self.timeout = float(cfg.get("timeout", DEFAULT_TIMEOUT))
        #: A test seam: inject a client double. Production builds the real one.
        self.client = client or NextcloudClient(
            self.url,
            self.username,
            self.app_password,
            timeout=self.timeout,
            verify_tls=self.verify_tls,
            default_calendar=self.default_calendar,
            default_addressbook=self.default_addressbook,
        )

    # ---- lifecycle ----

    def check_requirements(self) -> tuple[bool, str | None]:
        # Stdlib only: no missing dependency can make the bridge unstartable. A
        # missing connection surfaces as a 503 on the call, not a startup failure.
        return True, None

    # ---- routes ----

    def handle_get(self, path: str, query: dict) -> tuple[int, dict]:
        if path == "/health":
            return 200, {"ok": True, "platform": self.name}
        if not self.client.configured:
            return self._unconfigured()
        try:
            return self._get(path, query)
        except NextcloudError as exc:
            return 502, {"error": str(exc)[:300]}
        except ValueError as exc:
            return 400, {"error": str(exc)[:300]}

    def handle_post(self, path: str, payload: dict) -> tuple[int, dict]:
        if not self.client.configured:
            return self._unconfigured()
        try:
            return self._post(path, payload)
        except NextcloudError as exc:
            return 502, {"error": str(exc)[:300]}
        except ValueError as exc:
            return 400, {"error": str(exc)[:300]}

    def _unconfigured(self) -> tuple[int, dict]:
        return 503, {
            "error": (
                "the Nextcloud bridge has no connection configured (set "
                "bridges.nextcloud.url/username/app_password, or the top-level "
                "nextcloud_url/nextcloud_username/nextcloud_app_password)"
            )
        }

    def _get(self, path: str, query: dict) -> tuple[int, dict]:
        if path == "/user":
            return 200, {"user": self.client.user_info()}
        if path == "/capabilities":
            return 200, self.client.capabilities()
        if path == "/files":
            return 200, self.client.list_files(self._q(query, "path", "/") or "/")
        if path == "/files/stat":
            return 200, self.client.stat(self._required_query(query, "path"))
        if path == "/files/read":
            return 200, self.client.read_file(self._required_query(query, "path"))
        if path == "/files/search":
            query_text = self._required_query(query, "query")
            return 200, {
                "results": self.client.search(
                    query_text, self._q(query, "path", "/") or "/"
                )
            }
        if path == "/calendars":
            return 200, {"calendars": self.client.list_calendars()}
        if path == "/tasklists":
            return 200, {"tasklists": self._task_lists()}
        if path == "/calendars/events":
            calendar = self._resolve_calendar(self._q(query, "calendar"))
            return 200, {
                "calendar": calendar,
                "events": self.client.list_events(
                    calendar,
                    self._q(query, "start", "") or "",
                    self._q(query, "end", "") or "",
                ),
            }
        if path == "/tasks":
            calendar = self._resolve_calendar(self._q(query, "calendar"), "VTODO")
            return 200, {"calendar": calendar, "tasks": self.client.list_tasks(calendar)}
        if path == "/addressbooks":
            return 200, {"addressbooks": self.client.list_addressbooks()}
        if path == "/contacts":
            addressbook = self._resolve_addressbook(self._q(query, "addressbook"))
            return 200, {
                "addressbook": addressbook,
                "contacts": self.client.list_contacts(
                    addressbook, self._q(query, "query", "") or ""
                ),
            }
        if path == "/notes":
            return 200, {"notes": self.client.list_notes()}
        if path == "/notes/get":
            return 200, {"note": self.client.get_note(self._required_query(query, "id"))}
        return 404, {"error": "not found"}

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        if path == "/files/write":
            return 200, self.client.write_file(
                self._required(payload, "path"),
                self._required(payload, "content", allow_empty=True),
                self._optional_str(payload, "encoding", "utf-8"),
                self._optional_bool(payload, "overwrite", True),
            )
        if path == "/files/mkdir":
            return 200, self.client.mkdir(self._required(payload, "path"))
        if path == "/files/delete":
            return 200, self.client.delete(self._required(payload, "path"))
        if path == "/files/move":
            return 200, self.client.move(
                self._required(payload, "path"),
                self._required(payload, "destination"),
                self._optional_bool(payload, "overwrite", False),
            )
        if path == "/files/copy":
            return 200, self.client.copy(
                self._required(payload, "path"),
                self._required(payload, "destination"),
                self._optional_bool(payload, "overwrite", False),
            )
        if path == "/calendars/events":
            return 200, self._create_event(payload)
        if path == "/calendars/events/update":
            return 200, self._update_event(payload)
        if path == "/calendars/events/delete":
            calendar = self._resolve_calendar(self._optional_str(payload, "calendar"))
            return 200, self.client.delete_object(calendar, self._required(payload, "uid"))
        if path == "/tasks":
            return 200, self._create_task(payload)
        if path == "/tasks/update":
            return 200, self._update_task(payload)
        if path == "/tasks/complete":
            calendar = self._resolve_calendar(self._optional_str(payload, "calendar"), "VTODO")
            return 200, self.client.complete_task(calendar, self._required(payload, "uid"))
        if path == "/tasks/delete":
            calendar = self._resolve_calendar(self._optional_str(payload, "calendar"), "VTODO")
            return 200, self.client.delete_object(calendar, self._required(payload, "uid"))
        if path == "/contacts":
            return 200, self._create_contact(payload)
        if path == "/contacts/update":
            return 200, self._update_contact(payload)
        if path == "/contacts/delete":
            addressbook = self._resolve_addressbook(
                self._optional_str(payload, "addressbook")
            )
            return 200, self.client.delete_contact(
                addressbook, self._required(payload, "uid")
            )
        if path == "/notes":
            return 200, {
                "note": self.client.create_note(
                    self._required(payload, "title"),
                    self._optional_str(payload, "content"),
                    self._optional_str(payload, "category"),
                )
            }
        if path == "/notes/update":
            return 200, {"note": self._update_note(payload)}
        if path == "/notes/delete":
            return 200, self.client.delete_note(self._required_id(payload))
        return 404, {"error": "not found"}

    # ---- calendar/task helpers ----

    def _create_event(self, payload: dict) -> dict:
        calendar = self._resolve_calendar(self._optional_str(payload, "calendar"))
        uid = self._optional_str(payload, "uid") or uuid.uuid4().hex
        return self.client.create_event(
            calendar,
            uid,
            summary=self._required(payload, "summary"),
            start=self._required(payload, "start"),
            end=self._optional_str(payload, "end"),
            all_day=self._optional_bool(payload, "all_day", False),
            location=self._optional_str(payload, "location"),
            description=self._optional_str(payload, "description"),
            rrule=self._optional_str(payload, "rrule"),
            overwrite=self._optional_bool(payload, "overwrite", False),
        )

    def _update_event(self, payload: dict) -> dict:
        calendar = self._resolve_calendar(self._optional_str(payload, "calendar"))
        uid = self._required(payload, "uid")
        changes = {
            key: payload[key]
            for key in ("summary", "start", "end", "location", "description", "rrule")
            if key in payload
        }
        if "all_day" in payload:
            changes["all_day"] = self._optional_bool(payload, "all_day", False)
        return self.client.update_event(calendar, uid, **changes)

    def _create_task(self, payload: dict) -> dict:
        calendar = self._resolve_calendar(self._optional_str(payload, "calendar"), "VTODO")
        uid = self._optional_str(payload, "uid") or uuid.uuid4().hex
        return self.client.create_task(
            calendar,
            uid,
            summary=self._required(payload, "summary"),
            due=self._optional_str(payload, "due"),
            priority=self._optional_int(payload, "priority"),
            description=self._optional_str(payload, "description"),
        )

    def _update_task(self, payload: dict) -> dict:
        calendar = self._resolve_calendar(self._optional_str(payload, "calendar"), "VTODO")
        uid = self._required(payload, "uid")
        changes = {
            key: payload[key]
            for key in ("summary", "due", "priority", "description", "status", "percent_complete")
            if key in payload
        }
        return self.client.update_task(calendar, uid, **changes)

    # ---- contact helpers ----

    def _create_contact(self, payload: dict) -> dict:
        addressbook = self._resolve_addressbook(self._optional_str(payload, "addressbook"))
        uid = self._optional_str(payload, "uid") or uuid.uuid4().hex
        return self.client.create_contact(
            addressbook,
            uid,
            fn=self._required(payload, "fn"),
            n=self._optional_str(payload, "n"),
            emails=self._string_list(payload, "emails"),
            phones=self._string_list(payload, "phones"),
            org=self._optional_str(payload, "org"),
            title=self._optional_str(payload, "title"),
            note=self._optional_str(payload, "note"),
            url=self._optional_str(payload, "url"),
        )

    def _update_contact(self, payload: dict) -> dict:
        addressbook = self._resolve_addressbook(self._optional_str(payload, "addressbook"))
        uid = self._required(payload, "uid")
        changes = {}
        for key in ("fn", "n", "org", "title", "note", "url"):
            if key in payload:
                changes[key] = payload[key]
        for key in ("emails", "phones"):
            if key in payload:
                changes[key] = self._string_list(payload, key)
        return self.client.update_contact(addressbook, uid, **changes)

    # ---- notes helpers ----

    def _update_note(self, payload: dict) -> dict:
        note_id = self._required_id(payload)
        changes = {
            key: payload[key] for key in ("title", "content", "category") if key in payload
        }
        return self.client.update_note(note_id, **changes)

    # ---- default resolution ----

    def _resolve_calendar(self, name, component: str | None = None) -> str:
        """Resolve a calendar name, honoring the CalDAV component it must accept.

        An explicit name is validated when the collection's component set is
        known: asking a task operation to write to an event-only calendar is a
        real error (a `VTODO` PUT there is rejected), not a silent redirect. With
        no name, a task operation falls back to the configured task calendar, then
        the event default when it also accepts VTODO, then the sole user task
        list. `component=None` (events, files) keeps the historical behavior.
        """
        requested = str(name or "").strip()
        if requested:
            if component:
                calendars = self.client.list_calendars()
                match = next((c for c in calendars if c.get("name") == requested), None)
                if match is not None and not _supports(match, component):
                    raise ValueError(
                        f"{requested!r} is not a {_component_kind(component)} "
                        f"(it accepts {_join_components(match)}); "
                        f"{_task_list_hint(calendars)}"
                    )
            return requested

        # The configured defaults, most specific first. For a task operation the
        # event default is still consulted when it also accepts VTODO.
        defaults = [self.default_task_calendar] if component == "VTODO" else []
        defaults.append(self.default_calendar)
        calendars = None
        for default in defaults:
            if not default:
                continue
            if not component:
                return default
            if calendars is None:
                calendars = self.client.list_calendars()
            match = next((c for c in calendars if c.get("name") == default), None)
            if match is None or _supports(match, component):
                return default

        if calendars is None:
            calendars = self.client.list_calendars()
        if component:
            calendars = [c for c in calendars if _supports(c, component)]
            if component == "VTODO":
                calendars = [c for c in calendars if not _is_deck_board(c)]
        if len(calendars) == 1:
            return calendars[0]["name"]
        raise ValueError(
            f"calendar is required (name one in the request, or set "
            f"bridges.nextcloud.default_calendar"
            + (
                " / default_task_calendar"
                if component == "VTODO"
                else ""
            )
            + ")"
        )

    def _task_lists(self) -> list[dict]:
        """The user's task lists: VTODO-capable calendars, minus Deck boards."""
        return [
            calendar
            for calendar in self.client.list_calendars()
            if _supports(calendar, "VTODO") and not _is_deck_board(calendar)
        ]

    def _resolve_addressbook(self, name) -> str:
        name = str(name or self.default_addressbook or "").strip()
        if name:
            return name
        addressbooks = self.client.list_addressbooks()
        if len(addressbooks) == 1:
            return addressbooks[0]["name"]
        raise ValueError(
            "addressbook is required (name one in the request, or set "
            "bridges.nextcloud.default_addressbook)"
        )

    # ---- input helpers ----

    @staticmethod
    def _q(query: dict, key: str, default=None):
        values = query.get(key)
        if not values:
            return default
        return values[0]

    @staticmethod
    def _required_query(query: dict, key: str) -> str:
        value = NextcloudBridge._q(query, key)
        if value is None or not str(value).strip():
            raise ValueError(f"{key} is required")
        return str(value).strip()

    @staticmethod
    def _required(payload: dict, key: str, allow_empty: bool = False) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or (not allow_empty and not value.strip()):
            raise ValueError(f"{key} is required")
        return value if allow_empty else value.strip()

    @staticmethod
    def _required_id(payload: dict):
        value = payload.get("id")
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError("id is required")
        return value

    @staticmethod
    def _optional_str(payload: dict, key: str, default: str = "") -> str:
        value = payload.get(key)
        if value is None:
            return default
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value

    @staticmethod
    def _optional_bool(payload: dict, key: str, default: bool) -> bool:
        value = payload.get(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")
        return value

    @staticmethod
    def _optional_int(payload: dict, key: str):
        value = payload.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        return value

    @staticmethod
    def _string_list(payload: dict, key: str) -> list[str]:
        value = payload.get(key)
        if value is None:
            return []
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{key} must be a list of strings")
        return value
