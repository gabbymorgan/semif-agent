"""Outlook bridge: the user's Microsoft account (Graph) over HTTP.

Microsoft Graph is a plain HTTPS/JSON API, but a codegen-authored skill body
must still never speak it (or OAuth) directly. This bridge owns the connection —
the Entra ID client id, the tenant, and the stored refresh token — and exposes
one uniform, token-guarded localhost JSON API for mail, calendar, Microsoft To Do
tasks, contacts, and OneDrive files, so a body only reads `outlook_bridge_url`
and `outlook_bridge_token` from `ctx.config` and calls a documented route.

There is no service daemon and no third-party dependency: Graph is plain HTTPS,
so the bridge is stdlib-only (`urllib`) and starts instantly, like Nextcloud.
Authorization is the OAuth2 device code flow, run once with
`scripts/outlook-auth.py`; the resulting tokens are stored (0600) under the
checkout's `.runtime/outlook/token.json`. The bridge starts without them and an
unauthenticated route returns 503 telling the operator to run that helper.

The actual Graph work lives in `outlook_client.py`; the routes here validate
input and map a real failure (`OutlookError`) to a 502 — or a missing
authorization (`OutlookAuthRequired`) to a 503 — rather than a fabricated
success.
"""

from __future__ import annotations

from pathlib import Path

from .base import BridgeInfo, BridgeService
from .outlook_client import (
    DEFAULT_SCOPES,
    DEFAULT_TENANT,
    DEFAULT_TIMEOUT,
    OutlookAuthRequired,
    OutlookClient,
    OutlookError,
)

#: The checkout root (`semif_agent/bridges/outlook.py` -> repo root), used to
#: anchor the default token store under the gitignored `.runtime/` tree.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKEN_PATH = str(_REPO_ROOT / ".runtime" / "outlook" / "token.json")


class OutlookBridge(BridgeService):
    name = "outlook"
    INFO = BridgeInfo(
        name="outlook",
        service="outlook",
        description=(
            "Read and write the user's Microsoft account (Outlook / Microsoft "
            "365) through Microsoft Graph: mail, calendar events, Microsoft To "
            "Do tasks, contacts, and OneDrive files."
        ),
        url_config_var="outlook_bridge_url",
        endpoints=(
            "GET /health -> 200 {\"ok\": true, \"platform\": \"outlook\"} — the "
            "bridge is up",
            "GET /user -> 200 {\"user\": {id, display_name, mail, "
            "user_principal_name}} — the signed-in account",
            "GET /mail/folders -> 200 {\"folders\": [{id, name, total, unread}]}",
            "GET /mail/messages?folder=<name|id>&count=<n>&search=<text>&unread=<bool> "
            "-> 200 {\"folder\", \"messages\": [{id, subject, from, to, cc, "
            "received, is_read, has_attachments, importance, preview, web_link}]} "
            "— newest first; folder defaults to inbox (a well-known name like "
            "sentitems/drafts/archive or a folder display name/id)",
            "GET /mail/messages/get?id=<id> -> 200 {\"message\": {..., body, "
            "body_type}} — the full message",
            "POST /mail/send {\"to\": [addr, ...], \"cc\"?: [addr, ...], "
            "\"subject\"?, \"body\"?, \"content_type\"?: text|html} -> 200 "
            "{\"ok\": true, \"to\", \"cc\", \"subject\"} — sends immediately",
            "POST /mail/draft {\"to\"?, \"cc\"?, \"subject\"?, \"body\"?, "
            "\"content_type\"?} -> 200 {\"ok\": true, \"id\", \"subject\"} — "
            "saves a draft in the Drafts folder",
            "GET /calendars -> 200 {\"calendars\": [{id, name, color, "
            "is_default, can_edit}]}",
            "GET /calendars/events?calendar=<name|id>&start=<iso>&end=<iso> -> "
            "200 {\"calendar\", \"events\": [{id, subject, start, end, timezone, "
            "all_day, location, preview, show_as, is_online_meeting, web_link, "
            "organizer}]} — start/end are ISO 8601 (default: now to +30 days); "
            "recurring events are expanded, calendar defaults to the configured "
            "or primary calendar",
            "POST /calendars/events {\"calendar\"?, \"subject\", \"start\", "
            "\"end\"?, \"all_day\"?, \"location\"?, \"description\"?, "
            "\"attendees\"?: [addr, ...], \"timezone\"?} -> 200 {\"ok\": true, "
            "\"id\", \"subject\", \"calendar\", \"web_link\"} — start/end are "
            "local ISO 8601 in the named timezone (default UTC when absent); end "
            "defaults to +1h (+1 day when all_day)",
            "POST /calendars/events/update {\"id\", \"subject\"?, \"start\"?, "
            "\"end\"?, \"all_day\"?, \"location\"?, \"description\"?, "
            "\"timezone\"?} -> 200 {ok, id, subject} — only the named fields change",
            "POST /calendars/events/delete {\"id\"} -> 200 {\"ok\": true, \"id\"}",
            "GET /tasklists -> 200 {\"tasklists\": [{id, name, is_owner, "
            "is_default}]} — Microsoft To Do lists",
            "GET /tasks?list=<name|id> -> 200 {\"list\", \"tasks\": [{id, title, "
            "status, importance, due, completed, description, list_id}]} — list "
            "defaults to the configured or default list",
            "POST /tasks {\"list\"?, \"title\", \"due\"?, \"importance\"?, "
            "\"description\"?} -> 200 {\"ok\": true, \"id\", \"title\", "
            "\"list_id\"} — importance is low|normal|high; due is ISO 8601",
            "POST /tasks/update {\"id\", \"list\"?, \"title\"?, \"due\"?, "
            "\"importance\"?, \"status\"?, \"description\"?} -> 200 {ok, id, status}",
            "POST /tasks/complete {\"id\", \"list\"?} -> 200 {ok, id, status} — "
            "marks the task completed",
            "POST /tasks/delete {\"id\", \"list\"?} -> 200 {\"ok\": true, \"id\"}",
            "GET /contacts?query=<text>&count=<n> -> 200 {\"contacts\": [{id, "
            "display_name, given_name, surname, emails, phones, company, "
            "job_title}]} — query filters by name/email/company",
            "GET /files?path=<dir> -> 200 {\"path\", \"entries\": [{id, name, "
            "path, size, modified, is_folder, content_type, web_url}]} — OneDrive "
            "folder listing (default root)",
            "GET /files/read?path=<path> -> 200 {\"path\", \"name\", \"content\", "
            "\"encoding\" (utf-8|base64), \"size\", \"content_type\", \"modified\"} "
            "— file contents; binary is base64",
            "POST /files/write {\"path\", \"content\", \"encoding\"?: "
            "utf-8|base64, \"overwrite\"?: bool} -> 200 {\"ok\": true, \"path\", "
            "\"id\", \"size\", \"web_url\"}",
            "any request -> 401 {\"error\": \"unauthorized\"} when the token is "
            "configured and the auth header is missing or wrong",
            "503 {\"error\": \"...\"} when the bridge is not authenticated "
            "(run scripts/outlook-auth.py) or has no client id configured; 502 "
            "{\"error\": \"...\"} on a real Microsoft Graph failure (transport, "
            "HTTP error, or bad reply)",
        ),
        config_vars=("outlook_bridge_url", "outlook_bridge_token"),
        config_var_docs=(
            (
                "outlook_bridge_url",
                "Base URL of the local Outlook bridge (e.g. "
                "http://127.0.0.1:5231); never hardcode it in the body.",
            ),
            (
                "outlook_bridge_token",
                "Shared secret for the bridge, if one is configured; sent as the "
                "X-Semif-Token header. Leave blank when the bridge requires no auth.",
            ),
        ),
        auth_header="X-Semif-Token",
        auth_config_var="outlook_bridge_token",
    )

    def __init__(self, config: dict | None = None, trace=None, client=None):
        super().__init__(config, trace)
        cfg = config or {}
        self.client_id = str(cfg.get("client_id", "") or "").strip()
        self.tenant = str(cfg.get("tenant", DEFAULT_TENANT) or DEFAULT_TENANT).strip() or DEFAULT_TENANT
        self.token_path = str(cfg.get("token_path") or DEFAULT_TOKEN_PATH)
        self.default_timezone = str(cfg.get("default_timezone", "UTC") or "UTC").strip() or "UTC"
        self.default_mail_folder = str(cfg.get("default_mail_folder", "inbox") or "inbox").strip() or "inbox"
        #: A test seam: inject a client double. Production builds the real one.
        self.client = client or OutlookClient(
            self.client_id,
            tenant=self.tenant,
            token_path=self.token_path,
            scopes=cfg.get("scopes") or DEFAULT_SCOPES,
            timeout=float(cfg.get("timeout", DEFAULT_TIMEOUT)),
            verify_tls=bool(cfg.get("verify_tls", True)),
            default_calendar=str(cfg.get("default_calendar", "") or "").strip(),
            default_task_list=str(cfg.get("default_task_list", "") or "").strip(),
            default_mail_folder=self.default_mail_folder,
            default_timezone=self.default_timezone,
        )

    # ---- lifecycle ----

    def check_requirements(self) -> tuple[bool, str | None]:
        # Stdlib only: no missing dependency can make the bridge unstartable. A
        # missing authorization surfaces as a 503 on the call, not a startup failure.
        return True, None

    # ---- routes ----

    def handle_get(self, path: str, query: dict) -> tuple[int, dict]:
        if path == "/health":
            return 200, {"ok": True, "platform": self.name}
        if not self.client.configured:
            return self._unconfigured()
        try:
            return self._get(path, query)
        except OutlookAuthRequired as exc:
            return 503, {"error": str(exc)[:300]}
        except OutlookError as exc:
            return 502, {"error": str(exc)[:300]}
        except ValueError as exc:
            return 400, {"error": str(exc)[:300]}

    def handle_post(self, path: str, payload: dict) -> tuple[int, dict]:
        if not self.client.configured:
            return self._unconfigured()
        try:
            return self._post(path, payload)
        except OutlookAuthRequired as exc:
            return 503, {"error": str(exc)[:300]}
        except OutlookError as exc:
            return 502, {"error": str(exc)[:300]}
        except ValueError as exc:
            return 400, {"error": str(exc)[:300]}

    def _unconfigured(self) -> tuple[int, dict]:
        return 503, {
            "error": (
                "the Outlook bridge has no connection configured (set "
                "bridges.outlook.client_id, then run scripts/outlook-auth.py)"
            )
        }

    def _get(self, path: str, query: dict) -> tuple[int, dict]:
        if path == "/user":
            return 200, {"user": self.client.me()}
        if path == "/mail/folders":
            return 200, {"folders": self.client.list_mail_folders()}
        if path == "/mail/messages":
            folder = self._q(query, "folder", "") or ""
            return 200, {
                "folder": folder or self.default_mail_folder,
                "messages": self.client.list_messages(
                    folder,
                    self._int_query(query, "count", 25),
                    self._q(query, "search", "") or "",
                    self._bool_query(query, "unread", False),
                ),
            }
        if path == "/mail/messages/get":
            return 200, {"message": self.client.get_message(self._required_query(query, "id"))}
        if path == "/calendars":
            return 200, {"calendars": self.client.list_calendars()}
        if path == "/calendars/events":
            return 200, {
                "calendar": self._q(query, "calendar", "") or "",
                "events": self.client.list_events(
                    self._q(query, "calendar", "") or "",
                    self._q(query, "start", "") or "",
                    self._q(query, "end", "") or "",
                ),
            }
        if path == "/tasklists":
            return 200, {"tasklists": self.client.list_task_lists()}
        if path == "/tasks":
            task_list = self._q(query, "list", "") or ""
            return 200, {"list": task_list, "tasks": self.client.list_tasks(task_list)}
        if path == "/contacts":
            return 200, {
                "contacts": self.client.list_contacts(
                    self._q(query, "query", "") or "", self._int_query(query, "count", 50)
                )
            }
        if path == "/files":
            folder = self._q(query, "path", "/") or "/"
            return 200, {"path": folder, "entries": self.client.list_files(folder)}
        if path == "/files/read":
            return 200, self.client.read_file(self._required_query(query, "path"))
        return 404, {"error": "not found"}

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        if path == "/mail/send":
            recipients = self._string_list(payload, "to")
            if not recipients:
                raise ValueError("to is required")
            return 200, self.client.send_mail(
                recipients,
                self._optional_str(payload, "subject"),
                self._optional_str(payload, "body"),
                self._string_list(payload, "cc"),
                self._optional_str(payload, "content_type", "text") or "text",
            )
        if path == "/mail/draft":
            return 200, self.client.create_draft(
                self._string_list(payload, "to"),
                self._optional_str(payload, "subject"),
                self._optional_str(payload, "body"),
                self._string_list(payload, "cc"),
                self._optional_str(payload, "content_type", "text") or "text",
            )
        if path == "/calendars/events":
            return 200, self.client.create_event(
                calendar=self._optional_str(payload, "calendar"),
                subject=self._required(payload, "subject"),
                start=self._required(payload, "start"),
                end=self._optional_str(payload, "end"),
                all_day=self._optional_bool(payload, "all_day", False),
                location=self._optional_str(payload, "location"),
                description=self._optional_str(payload, "description"),
                attendees=self._string_list(payload, "attendees"),
                timezone_name=self._optional_str(payload, "timezone"),
            )
        if path == "/calendars/events/update":
            event_id = self._required(payload, "id")
            changes = {
                key: payload[key]
                for key in ("subject", "start", "end", "location", "description")
                if key in payload
            }
            if "all_day" in payload:
                changes["all_day"] = self._optional_bool(payload, "all_day", False)
            return 200, self.client.update_event(
                event_id, timezone_name=self._optional_str(payload, "timezone"), **changes
            )
        if path == "/calendars/events/delete":
            return 200, self.client.delete_event(self._required(payload, "id"))
        if path == "/tasks":
            return 200, self.client.create_task(
                task_list=self._optional_str(payload, "list"),
                title=self._required(payload, "title"),
                due=self._optional_str(payload, "due"),
                importance=self._optional_str(payload, "importance"),
                description=self._optional_str(payload, "description"),
            )
        if path == "/tasks/update":
            task_id = self._required(payload, "id")
            changes = {
                key: payload[key]
                for key in ("title", "due", "importance", "status", "description")
                if key in payload
            }
            return 200, self.client.update_task(
                task_id, task_list=self._optional_str(payload, "list"), **changes
            )
        if path == "/tasks/complete":
            return 200, self.client.complete_task(
                self._required(payload, "id"), task_list=self._optional_str(payload, "list")
            )
        if path == "/tasks/delete":
            return 200, self.client.delete_task(
                self._required(payload, "id"), task_list=self._optional_str(payload, "list")
            )
        if path == "/files/write":
            return 200, self.client.write_file(
                self._required(payload, "path"),
                self._required(payload, "content", allow_empty=True),
                self._optional_str(payload, "encoding", "utf-8") or "utf-8",
                self._optional_bool(payload, "overwrite", True),
            )
        return 404, {"error": "not found"}

    # ---- input helpers ----

    @staticmethod
    def _q(query: dict, key: str, default=None):
        values = query.get(key)
        if not values:
            return default
        return values[0]

    @staticmethod
    def _required_query(query: dict, key: str) -> str:
        value = OutlookBridge._q(query, key)
        if value is None or not str(value).strip():
            raise ValueError(f"{key} is required")
        return str(value).strip()

    @staticmethod
    def _int_query(query: dict, key: str, default: int) -> int:
        value = OutlookBridge._q(query, key)
        if value is None or str(value).strip() == "":
            return default
        try:
            return int(str(value))
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc

    @staticmethod
    def _bool_query(query: dict, key: str, default: bool) -> bool:
        value = OutlookBridge._q(query, key)
        if value is None or str(value).strip() == "":
            return default
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _required(payload: dict, key: str, allow_empty: bool = False) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or (not allow_empty and not value.strip()):
            raise ValueError(f"{key} is required")
        return value if allow_empty else value.strip()

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
    def _string_list(payload: dict, key: str) -> list[str]:
        value = payload.get(key)
        if value is None:
            return []
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{key} must be a list of strings")
        return value
