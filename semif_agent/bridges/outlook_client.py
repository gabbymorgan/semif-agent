"""A small, real Microsoft Graph client for Outlook over stdlib.

The Outlook bridge calls this instead of a skill body speaking Graph or OAuth
directly. It owns the connection (an Entra ID client id, tenant, and the stored
refresh token), the bearer-token lifecycle, Graph's paging/throttling, and the
JSON shapes, and returns plain dicts. Mail, calendar, To Do tasks, contacts, and
OneDrive files all go through the one Graph REST surface.

Authorization is the OAuth2 **device code flow** (a public client: no secret):
`run_device_code_flow` prints a code the user enters at a Microsoft URL, then
stores the resulting access + refresh tokens. The bridge starts without a token;
the first authenticated call raises `OutlookAuthRequired`, which the bridge maps
to a 503 telling the operator to run `scripts/outlook-auth.py`. Every other
failure is an `OutlookError` carrying the HTTP status when there was one, so the
bridge maps it to a 502 (a real Graph failure) rather than a fabricated success.

Only the stdlib is used: `urllib.request`, `json`, `ssl`, `os`, `time`,
`base64`, and `datetime`.
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

GRAPH = "https://graph.microsoft.com/v1.0"
LOGIN = "https://login.microsoftonline.com"
DEFAULT_TENANT = "common"
DEFAULT_TIMEOUT = 30.0
#: Delegated scopes the bridge needs. `offline_access` yields the refresh token.
DEFAULT_SCOPES = (
    "offline_access",
    "User.Read",
    "Mail.ReadWrite",
    "Mail.Send",
    "Calendars.ReadWrite",
    "Contacts.ReadWrite",
    "Tasks.ReadWrite",
    "Files.ReadWrite",
)
#: Refresh the access token this many seconds before Graph would reject it.
TOKEN_REFRESH_SKEW = 120.0
MAX_PAGES = 20
MAX_RETRIES = 3
AUTH_HINT = "run scripts/outlook-auth.py to authorize Outlook"


class OutlookError(Exception):
    """A real failure talking to Microsoft Graph (transport, HTTP, or a bad reply)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class OutlookAuthRequired(OutlookError):
    """No usable stored authorization; the operator must (re-)run the auth helper."""


# ---- small helpers ----------------------------------------------------------


def _ssl_context(verify_tls: bool):
    if verify_tls:
        return None
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _graph_error(detail: str) -> str:
    """Pull the human message out of a Graph error body, else the raw text."""
    try:
        data = json.loads(detail or "{}")
    except ValueError:
        return detail[:200]
    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or ""
        return str(message)[:200]
    if isinstance(error, str):
        return error[:200]
    return detail[:200]


def _retry_after(headers, attempt: int) -> float:
    """Seconds to wait before a throttled retry: Graph's Retry-After or backoff."""
    value = None
    if headers is not None:
        try:
            value = headers.get("Retry-After")
        except Exception:  # pragma: no cover - header objects vary
            value = None
    try:
        return max(0.0, float(str(value)))
    except (TypeError, ValueError):
        return float(2 ** attempt)


def normalize_scopes(scopes) -> tuple[str, ...]:
    """Accept a space string or a list; always a non-empty tuple."""
    if not scopes:
        return DEFAULT_SCOPES
    if isinstance(scopes, str):
        parts = scopes.replace(",", " ").split()
    else:
        parts = [str(part) for part in scopes]
    return tuple(parts) or DEFAULT_SCOPES


# ---- OAuth2 (device code, refresh) ------------------------------------------


def _oauth_post(url: str, fields: dict, timeout: float, verify_tls: bool) -> tuple[int, dict]:
    """POST an x-www-form-urlencoded form; return `(status, body)`.

    The Microsoft token endpoint reports pending/declined device codes as HTTP
    400 with a JSON `error`, so the body is parsed and returned rather than
    raised: the caller decides if it is retryable.
    """
    data = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=_ssl_context(verify_tls)
        ) as response:
            raw = response.read().decode("utf-8", "replace")
            return response.status, json.loads(raw or "{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw or "{}")
        except ValueError:
            body = {"error": "http_error", "error_description": raw[:300]}
        return exc.code, body
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise OutlookError(f"cannot reach Microsoft sign-in: {exc}") from exc


def request_device_code(
    client_id: str,
    tenant: str = DEFAULT_TENANT,
    scopes=DEFAULT_SCOPES,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    verify_tls: bool = True,
    login: str = LOGIN,
) -> dict:
    """Start a device-code flow: returns `user_code`, `verification_uri`, ..."""
    if not client_id:
        raise OutlookError("a Microsoft client id is required")
    url = f"{login}/{tenant or DEFAULT_TENANT}/oauth2/v2.0/devicecode"
    status, data = _oauth_post(
        url,
        {"client_id": client_id, "scope": " ".join(normalize_scopes(scopes))},
        timeout,
        verify_tls,
    )
    if status != 200 or "device_code" not in data:
        raise OutlookError(
            f"could not start Microsoft device authorization "
            f"({data.get('error', status)}): {data.get('error_description', '')}".strip()
        )
    return data


def refresh_access_token(
    client_id: str,
    tenant: str,
    refresh_token: str,
    scopes=DEFAULT_SCOPES,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    verify_tls: bool = True,
    login: str = LOGIN,
) -> tuple[int, dict]:
    """Exchange a refresh token for a new access token. Returns `(status, body)`."""
    url = f"{login}/{tenant or DEFAULT_TENANT}/oauth2/v2.0/token"
    return _oauth_post(
        url,
        {
            "client_id": client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": " ".join(normalize_scopes(scopes)),
        },
        timeout,
        verify_tls,
    )


def token_from_response(data: dict, *, tenant: str, client_id: str, fallback_refresh: str = "") -> dict:
    """Normalize a token-endpoint response into the stored token dict."""
    token = {
        "access_token": data.get("access_token", ""),
        "refresh_token": data.get("refresh_token") or fallback_refresh,
        "token_type": data.get("token_type", "Bearer"),
        "scope": data.get("scope", ""),
        "expires_at": time.time() + float(data.get("expires_in") or 3600),
        "tenant": tenant,
        "client_id": client_id,
    }
    return token


def poll_device_code(
    client_id: str,
    tenant: str,
    device_code: str,
    *,
    interval: float = 5.0,
    expires_in: float = 900.0,
    scopes=DEFAULT_SCOPES,
    timeout: float = DEFAULT_TIMEOUT,
    verify_tls: bool = True,
    sleep=time.sleep,
    login: str = LOGIN,
) -> dict:
    """Poll the token endpoint until the user completes the device-code flow.

    Handles the OAuth pending/slow-down/expiry error codes; any terminal error
    is a real `OutlookError`. Returns the normalized stored token dict.
    """
    url = f"{login}/{tenant or DEFAULT_TENANT}/oauth2/v2.0/token"
    deadline = time.time() + float(expires_in or 900.0)
    wait = max(1.0, float(interval or 5.0))
    while time.time() < deadline:
        sleep(wait)
        status, data = _oauth_post(
            url,
            {
                "client_id": client_id,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
                "scope": " ".join(normalize_scopes(scopes)),
            },
            timeout,
            verify_tls,
        )
        if status == 200 and "access_token" in data:
            return token_from_response(data, tenant=tenant, client_id=client_id)
        error = str(data.get("error") or "")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            wait += 5.0
            continue
        if error in ("expired_token", "bad_verification_code"):
            raise OutlookError("the Microsoft device code expired; start over")
        if error in ("authorization_declined", "access_denied"):
            raise OutlookError("the Microsoft authorization request was declined")
        raise OutlookError(
            f"Microsoft authorization failed ({error or status}): "
            f"{data.get('error_description', '')}".strip()
        )
    raise OutlookError("the Microsoft device code expired; start over")


def run_device_code_flow(
    client_id: str,
    tenant: str = DEFAULT_TENANT,
    scopes=DEFAULT_SCOPES,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    verify_tls: bool = True,
    out=print,
    sleep=time.sleep,
    login: str = LOGIN,
) -> dict:
    """Run the whole flow: start, show instructions, poll, return the token dict."""
    device = request_device_code(
        client_id, tenant, scopes, timeout=timeout, verify_tls=verify_tls, login=login
    )
    message = device.get("message") or (
        f"To authorize Outlook, visit {device.get('verification_uri')} and enter "
        f"the code {device.get('user_code')}."
    )
    out(message)
    return poll_device_code(
        client_id,
        tenant,
        device.get("device_code", ""),
        interval=float(device.get("interval") or 5.0),
        expires_in=float(device.get("expires_in") or 900.0),
        scopes=scopes,
        timeout=timeout,
        verify_tls=verify_tls,
        sleep=sleep,
        login=login,
    )


# ---- token storage ----------------------------------------------------------


def load_token(path: str) -> dict:
    """Read the stored token dict; raises `OutlookAuthRequired` when absent."""
    if not path or not os.path.exists(path):
        raise OutlookAuthRequired(f"Outlook is not authorized ({AUTH_HINT})")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise OutlookAuthRequired(f"the stored Outlook authorization is unreadable ({AUTH_HINT})") from exc
    if not isinstance(data, dict) or not data.get("refresh_token"):
        raise OutlookAuthRequired(f"the stored Outlook authorization is incomplete ({AUTH_HINT})")
    return data


def save_token(path: str, token: dict) -> None:
    """Write the token dict 0600 (it holds the refresh token, a real secret)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(token, handle, indent=2, sort_keys=True)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


# ---- Graph normalizers ------------------------------------------------------


def _address(recipient: dict) -> str:
    email = (recipient or {}).get("emailAddress") or {}
    name = str(email.get("name") or "").strip()
    address = str(email.get("address") or "").strip()
    if name and address:
        return f"{name} <{address}>"
    return address or name


def _addresses(recipients) -> list[str]:
    return [_address(item) for item in (recipients or []) if isinstance(item, dict)]


def _message(data: dict, full: bool = False) -> dict:
    result = {
        "id": data.get("id"),
        "subject": data.get("subject") or "",
        "from": _address(data.get("from") or {}),
        "to": _addresses(data.get("toRecipients")),
        "cc": _addresses(data.get("ccRecipients")),
        "received": data.get("receivedDateTime") or "",
        "sent": data.get("sentDateTime") or "",
        "is_read": bool(data.get("isRead")),
        "has_attachments": bool(data.get("hasAttachments")),
        "importance": data.get("importance") or "normal",
        "preview": data.get("bodyPreview") or "",
        "web_link": data.get("webLink") or "",
    }
    if full:
        body = data.get("body") or {}
        result["body"] = body.get("content") or ""
        result["body_type"] = body.get("contentType") or "text"
    return result


def _event(data: dict) -> dict:
    start = data.get("start") or {}
    end = data.get("end") or {}
    location = data.get("location") or {}
    return {
        "id": data.get("id"),
        "subject": data.get("subject") or "",
        "start": start.get("dateTime") or "",
        "end": end.get("dateTime") or "",
        "timezone": start.get("timeZone") or "",
        "all_day": bool(data.get("isAllDay")),
        "location": location.get("displayName") or "",
        "preview": data.get("bodyPreview") or "",
        "show_as": data.get("showAs") or "",
        "is_online_meeting": bool(data.get("isOnlineMeeting")),
        "web_link": data.get("webLink") or "",
        "organizer": _address(data.get("organizer") or {}),
    }


def _task(data: dict, list_id: str) -> dict:
    due = data.get("dueDateTime") or {}
    completed = data.get("completedDateTime") or {}
    body = data.get("body") or {}
    return {
        "id": data.get("id"),
        "title": data.get("title") or "",
        "status": data.get("status") or "notStarted",
        "importance": data.get("importance") or "normal",
        "due": due.get("dateTime") or "",
        "completed": completed.get("dateTime") or "",
        "description": body.get("content") or "",
        "list_id": list_id,
    }


def _contact(data: dict) -> dict:
    emails = [str(item.get("address") or "") for item in (data.get("emailAddresses") or [])]
    phones = list(data.get("businessPhones") or []) + list(data.get("homePhones") or [])
    mobile = data.get("mobilePhone")
    if mobile:
        phones.append(mobile)
    return {
        "id": data.get("id"),
        "display_name": data.get("displayName") or "",
        "given_name": data.get("givenName") or "",
        "surname": data.get("surname") or "",
        "emails": [email for email in emails if email],
        "phones": [phone for phone in phones if phone],
        "company": data.get("companyName") or "",
        "job_title": data.get("jobTitle") or "",
    }


def _file(data: dict) -> dict:
    folder = data.get("folder") or {}
    file_info = data.get("file") or {}
    parent = (data.get("parentReference") or {}).get("path") or ""
    return {
        "id": data.get("id"),
        "name": data.get("name") or "",
        "path": parent,
        "size": data.get("size"),
        "modified": data.get("lastModifiedDateTime") or "",
        "is_folder": bool(folder) or "childCount" in folder,
        "content_type": file_info.get("mimeType") or "",
        "web_url": data.get("webUrl") or "",
    }


def _graph_datetime(value, timezone_name: str) -> dict:
    """A Graph `dateTimeTimeZone` from an ISO string (offset-aware -> UTC)."""
    text = str(value or "").strip()
    if not text:
        return {"dateTime": text, "timeZone": timezone_name}
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return {"dateTime": text, "timeZone": timezone_name}
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
        return {"dateTime": moment.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": "UTC"}
    return {"dateTime": moment.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": timezone_name}


def _parse_naive(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def _quote_path(path: str) -> str:
    return urllib.parse.quote(str(path or "").strip("/"), safe="/")


_MAIL_WELL_KNOWN = {
    "inbox",
    "sentitems",
    "drafts",
    "deleteditems",
    "archive",
    "junkemail",
    "outbox",
    "conversationhistory",
}


class OutlookClient:
    """A configured connection to one Microsoft account via Graph."""

    def __init__(
        self,
        client_id: str,
        *,
        tenant: str = DEFAULT_TENANT,
        token_path: str = "",
        scopes=DEFAULT_SCOPES,
        timeout: float = DEFAULT_TIMEOUT,
        verify_tls: bool = True,
        default_calendar: str = "",
        default_task_list: str = "",
        default_mail_folder: str = "inbox",
        default_timezone: str = "UTC",
        graph: str = GRAPH,
        login: str = LOGIN,
    ):
        self.client_id = str(client_id or "").strip()
        self.tenant = str(tenant or DEFAULT_TENANT).strip() or DEFAULT_TENANT
        self.token_path = str(token_path or "")
        self.scopes = normalize_scopes(scopes)
        self.timeout = float(timeout or DEFAULT_TIMEOUT)
        self.verify_tls = bool(verify_tls)
        self.default_calendar = str(default_calendar or "").strip()
        self.default_task_list = str(default_task_list or "").strip()
        self.default_mail_folder = str(default_mail_folder or "inbox").strip() or "inbox"
        self.default_timezone = str(default_timezone or "UTC").strip() or "UTC"
        self.graph = str(graph or GRAPH).rstrip("/")
        self.login = str(login or LOGIN).rstrip("/")

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.token_path)

    # ---- token lifecycle ----

    def _access_token(self, force_refresh: bool = False) -> str:
        token = load_token(self.token_path)
        access = token.get("access_token")
        expires_at = float(token.get("expires_at") or 0)
        if access and not force_refresh and expires_at - time.time() > TOKEN_REFRESH_SKEW:
            return str(access)
        refresh = token.get("refresh_token")
        if not refresh:
            raise OutlookAuthRequired(f"the stored Outlook authorization has no refresh token ({AUTH_HINT})")
        status, data = refresh_access_token(
            self.client_id,
            self.tenant,
            refresh,
            self.scopes,
            timeout=self.timeout,
            verify_tls=self.verify_tls,
            login=self.login,
        )
        if status != 200 or "access_token" not in data:
            raise OutlookAuthRequired(
                f"could not refresh Microsoft authorization "
                f"({data.get('error', status)}); {AUTH_HINT}"
            )
        updated = token_from_response(
            data, tenant=self.tenant, client_id=self.client_id, fallback_refresh=refresh
        )
        save_token(self.token_path, updated)
        return str(updated["access_token"])

    # ---- transport ----

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        body=None,
        headers: dict | None = None,
        raw_body: bool = False,
        raw: bool = False,
    ):
        """One Graph call with bearer auth, a 401 refresh-retry, and 429 backoff."""
        url = path if str(path).startswith("http") else self.graph + path
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url += ("&" if "?" in url else "?") + urllib.parse.urlencode(clean)
        attempt = 0
        refreshed = False
        while True:
            token = self._access_token(force_refresh=refreshed)
            request_headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            data = None
            if body is not None:
                if raw_body:
                    data = body if isinstance(body, (bytes, bytearray)) else str(body).encode("utf-8")
                else:
                    data = json.dumps(body).encode("utf-8")
                    request_headers["Content-Type"] = "application/json"
            if headers:
                request_headers.update(headers)
            request = urllib.request.Request(url, data=data, method=method, headers=request_headers)
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout, context=_ssl_context(self.verify_tls)
                ) as response:
                    payload = response.read()
                    if raw:
                        return payload, response.headers
                    return json.loads(payload.decode("utf-8") or "{}"), response.headers
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace").strip()
                if exc.code == 401 and not refreshed:
                    refreshed = True
                    continue
                if exc.code in (429, 503) and attempt < MAX_RETRIES:
                    time.sleep(_retry_after(exc.headers, attempt))
                    attempt += 1
                    continue
                message = f"{method} {path}: HTTP {exc.code}"
                detail_text = _graph_error(detail)
                if detail_text:
                    message += f" {detail_text}"
                raise OutlookError(message, status=exc.code) from exc
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                raise OutlookError(f"{method} {path}: {exc}") from exc

    def _paged(self, path: str, params: dict | None = None, key: str = "value", limit: int = MAX_PAGES) -> list:
        items: list = []
        data, _headers = self._request("GET", path, params=params)
        for _ in range(max(1, limit)):
            if isinstance(data, dict):
                values = data.get(key)
                if isinstance(values, list):
                    items.extend(values)
                next_link = data.get("@odata.nextLink")
            else:
                next_link = None
            if not next_link:
                break
            data, _headers = self._request("GET", next_link)
        return items

    # ---- account ----

    def me(self) -> dict:
        data, _ = self._request(
            "GET", "/me", params={"$select": "id,displayName,mail,userPrincipalName"}
        )
        return {
            "id": data.get("id"),
            "display_name": data.get("displayName") or "",
            "mail": data.get("mail") or data.get("userPrincipalName") or "",
            "user_principal_name": data.get("userPrincipalName") or "",
        }

    # ---- mail ----

    def list_mail_folders(self) -> list[dict]:
        items = self._paged(
            "/me/mailFolders",
            {"$select": "id,displayName,totalItemCount,unreadItemCount", "$top": 100},
        )
        return [
            {
                "id": item.get("id"),
                "name": item.get("displayName") or "",
                "total": item.get("totalItemCount"),
                "unread": item.get("unreadItemCount"),
            }
            for item in items
        ]

    def _resolve_mail_folder(self, folder) -> str:
        name = str(folder or self.default_mail_folder or "inbox").strip()
        if not name:
            name = "inbox"
        if name.lower() in _MAIL_WELL_KNOWN:
            return name.lower()
        for candidate in self.list_mail_folders():
            if name.lower() in (str(candidate["id"]).lower(), str(candidate["name"]).lower()):
                return str(candidate["id"])
        raise OutlookError(f"mail folder {name!r} not found")

    def list_messages(self, folder="", count=25, search="", unread_only=False) -> list[dict]:
        params = {
            "$top": max(1, min(int(count or 25), 100)),
            "$orderby": "receivedDateTime desc",
            "$select": (
                "id,subject,from,toRecipients,ccRecipients,receivedDateTime,isRead,"
                "hasAttachments,bodyPreview,importance,webLink"
            ),
        }
        if search:
            params["$search"] = '"' + str(search).replace('"', "") + '"'
        elif unread_only:
            params["$filter"] = "isRead eq false"
        folder_id = self._resolve_mail_folder(folder)
        items = self._paged(
            f"/me/mailFolders/{urllib.parse.quote(folder_id)}/messages", params, limit=1
        )
        return [_message(item) for item in items]

    def get_message(self, message_id: str) -> dict:
        data, _ = self._request(
            "GET",
            f"/me/messages/{urllib.parse.quote(message_id)}",
            params={
                "$select": (
                    "id,subject,from,toRecipients,ccRecipients,receivedDateTime,"
                    "sentDateTime,isRead,hasAttachments,body,bodyPreview,importance,webLink"
                )
            },
        )
        return _message(data, full=True)

    def send_mail(self, to, subject, body, cc=(), content_type="text", save_to_sent=True) -> dict:
        recipients = [str(item) for item in (to or []) if str(item).strip()]
        if not recipients:
            raise ValueError("to is required")
        message = {
            "subject": str(subject or ""),
            "body": {
                "contentType": "HTML" if str(content_type).lower() == "html" else "Text",
                "content": str(body or ""),
            },
            "toRecipients": [{"emailAddress": {"address": address}} for address in recipients],
            "ccRecipients": [
                {"emailAddress": {"address": str(item)}}
                for item in (cc or [])
                if str(item).strip()
            ],
        }
        self._request(
            "POST", "/me/sendMail", body={"message": message, "saveToSentItems": bool(save_to_sent)}
        )
        return {"ok": True, "to": recipients, "cc": list(cc or []), "subject": message["subject"]}

    def create_draft(self, to, subject, body, cc=(), content_type="text") -> dict:
        message = {
            "subject": str(subject or ""),
            "body": {
                "contentType": "HTML" if str(content_type).lower() == "html" else "Text",
                "content": str(body or ""),
            },
            "toRecipients": [
                {"emailAddress": {"address": str(item)}}
                for item in (to or [])
                if str(item).strip()
            ],
            "ccRecipients": [
                {"emailAddress": {"address": str(item)}}
                for item in (cc or [])
                if str(item).strip()
            ],
        }
        data, _ = self._request("POST", "/me/messages", body=message)
        return {"ok": True, "id": data.get("id"), "subject": message["subject"]}

    # ---- calendar ----

    def list_calendars(self) -> list[dict]:
        items = self._paged(
            "/me/calendars",
            {"$select": "id,name,color,isDefaultCalendar,canEdit", "$top": 100},
        )
        return [
            {
                "id": item.get("id"),
                "name": item.get("name") or "",
                "color": item.get("color") or "",
                "is_default": bool(item.get("isDefaultCalendar")),
                "can_edit": bool(item.get("canEdit")),
            }
            for item in items
        ]

    def _resolve_calendar(self, calendar) -> str:
        requested = str(calendar or self.default_calendar or "").strip()
        if not requested:
            return ""
        lowered = requested.lower()
        for candidate in self.list_calendars():
            if lowered in (str(candidate["id"]).lower(), str(candidate["name"]).lower()):
                return str(candidate["id"])
        # Not a known name; let Graph reject an id it does not own (mapped to 502).
        return requested

    def _calendar_view_path(self, calendar_id: str) -> str:
        if calendar_id:
            return f"/me/calendars/{urllib.parse.quote(calendar_id)}/calendarView"
        return "/me/calendarView"

    def list_events(self, calendar="", start="", end="") -> list[dict]:
        calendar_id = self._resolve_calendar(calendar)
        now = datetime.now(timezone.utc)
        window_start = start or now.strftime("%Y-%m-%dT%H:%M:%S")
        window_end = end or (now + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
        params = {
            "startDateTime": window_start,
            "endDateTime": window_end,
            "$select": (
                "id,subject,start,end,isAllDay,location,bodyPreview,showAs,"
                "isOnlineMeeting,webLink,organizer"
            ),
            "$orderby": "start/dateTime",
            "$top": 50,
        }
        items = self._paged(self._calendar_view_path(calendar_id), params, limit=4)
        return [_event(item) for item in items]

    def create_event(
        self,
        calendar="",
        subject="",
        start="",
        end="",
        all_day=False,
        location="",
        description="",
        attendees=(),
        timezone_name="",
    ) -> dict:
        calendar_id = self._resolve_calendar(calendar)
        tz = str(timezone_name or self.default_timezone or "UTC").strip() or "UTC"
        if not end:
            parsed = _parse_naive(start)
            if parsed is not None:
                delta = timedelta(days=1) if all_day else timedelta(hours=1)
                end = (parsed + delta).strftime("%Y-%m-%dT%H:%M:%S")
        event = {
            "subject": str(subject or ""),
            "start": _graph_datetime(start, tz),
            "end": _graph_datetime(end or start, tz),
            "isAllDay": bool(all_day),
        }
        if location:
            event["location"] = {"displayName": str(location)}
        if description:
            event["body"] = {"contentType": "Text", "content": str(description)}
        attendee_list = [str(item) for item in (attendees or []) if str(item).strip()]
        if attendee_list:
            event["attendees"] = [
                {"emailAddress": {"address": address}, "type": "required"}
                for address in attendee_list
            ]
        path = (
            f"/me/calendars/{urllib.parse.quote(calendar_id)}/events"
            if calendar_id
            else "/me/events"
        )
        data, _ = self._request("POST", path, body=event)
        return {
            "ok": True,
            "id": data.get("id"),
            "subject": data.get("subject") or event["subject"],
            "calendar": calendar_id,
            "web_link": data.get("webLink") or "",
        }

    def update_event(self, event_id: str, *, timezone_name: str = "", **fields) -> dict:
        patch: dict = {}
        if "subject" in fields and fields["subject"] is not None:
            patch["subject"] = str(fields["subject"])
        tz = str(timezone_name or self.default_timezone or "UTC").strip() or "UTC"
        if fields.get("start"):
            patch["start"] = _graph_datetime(fields["start"], tz)
        if fields.get("end"):
            patch["end"] = _graph_datetime(fields["end"], tz)
        if "all_day" in fields and fields["all_day"] is not None:
            patch["isAllDay"] = bool(fields["all_day"])
        if fields.get("location") is not None:
            patch["location"] = {"displayName": str(fields.get("location") or "")}
        if fields.get("description") is not None:
            patch["body"] = {"contentType": "Text", "content": str(fields.get("description") or "")}
        data, _ = self._request(
            "PATCH", f"/me/events/{urllib.parse.quote(event_id)}", body=patch
        )
        return {"ok": True, "id": data.get("id") or event_id, "subject": data.get("subject") or ""}

    def delete_event(self, event_id: str) -> dict:
        self._request("DELETE", f"/me/events/{urllib.parse.quote(event_id)}")
        return {"ok": True, "id": event_id}

    # ---- To Do tasks ----

    def list_task_lists(self) -> list[dict]:
        # No $select: Graph's To Do endpoints reject OData $select for personal
        # (consumer) Microsoft accounts with RequestBroker--ParseUri, while $top
        # is accepted. Fetch the full shape and read the fields we need.
        items = self._paged("/me/todo/lists", {"$top": 100})
        return [
            {
                "id": item.get("id"),
                "name": item.get("displayName") or "",
                "is_owner": bool(item.get("isOwner")),
                "is_default": str(item.get("wellknownListName") or "") == "defaultList",
            }
            for item in items
        ]

    def _resolve_task_list(self, task_list) -> str:
        requested = str(task_list or self.default_task_list or "").strip()
        lists = self.list_task_lists()
        if requested:
            lowered = requested.lower()
            for candidate in lists:
                if lowered in (str(candidate["id"]).lower(), str(candidate["name"]).lower()):
                    return str(candidate["id"])
            return requested
        for candidate in lists:
            if candidate.get("is_default"):
                return str(candidate["id"])
        if lists:
            return str(lists[0]["id"])
        raise OutlookError("no Microsoft To Do task lists are available")

    def list_tasks(self, task_list="") -> list[dict]:
        list_id = self._resolve_task_list(task_list)
        # No $select here either: see list_task_lists (consumer To Do quirk).
        items = self._paged(
            f"/me/todo/lists/{urllib.parse.quote(list_id)}/tasks", {"$top": 100}
        )
        return [_task(item, list_id) for item in items]

    def create_task(self, task_list="", title="", due="", importance="", description="") -> dict:
        list_id = self._resolve_task_list(task_list)
        task: dict = {"title": str(title or "")}
        if due:
            task["dueDateTime"] = _graph_datetime(due, self.default_timezone)
        if importance:
            task["importance"] = str(importance)
        if description:
            task["body"] = {"contentType": "Text", "content": str(description)}
        data, _ = self._request(
            "POST", f"/me/todo/lists/{urllib.parse.quote(list_id)}/tasks", body=task
        )
        return {"ok": True, "id": data.get("id"), "title": data.get("title") or task["title"], "list_id": list_id}

    def update_task(self, task_id: str, *, task_list="", **fields) -> dict:
        list_id = self._resolve_task_list(task_list)
        patch: dict = {}
        for key in ("title", "importance", "status"):
            if fields.get(key) is not None:
                patch[key] = str(fields[key])
        if fields.get("due"):
            patch["dueDateTime"] = _graph_datetime(fields["due"], self.default_timezone)
        if fields.get("description") is not None:
            patch["body"] = {"contentType": "Text", "content": str(fields.get("description") or "")}
        data, _ = self._request(
            "PATCH",
            f"/me/todo/lists/{urllib.parse.quote(list_id)}/tasks/{urllib.parse.quote(task_id)}",
            body=patch,
        )
        return {"ok": True, "id": data.get("id") or task_id, "status": data.get("status") or ""}

    def complete_task(self, task_id: str, task_list="") -> dict:
        return self.update_task(task_id, task_list=task_list, status="completed")

    def delete_task(self, task_id: str, task_list="") -> dict:
        list_id = self._resolve_task_list(task_list)
        self._request(
            "DELETE",
            f"/me/todo/lists/{urllib.parse.quote(list_id)}/tasks/{urllib.parse.quote(task_id)}",
        )
        return {"ok": True, "id": task_id}

    # ---- contacts ----

    def list_contacts(self, query="", count=50) -> list[dict]:
        params = {
            "$select": (
                "id,displayName,givenName,surname,emailAddresses,businessPhones,"
                "homePhones,mobilePhone,companyName,jobTitle"
            ),
            "$top": max(1, min(int(count or 50), 100)),
        }
        if query:
            params["$search"] = '"' + str(query).replace('"', "") + '"'
        items = self._paged("/me/contacts", params, limit=1)
        return [_contact(item) for item in items]

    # ---- OneDrive files ----

    def list_files(self, path="/") -> list[dict]:
        clean = str(path or "/").strip("/")
        if clean:
            endpoint = f"/me/drive/root:/{_quote_path(clean)}:/children"
        else:
            endpoint = "/me/drive/root/children"
        items = self._paged(
            endpoint,
            {
                "$select": (
                    "id,name,size,lastModifiedDateTime,folder,file,webUrl,parentReference"
                ),
                "$top": 200,
            },
            limit=2,
        )
        return [_file(item) for item in items]

    def read_file(self, path: str) -> dict:
        clean = str(path or "").strip("/")
        if not clean:
            raise ValueError("path is required")
        meta, _ = self._request(
            "GET",
            f"/me/drive/root:/{_quote_path(clean)}:",
            params={"$select": "id,name,size,file,lastModifiedDateTime,webUrl"},
        )
        payload, _ = self._request(
            "GET", f"/me/drive/root:/{_quote_path(clean)}:/content", raw=True
        )
        content_type = ((meta.get("file") or {}).get("mimeType")) or ""
        try:
            text = payload.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            text = base64.b64encode(payload).decode("ascii")
            encoding = "base64"
        return {
            "path": clean,
            "name": meta.get("name") or clean.rsplit("/", 1)[-1],
            "content": text,
            "encoding": encoding,
            "size": meta.get("size"),
            "content_type": content_type,
            "modified": meta.get("lastModifiedDateTime") or "",
        }

    def write_file(self, path: str, content: str, encoding: str = "utf-8", overwrite: bool = True) -> dict:
        clean = str(path or "").strip("/")
        if not clean:
            raise ValueError("path is required")
        if str(encoding).lower() == "base64":
            payload = base64.b64decode(content or "")
        else:
            payload = str(content or "").encode("utf-8")
        headers = {"Content-Type": "application/octet-stream"}
        if not overwrite:
            headers["If-None-Match"] = "*"
        data, _ = self._request(
            "PUT",
            f"/me/drive/root:/{_quote_path(clean)}:/content",
            body=payload,
            raw_body=True,
            headers=headers,
        )
        return {
            "ok": True,
            "path": clean,
            "id": data.get("id"),
            "size": data.get("size"),
            "web_url": data.get("webUrl") or "",
        }
