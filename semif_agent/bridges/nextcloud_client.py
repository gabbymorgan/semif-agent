"""A small, real Nextcloud client: WebDAV, CalDAV, CardDAV, and OCS over stdlib.

The Nextcloud bridge calls this instead of a skill body speaking the DAV/OCS
protocols directly. It owns the connection (base URL, username, app password),
the HTTP Basic auth, the XML request bodies, and the iCalendar/vCard parsing, and
returns plain dicts. Every failure is a `NextcloudError` carrying the HTTP status
when there was one, so the bridge can map it to a 502 (a real Nextcloud failure)
rather than a fabricated success.

Only the stdlib is used: `urllib.request` (arbitrary DAV methods), `xml.etree`,
`ssl`, `base64`, and `json`. No third-party DAV/iCalendar library.
"""

from __future__ import annotations

import base64
import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from xml.sax.saxutils import escape as _xml_escape

DAV = "DAV:"
CALDAV = "urn:ietf:params:xml:ns:caldav"
CARDDAV = "urn:ietf:params:xml:ns:carddav"
OC = "http://owncloud.org/ns"
NC = "http://nextcloud.org/ns"

DEFAULT_TIMEOUT = 30.0


class NextcloudError(Exception):
    """A real failure talking to Nextcloud (transport, HTTP, or a bad reply)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


# ---- small helpers ----------------------------------------------------------


def _text(element) -> str:
    if element is None:
        return ""
    return (element.text or "").strip()


def _int(value, default=None):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _norm_path(path) -> str:
    """A user-facing DAV path: always leading `/`, no trailing slash (root `/`)."""
    text = "/" + str(path or "").strip("/")
    return "/" if text == "/" else text


def _quote_path(path) -> str:
    segments = [s for s in str(path or "").split("/") if s]
    if not segments:
        return "/"
    return "/" + "/".join(urllib.parse.quote(segment, safe="") for segment in segments)


def _unescape_text(value: str) -> str:
    """Undo iCalendar/vCard backslash escapes (`\\n`, `\\,`, `\\;`, `\\\\`)."""
    out: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\" and index + 1 < len(value):
            nxt = value[index + 1]
            out.append({"n": "\n", "N": "\n", ",": ",", ";": ";", "\\": "\\"}.get(nxt, nxt))
            index += 2
        else:
            out.append(char)
            index += 1
    return "".join(out)


def _escape_text(value) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )


def _unfold(text: str) -> list[str]:
    """Split into unfolded logical lines (RFC 5545/6350 line folding)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    for line in lines:
        if line[:1] in (" ", "\t") and out:
            out[-1] += line[1:]
        else:
            out.append(line)
    return out


def _parse_prop(line: str) -> tuple[str, dict, str]:
    """`NAME;PARAM=VAL:VALUE` -> (NAME, params, value)."""
    name, _, value = line.partition(":")
    params: dict[str, str] = {}
    if ";" in name:
        head, *rest = name.split(";")
        name = head
        for chunk in rest:
            key, _, val = chunk.partition("=")
            params[key.strip().upper()] = val.strip().strip('"')
    return name.strip().upper(), params, value


def _prop_name(line: str) -> str:
    return line.split(":", 1)[0].split(";", 1)[0].strip().upper()


def parse_components(text: str, component: str) -> list[dict]:
    """All top-level `component` blocks in an iCalendar stream, as prop dicts.

    Each block maps property NAME -> list of `(params, value)` so repeated
    properties (ATTENDEE, ...) survive. Nested components (VALARM) contribute
    their own property lines, which is harmless: their names never collide with
    the ones the bridge reads.
    """
    component = component.upper()
    blocks: list[dict] = []
    current: list[str] | None = None
    for line in _unfold(text):
        if line.startswith("BEGIN:"):
            if line[6:].strip().upper() == component:
                current = []
            elif current is not None:
                current.append(line)
        elif line.startswith("END:"):
            if line[4:].strip().upper() == component and current is not None:
                blocks.append(_props(current))
                current = None
        elif current is not None:
            current.append(line)
    return blocks


def _props(lines: list[str]) -> dict:
    props: dict[str, list[tuple[dict, str]]] = {}
    for line in lines:
        if not line.strip():
            continue
        name, params, value = _parse_prop(line)
        props.setdefault(name, []).append((params, value))
    return props


def _first(props: dict, name: str):
    entries = props.get(name.upper())
    return entries[0] if entries else None


def _value(props: dict, name: str) -> str:
    entry = _first(props, name)
    return _unescape_text(entry[1]) if entry is not None else ""


def _dt_value(params: dict, value: str) -> dict:
    value = (value or "").strip()
    if params.get("VALUE", "").upper() == "DATE" or (len(value) == 8 and value.isdigit()):
        return {
            "value": f"{value[0:4]}-{value[4:6]}-{value[6:8]}",
            "all_day": True,
        }
    match = re.match(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(Z?)", value)
    if not match:
        return {"value": value, "all_day": False}
    iso = (
        f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        f"T{match.group(4)}:{match.group(5)}:{match.group(6)}"
    )
    if match.group(7) == "Z":
        iso += "+00:00"
    return {"value": iso, "all_day": False}


def _parse_dt(value) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _parse_date(value) -> date:
    return date.fromisoformat(str(value or "").strip()[:10])


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _dt_lines(name: str, value, all_day: bool) -> list[str]:
    if all_day:
        return [f"{name};VALUE=DATE:{_parse_date(value).strftime('%Y%m%d')}"]
    return [f"{name}:{_stamp(_parse_dt(value))}"]


# ---- builders ---------------------------------------------------------------


def build_event_ics(
    uid: str,
    summary: str,
    start: str,
    end: str = "",
    all_day: bool = False,
    location: str = "",
    description: str = "",
    rrule: str = "",
    dtstamp: str | None = None,
) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//semif-agent//nextcloud bridge//EN",
        "CALSCALE:GREGORIAN",
        "BEGIN:VEVENT",
        f"UID:{_escape_text(uid)}",
        f"DTSTAMP:{dtstamp or _now_stamp()}",
    ]
    lines += _dt_lines("DTSTART", start, all_day)
    if end:
        lines += _dt_lines("DTEND", end, all_day)
    lines.append(f"SUMMARY:{_escape_text(summary)}")
    if description:
        lines.append(f"DESCRIPTION:{_escape_text(description)}")
    if location:
        lines.append(f"LOCATION:{_escape_text(location)}")
    if rrule:
        lines.append(f"RRULE:{rrule}")
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(lines) + "\r\n"


def build_todo_ics(
    uid: str,
    summary: str,
    due: str = "",
    priority: int | None = None,
    description: str = "",
    status: str = "NEEDS-ACTION",
    percent_complete: int | None = None,
    completed: str = "",
    dtstamp: str | None = None,
) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//semif-agent//nextcloud bridge//EN",
        "CALSCALE:GREGORIAN",
        "BEGIN:VTODO",
        f"UID:{_escape_text(uid)}",
        f"DTSTAMP:{dtstamp or _now_stamp()}",
        f"SUMMARY:{_escape_text(summary)}",
        f"STATUS:{status}",
    ]
    if due:
        lines += _dt_lines("DUE", due, False)
    if priority is not None:
        lines.append(f"PRIORITY:{int(priority)}")
    if percent_complete is not None:
        lines.append(f"PERCENT-COMPLETE:{int(percent_complete)}")
    if completed:
        lines.append(f"COMPLETED:{_stamp(_parse_dt(completed))}")
    if description:
        lines.append(f"DESCRIPTION:{_escape_text(description)}")
    lines += ["END:VTODO", "END:VCALENDAR"]
    return "\r\n".join(lines) + "\r\n"


def build_vcard(
    uid: str,
    fn: str,
    n: str = "",
    emails=(),
    phones=(),
    org: str = "",
    title: str = "",
    note: str = "",
    url: str = "",
) -> str:
    lines = [
        "BEGIN:VCARD",
        "VERSION:3.0",
        f"UID:{_escape_text(uid)}",
        f"FN:{_escape_text(fn)}",
    ]
    if n:
        lines.append(f"N:{_escape_text(n)}")
    for email in emails:
        lines.append(f"EMAIL;TYPE=INTERNET:{_escape_text(email)}")
    for phone in phones:
        lines.append(f"TEL:{_escape_text(phone)}")
    if org:
        lines.append(f"ORG:{_escape_text(org)}")
    if title:
        lines.append(f"TITLE:{_escape_text(title)}")
    if note:
        lines.append(f"NOTE:{_escape_text(note)}")
    if url:
        lines.append(f"URL:{_escape_text(url)}")
    lines.append("END:VCARD")
    return "\r\n".join(lines) + "\r\n"


def parse_vcard(text: str) -> dict:
    props = _props(_unfold(text))
    return {
        "uid": _value(props, "UID"),
        "fn": _value(props, "FN"),
        "n": _value(props, "N"),
        "emails": [_unescape_text(v) for _, v in props.get("EMAIL", [])],
        "phones": [_unescape_text(v) for _, v in props.get("TEL", [])],
        "org": _value(props, "ORG"),
        "title": _value(props, "TITLE"),
        "note": _value(props, "NOTE"),
        "url": _value(props, "URL"),
    }


def _event_dict(props: dict) -> dict:
    start = _first(props, "DTSTART")
    end = _first(props, "DTEND")
    start_info = _dt_value(*start) if start else {"value": "", "all_day": False}
    end_info = _dt_value(*end) if end else {"value": "", "all_day": False}
    return {
        "uid": _value(props, "UID"),
        "summary": _value(props, "SUMMARY"),
        "description": _value(props, "DESCRIPTION"),
        "location": _value(props, "LOCATION"),
        "start": start_info["value"],
        "end": end_info["value"],
        "all_day": bool(start_info["all_day"]),
        "rrule": _value(props, "RRULE"),
        "status": _value(props, "STATUS"),
    }


def _todo_dict(props: dict) -> dict:
    due = _first(props, "DUE")
    due_info = _dt_value(*due) if due else {"value": "", "all_day": False}
    return {
        "uid": _value(props, "UID"),
        "summary": _value(props, "SUMMARY"),
        "description": _value(props, "DESCRIPTION"),
        "due": due_info["value"],
        "status": _value(props, "STATUS"),
        "percent_complete": _int(_value(props, "PERCENT-COMPLETE")),
        "priority": _int(_value(props, "PRIORITY")),
    }


def _extract_block(lines: list[str], component: str):
    """Return `(start, end, block_lines)` for the first component, or None."""
    component = component.upper()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("BEGIN:") and line[6:].strip().upper() == component:
            start = index
            break
    if start is None:
        return None
    depth = 1
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("BEGIN:"):
            depth += 1
        elif lines[index].startswith("END:"):
            depth -= 1
            if depth == 0:
                return start, index, lines[start : index + 1]
    return None


def _replace_props(block: list[str], updates: dict) -> list[str]:
    """Replace (or remove) properties in a component block, preserving the rest."""
    wanted = {name.upper(): lines for name, lines in updates.items()}
    kept = [line for line in block if _prop_name(line) not in wanted]
    insert_at = len(kept) - 1 if kept and kept[-1].startswith("END:") else len(kept)
    inserted: list[str] = []
    for name, lines in wanted.items():
        if lines:
            inserted.extend(lines)
    kept[insert_at:insert_at] = inserted
    return kept


def _rebuild_component(original: list[str], component: str, block: list[str]) -> str:
    found = _extract_block(original, component)
    if found is None:
        return "\r\n".join(block) + "\r\n"
    start, end, _ = found
    merged = original[:start] + block + original[end + 1 :]
    return "\r\n".join(merged) + "\r\n"


def _multistatus(data: bytes):
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise NextcloudError(f"unparseable DAV response: {exc}") from exc
    if root.tag != f"{{{DAV}}}multistatus":
        raise NextcloudError(f"unexpected DAV response root {root.tag!r}")
    return root


def _response_prop(response):
    for propstat in response.findall(f"{{{DAV}}}propstat"):
        status = _text(propstat.find(f"{{{DAV}}}status"))
        if " 200 " in status:
            return propstat.find(f"{{{DAV}}}prop")
    propstat = response.find(f"{{{DAV}}}propstat")
    return propstat.find(f"{{{DAV}}}prop") if propstat is not None else None


def _dav_href_to_path(href: str, base: str) -> str:
    path = urllib.parse.unquote(urllib.parse.urlparse(href or "").path)
    if base and path.startswith(base):
        path = path[len(base) :]
    return _norm_path(path)


# ---- request bodies ---------------------------------------------------------

_FILE_PROPFIND = (
    '<?xml version="1.0"?>'
    '<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns" '
    'xmlns:nc="http://nextcloud.org/ns">'
    "<d:prop><d:resourcetype/><d:getcontentlength/><d:getlastmodified/>"
    "<d:getetag/><d:getcontenttype/><d:displayname/></d:prop></d:propfind>"
).encode("utf-8")

_CALENDAR_PROPFIND = (
    '<?xml version="1.0"?>'
    '<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav" '
    'xmlns:oc="http://owncloud.org/ns" xmlns:nc="http://nextcloud.org/ns">'
    "<d:prop><d:resourcetype/><d:displayname/><c:calendar-description/>"
    "<c:getctag/><oc:getctag/><nc:calendar-color/><oc:calendar-color/></d:prop>"
    "</d:propfind>"
).encode("utf-8")

_ADDRESSBOOK_PROPFIND = (
    '<?xml version="1.0"?>'
    '<d:propfind xmlns:d="DAV:" xmlns:card="urn:ietf:params:xml:ns:carddav">'
    "<d:prop><d:resourcetype/><d:displayname/>"
    "<card:addressbook-description/></d:prop></d:propfind>"
).encode("utf-8")

_ADDRESSBOOK_QUERY = (
    '<?xml version="1.0"?>'
    '<card:addressbook-query xmlns:d="DAV:" '
    'xmlns:card="urn:ietf:params:xml:ns:carddav">'
    "<d:prop><d:getetag/><card:address-data/></d:prop><card:filter/>"
    "</card:addressbook-query>"
).encode("utf-8")


def _calendar_query_body(component: str, start: str = "", end: str = "") -> bytes:
    time_range = ""
    if start or end:
        attrs = ""
        if start:
            attrs += f' start="{_stamp(_parse_dt(start))}"'
        if end:
            attrs += f' end="{_stamp(_parse_dt(end))}"'
        time_range = f"<c:time-range{attrs}/>"
    return (
        '<?xml version="1.0"?>'
        '<c:calendar-query xmlns:d="DAV:" '
        'xmlns:c="urn:ietf:params:xml:ns:caldav">'
        "<d:prop><d:getetag/><c:calendar-data/></d:prop>"
        "<c:filter><c:comp-filter name=\"VCALENDAR\">"
        f'<c:comp-filter name="{component}">{time_range}</c:comp-filter>'
        "</c:comp-filter></c:filter></c:calendar-query>"
    ).encode("utf-8")


def _search_body(user: str, path: str, query: str) -> bytes:
    scope = f"/files/{urllib.parse.quote(user)}{_quote_path(path)}"
    return (
        '<?xml version="1.0"?>'
        '<d:searchrequest xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns" '
        'xmlns:nc="http://nextcloud.org/ns"><d:basicsearch>'
        "<d:select><d:prop><d:displayname/><d:getcontentlength/>"
        "<d:getlastmodified/><d:getetag/><d:getcontenttype/>"
        "<d:resourcetype/></d:prop></d:select>"
        "<d:from><d:scope>"
        f"<d:href>{_xml_escape(scope)}</d:href><d:depth>infinity</d:depth>"
        "</d:scope></d:from>"
        "<d:where><d:like><d:prop><d:displayname/></d:prop>"
        f"<d:literal>%{_xml_escape(query)}%</d:literal></d:like></d:where>"
        "<d:orderby/></d:basicsearch></d:searchrequest>"
    ).encode("utf-8")


class NextcloudClient:
    """A configured connection to one Nextcloud account."""

    def __init__(
        self,
        url: str,
        username: str,
        app_password: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        verify_tls: bool = True,
        default_calendar: str = "",
        default_addressbook: str = "",
    ):
        self.url = str(url or "").strip().rstrip("/")
        self.username = str(username or "").strip()
        self.app_password = str(app_password or "")
        self.timeout = float(timeout or DEFAULT_TIMEOUT)
        self.verify_tls = bool(verify_tls)
        self.default_calendar = str(default_calendar or "").strip()
        self.default_addressbook = str(default_addressbook or "").strip()

    @property
    def configured(self) -> bool:
        return bool(self.url and self.username and self.app_password)

    # ---- transport ----

    def _auth_header(self) -> str:
        token = base64.b64encode(
            f"{self.username}:{self.app_password}".encode("utf-8")
        ).decode("ascii")
        return f"Basic {token}"

    def _url(self, path: str) -> str:
        if not self.url:
            raise NextcloudError("the Nextcloud base URL is not configured")
        return self.url + path

    def _ssl_context(self):
        if self.verify_tls:
            return None
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict | None = None,
        depth: str | None = None,
        content_type: str | None = None,
    ):
        request_headers = {"Authorization": self._auth_header()}
        if depth is not None:
            request_headers["Depth"] = depth
        if content_type:
            request_headers["Content-Type"] = content_type
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            self._url(path), data=body, method=method, headers=request_headers
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=self._ssl_context()
            ) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace").strip()[:200]
            except Exception:
                detail = ""
            message = f"{method} {path}: HTTP {exc.code}"
            if detail:
                message += f" {detail}"
            raise NextcloudError(message, status=exc.code) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise NextcloudError(f"{method} {path}: {exc}") from exc

    def _ocs(self, method: str, path: str, *, params: dict | None = None, payload: dict | None = None):
        url = self._url(path)
        params = dict(params or {})
        params.setdefault("format", "json")
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        headers = {
            "Authorization": self._auth_header(),
            "OCS-APIRequest": "true",
            "Accept": "application/json",
        }
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=self._ssl_context()
            ) as response:
                data = json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace").strip()[:200]
            except Exception:
                detail = ""
            raise NextcloudError(
                f"{method} {path}: HTTP {exc.code} {detail}".strip(), status=exc.code
            ) from exc
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            raise NextcloudError(f"{method} {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise NextcloudError(f"{method} {path}: unexpected OCS reply")
        ocs = data.get("ocs")
        if isinstance(ocs, dict):
            meta = ocs.get("meta") or {}
            statuscode = _int(meta.get("statuscode"))
            if statuscode not in (None, 100, 200):
                raise NextcloudError(
                    f"OCS {path}: {meta.get('statuscode')} {meta.get('message')}".strip(),
                    status=statuscode if statuscode and statuscode >= 400 else None,
                )
            return ocs.get("data")
        return data

    # ---- paths ----

    def _files_base(self) -> str:
        return f"/remote.php/dav/files/{urllib.parse.quote(self.username)}"

    def _files_path(self, path: str) -> str:
        return self._files_base() + _quote_path(path)

    def _calendars_base(self) -> str:
        return f"/remote.php/dav/calendars/{urllib.parse.quote(self.username)}"

    def _calendar_path(self, calendar: str) -> str:
        return f"{self._calendars_base()}/{urllib.parse.quote(calendar)}/"

    def _calendar_object_path(self, calendar: str, uid: str) -> str:
        return f"{self._calendar_path(calendar)}{urllib.parse.quote(uid, safe='')}.ics"

    def _addressbooks_base(self) -> str:
        return f"/remote.php/dav/addressbooks/users/{urllib.parse.quote(self.username)}"

    def _addressbook_path(self, addressbook: str) -> str:
        return f"{self._addressbooks_base()}/{urllib.parse.quote(addressbook)}/"

    def _contact_path(self, addressbook: str, uid: str) -> str:
        return f"{self._addressbook_path(addressbook)}{urllib.parse.quote(uid, safe='')}.vcf"

    # ---- server info ----

    def user_info(self) -> dict:
        data = self._ocs("GET", "/ocs/v2.php/cloud/user") or {}
        if not isinstance(data, dict):
            return {}
        return {
            "id": str(data.get("id") or ""),
            "display_name": str(data.get("display-name") or data.get("displayname") or ""),
            "email": str(data.get("email") or ""),
            "language": str(data.get("language") or ""),
            "locale": str(data.get("locale") or ""),
        }

    def capabilities(self) -> dict:
        data = self._ocs("GET", "/ocs/v2.php/cloud/capabilities") or {}
        return data if isinstance(data, dict) else {}

    # ---- files ----

    def list_files(self, path: str = "/") -> dict:
        _status, _headers, data = self._request(
            "PROPFIND",
            self._files_path(path),
            body=_FILE_PROPFIND,
            depth="1",
            content_type="application/xml; charset=utf-8",
        )
        root = _multistatus(data)
        requested = _norm_path(path)
        entries = []
        for response in root.findall(f"{{{DAV}}}response"):
            href = _text(response.find(f"{{{DAV}}}href"))
            entry_path = _dav_href_to_path(href, self._files_base())
            if entry_path == requested:
                continue
            prop = _response_prop(response)
            if prop is None:
                continue
            entries.append(self._file_entry(entry_path, prop))
        entries.sort(key=lambda entry: (entry["type"] != "dir", entry["name"].lower()))
        return {"path": requested, "entries": entries}

    def stat(self, path: str) -> dict:
        _status, _headers, data = self._request(
            "PROPFIND",
            self._files_path(path),
            body=_FILE_PROPFIND,
            depth="0",
            content_type="application/xml; charset=utf-8",
        )
        root = _multistatus(data)
        response = root.find(f"{{{DAV}}}response")
        if response is None:
            raise NextcloudError(f"stat {path}: no DAV response", status=404)
        prop = _response_prop(response)
        if prop is None:
            raise NextcloudError(f"stat {path}: no properties returned", status=404)
        return self._file_entry(_norm_path(path), prop)

    @staticmethod
    def _file_entry(path: str, prop) -> dict:
        resourcetype = prop.find(f"{{{DAV}}}resourcetype")
        is_dir = resourcetype is not None and resourcetype.find(f"{{{DAV}}}collection") is not None
        name = _text(prop.find(f"{{{DAV}}}displayname")) or path.rstrip("/").rsplit("/", 1)[-1]
        return {
            "name": name,
            "path": path,
            "type": "dir" if is_dir else "file",
            "size": _int(_text(prop.find(f"{{{DAV}}}getcontentlength"))),
            "modified": _text(prop.find(f"{{{DAV}}}getlastmodified")),
            "etag": _text(prop.find(f"{{{DAV}}}getetag")),
            "content_type": _text(prop.find(f"{{{DAV}}}getcontenttype")),
        }

    def read_file(self, path: str) -> dict:
        _status, _headers, data = self._request("GET", self._files_path(path))
        try:
            content = data.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = base64.b64encode(data).decode("ascii")
            encoding = "base64"
        return {
            "path": _norm_path(path),
            "content": content,
            "encoding": encoding,
            "size": len(data),
        }

    def write_file(
        self, path: str, content: str, encoding: str = "utf-8", overwrite: bool = True
    ) -> dict:
        if encoding == "base64":
            try:
                data = base64.b64decode(content, validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"content is not valid base64: {exc}") from exc
        else:
            data = str(content).encode("utf-8")
        headers = {} if overwrite else {"If-None-Match": "*"}
        self._request(
            "PUT",
            self._files_path(path),
            body=data,
            headers=headers,
            content_type="application/octet-stream",
        )
        return {"ok": True, "path": _norm_path(path)}

    def mkdir(self, path: str) -> dict:
        self._request("MKCOL", self._files_path(path))
        return {"ok": True, "path": _norm_path(path)}

    def delete(self, path: str) -> dict:
        self._request("DELETE", self._files_path(path))
        return {"ok": True, "path": _norm_path(path)}

    def move(self, path: str, destination: str, overwrite: bool = False) -> dict:
        self._request(
            "MOVE",
            self._files_path(path),
            headers={
                "Destination": self._url(self._files_path(destination)),
                "Overwrite": "T" if overwrite else "F",
            },
        )
        return {"ok": True, "path": _norm_path(path), "destination": _norm_path(destination)}

    def copy(self, path: str, destination: str, overwrite: bool = False) -> dict:
        self._request(
            "COPY",
            self._files_path(path),
            headers={
                "Destination": self._url(self._files_path(destination)),
                "Overwrite": "T" if overwrite else "F",
            },
        )
        return {"ok": True, "path": _norm_path(path), "destination": _norm_path(destination)}

    def search(self, query: str, path: str = "/") -> list[dict]:
        _status, _headers, data = self._request(
            "SEARCH",
            "/remote.php/dav/",
            body=_search_body(self.username, path, query),
            content_type="application/xml; charset=utf-8",
        )
        root = _multistatus(data)
        results = []
        for response in root.findall(f"{{{DAV}}}response"):
            href = _text(response.find(f"{{{DAV}}}href"))
            prop = _response_prop(response)
            if prop is None:
                continue
            results.append(
                self._file_entry(_dav_href_to_path(href, self._files_base()), prop)
            )
        results.sort(key=lambda entry: entry["path"])
        return results

    # ---- calendars ----

    def list_calendars(self) -> list[dict]:
        _status, _headers, data = self._request(
            "PROPFIND",
            self._calendars_base() + "/",
            body=_CALENDAR_PROPFIND,
            depth="1",
            content_type="application/xml; charset=utf-8",
        )
        root = _multistatus(data)
        calendars = []
        for response in root.findall(f"{{{DAV}}}response"):
            prop = _response_prop(response)
            if prop is None:
                continue
            resourcetype = prop.find(f"{{{DAV}}}resourcetype")
            if resourcetype is None or resourcetype.find(f"{{{CALDAV}}}calendar") is None:
                continue
            href = urllib.parse.unquote(
                urllib.parse.urlparse(_text(response.find(f"{{{DAV}}}href"))).path
            )
            name = href.rstrip("/").rsplit("/", 1)[-1]
            if not name:
                continue
            color = _text(prop.find(f"{{{NC}}}calendar-color")) or _text(
                prop.find(f"{{{OC}}}calendar-color")
            )
            ctag = _text(prop.find(f"{{{CALDAV}}}getctag")) or _text(
                prop.find(f"{{{OC}}}getctag")
            )
            calendars.append(
                {
                    "name": name,
                    "label": _text(prop.find(f"{{{DAV}}}displayname")) or name,
                    "description": _text(prop.find(f"{{{CALDAV}}}calendar-description")),
                    "color": color,
                    "ctag": ctag,
                }
            )
        calendars.sort(key=lambda calendar: calendar["name"].lower())
        return calendars

    def _report_objects(self, calendar: str, component: str, start: str = "", end: str = "") -> list[dict]:
        _status, _headers, data = self._request(
            "REPORT",
            self._calendar_path(calendar),
            body=_calendar_query_body(component, start, end),
            depth="1",
            content_type="application/xml; charset=utf-8",
        )
        root = _multistatus(data)
        objects = []
        for response in root.findall(f"{{{DAV}}}response"):
            prop = _response_prop(response)
            if prop is None:
                continue
            href = _text(response.find(f"{{{DAV}}}href"))
            etag = _text(prop.find(f"{{{DAV}}}getetag"))
            calendar_data = prop.find(f"{{{CALDAV}}}calendar-data")
            ics = calendar_data.text if calendar_data is not None and calendar_data.text else ""
            for block in parse_components(ics, component):
                item = _event_dict(block) if component == "VEVENT" else _todo_dict(block)
                item["calendar"] = calendar
                item["href"] = href
                item["etag"] = etag
                objects.append(item)
        return objects

    def list_events(self, calendar: str, start: str = "", end: str = "") -> list[dict]:
        return self._report_objects(calendar, "VEVENT", start, end)

    def list_tasks(self, calendar: str) -> list[dict]:
        return self._report_objects(calendar, "VTODO")

    def get_object(self, calendar: str, uid: str) -> tuple[str | None, str | None]:
        try:
            _status, headers, data = self._request(
                "GET", self._calendar_object_path(calendar, uid)
            )
        except NextcloudError as exc:
            if exc.status == 404:
                return None, None
            raise
        return (headers.get("ETag") or ""), data.decode("utf-8", "replace")

    def put_object(
        self, calendar: str, uid: str, ics: str, etag: str | None = None, overwrite: bool = False
    ) -> dict:
        headers = {"Content-Type": "text/calendar; charset=utf-8"}
        if etag:
            headers["If-Match"] = etag
        elif not overwrite:
            headers["If-None-Match"] = "*"
        _status, response_headers, _data = self._request(
            "PUT",
            self._calendar_object_path(calendar, uid),
            body=ics.encode("utf-8"),
            headers=headers,
        )
        return {
            "ok": True,
            "uid": uid,
            "calendar": calendar,
            "etag": response_headers.get("ETag") or "",
        }

    def delete_object(self, calendar: str, uid: str) -> dict:
        self._request("DELETE", self._calendar_object_path(calendar, uid))
        return {"ok": True, "uid": uid, "calendar": calendar}

    def create_event(
        self,
        calendar: str,
        uid: str,
        summary: str,
        start: str,
        end: str = "",
        all_day: bool = False,
        location: str = "",
        description: str = "",
        rrule: str = "",
        overwrite: bool = False,
    ) -> dict:
        ics = build_event_ics(
            uid,
            summary,
            start,
            end,
            all_day=all_day,
            location=location,
            description=description,
            rrule=rrule,
        )
        return self.put_object(calendar, uid, ics, overwrite=overwrite)

    def update_event(self, calendar: str, uid: str, **changes) -> dict:
        etag, existing = self.get_object(calendar, uid)
        if existing is None:
            raise NextcloudError(f"event {uid!r} was not found on calendar {calendar!r}", status=404)
        block = _extract_block(_unfold(existing), "VEVENT")
        if block is None:
            raise NextcloudError(f"event {uid!r} has no VEVENT component")
        block_lines = block[2]
        all_day = changes.get("all_day")
        updates: dict[str, list[str]] = {}
        if "summary" in changes and changes["summary"] is not None:
            updates["SUMMARY"] = [f"SUMMARY:{_escape_text(changes['summary'])}"]
        if "description" in changes and changes["description"] is not None:
            updates["DESCRIPTION"] = [f"DESCRIPTION:{_escape_text(changes['description'])}"]
        if "location" in changes and changes["location"] is not None:
            updates["LOCATION"] = [f"LOCATION:{_escape_text(changes['location'])}"]
        if "rrule" in changes and changes["rrule"] is not None:
            updates["RRULE"] = [f"RRULE:{changes['rrule']}"] if changes["rrule"] else []
        if "start" in changes and changes["start"] is not None:
            updates["DTSTART"] = _dt_lines(
                "DTSTART", changes["start"], bool(all_day)
            )
        if "end" in changes and changes["end"] is not None:
            updates["DTEND"] = _dt_lines("DTEND", changes["end"], bool(all_day))
        new_block = _replace_props(block_lines, updates)
        ics = _rebuild_component(_unfold(existing), "VEVENT", new_block)
        return self.put_object(calendar, uid, ics, etag=etag)

    def create_task(
        self,
        calendar: str,
        uid: str,
        summary: str,
        due: str = "",
        priority: int | None = None,
        description: str = "",
    ) -> dict:
        ics = build_todo_ics(
            uid, summary, due=due, priority=priority, description=description
        )
        return self.put_object(calendar, uid, ics)

    def update_task(self, calendar: str, uid: str, **changes) -> dict:
        etag, existing = self.get_object(calendar, uid)
        if existing is None:
            raise NextcloudError(f"task {uid!r} was not found on calendar {calendar!r}", status=404)
        block = _extract_block(_unfold(existing), "VTODO")
        if block is None:
            raise NextcloudError(f"task {uid!r} has no VTODO component")
        updates: dict[str, list[str]] = {}
        if "summary" in changes and changes["summary"] is not None:
            updates["SUMMARY"] = [f"SUMMARY:{_escape_text(changes['summary'])}"]
        if "description" in changes and changes["description"] is not None:
            updates["DESCRIPTION"] = [f"DESCRIPTION:{_escape_text(changes['description'])}"]
        if "due" in changes and changes["due"] is not None:
            updates["DUE"] = _dt_lines("DUE", changes["due"], False) if changes["due"] else []
        if "priority" in changes and changes["priority"] is not None:
            updates["PRIORITY"] = [f"PRIORITY:{int(changes['priority'])}"]
        if "status" in changes and changes["status"] is not None:
            updates["STATUS"] = [f"STATUS:{changes['status']}"]
        if "percent_complete" in changes and changes["percent_complete"] is not None:
            updates["PERCENT-COMPLETE"] = [f"PERCENT-COMPLETE:{int(changes['percent_complete'])}"]
        new_block = _replace_props(block[2], updates)
        ics = _rebuild_component(_unfold(existing), "VTODO", new_block)
        return self.put_object(calendar, uid, ics, etag=etag)

    def complete_task(self, calendar: str, uid: str) -> dict:
        etag, existing = self.get_object(calendar, uid)
        if existing is None:
            raise NextcloudError(f"task {uid!r} was not found on calendar {calendar!r}", status=404)
        block = _extract_block(_unfold(existing), "VTODO")
        if block is None:
            raise NextcloudError(f"task {uid!r} has no VTODO component")
        updates = {
            "STATUS": ["STATUS:COMPLETED"],
            "PERCENT-COMPLETE": ["PERCENT-COMPLETE:100"],
            "COMPLETED": [f"COMPLETED:{_now_stamp()}"],
        }
        new_block = _replace_props(block[2], updates)
        ics = _rebuild_component(_unfold(existing), "VTODO", new_block)
        return self.put_object(calendar, uid, ics, etag=etag)

    # ---- contacts ----

    def list_addressbooks(self) -> list[dict]:
        _status, _headers, data = self._request(
            "PROPFIND",
            self._addressbooks_base() + "/",
            body=_ADDRESSBOOK_PROPFIND,
            depth="1",
            content_type="application/xml; charset=utf-8",
        )
        root = _multistatus(data)
        addressbooks = []
        for response in root.findall(f"{{{DAV}}}response"):
            prop = _response_prop(response)
            if prop is None:
                continue
            resourcetype = prop.find(f"{{{DAV}}}resourcetype")
            if resourcetype is None or resourcetype.find(f"{{{CARDDAV}}}addressbook") is None:
                continue
            href = urllib.parse.unquote(
                urllib.parse.urlparse(_text(response.find(f"{{{DAV}}}href"))).path
            )
            name = href.rstrip("/").rsplit("/", 1)[-1]
            if not name:
                continue
            addressbooks.append(
                {
                    "name": name,
                    "label": _text(prop.find(f"{{{DAV}}}displayname")) or name,
                    "description": _text(prop.find(f"{{{CARDDAV}}}addressbook-description")),
                }
            )
        addressbooks.sort(key=lambda book: book["name"].lower())
        return addressbooks

    def list_contacts(self, addressbook: str, query: str = "") -> list[dict]:
        _status, _headers, data = self._request(
            "REPORT",
            self._addressbook_path(addressbook),
            body=_ADDRESSBOOK_QUERY,
            depth="1",
            content_type="application/xml; charset=utf-8",
        )
        root = _multistatus(data)
        contacts = []
        for response in root.findall(f"{{{DAV}}}response"):
            prop = _response_prop(response)
            if prop is None:
                continue
            href = _text(response.find(f"{{{DAV}}}href"))
            etag = _text(prop.find(f"{{{DAV}}}getetag"))
            card_data = prop.find(f"{{{CARDDAV}}}address-data")
            vcf = card_data.text if card_data is not None and card_data.text else ""
            contact = parse_vcard(vcf)
            contact["addressbook"] = addressbook
            contact["href"] = href
            contact["etag"] = etag
            contacts.append(contact)
        if query:
            needle = query.lower()
            contacts = [
                contact
                for contact in contacts
                if needle
                in " ".join(
                    [contact["fn"], contact["org"], contact["note"], *contact["emails"], *contact["phones"]]
                ).lower()
            ]
        contacts.sort(key=lambda contact: contact["fn"].lower())
        return contacts

    def get_contact(self, addressbook: str, uid: str) -> tuple[str | None, dict | None]:
        try:
            _status, headers, data = self._request(
                "GET", self._contact_path(addressbook, uid)
            )
        except NextcloudError as exc:
            if exc.status == 404:
                return None, None
            raise
        return (headers.get("ETag") or ""), parse_vcard(data.decode("utf-8", "replace"))

    def create_contact(self, addressbook: str, uid: str, **fields) -> dict:
        vcf = build_vcard(
            uid,
            fields.get("fn", ""),
            n=fields.get("n", ""),
            emails=fields.get("emails") or (),
            phones=fields.get("phones") or (),
            org=fields.get("org", ""),
            title=fields.get("title", ""),
            note=fields.get("note", ""),
            url=fields.get("url", ""),
        )
        self._request(
            "PUT",
            self._contact_path(addressbook, uid),
            body=vcf.encode("utf-8"),
            headers={"Content-Type": "text/vcard; charset=utf-8", "If-None-Match": "*"},
        )
        return {"ok": True, "uid": uid, "addressbook": addressbook}

    def update_contact(self, addressbook: str, uid: str, **changes) -> dict:
        etag, existing = self.get_contact(addressbook, uid)
        if existing is None:
            raise NextcloudError(
                f"contact {uid!r} was not found in addressbook {addressbook!r}", status=404
            )
        updates: dict[str, list[str]] = {}
        if "fn" in changes and changes["fn"] is not None:
            updates["FN"] = [f"FN:{_escape_text(changes['fn'])}"]
        if "n" in changes and changes["n"] is not None:
            updates["N"] = [f"N:{_escape_text(changes['n'])}"] if changes["n"] else []
        if "org" in changes and changes["org"] is not None:
            updates["ORG"] = [f"ORG:{_escape_text(changes['org'])}"] if changes["org"] else []
        if "title" in changes and changes["title"] is not None:
            updates["TITLE"] = [f"TITLE:{_escape_text(changes['title'])}"] if changes["title"] else []
        if "note" in changes and changes["note"] is not None:
            updates["NOTE"] = [f"NOTE:{_escape_text(changes['note'])}"] if changes["note"] else []
        if "emails" in changes and changes["emails"] is not None:
            updates["EMAIL"] = [f"EMAIL;TYPE=INTERNET:{_escape_text(email)}" for email in changes["emails"]]
        if "phones" in changes and changes["phones"] is not None:
            updates["TEL"] = [f"TEL:{_escape_text(phone)}" for phone in changes["phones"]]
        try:
            _status, _headers, data = self._request(
                "GET", self._contact_path(addressbook, uid)
            )
        except NextcloudError as exc:  # pragma: no cover - raced away after the etag read
            raise NextcloudError(f"contact {uid!r} could not be read: {exc}", status=exc.status) from exc
        original = _unfold(data.decode("utf-8", "replace"))
        block = _extract_block(original, "VCARD")
        if block is None:
            raise NextcloudError(f"contact {uid!r} has no VCARD component")
        new_block = _replace_props(block[2], updates)
        vcf = _rebuild_component(original, "VCARD", new_block)
        headers = {"Content-Type": "text/vcard; charset=utf-8"}
        if etag:
            headers["If-Match"] = etag
        self._request(
            "PUT", self._contact_path(addressbook, uid), body=vcf.encode("utf-8"), headers=headers
        )
        return {"ok": True, "uid": uid, "addressbook": addressbook}

    def delete_contact(self, addressbook: str, uid: str) -> dict:
        self._request("DELETE", self._contact_path(addressbook, uid))
        return {"ok": True, "uid": uid, "addressbook": addressbook}

    # ---- notes (Notes app OCS API) ----

    def list_notes(self) -> list[dict]:
        data = self._ocs("GET", "/ocs/v2.php/apps/notes/api/v1/notes")
        return data if isinstance(data, list) else []

    def get_note(self, note_id) -> dict:
        data = self._ocs("GET", f"/ocs/v2.php/apps/notes/api/v1/notes/{int(note_id)}")
        if not isinstance(data, dict):
            raise NextcloudError(f"note {note_id} returned no data")
        return data

    def create_note(self, title: str, content: str = "", category: str = "") -> dict:
        payload = {"title": title, "content": content}
        if category:
            payload["category"] = category
        data = self._ocs("POST", "/ocs/v2.php/apps/notes/api/v1/notes", payload=payload)
        if not isinstance(data, dict):
            raise NextcloudError("creating the note returned no data")
        return data

    def update_note(self, note_id, **changes) -> dict:
        payload = {key: value for key, value in changes.items() if value is not None}
        data = self._ocs(
            "PUT", f"/ocs/v2.php/apps/notes/api/v1/notes/{int(note_id)}", payload=payload
        )
        if not isinstance(data, dict):
            raise NextcloudError(f"updating note {note_id} returned no data")
        return data

    def delete_note(self, note_id) -> dict:
        self._ocs("DELETE", f"/ocs/v2.php/apps/notes/api/v1/notes/{int(note_id)}")
        return {"ok": True, "id": int(note_id)}
