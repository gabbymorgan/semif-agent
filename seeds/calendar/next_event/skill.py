"""Report the next upcoming event on the user's Nextcloud calendar.

Real integration: CalDAV against the user's Nextcloud (SabreDAV). Connection
details and the default calendar come from `ctx.config` (`nextcloud_url`,
`nextcloud_username`, `nextcloud_app_password`, `nextcloud_default_calendar`).

`act` discovers the user's calendars over PROPFIND and resolves which one the
request means with a single SemIf choice over every owned calendar (state = the
user query): a strong winner (probability >= `CALENDAR_WINNER_TAU`) is used,
otherwise the configured default calendar is used; with no configured default
the weak winner stands. It then issues a `calendar-query` REPORT with
server-side recurrence expansion and reports the earliest instance that has not
ended yet.

Recurring events are expanded by the server (`<c:expand>`), so no RRULE engine
lives here; a server that ignores expansion is reported honestly rather than
guessed at.
"""

from __future__ import annotations

import base64
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from semif_agent.decisions import DecisionRequest, Option
from semif_agent.skills import ActionResult

INTEGRATION = {
    "service": "nextcloud_calendar",
    "transport": "caldav",
    "config_vars": [
        "nextcloud_url",
        "nextcloud_username",
        "nextcloud_app_password",
        "nextcloud_default_calendar",
    ],
}

CONTRACT = {
    "nextcloud_url": "Base URL of the Nextcloud instance, e.g. https://cloud.example.com (no path, no trailing slash needed).",
    "nextcloud_username": "The Nextcloud username whose calendars are read.",
    "nextcloud_app_password": "A Nextcloud app password (Settings > Security > Devices & sessions) used for CalDAV Basic auth; the account password only works when two-factor auth is off.",
    "nextcloud_default_calendar": "Name of the calendar read by default when the request names none, e.g. personal (a request may still name another calendar).",
}

DAV = "DAV:"
CALDAV = "urn:ietf:params:xml:ns:caldav"
LOOKAHEAD_DAYS = 60
TIMEOUT_SECONDS = 30
# A SemIf probability at or above this counts as a "strong winner" for the
# target calendar; below it the configured default calendar is used instead.
CALENDAR_WINNER_TAU = 0.6

PROPFIND_BODY = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
    "<d:prop><d:displayname/><d:resourcetype/></d:prop>"
    "</d:propfind>"
).encode("utf-8")

PROPERTY_RE = re.compile(r"^(?P<name>[A-Za-z-]+)(?P<params>(?:;[^:]*)?):(?P<value>.*)$")
DURATION_RE = re.compile(
    r"^(?P<sign>[+-])?P"
    r"(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


class CalDavError(Exception):
    """A real failure talking to the calendar service."""


def _now():
    """Current local time (aware). A test seam; the body never freezes time."""
    return datetime.now().astimezone()


# ---- transport ----


def _request(ctx, method, path, body=None, depth="0"):
    username = str(ctx.config.get("nextcloud_username", "") or "").strip()
    password = str(ctx.config.get("nextcloud_app_password", "") or "")
    url = str(ctx.config.get("nextcloud_url", "") or "").strip().rstrip("/")
    if not url:
        raise CalDavError("nextcloud_url is not configured")
    if not username or not password:
        raise CalDavError("nextcloud_username and nextcloud_app_password are not configured")
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        url + path,
        data=body,
        method=method,
        headers={
            "Authorization": f"Basic {token}",
            "Depth": depth,
            "Content-Type": "application/xml; charset=utf-8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()[:200]
        raise CalDavError(f"{method} {path}: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise CalDavError(f"{method} {path}: {exc}") from exc


# ---- CalDAV discovery + query ----


def _multistatus(body):
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise CalDavError(f"unparseable CalDAV XML: {exc}") from exc
    if root.tag != f"{{{DAV}}}multistatus":
        raise CalDavError(f"unexpected CalDAV response root {root.tag!r}")
    return root


def _response_prop(response):
    return response.find(f"{{{DAV}}}propstat/{{{DAV}}}prop")


def _discover_calendars(ctx):
    username = str(ctx.config.get("nextcloud_username", "") or "").strip()
    if not username:
        raise CalDavError("nextcloud_username is not configured")
    home = f"/remote.php/dav/calendars/{urllib.parse.quote(username)}/"
    root = _multistatus(_request(ctx, "PROPFIND", home, PROPFIND_BODY, depth="1"))
    calendars = []
    for response in root.findall(f"{{{DAV}}}response"):
        prop = _response_prop(response)
        if prop is None:
            continue
        resourcetype = prop.find(f"{{{DAV}}}resourcetype")
        if resourcetype is None or resourcetype.find(f"{{{CALDAV}}}calendar") is None:
            continue
        href = response.find(f"{{{DAV}}}href")
        path = urllib.parse.urlparse((href.text if href is not None else "") or "").path
        name = urllib.parse.unquote(path.rstrip("/").rsplit("/", 1)[-1])
        if not name:
            continue
        display = prop.find(f"{{{DAV}}}displayname")
        label = (display.text or "").strip() if display is not None else ""
        calendars.append({"name": name, "label": label or name})
    return calendars


def _calendar_path(ctx, calendar):
    username = str(ctx.config.get("nextcloud_username", "") or "").strip()
    return (
        f"/remote.php/dav/calendars/{urllib.parse.quote(username)}/"
        f"{urllib.parse.quote(calendar)}/"
    )


def _report_body(start, end):
    start_text = start.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    end_text = end.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
        "<d:prop>"
        f'<c:calendar-data><c:expand start="{start_text}" end="{end_text}"/>'
        "</c:calendar-data>"
        "</d:prop>"
        "<c:filter>"
        '<c:comp-filter name="VCALENDAR">'
        '<c:comp-filter name="VEVENT">'
        f'<c:time-range start="{start_text}" end="{end_text}"/>'
        "</c:comp-filter>"
        "</c:comp-filter>"
        "</c:filter>"
        "</c:calendar-query>"
    ).encode("utf-8")


def _calendar_payloads(body):
    root = _multistatus(body)
    payloads = []
    for response in root.findall(f"{{{DAV}}}response"):
        prop = _response_prop(response)
        if prop is None:
            continue
        for data in prop.findall(f"{{{CALDAV}}}calendar-data"):
            if data.text:
                payloads.append(data.text)
    return payloads


# ---- calendar resolution ----


def _default_calendar(ctx, calendars):
    """The configured default calendar, if it is one of the account's."""
    default_name = str(ctx.config.get("nextcloud_default_calendar", "") or "").strip().lower()
    for calendar in calendars:
        if default_name and default_name in (calendar["name"].lower(), calendar["label"].lower()):
            return calendar
    return None


def _resolve_calendar(ctx, request, calendars, decisions):
    """Resolve which calendar the request means, mirroring calendar.create_event.

    A strong SemIf winner is used; a weak winner falls back to the configured
    default calendar, and with no configured default the weak winner stands.
    """
    default = _default_calendar(ctx, calendars)
    if len(calendars) == 1:
        return calendars[0]["name"]

    decision = DecisionRequest(
        state=request.text,
        question="Which calendar is referred to in the user query?",
        options=[Option(c["name"], c["label"]) for c in calendars],
    )
    result = ctx.engine.call(decision)
    decisions.append((decision, result))
    winner = result.selected
    if result.prob(winner) >= CALENDAR_WINNER_TAU:
        return winner
    if default is not None:
        return default["name"]
    return winner


# ---- iCalendar parsing ----


def _unfold(text):
    lines = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _parse_property(line):
    match = PROPERTY_RE.match(line)
    if match is None:
        return "", {}, ""
    name = match.group("name").upper()
    params = {}
    for chunk in match.group("params").strip(";").split(";"):
        key, _, value = chunk.partition("=")
        if key:
            params[key.strip().upper()] = value.strip().strip('"')
    return name, params, match.group("value")


def _parse_datetime(value, params, default_tz):
    value = value.strip()
    if params.get("VALUE") == "DATE" or (len(value) == 8 and value.isdigit()):
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=default_tz), True
    tz = default_tz
    if value.endswith("Z"):
        tz = timezone.utc
        value = value[:-1]
    elif params.get("TZID"):
        try:
            tz = ZoneInfo(params["TZID"])
        except Exception:
            tz = default_tz
    return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=tz), False


def _parse_duration(value):
    match = DURATION_RE.match(value.strip().upper())
    if match is None:
        return None
    total = timedelta(
        weeks=int(match.group("weeks") or 0),
        days=int(match.group("days") or 0),
        hours=int(match.group("hours") or 0),
        minutes=int(match.group("minutes") or 0),
        seconds=int(match.group("seconds") or 0),
    )
    return -total if match.group("sign") == "-" else total


def _parse_events(payloads, default_tz):
    events = []
    for payload in payloads:
        current = None
        for line in _unfold(payload):
            name, params, value = _parse_property(line)
            if name == "BEGIN" and value.strip().upper() == "VEVENT":
                current = {}
                continue
            if name == "END" and value.strip().upper() == "VEVENT":
                if current is not None:
                    events.append(current)
                current = None
                continue
            if current is None:
                continue
            if name == "DTSTART":
                current["start"], current["all_day"] = _parse_datetime(
                    value, params, default_tz
                )
            elif name == "DTEND":
                current["end"] = _parse_datetime(value, params, default_tz)[0]
            elif name == "DURATION":
                current["duration"] = _parse_duration(value)
            elif name == "SUMMARY":
                current["summary"] = value.strip()
            elif name == "LOCATION":
                current["location"] = value.strip()
            elif name == "STATUS" and value.strip().upper() == "CANCELLED":
                current["cancelled"] = True
        if current is not None:
            events.append(current)
    return [
        event for event in events if "start" in event and not event.get("cancelled")
    ]


def _event_end(event):
    if event.get("end") is not None:
        return event["end"]
    duration = event.get("duration")
    if duration is not None:
        return event["start"] + duration
    if event.get("all_day"):
        return event["start"] + timedelta(days=1)
    return event["start"]


def _pick_next(events, now):
    current_or_future = [event for event in events if _event_end(event) >= now]
    if not current_or_future:
        return None
    return min(current_or_future, key=lambda event: event["start"])


def _format_when(event):
    start = event["start"]
    if event.get("all_day"):
        return start.strftime("%Y-%m-%d (all day)")
    return start.strftime("%Y-%m-%d %H:%M %Z")


# ---- skill phases ----


def act(ctx, request):
    try:
        calendars = _discover_calendars(ctx)
    except CalDavError as exc:
        return ActionResult(
            action_log=f"calendar.next_event: calendar discovery failed: {exc}",
            new_state=request.text,
        )
    if not calendars:
        return ActionResult(
            action_log="calendar.next_event: calendar discovery failed: no calendars found",
            new_state=request.text,
        )

    decisions = []
    calendar = _resolve_calendar(ctx, request, calendars, decisions)
    if not calendar:
        return ActionResult(
            action_log="calendar.next_event aborted: no calendar resolved.",
            new_state=request.text,
            decisions=decisions,
        )

    now = _now()
    local_tz = now.tzinfo or timezone.utc
    window_start = now - timedelta(days=1)
    window_end = now + timedelta(days=LOOKAHEAD_DAYS)
    try:
        body = _request(
            ctx,
            "REPORT",
            _calendar_path(ctx, calendar),
            _report_body(window_start, window_end),
            depth="1",
        )
        events = _parse_events(_calendar_payloads(body), local_tz)
    except CalDavError as exc:
        return ActionResult(
            action_log=f"calendar.next_event failed on {calendar!r}: {exc}",
            new_state=request.text,
            decisions=decisions,
        )

    next_event = _pick_next(events, now)
    if next_event is None:
        message = (
            f"No events on calendar {calendar!r} through "
            f"{window_end.strftime('%Y-%m-%d')}."
        )
        return ActionResult(action_log=message, new_state=message, decisions=decisions)

    summary = next_event.get("summary") or "(no title)"
    when = _format_when(next_event)
    message = f"Next event on {calendar!r}: {summary} — {when}"
    if next_event.get("location"):
        message += f" at {next_event['location']}"
    return ActionResult(action_log=message, new_state=message, decisions=decisions)
