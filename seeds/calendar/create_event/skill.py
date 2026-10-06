"""Create a calendar event on the user's Nextcloud calendar from a request.

Real integration: CalDAV against the user's Nextcloud (SabreDAV). Connection
details and the default calendar come from `ctx.config` (`nextcloud_url`,
`nextcloud_username`, `nextcloud_app_password`, `nextcloud_default_calendar`).

`act` reads `request.text` and derives the event's per-request arguments. The
title and description are extracted by the local LLM bridge (`llm_bridge_url`,
`llm_bridge_token`) with one generic chat call, cached in `request.meta` so a
`needs_input` resume does not re-call it; the start date/time and duration are
parsed deterministically. Missing arguments are asked one at a time and the
answer is stashed in `request.meta`; because `act` is single-phase and re-runs
from the top after a `needs_input` pause, the body never re-asks a question it
already has an answer for. Calendars are discovered over PROPFIND and the target
is resolved by a single SemIf choice over every owned calendar (state = the user
query), asking which one the request refers to; a strong winner (probability >=
`CALENDAR_WINNER_TAU`) is used, otherwise the configured default calendar is
used. The event is created with a real CalDAV PUT (iCalendar, UTC timestamps,
`If-None-Match: *`) and the real outcome is reported.

The date/time parse is deliberately bounded; the title/description come from the
model. When the model bridge is unavailable the body reports the real failure
rather than guessing.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, time, timedelta, timezone

from semif_agent.decisions import DecisionRequest, Option
from semif_agent.skills import ActionResult
from semif_agent.timers import local_now

INTEGRATION = {
    "service": "nextcloud_calendar",
    "transport": "caldav",
    "config_vars": [
        "nextcloud_url",
        "nextcloud_username",
        "nextcloud_app_password",
        "nextcloud_default_calendar",
        "llm_bridge_url",
        "llm_bridge_token",
    ],
}

CONTRACT = {
    "nextcloud_url": "Base URL of the Nextcloud instance, e.g. https://cloud.example.com (no path, no trailing slash needed).",
    "nextcloud_username": "The Nextcloud username whose calendars are used.",
    "nextcloud_app_password": "A Nextcloud app password (Settings > Security > Devices & sessions) used for CalDAV Basic auth; the account password only works when two-factor auth is off.",
    "nextcloud_default_calendar": "Name of the calendar new events go to by default, e.g. personal (a request may still name another calendar).",
    "llm_bridge_url": "Base URL of the local LLM bridge, e.g. http://127.0.0.1:5229 (no trailing slash needed); used to extract the event title and description.",
    "llm_bridge_token": "Shared secret for the LLM bridge, if one is configured; sent as the X-Semif-Token header. Leave blank when the bridge requires no auth.",
}

DAV = "DAV:"
CALDAV = "urn:ietf:params:xml:ns:caldav"
TIMEOUT_SECONDS = 30
DEFAULT_DURATION_MINUTES = 60
# A SemIf probability at or above this counts as a "strong winner" for the
# target calendar; below it the configured default calendar is used instead.
CALENDAR_WINNER_TAU = 0.6

PROPFIND_BODY = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">'
    "<d:prop><d:displayname/><d:resourcetype/></d:prop>"
    "</d:propfind>"
).encode("utf-8")

_WEEKDAYS = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}
_WEEKDAY_RE = re.compile(
    r"\b(" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True)) + r")\b", re.I
)

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH_ALT = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

class CalDavError(Exception):
    """A real failure talking to the calendar service."""


class BridgeError(Exception):
    """A real failure talking to the local LLM bridge."""


def _now(ctx):
    """Current time (aware) in the user's timezone: the top-level `timezone`
    config value when set, else host local. A test seam; never freezes time."""
    return local_now(ctx.config)


# ---- transport ----


def _request(ctx, method, path, body=None, depth="0", content_type="application/xml; charset=utf-8", extra_headers=None):
    url = str(ctx.config.get("nextcloud_url", "") or "").strip().rstrip("/")
    username = str(ctx.config.get("nextcloud_username", "") or "").strip()
    password = str(ctx.config.get("nextcloud_app_password", "") or "")
    if not url:
        raise CalDavError("nextcloud_url is not configured")
    if not username or not password:
        raise CalDavError("nextcloud_username and nextcloud_app_password are not configured")
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    headers = {"Authorization": f"Basic {token}", "Depth": depth}
    if body is not None:
        headers["Content-Type"] = content_type
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(
        url + path, data=body, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()[:200]
        raise CalDavError(f"{method} {path}: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise CalDavError(f"{method} {path}: {exc}") from exc


# ---- CalDAV discovery ----


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
    _status, body = _request(ctx, "PROPFIND", home, PROPFIND_BODY, depth="1")
    calendars = []
    for response in _multistatus(body).findall(f"{{{DAV}}}response"):
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


# ---- LLM field extraction ----


def _extract_fields(ctx, request):
    """Ask the LLM bridge for the event title and description.

    One generic `POST /chat` call to the local LLM bridge; the reply is the JSON
    object the prompt asks for. Raises `BridgeError` when the bridge is
    unreachable or the reply carries no usable object.
    """
    base = str(ctx.config.get("llm_bridge_url", "") or "").strip().rstrip("/")
    if not base:
        raise BridgeError("llm_bridge_url is not configured")
    token = str(ctx.config.get("llm_bridge_token", "") or "").strip()
    messages = [
        {
            "role": "system",
            "content": (
                "Extract the calendar event from the user's request. Reply with "
                'ONLY a JSON object: {"title": "<short event title>", '
                '"description": "<one short sentence, or empty>"}. The title must '
                "not contain the date, the time, or words like add/schedule/create."
            ),
        },
        {"role": "user", "content": request.text},
    ]
    payload = json.dumps({"messages": messages}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Semif-Token"] = token
    http_request = urllib.request.Request(
        base + "/chat", data=payload, method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(http_request, timeout=TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()[:200]
        raise BridgeError(f"llm bridge /chat: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise BridgeError(f"llm bridge /chat: {exc}") from exc
    return _parse_fields(str(data.get("text") or ""))


def _parse_fields(text):
    """Pull the `{"title","description"}` object out of the model's reply."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise BridgeError("llm bridge returned no JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except ValueError as exc:
        raise BridgeError(f"llm bridge returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BridgeError("llm bridge returned JSON that is not an object")
    return {
        "title": str(data.get("title") or "").strip(),
        "description": str(data.get("description") or "").strip(),
    }


def _month_number(token):
    return _MONTHS.get(token.lower()[:3])


def _resolve_month_day(month, day, year, now):
    if not (month and 1 <= day <= 31):
        return None
    given_year = year is not None
    year = year or now.year
    try:
        candidate = datetime(year, month, day).date()
    except ValueError:
        return None
    if not given_year and candidate < now.date():
        try:
            candidate = datetime(year + 1, month, day).date()
        except ValueError:
            return None
    return candidate


def _parse_when(text, now):
    """Best-effort extraction of date/time/duration from free text.

    Returns a dict with `date` (datetime.date | None), `time` (datetime.time |
    None), `all_day` (bool), `duration_minutes` (int | None) and `from_weekday`
    (bool). The title and description come from the LLM bridge, so no matched
    spans are tracked here.
    """
    text = text or ""
    result = {
        "date": None,
        "time": None,
        "all_day": False,
        "duration_minutes": None,
        "from_weekday": False,
    }

    match = re.search(r"\ball[\s-]?day\b|\bwhole\s+day\b", text, re.I)
    if match:
        result["all_day"] = True

    match = re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b", text, re.I)
    if match:
        result["duration_minutes"] = int(round(float(match.group(1)) * 60))
    else:
        match = re.search(r"\bfor\s+(\d+)\s*(minutes?|mins?|m)\b", text, re.I)
        if match:
            result["duration_minutes"] = int(match.group(1))
        else:
            for phrase, minutes in (
                (r"\bfor\s+half\s+an?\s+hour\b", 30),
                (r"\bfor\s+an?\s+hour\b", 60),
                (r"\bfor\s+a\s+couple\s+of\s+hours\b", 120),
            ):
                match = re.search(phrase, text, re.I)
                if match:
                    result["duration_minutes"] = minutes
                    break

    date = None
    match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if match:
        try:
            date = datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3))
            ).date()
        except ValueError:
            date = None

    if date is None:
        match = re.search(
            r"\b(" + _MONTH_ALT + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b",
            text,
            re.I,
        )
        if match:
            month = _month_number(match.group(1))
            year = int(match.group(3)) if match.group(3) else None
            date = _resolve_month_day(month, int(match.group(2)), year, now)

    if date is None:
        match = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + _MONTH_ALT + r")(?:,?\s*(\d{4}))?\b",
            text,
            re.I,
        )
        if match:
            month = _month_number(match.group(2))
            year = int(match.group(3)) if match.group(3) else None
            date = _resolve_month_day(month, int(match.group(1)), year, now)

    if date is None:
        match = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text)
        if match:
            first, second = int(match.group(1)), int(match.group(2))
            month, day = (first, second) if first <= 12 else (second, first)
            year = int(match.group(3)) if match.group(3) else None
            if year is not None and year < 100:
                year += 2000
            date = _resolve_month_day(month, day, year, now)

    if date is None:
        match = re.search(
            r"\b(today|tonight|tomorrow|tmrw|tomorow|next\s+week)\b", text, re.I
        )
        if match:
            word = re.sub(r"\s+", " ", match.group(1).lower())
            if word in ("today", "tonight"):
                date = now.date()
            elif word in ("tomorrow", "tmrw", "tomorow"):
                date = now.date() + timedelta(days=1)
            else:
                date = now.date() + timedelta(days=7)

    if date is None:
        match = _WEEKDAY_RE.search(text)
        if match:
            target = _WEEKDAYS[match.group(1).lower()]
            date = now.date() + timedelta(days=(target - now.date().weekday()) % 7)
            prefix = text[max(0, match.start() - 6):match.start()]
            if re.search(r"\bnext\s*$", prefix, re.I):
                date += timedelta(days=7)
            result["from_weekday"] = True

    parsed_time = None
    match = re.search(r"\bnoon\b", text, re.I)
    if match:
        parsed_time = time(12, 0)
    elif re.search(r"\bmidnight\b", text, re.I):
        parsed_time = time(0, 0)
    else:
        match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\b", text, re.I)
        if match:
            hour = int(match.group(1)) % 12
            minute = int(match.group(2) or 0)
            if match.group(3).lower() == "p":
                hour += 12
            if hour < 24 and minute < 60:
                parsed_time = time(hour, minute)
        else:
            match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
            if match:
                hour, minute = int(match.group(1)), int(match.group(2))
                if hour < 24 and minute < 60:
                    parsed_time = time(hour, minute)
            else:
                match = re.search(r"\bat\s+(\d{1,2})\b", text, re.I)
                if match and 13 <= int(match.group(1)) < 24:
                    parsed_time = time(int(match.group(1)), 0)

    result["date"] = date
    result["time"] = parsed_time
    return result


# ---- calendar resolution ----


def _resolve_calendar(ctx, request, calendars, decisions):
    default_name = str(ctx.config.get("nextcloud_default_calendar", "") or "").strip().lower()
    default = None
    for calendar in calendars:
        if default_name and default_name in (calendar["name"].lower(), calendar["label"].lower()):
            default = calendar
            break

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


# ---- iCalendar ----


def _escape_ics(value):
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )


def _utc_stamp(moment):
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_ics(uid, summary, start, end, all_day, stamp, description=""):
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//semif-agent//calendar.create_event//EN",
        "CALSCALE:GREGORIAN",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{stamp}",
    ]
    if all_day:
        lines.append(f"DTSTART;VALUE=DATE:{start.strftime('%Y%m%d')}")
        lines.append(f"DTEND;VALUE=DATE:{end.strftime('%Y%m%d')}")
    else:
        lines.append(f"DTSTART:{_utc_stamp(start)}")
        lines.append(f"DTEND:{_utc_stamp(end)}")
    lines.append(f"SUMMARY:{_escape_ics(summary)}")
    if description:
        lines.append(f"DESCRIPTION:{_escape_ics(description)}")
    lines += ["END:VEVENT", "END:VCALENDAR", ""]
    return "\r\n".join(lines).encode("utf-8")


# ---- skill phases ----


def _ask(request, field, question):
    request.meta["create_event_awaiting"] = field
    return ActionResult(
        action_log=f"calendar.create_event: waiting for {field}",
        new_state=request.text,
        needs_input=question,
    )


def act(ctx, request):
    answers = request.meta.setdefault("create_event_answers", {})
    awaiting = request.meta.get("create_event_awaiting")
    if request.user_input is not None and awaiting:
        if answers.get(awaiting) != request.user_input.strip():
            answers[awaiting] = request.user_input.strip()
        request.meta.pop("create_event_awaiting", None)

    fields = request.meta.get("create_event_fields")
    if fields is None:
        try:
            fields = _extract_fields(ctx, request)
        except BridgeError as exc:
            return ActionResult(
                action_log=f"calendar.create_event: {exc}",
                new_state=request.text,
            )
        request.meta["create_event_fields"] = fields
    title = fields["title"]
    description = fields["description"]

    now = _now(ctx)
    parsed = _parse_when(request.text, now)

    if answers.get("title"):
        title = answers["title"].strip()
    for key in ("date", "time"):
        if answers.get(key):
            extra = _parse_when(answers[key], now)
            if extra["date"] is not None:
                parsed["date"] = extra["date"]
                parsed["from_weekday"] = extra["from_weekday"]
            if extra["time"] is not None:
                parsed["time"] = extra["time"]
            if extra["all_day"]:
                parsed["all_day"] = True
            if extra["duration_minutes"]:
                parsed["duration_minutes"] = extra["duration_minutes"]

    if not title:
        return _ask(request, "title", "What should I call the event?")
    if parsed["date"] is None:
        return _ask(request, "date", f"What day should I schedule {title!r}?")
    if not parsed["all_day"] and parsed["time"] is None:
        return _ask(request, "time", f"What time should {title!r} start? (or say 'all day')")

    local_tz = now.tzinfo or timezone.utc
    all_day = bool(parsed["all_day"])
    if all_day:
        start_date = parsed["date"]
        end_date = start_date + timedelta(days=1)
        start = datetime.combine(start_date, time(0, 0), tzinfo=local_tz)
        end = datetime.combine(end_date, time(0, 0), tzinfo=local_tz)
    else:
        start = datetime.combine(parsed["date"], parsed["time"], tzinfo=local_tz)
        if start <= now and parsed.get("from_weekday"):
            start += timedelta(days=7)
        duration = timedelta(minutes=parsed["duration_minutes"] or DEFAULT_DURATION_MINUTES)
        end = start + duration

    try:
        calendars = _discover_calendars(ctx)
    except CalDavError as exc:
        return ActionResult(
            action_log=f"calendar.create_event: calendar discovery failed: {exc}",
            new_state=request.text,
        )
    if not calendars:
        return ActionResult(
            action_log="calendar.create_event: no calendars found on the account",
            new_state=request.text,
        )

    decisions = []
    calendar = _resolve_calendar(ctx, request, calendars, decisions)

    uid = f"{uuid.uuid4().hex}@semif-agent"
    body = _build_ics(
        uid, title, start, end, all_day, _utc_stamp(now), description=description
    )
    path = _calendar_path(ctx, calendar) + urllib.parse.quote(uid, safe="") + ".ics"
    try:
        _request(
            ctx,
            "PUT",
            path,
            body=body,
            depth="0",
            content_type="text/calendar; charset=utf-8",
            extra_headers={"If-None-Match": "*"},
        )
    except CalDavError as exc:
        return ActionResult(
            action_log=(
                f"calendar.create_event: could not create {title!r} on "
                f"{calendar!r}: {exc}"
            ),
            new_state=request.text,
            decisions=decisions,
        )

    if all_day:
        when_text = f"{start_date.isoformat()} (all day)"
    else:
        when_text = (
            f"{start.strftime('%Y-%m-%d %H:%M %Z')}–"
            f"{end.strftime('%H:%M %Z')}"
        )
    report = f"Created {title!r} on calendar {calendar!r}: {when_text} (UID {uid})."
    return ActionResult(
        action_log=f"calendar.create_event: {report}",
        new_state=report,
        decisions=decisions,
    )
