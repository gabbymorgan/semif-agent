"""Create a calendar event on the user's Nextcloud calendar from a request.

Real integration: CalDAV against the user's Nextcloud (SabreDAV). Connection
details and the default calendar come from `ctx.config` (`nextcloud_url`,
`nextcloud_username`, `nextcloud_app_password`, `nextcloud_default_calendar`).

`act` reads `request.text` and derives the event's per-request arguments — the
title, the start date/time, the duration, and (when named) the target calendar.
Missing arguments are asked one at a time and the answer is stashed in
`request.meta`; because `act` is single-phase and re-runs from the top after a
`needs_input` pause, the body re-parses the query each pass and never re-asks a
question it already has an answer for. Calendars are discovered over PROPFIND and
a named target is resolved with a SemIf sub-decision when more than one matches;
otherwise the configured default calendar is used. The event is created with a
real CalDAV PUT (iCalendar, UTC timestamps, `If-None-Match: *`) and the real
outcome is reported.

Parsing is deliberately bounded and stdlib-only (no text generation is available
to a skill body): when a required field cannot be derived confidently the body
asks rather than guessing.
"""

from __future__ import annotations

import base64
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, time, timedelta, timezone

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
    "nextcloud_username": "The Nextcloud username whose calendars are used.",
    "nextcloud_app_password": "A Nextcloud app password (Settings > Security > Devices & sessions) used for CalDAV Basic auth; the account password only works when two-factor auth is off.",
    "nextcloud_default_calendar": "Name of the calendar new events go to by default, e.g. personal (a request may still name another calendar).",
}

DAV = "DAV:"
CALDAV = "urn:ietf:params:xml:ns:caldav"
TIMEOUT_SECONDS = 30
DEFAULT_DURATION_MINUTES = 60

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

_LEADING_NOISE = re.compile(
    r"^\s*(please\s+)?(add|create|schedule|put|make|set\s+up|new|insert|book|a|an|the)\b[\s:,-]*",
    re.I,
)
_TITLE_NOISE = [
    re.compile(r"\b(to|on|in)\s+(my|the)\s+[A-Za-z0-9_'-]+\s+calendar\b", re.I),
    re.compile(r"\b(to|on|in)\s+(my|the)\s+calendar\b", re.I),
    re.compile(r"\b(new\s+|calendar\s+)*event\b", re.I),
    re.compile(r"\b(called|titled|named)\b", re.I),
]
_TRAILING_NOISE = re.compile(
    r"[\s,;:-]*\b(for|on|at|to|the|my|a|an|in|of)\b[\s,;:-]*$", re.I
)


class CalDavError(Exception):
    """A real failure talking to the calendar service."""


def _now():
    """Current local time (aware). A test seam; the body never freezes time."""
    return datetime.now().astimezone()


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


# ---- request -> arguments ----


def _remove_spans(text, spans):
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + " " + text[end:]
    return text


def _clean_title(text):
    text = re.sub(r"\s{2,}", " ", text).strip()
    previous = None
    while previous != text:
        previous = text
        text = _LEADING_NOISE.sub("", text)
        for pattern in _TITLE_NOISE:
            text = pattern.sub(" ", text)
        text = _TRAILING_NOISE.sub("", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
    return text.strip(" \t,;:-.")


def _extract_title(text, spans):
    return _clean_title(_remove_spans(text or "", spans))


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
    None), `all_day` (bool), `duration_minutes` (int | None), `from_weekday`
    (bool) and `spans` (matched ranges, so the title can be cleaned).
    """
    text = text or ""
    result = {
        "date": None,
        "time": None,
        "all_day": False,
        "duration_minutes": None,
        "from_weekday": False,
    }
    spans = []

    match = re.search(r"\ball[\s-]?day\b|\bwhole\s+day\b", text, re.I)
    if match:
        result["all_day"] = True
        spans.append(match.span())

    match = re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b", text, re.I)
    if match:
        result["duration_minutes"] = int(round(float(match.group(1)) * 60))
        spans.append(match.span())
    else:
        match = re.search(r"\bfor\s+(\d+)\s*(minutes?|mins?|m)\b", text, re.I)
        if match:
            result["duration_minutes"] = int(match.group(1))
            spans.append(match.span())
        else:
            for phrase, minutes in (
                (r"\bfor\s+half\s+an?\s+hour\b", 30),
                (r"\bfor\s+an?\s+hour\b", 60),
                (r"\bfor\s+a\s+couple\s+of\s+hours\b", 120),
            ):
                match = re.search(phrase, text, re.I)
                if match:
                    result["duration_minutes"] = minutes
                    spans.append(match.span())
                    break

    date = None
    match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if match:
        try:
            date = datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3))
            ).date()
            spans.append(match.span())
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
            if date is not None:
                spans.append(match.span())

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
            if date is not None:
                spans.append(match.span())

    if date is None:
        match = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text)
        if match:
            first, second = int(match.group(1)), int(match.group(2))
            month, day = (first, second) if first <= 12 else (second, first)
            year = int(match.group(3)) if match.group(3) else None
            if year is not None and year < 100:
                year += 2000
            date = _resolve_month_day(month, day, year, now)
            if date is not None:
                spans.append(match.span())

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
            spans.append(match.span())

    if date is None:
        match = _WEEKDAY_RE.search(text)
        if match:
            target = _WEEKDAYS[match.group(1).lower()]
            date = now.date() + timedelta(days=(target - now.date().weekday()) % 7)
            prefix = text[max(0, match.start() - 6):match.start()]
            if re.search(r"\bnext\s*$", prefix, re.I):
                date += timedelta(days=7)
                spans.append((match.start() - len("next "), match.end()))
            else:
                spans.append(match.span())
            result["from_weekday"] = True

    parsed_time = None
    match = re.search(r"\bnoon\b", text, re.I)
    if match:
        parsed_time = time(12, 0)
        spans.append(match.span())
    elif re.search(r"\bmidnight\b", text, re.I):
        parsed_time = time(0, 0)
        spans.append(re.search(r"\bmidnight\b", text, re.I).span())
    else:
        match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\b", text, re.I)
        if match:
            hour = int(match.group(1)) % 12
            minute = int(match.group(2) or 0)
            if match.group(3).lower() == "p":
                hour += 12
            if hour < 24 and minute < 60:
                parsed_time = time(hour, minute)
                spans.append(match.span())
        else:
            match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
            if match:
                hour, minute = int(match.group(1)), int(match.group(2))
                if hour < 24 and minute < 60:
                    parsed_time = time(hour, minute)
                    spans.append(match.span())
            else:
                match = re.search(r"\bat\s+(\d{1,2})\b", text, re.I)
                if match and 13 <= int(match.group(1)) < 24:
                    parsed_time = time(int(match.group(1)), 0)
                    spans.append(match.span())

    result["date"] = date
    result["time"] = parsed_time
    result["spans"] = spans
    return result


# ---- calendar resolution ----


def _mentioned_calendars(calendars, text):
    low = (text or "").lower()
    found = []
    for calendar in calendars:
        for token in {calendar["name"].lower(), calendar["label"].lower()}:
            if len(token) < 3:
                continue
            pattern = (
                rf"\b{re.escape(token)}\b\s+calendar\b"
                rf"|\bcalendar\b\s+\b{re.escape(token)}\b"
            )
            if re.search(pattern, low):
                found.append(calendar)
                break
    return found


def _resolve_calendar(ctx, request, calendars, decisions):
    default_name = str(ctx.config.get("nextcloud_default_calendar", "") or "").strip().lower()
    default = None
    for calendar in calendars:
        if default_name and default_name in (calendar["name"].lower(), calendar["label"].lower()):
            default = calendar
            break

    mentioned = _mentioned_calendars(calendars, request.text)
    if len(mentioned) == 1:
        return mentioned[0]["name"]
    if len(mentioned) > 1:
        return _choose_calendar(ctx, request, mentioned, default, decisions)
    if default is not None:
        return default["name"]
    if len(calendars) == 1:
        return calendars[0]["name"]
    return _choose_calendar(ctx, request, calendars, default, decisions)


def _choose_calendar(ctx, request, candidates, default, decisions):
    options = []
    for calendar in candidates:
        label = calendar["label"]
        if default is not None and calendar["name"] == default["name"]:
            label = f"{label} (default)"
        options.append(Option(calendar["name"], label))
    decision = DecisionRequest(
        state=request.text,
        question="Which calendar should the event go on?",
        options=options,
    )
    result = ctx.engine.call(decision)
    decisions.append((decision, result))
    return result.selected


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

    now = _now()
    parsed = _parse_when(request.text, now)
    title = _extract_title(request.text, parsed["spans"])

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
    body = _build_ics(uid, title, start, end, all_day, _utc_stamp(now))
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
