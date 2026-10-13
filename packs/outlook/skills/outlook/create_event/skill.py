"""Create an event on the user's Outlook calendar from a request.

Real integration: the local Outlook bridge (Microsoft Graph). Connection comes
from `ctx.config` (`outlook_bridge_url`, `outlook_bridge_token`); the body never
speaks Graph or OAuth directly. `act` derives the event's title, date, time and
duration from `request.text`; missing pieces are asked one at a time and the
answers are stashed in `request.meta` (act re-runs from the top after a
`needs_input` pause, so a question is never asked twice). The event is created
with a real `POST /calendars/events` and the real outcome reported. The user's
IANA timezone (the top-level `timezone` config value, when set) labels the start.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time, timedelta

from semif_agent.skills import ActionResult
from semif_agent.timers import local_now

INTEGRATION = {
    "service": "outlook",
    "transport": "http",
    "config_vars": ["outlook_bridge_url", "outlook_bridge_token"],
}

CONTRACT = {
    "outlook_bridge_url": "Base URL of the local Outlook bridge, e.g. http://127.0.0.1:5231 (no trailing slash needed).",
    "outlook_bridge_token": "Shared secret for the Outlook bridge, if one is configured; sent as the X-Semif-Token header. Leave blank when the bridge requires no auth.",
}

TIMEOUT_SECONDS = 30
DEFAULT_DURATION_MINUTES = 60

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
_COMMAND_WORDS = re.compile(
    r"\b(add|create|new|put|make|set\s*up|set|schedule|book|"
    r"i\s+need\s+to|i\s+have\s+to|calendar|"
    r"for|on|at|from|my|the|a|an|to)\b",
    re.I,
)


class OutlookBridgeError(Exception):
    """A real failure talking to the local Outlook bridge."""


def _now(ctx):
    return local_now(ctx.config)


def _post(ctx, path, payload):
    base = str(ctx.config.get("outlook_bridge_url", "") or "").strip().rstrip("/")
    if not base:
        raise OutlookBridgeError("outlook_bridge_url is not configured")
    token = str(ctx.config.get("outlook_bridge_token", "") or "").strip()
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["X-Semif-Token"] = token
    http_request = urllib.request.Request(base + path, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(http_request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()[:200]
        raise OutlookBridgeError(f"POST {path}: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise OutlookBridgeError(f"POST {path}: {exc}") from exc


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
    """`{"date","date_span","time","time_span","all_day","duration_minutes","from_weekday"}`."""
    text = text or ""
    result = {
        "date": None,
        "date_span": None,
        "time": None,
        "time_span": None,
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
            ):
                match = re.search(phrase, text, re.I)
                if match:
                    result["duration_minutes"] = minutes
                    break

    date = None
    span = None
    match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if match:
        try:
            date = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3))).date()
            span = match.span()
        except ValueError:
            date = None
    if date is None:
        match = re.search(
            r"\b(" + _MONTH_ALT + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b", text, re.I
        )
        if match:
            month = _month_number(match.group(1))
            year = int(match.group(3)) if match.group(3) else None
            date = _resolve_month_day(month, int(match.group(2)), year, now)
            span = match.span() if date else None
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
            span = match.span() if date else None
    if date is None:
        match = re.search(r"\b(today|tonight|tomorrow|tmrw|tomorow|next\s+week)\b", text, re.I)
        if match:
            word = re.sub(r"\s+", " ", match.group(1).lower())
            if word in ("today", "tonight"):
                date = now.date()
            elif word in ("tomorrow", "tmrw", "tomorow"):
                date = now.date() + timedelta(days=1)
            else:
                date = now.date() + timedelta(days=7)
            span = match.span()
    if date is None:
        match = _WEEKDAY_RE.search(text)
        if match:
            target = _WEEKDAYS[match.group(1).lower()]
            date = now.date() + timedelta(days=(target - now.date().weekday()) % 7)
            prefix = text[max(0, match.start() - 6) : match.start()]
            if re.search(r"\bnext\s*$", prefix, re.I):
                date += timedelta(days=7)
            result["from_weekday"] = True
            span = match.span()

    parsed_time = None
    time_span = None
    match = re.search(r"\bnoon\b", text, re.I)
    if match:
        parsed_time = time(12, 0)
        time_span = match.span()
    elif re.search(r"\bmidnight\b", text, re.I):
        parsed_time = time(0, 0)
        time_span = re.search(r"\bmidnight\b", text, re.I).span()
    else:
        match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\b", text, re.I)
        if match:
            hour = int(match.group(1)) % 12
            minute = int(match.group(2) or 0)
            if match.group(3).lower() == "p":
                hour += 12
            if hour < 24 and minute < 60:
                parsed_time = time(hour, minute)
                time_span = match.span()
        else:
            match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
            if match:
                hour, minute = int(match.group(1)), int(match.group(2))
                if hour < 24 and minute < 60:
                    parsed_time = time(hour, minute)
                    time_span = match.span()

    result["date"] = date
    result["date_span"] = span
    result["time"] = parsed_time
    result["time_span"] = time_span
    return result


def _title(text, parsed):
    remove = [parsed.get("date_span"), parsed.get("time_span")]
    out = text
    for span in sorted((s for s in remove if s), key=lambda s: s[0], reverse=True):
        out = out[: span[0]] + " " + out[span[1] :]
    phrase = re.search(r"\ball[\s-]?day\b|\bwhole\s+day\b", out, re.I)
    if phrase:
        out = out[: phrase.start()] + " " + out[phrase.end() :]
    cleaned = re.sub(r"\s+", " ", _COMMAND_WORDS.sub(" ", out)).strip(" .,:;-")
    return cleaned


def _ask(request, field, question):
    request.meta["create_event_awaiting"] = field
    return ActionResult(action_log=f"waiting for {field}", new_state=request.text, needs_input=question)


def act(ctx, request):
    answers = request.meta.setdefault("create_event_answers", {})
    awaiting = request.meta.get("create_event_awaiting")
    if request.user_input is not None and awaiting:
        answers[awaiting] = request.user_input.strip()
        request.meta.pop("create_event_awaiting", None)

    now = _now(ctx)
    parsed = _parse_when(request.text, now)
    title = _title(request.text, parsed)

    if answers.get("title"):
        title = answers["title"].strip()
    for key in ("date", "time"):
        if answers.get(key):
            extra = _parse_when(answers[key], now)
            if extra["date"] is not None:
                parsed["date"] = extra["date"]
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

    all_day = bool(parsed["all_day"])
    if all_day:
        start = datetime.combine(parsed["date"], time(0, 0))
    else:
        start = datetime.combine(parsed["date"], parsed["time"])
        if start <= now.replace(tzinfo=None) and parsed.get("from_weekday"):
            start += timedelta(days=7)

    timezone_name = str(ctx.config.get("timezone", "") or "").strip()
    payload = {
        "subject": title,
        "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "all_day": all_day,
    }
    if not all_day:
        duration = timedelta(minutes=parsed["duration_minutes"] or DEFAULT_DURATION_MINUTES)
        payload["end"] = (start + duration).strftime("%Y-%m-%dT%H:%M:%S")
    if timezone_name:
        payload["timezone"] = timezone_name

    try:
        data = _post(ctx, "/calendars/events", payload)
    except OutlookBridgeError as exc:
        return ActionResult(action_log=str(exc), new_state=request.text)

    if all_day:
        when_text = f"{start.date().isoformat()} (all day)"
    else:
        when_text = start.strftime("%Y-%m-%d %H:%M")
    report = f"Created {title!r} on {when_text}."
    return ActionResult(action_log=report, new_state=report)
