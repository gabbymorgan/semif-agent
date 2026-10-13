"""Add a task to the user's Microsoft To Do list.

Real integration: the local Outlook bridge (Microsoft Graph). Connection comes
from `ctx.config` (`outlook_bridge_url`, `outlook_bridge_token`); the body never
speaks Graph or OAuth directly. `act` derives the task title and an optional due
date from `request.text`; a missing title is asked one at a time and the answer
is stashed in `request.meta` (act re-runs from the top after a `needs_input`
pause, so a question is never asked twice). The task is created with a real
`POST /tasks` and the real outcome reported.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

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
    r"\b(add|create|new|put|make|set\s*up|set|schedule|remind\s+me(?:\s+to)?|"
    r"i\s+need\s+to|i\s+have\s+to|tasks|task|to-?dos|to-?do|todo|lists|list|for|by|due|on|"
    r"my|the|a|an|to)\b",
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


def _match_due(text, now):
    """Return `(due_date | None, matched_span | None)` for the due date."""
    match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3))).date(), match.span()
        except ValueError:
            return None, None
    match = re.search(r"\b(" + _MONTH_ALT + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b", text, re.I)
    if match:
        month = _MONTHS.get(match.group(1).lower()[:3])
        year = int(match.group(3)) if match.group(3) else now.year
        try:
            candidate = datetime(year, month, int(match.group(2))).date()
        except (TypeError, ValueError):
            return None, None
        if not match.group(3) and candidate < now.date():
            candidate = candidate.replace(year=year + 1)
        return candidate, match.span()
    match = re.search(r"\b(today|tonight|tomorrow|tmrw|next\s+week)\b", text, re.I)
    if match:
        word = re.sub(r"\s+", " ", match.group(1).lower())
        if word in ("today", "tonight"):
            return now.date(), match.span()
        if word in ("tomorrow", "tmrw"):
            return now.date() + timedelta(days=1), match.span()
        return now.date() + timedelta(days=7), match.span()
    match = _WEEKDAY_RE.search(text)
    if match:
        target = _WEEKDAYS[match.group(1).lower()]
        date = now.date() + timedelta(days=(target - now.date().weekday()) % 7)
        return date, match.span()
    return None, None


def _title(text, span):
    stripped = text
    if span is not None:
        stripped = text[: span[0]] + " " + text[span[1] :]
    cleaned = re.sub(r"\s+", " ", _COMMAND_WORDS.sub(" ", stripped)).strip(" .,:;-")
    return cleaned


def _ask(request, field, question):
    request.meta["create_task_awaiting"] = field
    return ActionResult(action_log=f"waiting for {field}", new_state=request.text, needs_input=question)


def act(ctx, request):
    answers = request.meta.setdefault("create_task_answers", {})
    awaiting = request.meta.get("create_task_awaiting")
    if request.user_input is not None and awaiting:
        answers[awaiting] = request.user_input.strip()
        request.meta.pop("create_task_awaiting", None)

    now = _now(ctx)
    due, span = _match_due(request.text, now)
    title = _title(request.text, span)

    if answers.get("title"):
        title = answers["title"].strip()
    if answers.get("due"):
        extra_due, _ = _match_due(answers["due"], now)
        if extra_due is not None:
            due = extra_due

    if not title:
        return _ask(request, "title", "What's the task?")

    payload = {"title": title}
    if due is not None:
        payload["due"] = f"{due.isoformat()}T00:00:00"

    try:
        data = _post(ctx, "/tasks", payload)
    except OutlookBridgeError as exc:
        return ActionResult(action_log=str(exc), new_state=request.text)

    when = f" due {due.isoformat()}" if due is not None else ""
    report = f"Added task {title!r}{when}."
    return ActionResult(action_log=report, new_state=report)
