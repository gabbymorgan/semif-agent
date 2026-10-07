"""Set a countdown timer that notifies the user when it finishes.

Real local compute: parses a duration out of the request (digits or words) and
schedules it on the in-process timer service (`ctx.timers`). When it fires, the
front end that set it notifies the user — the REPL prints it, the gateway sends
it back to the originating chat, the dashboard shows it. The due time is shown
in the user's timezone (the top-level `timezone` config value, else host local).
A timer lives in the process that set it and does not survive a restart.

If no duration can be parsed the body pauses and asks for one, stashing the
answer in `request.meta` so the single-phase `act` re-run does not ask again.
"""

from __future__ import annotations

import re

from semif_agent.skills import ActionResult
from semif_agent.timers import format_duration

INTEGRATION = {"service": "local_clock", "transport": "compute", "config_vars": []}

CONTRACT = {}

_DURATION_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fortyfive": 45,
    "sixty": 60,
    "ninety": 90,
}

_UNIT_SECONDS = {
    "second": 1,
    "seconds": 1,
    "sec": 1,
    "secs": 1,
    "s": 1,
    "minute": 60,
    "minutes": 60,
    "min": 60,
    "mins": 60,
    "m": 60,
    "hour": 3600,
    "hours": 3600,
    "hr": 3600,
    "hrs": 3600,
    "h": 3600,
}

_DIGIT_DURATION = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b",
    re.I,
)
_WORD_DURATION = re.compile(
    r"\b(" + "|".join(sorted(_DURATION_WORDS, key=len, reverse=True)) + r")\s+"
    r"(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
    re.I,
)


def _parse_duration(text):
    """Seconds for a duration phrase, or None when none is present."""
    text = text or ""
    if re.search(r"\bhalf\s+an?\s+hour\b", text, re.I):
        return 1800
    if re.search(r"\bquarter\s+(?:of\s+)?an?\s+hour\b", text, re.I):
        return 900
    if re.search(r"\ba\s+couple\s+of\s+(?:minutes?|mins?)\b", text, re.I):
        return 120
    if re.search(r"\ba\s+few\s+(?:minutes?|mins?)\b", text, re.I):
        return 180
    match = _DIGIT_DURATION.search(text)
    if match:
        unit = _UNIT_SECONDS.get(match.group(2).lower())
        if unit:
            return int(round(float(match.group(1)) * unit))
    match = _WORD_DURATION.search(text)
    if match:
        value = _DURATION_WORDS.get(match.group(1).lower())
        unit = _UNIT_SECONDS.get(match.group(2).lower())
        if value and unit:
            return value * unit
    return None


def act(ctx, request):
    if request.user_input is not None and request.meta.get("timer_awaiting"):
        if request.user_input.strip():
            request.meta["timer_duration_text"] = request.user_input.strip()
        request.meta.pop("timer_awaiting", None)

    text = request.meta.get("timer_duration_text") or request.text
    seconds = _parse_duration(text)
    if seconds is None:
        request.meta["timer_awaiting"] = True
        return ActionResult(
            action_log="waiting for a duration",
            new_state=request.text,
            needs_input="How long should the timer run? (e.g. '5 minutes')",
        )
    if ctx.timers is None:
        return ActionResult(
            action_log="no timer service is available in this front end",
            new_state=request.text,
        )

    label = format_duration(seconds)
    timer = ctx.timers.set_timer(seconds, label, run_id=request.id, source=request.source)
    due = timer.due_datetime().strftime("%H:%M")
    message = f"Timer set for {label}; it will fire at {due}."
    return ActionResult(action_log=message, new_state=message)
