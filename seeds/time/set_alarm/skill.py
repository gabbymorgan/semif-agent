"""Set an alarm for a specific time of day that notifies when it goes off.

Real local compute: parses a wall-clock time out of the request and schedules it
on the in-process timer service (`ctx.timers`). The wall-clock time is read in
the user's timezone — the top-level `timezone` config value when set, otherwise
the host's local timezone. A time already past rolls to the next day. When it
fires, the front end that set it notifies the user.

If no time can be parsed the body pauses and asks for one, stashing the answer
in `request.meta` so the single-phase `act` re-run does not ask again.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta

from semif_agent.skills import ActionResult
from semif_agent.timers import local_now

INTEGRATION = {"service": "local_clock", "transport": "compute", "config_vars": []}

CONTRACT = {}


def _parse_time(text, now):
    """The next datetime matching a wall-clock time in `text`, or None."""
    text = text or ""
    parsed = None
    if re.search(r"\bnoon\b", text, re.I):
        parsed = time(12, 0)
    elif re.search(r"\bmidnight\b", text, re.I):
        parsed = time(0, 0)
    else:
        match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\b", text, re.I)
        if match:
            hour = int(match.group(1)) % 12
            minute = int(match.group(2) or 0)
            if match.group(3).lower() == "p":
                hour += 12
            if hour < 24 and minute < 60:
                parsed = time(hour, minute)
        else:
            match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
            if match:
                hour, minute = int(match.group(1)), int(match.group(2))
                if hour < 24 and minute < 60:
                    parsed = time(hour, minute)
            else:
                # A bare hour is only accepted when unambiguous (24-hour).
                match = re.search(r"\bat\s+(\d{1,2})\b", text, re.I)
                if match and 13 <= int(match.group(1)) < 24:
                    parsed = time(int(match.group(1)), 0)
    if parsed is None:
        return None
    candidate = datetime.combine(now.date(), parsed, tzinfo=now.tzinfo)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def act(ctx, request):
    if request.user_input is not None and request.meta.get("alarm_awaiting"):
        if request.user_input.strip():
            request.meta["alarm_time_text"] = request.user_input.strip()
        request.meta.pop("alarm_awaiting", None)

    text = request.meta.get("alarm_time_text") or request.text
    now = local_now(ctx.config)
    when = _parse_time(text, now)
    if when is None:
        request.meta["alarm_awaiting"] = True
        return ActionResult(
            action_log="waiting for a time",
            new_state=request.text,
            needs_input="What time should the alarm go off? (e.g. '5pm')",
        )
    if ctx.timers is None:
        return ActionResult(
            action_log="no timer service is available in this front end",
            new_state=request.text,
        )

    label = when.strftime("%H:%M")
    ctx.timers.set_alarm(when, label, run_id=request.id, source=request.source)
    message = f"Alarm set for {when.strftime('%H:%M')} on {when.strftime('%A')}."
    return ActionResult(action_log=message, new_state=message)
