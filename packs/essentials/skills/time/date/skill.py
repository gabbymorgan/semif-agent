"""Report today's date.

Real local compute: reads the host clock and formats the calendar date in the
user's timezone — the top-level `timezone` config value (an IANA name) when set,
otherwise the host's local timezone. No external service.
"""

from __future__ import annotations

from semif_agent.skills import ActionResult
from semif_agent.timers import local_now

INTEGRATION = {"service": "local_clock", "transport": "compute", "config_vars": []}

CONTRACT = {}


def act(ctx, request):
    today = local_now(ctx.config).date()
    message = f"Today is {today.strftime('%A, %B')} {today.day}, {today.year}."
    return ActionResult(action_log=message, new_state=message)
