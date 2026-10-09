"""Report the current local time.

Real local compute: reads the host clock and formats it in the user's
timezone — the top-level `timezone` config value (an IANA name such as
`America/Chicago`) when set, otherwise the host's local timezone. No external
service; the `compute` transport is the codebase's home for a body that acts on
the machine itself.
"""

from __future__ import annotations

from semif_agent.skills import ActionResult
from semif_agent.timers import local_now

INTEGRATION = {"service": "local_clock", "transport": "compute", "config_vars": []}

CONTRACT = {}


def act(ctx, request):
    now = local_now(ctx.config)
    message = f"It is {now.strftime('%I:%M %p').lstrip('0')}."
    return ActionResult(action_log=message, new_state=message)
