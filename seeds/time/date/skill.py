"""Report today's date.

Real local compute: reads the host clock and formats the local calendar date.
No external service and no configuration.
"""

from __future__ import annotations

from datetime import datetime

from semif_agent.skills import ActionResult

INTEGRATION = {"service": "local_clock", "transport": "compute", "config_vars": []}

CONTRACT = {}


def act(ctx, request):
    today = datetime.now().astimezone().date()
    message = f"Today is {today.strftime('%A, %B %d, %Y')} ({today.isoformat()})."
    return ActionResult(action_log=f"time.date: {message}", new_state=message)
