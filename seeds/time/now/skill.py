"""Report the current local time.

Real local compute: reads the host clock and formats it in the host's local
timezone. No external service and no configuration — the `compute` transport is
the codebase's home for a body that acts on the machine itself.
"""

from __future__ import annotations

from datetime import datetime

from semif_agent.skills import ActionResult

INTEGRATION = {"service": "local_clock", "transport": "compute", "config_vars": []}

CONTRACT = {}


def _offset_text(now: datetime) -> str:
    offset = now.strftime("%z")
    if not offset:
        return ""
    return f" UTC{offset[:3]}:{offset[3:]}"


def act(ctx, request):
    now = datetime.now().astimezone()
    zone = now.tzname() or "local"
    message = (
        f"It is {now.strftime('%H:%M')} "
        f"({now.strftime('%I:%M %p').lstrip('0')} {zone}{_offset_text(now)})."
    )
    return ActionResult(action_log=f"time.now: {message}", new_state=message)
