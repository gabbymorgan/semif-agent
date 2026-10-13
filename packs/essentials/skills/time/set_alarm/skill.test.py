"""Hermetic mechanics test for time.set_alarm.

Run from this folder: `python skill.test.py`. No external network, no real
sleeping: a fake timer service records what the body schedules. This proves the
body parses wall-clock times (am/pm, 24-hour, noon/midnight), rolls a past time
to the next day, and pauses for input when no time is present.
"""

import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

from semif_agent.decisions import Request
from semif_agent.skills import ActionContext

import skill


class FakeTimers:
    def __init__(self):
        self.alarms = []

    def set_alarm(self, at, label, run_id="", source=""):
        self.alarms.append((at, label))
        return SimpleNamespace(id="a1", due_at=at.timestamp())


def ctx_with_timers():
    timers = FakeTimers()
    return ActionContext(engine=None, config={}, timers=timers), timers


def main():
    now = datetime.now().astimezone()

    ctx, timers = ctx_with_timers()
    action = skill.act(ctx, Request("set an alarm for 5pm"))
    assert action.needs_input is None, action.needs_input
    assert len(timers.alarms) == 1, timers.alarms
    at, label = timers.alarms[0]
    assert (at.hour, at.minute) == (17, 0), at
    assert label == "17:00"
    assert at > now, "an alarm must be in the future"
    assert at <= now + timedelta(days=1)
    print(action.action_log)

    ctx, timers = ctx_with_timers()
    skill.act(ctx, Request("set an alarm for 17:30"))
    assert timers.alarms[0][0].strftime("%H:%M") == "17:30", timers.alarms

    ctx, timers = ctx_with_timers()
    skill.act(ctx, Request("wake me at noon"))
    assert timers.alarms[0][0].strftime("%H:%M") == "12:00", timers.alarms

    ctx, timers = ctx_with_timers()
    skill.act(ctx, Request("alarm at midnight"))
    assert timers.alarms[0][0].strftime("%H:%M") == "00:00", timers.alarms

    # An unparseable request pauses, then the answer is used on the re-run.
    ctx, timers = ctx_with_timers()
    request = Request("set an alarm")
    action = skill.act(ctx, request)
    assert action.needs_input, "an alarm with no time must ask for one"
    assert timers.alarms == []
    request.user_input = "6:15am"
    action = skill.act(ctx, request)
    assert action.needs_input is None
    assert timers.alarms[0][0].strftime("%H:%M") == "06:15", timers.alarms
    print(action.action_log)

    # No timer service fails honestly.
    ctx = ActionContext(engine=None, config={}, timers=None)
    action = skill.act(ctx, Request("set an alarm for 5pm"))
    assert "no timer service" in action.action_log, action.action_log

    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
