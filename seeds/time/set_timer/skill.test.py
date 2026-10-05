"""Hermetic mechanics test for time.set_timer.

Run from this folder: `python skill.test.py`. No external network, no real
sleeping: a fake timer service records what the body schedules. This proves the
body parses durations (digits and words) and schedules the right number of
seconds, and that an unparseable request pauses for input; it does NOT prove
the in-process timer actually fires (tests/test_timers.py covers the service).
"""

import sys
import time
from datetime import datetime
from types import SimpleNamespace

from semif_agent.decisions import Request
from semif_agent.skills import ActionContext

import skill


class FakeTimer:
    def __init__(self, seconds):
        self.due_at = time.time() + seconds

    def due_datetime(self):
        return datetime.fromtimestamp(self.due_at).astimezone()


class FakeTimers:
    def __init__(self):
        self.timers = []

    def set_timer(self, seconds, label, run_id="", source=""):
        self.timers.append((seconds, label))
        return FakeTimer(seconds)


def ctx_with_timers():
    timers = FakeTimers()
    return ActionContext(engine=None, config={}, timers=timers), timers


def main():
    ctx, timers = ctx_with_timers()
    action = skill.act(ctx, Request("please set a timer for five minutes"))
    assert action.needs_input is None, action.needs_input
    assert timers.timers == [(300, "5 minutes")], timers.timers
    assert "300" not in action.action_log and "5 minutes" in action.action_log
    print(action.action_log)

    ctx, timers = ctx_with_timers()
    action = skill.act(ctx, Request("set a timer for 90 seconds"))
    assert timers.timers == [(90, "1 minute 30 seconds")], timers.timers
    print(action.action_log)

    ctx, timers = ctx_with_timers()
    skill.act(ctx, Request("timer for 1.5 hours"))
    assert timers.timers == [(5400, "1 hour 30 minutes")], timers.timers

    ctx, timers = ctx_with_timers()
    skill.act(ctx, Request("set a timer for half an hour"))
    assert timers.timers == [(1800, "30 minutes")], timers.timers

    # An unparseable request pauses, then the answer is used on the re-run.
    ctx, timers = ctx_with_timers()
    request = Request("set a timer")
    action = skill.act(ctx, request)
    assert action.needs_input, "a timer with no duration must ask for one"
    assert timers.timers == []
    request.user_input = "ten minutes"
    action = skill.act(ctx, request)
    assert action.needs_input is None
    assert timers.timers == [(600, "10 minutes")], timers.timers
    print(action.action_log)

    # No timer service (e.g. a front end without one) fails honestly.
    ctx = ActionContext(engine=None, config={}, timers=None)
    action = skill.act(ctx, Request("set a timer for 5 minutes"))
    assert action.needs_input is None
    assert "no timer service" in action.action_log, action.action_log

    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
