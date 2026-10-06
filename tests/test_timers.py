"""Pure-stdlib tests for the in-process timer/alarm service.

The service is real local compute: a background thread fires on the host clock.
These tests use sub-second durations so they stay fast, and they assert the
mechanics (fire, drain, cancel, trace, honest message) rather than any external
service. Skill bodies that schedule timers are covered by the seed tests.
"""

import time
from datetime import datetime, timedelta, timezone

from semif_agent.timers import TimerService, format_duration
from semif_agent.trace import TraceLog


def wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_format_duration():
    assert format_duration(1) == "1 second"
    assert format_duration(30) == "30 seconds"
    assert format_duration(60) == "1 minute"
    assert format_duration(90) == "1 minute 30 seconds"
    assert format_duration(3600) == "1 hour"
    assert format_duration(5400) == "1 hour 30 minutes"


def test_timer_fires_and_drains_once(tmp_path):
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    service = TimerService(trace=trace)
    try:
        timer = service.set_timer(0.1, "0.1 seconds", run_id="r1", source="typed")
        assert [t["id"] for t in service.pending()] == [timer.id]

        assert wait_for(lambda: service.fired()), "the timer must fire"
        fired = service.drain()
        assert len(fired) == 1
        assert fired[0].run_id == "r1"
        assert fired[0].kind == "timer"
        assert "0.1 seconds" in fired[0].message

        assert service.drain() == [], "drain is one-shot"
        assert service.pending() == []

        events = [e for e in trace.read() if e["kind"] == "timer_fired"]
        assert events, "firing must be traced"
        assert events[0]["timer_id"] == timer.id
        assert events[0]["timer_kind"] == "timer"
        assert events[0]["run_id"] == "r1"
    finally:
        service.stop()


def test_alarm_fires():
    service = TimerService()
    try:
        at = datetime.now(timezone.utc) + timedelta(seconds=0.1)
        service.set_alarm(at, "5:00", run_id="r2")
        assert wait_for(lambda: service.fired()), "the alarm must fire"
        fired = service.fired()
        assert fired[0]["kind"] == "alarm"
        assert "Alarm" in fired[0]["message"]
    finally:
        service.stop()


def test_cancel_prevents_firing():
    service = TimerService()
    try:
        timer = service.set_timer(0.2, "cancel me")
        assert service.cancel(timer.id) is True
        assert service.cancel(timer.id) is False
        time.sleep(0.3)
        assert service.drain() == []
        assert service.pending() == []
    finally:
        service.stop()


def test_non_positive_duration_rejected():
    service = TimerService()
    try:
        try:
            service.set_timer(0, "nope")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for a non-positive duration")
    finally:
        service.stop()


def test_resolve_timezone_falls_back_to_host_local():
    from zoneinfo import ZoneInfo

    from semif_agent.timers import resolve_timezone

    assert resolve_timezone(None) is None
    assert resolve_timezone({}) is None
    assert resolve_timezone({"timezone": ""}) is None
    assert resolve_timezone({"timezone": "   "}) is None
    assert resolve_timezone({"timezone": "Not/AZone"}) is None
    assert resolve_timezone({"timezone": "America/Chicago"}) == ZoneInfo("America/Chicago")


def test_local_now_uses_configured_timezone():
    from semif_agent.timers import local_now

    now = local_now({"timezone": "America/Chicago"})
    assert now.tzinfo is not None
    assert now.utcoffset() in (timedelta(hours=-5), timedelta(hours=-6))
    # No configured zone -> an aware host-local time, never naive.
    assert local_now({}).tzinfo is not None


def test_timer_due_and_fired_time_use_configured_timezone():
    from zoneinfo import ZoneInfo

    service = TimerService(timezone="America/Chicago")
    try:
        timer = service.set_timer(3600, "1 hour")
        assert timer.due_datetime().tzinfo == ZoneInfo("America/Chicago")
        # 1700000000 == 2023-11-14 22:13:20 UTC == 16:13:20 in Chicago (CST).
        message = service._message(timer, 1700000000)
        expected = datetime.fromtimestamp(1700000000, tz=ZoneInfo("America/Chicago")).strftime("%H:%M")
        assert expected in message, message
    finally:
        service.stop()
