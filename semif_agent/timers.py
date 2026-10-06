"""In-process timer and alarm service.

A skill body schedules a timer or alarm through `ctx.timers`. One background
daemon thread wakes at the earliest due time, records a `timer_fired` trace
event, and exposes the fired timer so a front end can notify the user (the REPL
prints it, the gateway routes it to the originating chat, the dashboard shows
it). Timers live in the process that set them and do **not** survive a restart
(persistence is future work) — so the agent never claims a timer outlived a
shutdown it did not.

This is real local compute, not a simulation: the thread fires on the host
clock and the notification carries the real due/fired times.
"""

from __future__ import annotations

import heapq
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def resolve_timezone(config) -> tzinfo | None:
    """The tzinfo named by `config["timezone"]`, or None for host local.

    An empty/absent name or an unknown zone falls back to the host's local
    timezone (None), so a misconfigured value never breaks a run.
    """
    name = str((config or {}).get("timezone", "") or "").strip()
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return None


def local_now(config=None) -> datetime:
    """Current time, aware: the configured timezone, else the host's local zone."""
    tz = resolve_timezone(config)
    if tz is None:
        return datetime.now().astimezone()
    return datetime.now(tz)


def format_duration(seconds: float) -> str:
    """Human text for a duration, e.g. 300 -> '5 minutes', 5400 -> '1 hour 30 minutes'."""
    total = int(round(float(seconds)))
    if total < 60:
        return f"{total} second" + ("s" if total != 1 else "")
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        text = f"{minutes} minute" + ("s" if minutes != 1 else "")
        if secs:
            text += f" {secs} second" + ("s" if secs != 1 else "")
        return text
    hours, minutes = divmod(minutes, 60)
    text = f"{hours} hour" + ("s" if hours != 1 else "")
    if minutes:
        text += f" {minutes} minute" + ("s" if minutes != 1 else "")
    return text


@dataclass
class Timer:
    """One scheduled timer or alarm."""

    id: str
    run_id: str
    kind: str  # "timer" | "alarm"
    label: str
    source: str
    due_at: float  # epoch seconds
    created_at: float = field(default_factory=time.time)
    # tzinfo used to format the due time for the user (None = host local).
    tz: tzinfo | None = None

    def due_datetime(self) -> datetime:
        if self.tz is not None:
            return datetime.fromtimestamp(self.due_at, tz=self.tz)
        return datetime.fromtimestamp(self.due_at).astimezone()

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "kind": self.kind,
            "label": self.label,
            "source": self.source,
            "due_at": self.due_at,
            "due_at_local": self.due_datetime().isoformat(),
            "created_at": self.created_at,
        }


@dataclass
class FiredTimer:
    """A timer that has fired; the notification a front end delivers."""

    id: str
    run_id: str
    kind: str
    label: str
    source: str
    due_at: float
    fired_at: float
    message: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "kind": self.kind,
            "label": self.label,
            "source": self.source,
            "due_at": self.due_at,
            "fired_at": self.fired_at,
            "message": self.message,
        }


class TimerService:
    """Owns scheduled timers/alarms and the single firing thread.

    Thread-safe: bodies schedule from the scheduler thread while the firing
    thread delivers. `drain()` hands fired timers to the active front end once;
    `fired()` keeps a bounded history for the dashboard.
    """

    def __init__(self, trace=None, history: int = 50, timezone: str | None = None):
        self.trace = trace
        self._history_limit = int(history)
        self._tz = resolve_timezone({"timezone": timezone})
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._heap: list[tuple[float, int, Timer]] = []
        self._seq = 0
        self._timers: dict[str, Timer] = {}
        self._fired: list[FiredTimer] = []
        self._undrained: list[FiredTimer] = []
        self._thread: threading.Thread | None = None
        self._stop = False

    # ---- scheduling ----

    def set_timer(self, seconds: float, label: str, run_id: str = "", source: str = "") -> Timer:
        """Schedule a countdown timer that fires `seconds` from now."""
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError("timer duration must be positive")
        return self._schedule("timer", time.time() + seconds, label, run_id, source)

    def set_alarm(self, at: datetime, label: str, run_id: str = "", source: str = "") -> Timer:
        """Schedule an alarm for a wall-clock moment.

        A naive `at` is interpreted in the host's local timezone; an aware one is
        converted to epoch directly.
        """
        return self._schedule("alarm", at.timestamp(), label, run_id, source)

    def _schedule(self, kind: str, due_at: float, label: str, run_id: str, source: str) -> Timer:
        timer = Timer(
            id=uuid.uuid4().hex[:12],
            run_id=str(run_id or ""),
            kind=kind,
            label=str(label or ""),
            source=str(source or ""),
            due_at=float(due_at),
            tz=self._tz,
        )
        with self._cv:
            self._seq += 1
            heapq.heappush(self._heap, (timer.due_at, self._seq, timer))
            self._timers[timer.id] = timer
            self._ensure_thread()
            self._cv.notify_all()
        return timer

    def cancel(self, timer_id: str) -> bool:
        """Cancel a scheduled timer. Returns False when it already fired/gone."""
        with self._lock:
            return self._timers.pop(timer_id, None) is not None

    # ---- inspection ----

    def pending(self) -> list[dict]:
        with self._lock:
            timers = sorted(self._timers.values(), key=lambda t: t.due_at)
            return [t.as_dict() for t in timers]

    def fired(self) -> list[dict]:
        """Bounded history of fired timers, oldest first."""
        with self._lock:
            return [f.as_dict() for f in self._fired]

    def drain(self) -> list[FiredTimer]:
        """Pop and return timers fired since the last drain (for push front ends)."""
        with self._lock:
            fired = self._undrained
            self._undrained = []
            return fired

    def stop(self) -> None:
        """Stop the firing thread (does not cancel scheduled timers)."""
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    # ---- firing thread ----

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop = False
            self._thread = threading.Thread(target=self._run, name="timers", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            with self._cv:
                if self._stop:
                    return
                # Drop cancelled timers from the head of the heap.
                while self._heap and self._heap[0][2].id not in self._timers:
                    heapq.heappop(self._heap)
                if not self._heap:
                    self._cv.wait()
                    continue
                due_at, _seq, timer = self._heap[0]
                remaining = due_at - time.time()
                if remaining > 0:
                    self._cv.wait(timeout=remaining)
                    continue
                heapq.heappop(self._heap)
                if self._timers.pop(timer.id, None) is None:
                    continue
            self._fire(timer)

    def _fire(self, timer: Timer) -> None:
        fired_at = time.time()
        fired = FiredTimer(
            id=timer.id,
            run_id=timer.run_id,
            kind=timer.kind,
            label=timer.label,
            source=timer.source,
            due_at=timer.due_at,
            fired_at=fired_at,
            message=self._message(timer, fired_at),
        )
        with self._lock:
            self._fired.append(fired)
            if len(self._fired) > self._history_limit:
                self._fired = self._fired[-self._history_limit:]
            self._undrained.append(fired)
        if self.trace is not None:
            try:
                self.trace.append(
                    "timer_fired",
                    timer.run_id,
                    timer_id=timer.id,
                    timer_kind=timer.kind,
                    label=timer.label,
                    source=timer.source,
                    due_at=timer.due_at,
                    fired_at=fired_at,
                )
            except Exception:
                pass

    def _to_local(self, ts: float) -> datetime:
        if self._tz is not None:
            return datetime.fromtimestamp(ts, tz=self._tz)
        return datetime.fromtimestamp(ts).astimezone()

    def _message(self, timer: Timer, fired_at: float) -> str:
        when = self._to_local(fired_at).strftime("%H:%M")
        if timer.kind == "alarm":
            body = timer.label or "Alarm"
            return f"Alarm: {body} — it's {when}."
        body = timer.label or "Timer"
        return f"Timer finished: {body} ({when})."
