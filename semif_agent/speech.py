"""The scheduler's speech channel: a long-form read-aloud request for a front end.

A skill body cannot speak: it runs inside the scheduler, which is transport-
agnostic and must never reach into a front end. So a body that resolves a long
text to be *read aloud* registers a `ReadAloud` here (through `ctx.speech`) and
returns immediately. The gateway that owns the originating chat drains the
request (`drain()`) and hands it to the platform adapter, which streams it with
text-to-speech until the user stops it. Position is tracked in a
``bookmarks.json`` beside the text (see `semif_agent.reading`).

This mirrors `semif_agent.timers`: the scheduler exposes a small thread-safe
service; a front end drains what it can deliver. It is neutral — stdlib only,
and it knows nothing about audio or any specific gateway.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from .reading import DEFAULT_CHUNK_CHARS


@dataclass
class ReadAloud:
    """One queued request to read a saved text aloud from a position.

    `path` is the text file; `bookmarks_path` the ``bookmarks.json`` beside it;
    `key` the book's bookmark key; `start` the character offset to resume from.
    `source` is the originating front-end source (`"voice:<chat_id>"`) and
    `run_id` the run that asked, so the gateway can route it to the right chat.
    """

    path: str
    key: str
    bookmarks_path: str
    run_id: str = ""
    source: str = ""
    title: str = ""
    start: int = 0
    chunk_chars: int = DEFAULT_CHUNK_CHARS
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)


class SpeechService:
    """Holds read-aloud requests until a front end drains them.

    Bounded like the request queue: a front end that never drains (the REPL)
    cannot grow it without limit.
    """

    def __init__(self, trace=None, max_pending: int = 64):
        self.trace = trace
        self._lock = threading.Lock()
        self._pending: deque[ReadAloud] = deque(maxlen=max(1, int(max_pending)))
        #: Optional predicate set by a front end: whether a given source can be
        #: streamed to (the voice gateway; an LXMF voice chat). None means no
        #: front end is attached (the REPL, a test), so nothing supports it.
        self._stream_check = None

    def set_stream_check(self, check) -> None:
        """Register the front end's per-source streaming capability predicate.

        Called by the gateway service with a callable `(source) -> bool` so a
        skill can ask `supports(source)` before it posts a read-aloud. A front
        end that cannot stream (or has none attached) leaves it None.
        """
        self._stream_check = check if callable(check) else None

    def supports(self, source: str) -> bool:
        """Whether the front end can stream a read-aloud to `source`.

        False when no front end is attached, when the source's platform cannot
        stream, or when the check raises — the caller then keeps its normal
        (non-read-aloud) behaviour.
        """
        check = self._stream_check
        if check is None:
            return False
        try:
            return bool(check(str(source or "")))
        except Exception:
            return False

    def read_aloud(
        self,
        *,
        path: str,
        key: str,
        bookmarks_path: str,
        run_id: str = "",
        source: str = "",
        title: str = "",
        start: int = 0,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
    ) -> ReadAloud:
        """Queue a text to be read aloud. Returns the queued request."""
        read = ReadAloud(
            path=str(path),
            key=str(key),
            bookmarks_path=str(bookmarks_path),
            run_id=str(run_id or ""),
            source=str(source or ""),
            title=str(title or ""),
            start=max(0, int(start or 0)),
            chunk_chars=max(1, int(chunk_chars or DEFAULT_CHUNK_CHARS)),
        )
        with self._lock:
            self._pending.append(read)
        self._event(
            "read_aloud_queued",
            read.run_id,
            read_id=read.id,
            key=read.key,
            source=read.source,
            start=read.start,
        )
        return read

    def drain(self) -> list[ReadAloud]:
        """Pop and return the queued requests, oldest first (for a front end)."""
        with self._lock:
            reads = list(self._pending)
            self._pending.clear()
        return reads

    def pending_for(self, run_id: str) -> bool:
        """Whether a read-aloud is queued for a run (not yet drained).

        Lets the scheduler treat a run that handed off to a long-form read as
        complete: the read is the answer, so `assess:outcome` has no further
        work to find and must not re-dispatch the request.
        """
        with self._lock:
            return any(r.run_id == str(run_id) for r in self._pending)

    def pending(self) -> list[dict]:
        """A snapshot of the queue (for the dashboard)."""
        with self._lock:
            return [
                {
                    "id": r.id,
                    "run_id": r.run_id,
                    "source": r.source,
                    "key": r.key,
                    "title": r.title,
                    "start": r.start,
                    "path": r.path,
                    "created_at": r.created_at,
                }
                for r in self._pending
            ]

    def _event(self, kind: str, run_id: str, **fields) -> None:
        if self.trace is None:
            return
        try:
            self.trace.append(kind, run_id, **fields)
        except Exception:
            pass
