"""FIFO request queue with a maximum depth.

Requests are dispatched strictly in arrival order: there is no urgency scoring
and no preemption, so the running process always finishes before the next queued
request is picked up.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

from .decisions import Request


class RequestQueue:
    """A bounded first-in-first-out queue of pending requests."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._entries: deque[Request] = deque()

    def push(self, request: Request) -> bool:
        """Append an item. Returns False if the queue is full."""
        if len(self._entries) >= self.max_size:
            return False
        self._entries.append(request)
        return True

    def pop(self) -> Request | None:
        if not self._entries:
            return None
        return self._entries.popleft()

    def peek(self) -> Request | None:
        if not self._entries:
            return None
        return self._entries[0]

    def __len__(self) -> int:
        return len(self._entries)

    def items(self) -> Iterator[Request]:
        yield from self._entries
