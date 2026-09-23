"""Urgency priority queue.

Items are sorted by descending urgency weight, then FIFO arrival, then most
recent first. Queued items age upward so they cannot starve, and the queue has
a maximum depth.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass
from typing import Iterator

from .decisions import Request

MIN_RECENCY = 0.0


@dataclass
class QueueItem:
    weight: float
    seq: int
    recency: float
    request: Request

    def key(self) -> tuple:
        return (-self.weight, self.seq)


class UrgencyQueue:
    """A max-by-urgency priority queue with FIFO/recency tie-breaking."""

    def __init__(self, max_size: int = 100, age_rate: float = 0.0):
        self.max_size = max_size
        self.age_rate = age_rate
        self._entries: list[tuple] = []
        self._counter = itertools.count()

    def push(self, request: Request, weight: float, recency: float | None = None) -> bool:
        """Insert an item. Returns False if the queue is full."""
        if len(self._entries) >= self.max_size:
            return False
        item = QueueItem(
            weight=weight,
            seq=next(self._counter),
            recency=recency if recency is not None else time.time(),
            request=request,
        )
        entry = (*item.key(), next(self._counter), item)
        heapq.heappush(self._entries, entry)
        return True

    def pop(self) -> Request | None:
        if not self._entries:
            return None
        return heapq.heappop(self._entries)[-1].request

    def peek(self) -> Request | None:
        if not self._entries:
            return None
        return self._entries[0][-1].request

    def age(self, dt: float = 1.0) -> None:
        """Pull queued priorities toward critical so old items catch up."""
        if self.age_rate <= 0:
            return
        rebuilt = []
        for entry in self._entries:
            item = entry[-1]
            item.weight += self.age_rate * dt * (1.0 - item.weight)
            rebuilt.append((*item.key(), next(self._counter), item))
        self._entries = rebuilt
        heapq.heapify(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self) -> Iterator[tuple[float, Request]]:
        for entry in sorted(self._entries, key=lambda e: e[:-1]):
            item = entry[-1]
            yield item.weight, item.request