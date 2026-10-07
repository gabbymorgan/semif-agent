"""A bounded, thread-safe FIFO of buffered inbound messages + known contacts.

The bridge owns the read cursor so a read skill stays stateless. Generic over
transports: entries are normalized message dicts (`contact_id`, `display_name`,
`text`), which is exactly what `semif_agent.simplex_ws` yields.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque


class MessagingInbox:
    def __init__(self, max_size: int = 100):
        self._lock = threading.Lock()
        self._items: deque[dict] = deque()
        self._max = max(1, int(max_size))
        self._contacts: dict[str, dict] = {}

    def record(self, message: dict) -> dict:
        contact_id = str(message.get("contact_id") or "")
        display_name = message.get("display_name") or ""
        entry = {
            "id": uuid.uuid4().hex[:12],
            "contact_id": contact_id,
            "display_name": display_name,
            "text": message.get("text") or "",
            "received_at": time.time(),
        }
        with self._lock:
            self._remember(contact_id, message)
            self._items.append(entry)
            while len(self._items) > self._max:
                self._items.popleft()
        return entry

    def _remember(self, contact_id: str, contact: dict) -> None:
        """Upsert the cached contact for an id (caller holds the lock)."""
        current = self._contacts.setdefault(
            contact_id, {"id": contact_id, "display_name": ""}
        )
        name = contact.get("display_name") or ""
        if name:
            current["display_name"] = name
        # Keep the unique local name and connection health when the source knows
        # them; never invent them (an inbound item has no connection state).
        for key in ("local_name", "connected", "auth_errors"):
            if key in contact and contact[key] not in (None, ""):
                current[key] = contact[key]

    def peek(self) -> list[dict]:
        with self._lock:
            return list(self._items)

    def pop(self, contact_id: str | None = None) -> dict | None:
        with self._lock:
            if contact_id is None:
                return self._items.popleft() if self._items else None
            target = str(contact_id)
            for index, entry in enumerate(self._items):
                if entry["contact_id"] == target:
                    del self._items[index]
                    return entry
            return None

    def merge_contacts(self, contacts: list[dict]) -> None:
        """Upsert contacts learned from the daemon (not from a message).

        A non-empty incoming name overwrites a cached one; an empty name never
        clobbers a name already learned from an inbound message. The unique
        local name and connection health ride along when the daemon reports
        them.
        """
        with self._lock:
            for contact in contacts:
                contact_id = str(contact.get("id") or "")
                if not contact_id:
                    continue
                self._remember(contact_id, contact)

    def contacts(self) -> list[dict]:
        with self._lock:
            return [dict(contact) for contact in self._contacts.values()]
