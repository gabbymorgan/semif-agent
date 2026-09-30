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
        self._contacts: dict[str, str] = {}

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
            if display_name:
                self._contacts[contact_id] = display_name
            else:
                self._contacts.setdefault(contact_id, "")
            self._items.append(entry)
            while len(self._items) > self._max:
                self._items.popleft()
        return entry

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
        clobbers a name already learned from an inbound message.
        """
        with self._lock:
            for contact in contacts:
                contact_id = str(contact.get("id") or "")
                if not contact_id:
                    continue
                name = contact.get("display_name") or ""
                if name:
                    self._contacts[contact_id] = name
                else:
                    self._contacts.setdefault(contact_id, "")

    def contacts(self) -> list[dict]:
        with self._lock:
            return [
                {"id": contact_id, "display_name": name}
                for contact_id, name in self._contacts.items()
            ]
