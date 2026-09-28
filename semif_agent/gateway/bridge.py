"""Local HTTP bridge that exposes the messenger gateway to skill bodies.

A skill never opens a WebSocket to the `simplex-chat` daemon itself. Instead
the running gateway serves this small stdlib HTTP API on localhost:

    GET  /health                     {"ok": true, "platform": "simplex"}
    GET  /contacts                   {"contacts": [{"id", "display_name"}]}
    GET  /inbox                      peek buffered inbound messages
    GET  /inbox/next?contact=<id>    pop the oldest unread message (optionally
                                     for one contact) -> {"message": {...}|null}
    GET  /address                    the bot's user contact link, creating it if
                                     needed -> {"short_link", "full_link", "created"}
    POST /send                       {"recipient": "<id|name>", "text": "..."}

The bridge owns the read cursor (a bounded FIFO of authorized inbound DMs), so
a read skill stays stateless. Outbound sends reuse the gateway's existing
`OutboundMessage` queue, i.e. the same transport that already replies to chats.
The bridge is deliberately the only messaging surface a generated body touches,
which keeps bodies stdlib-only and their hermetic tests ordinary loopback HTTP.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .base import InboundMessage, OutboundMessage


class MessagingInbox:
    """Thread-safe, bounded FIFO of buffered inbound messages + contacts."""

    def __init__(self, max_size: int = 100):
        self._lock = threading.Lock()
        self._items: deque[dict] = deque()
        self._max = max(1, int(max_size))
        self._contacts: dict[str, str] = {}

    def record(self, msg: InboundMessage) -> dict:
        contact_id = str(msg.contact_id or msg.chat_id)
        entry = {
            "id": uuid.uuid4().hex[:12],
            "contact_id": contact_id,
            "display_name": msg.display_name or "",
            "text": msg.text,
            "received_at": time.time(),
        }
        with self._lock:
            if msg.display_name:
                self._contacts[contact_id] = msg.display_name
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

    def contacts(self) -> list[dict]:
        with self._lock:
            return [
                {"id": contact_id, "display_name": name}
                for contact_id, name in self._contacts.items()
            ]


class MessagingBridge:
    """The gateway's localhost messaging API for skills.

    Serves the endpoints above in a background thread. `outbound` is the
    adapter's send queue (the gateway drains it); posting an `OutboundMessage`
    onto it is exactly how replies are sent. `config` is the
    `gateway.<platform>.bridge` block: `host` (default 127.0.0.1), `port`
    (default 5227, `0` picks an ephemeral port), `token` (optional shared
    secret checked against the `X-Semif-Token` header), `max_inbox`.

    `address_provider` is an optional zero-arg callable returning
    `{short_link, full_link, created}` — the gateway wires it to the adapter's
    contact-address lookup. When absent, `GET /address` answers 503 so a skill
    reports that the link is unavailable rather than inventing one.
    """

    name = "simplex"

    def __init__(
        self,
        outbound: "queue.Queue[OutboundMessage | None]",
        config: dict | None = None,
        address_provider=None,
    ):
        cfg = config or {}
        self.host = str(cfg.get("host", "127.0.0.1"))
        self.port = int(cfg.get("port", 5227))
        self.token = str(cfg.get("token", "") or "")
        self.inbox = MessagingInbox(int(cfg.get("max_inbox", 100)))
        self.outbound = outbound
        self.address_provider = address_provider
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----

    def start(self) -> int:
        """Start serving; returns the bound port (useful when `port` is 0)."""
        handler = type("BridgeHandler", (_BridgeHandler,), {})
        server = ThreadingHTTPServer((self.host, self.port), handler)
        server.bridge = self  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="messaging-bridge", daemon=True
        )
        self._thread.start()
        return int(server.server_address[1])

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    # ---- used by the gateway service ----

    def record_inbound(self, msg: InboundMessage) -> None:
        """Observer callback: buffer an authorized inbound message for reads."""
        self.inbox.record(msg)

    def send(self, recipient: str, text: str) -> str:
        """Resolve a recipient and enqueue the message. Returns the contact id."""
        contact_id = self.resolve(recipient)
        if contact_id is None:
            raise ValueError(f"unknown SimpleX recipient: {recipient!r}")
        self.outbound.put(OutboundMessage(chat_id=contact_id, text=text))
        return contact_id

    def resolve(self, recipient: str) -> str | None:
        """A numeric id (with or without `@`) or a known display name."""
        recipient = str(recipient or "").strip()
        if not recipient:
            return None
        bare = recipient.lstrip("@")
        if bare.isdigit():
            return bare
        for contact in self.inbox.contacts():
            if contact["display_name"] == recipient or contact["id"] == recipient:
                return contact["id"]
        return None


class _BridgeHandler(BaseHTTPRequestHandler):
    """Request handler; the owning bridge is on `self.server.bridge`."""

    def log_message(self, *args) -> None:  # silence the default stderr spam
        pass

    # ---- helpers ----

    def _bridge(self) -> MessagingBridge:
        return self.server.bridge  # type: ignore[attr-defined]

    def _authorized(self) -> bool:
        token = self._bridge().token
        if not token:
            return True
        return self.headers.get("X-Semif-Token") == token

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._json(401, {"error": "unauthorized"})
        return False

    # ---- routes ----

    def do_GET(self) -> None:
        if not self._require_auth():
            return
        url = urlparse(self.path)
        bridge = self._bridge()
        if url.path == "/health":
            self._json(200, {"ok": True, "platform": bridge.name})
        elif url.path == "/contacts":
            self._json(200, {"contacts": bridge.inbox.contacts()})
        elif url.path == "/inbox":
            self._json(200, {"messages": bridge.inbox.peek()})
        elif url.path == "/inbox/next":
            contact = (parse_qs(url.query).get("contact") or [None])[0]
            self._json(200, {"message": bridge.inbox.pop(contact)})
        elif url.path == "/address":
            provider = bridge.address_provider
            if provider is None:
                self._json(503, {"error": "address lookup is not available"})
                return
            try:
                link = provider()
            except Exception as exc:
                self._json(502, {"error": f"address lookup failed: {exc}"[:300]})
                return
            if not isinstance(link, dict):
                self._json(502, {"error": "address lookup returned no link"})
                return
            self._json(200, link)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._require_auth():
            return
        url = urlparse(self.path)
        if url.path != "/send":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            self._json(400, {"error": "invalid JSON body"})
            return
        if not isinstance(payload, dict):
            self._json(400, {"error": "body must be a JSON object"})
            return
        recipient = payload.get("recipient")
        text = payload.get("text")
        if not isinstance(recipient, str) or not recipient.strip():
            self._json(400, {"error": "recipient is required"})
            return
        if not isinstance(text, str) or not text.strip():
            self._json(400, {"error": "text is required"})
            return
        try:
            contact_id = self._bridge().send(recipient, text)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(200, {"ok": True, "contact_id": contact_id})


def start_bridge(
    outbound: "queue.Queue[OutboundMessage | None]",
    config: dict | None = None,
    address_provider=None,
) -> tuple[MessagingBridge, int]:
    bridge = MessagingBridge(outbound, config=config, address_provider=address_provider)
    port = bridge.start()
    return bridge, port
