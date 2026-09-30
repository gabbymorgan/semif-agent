"""SimpleX forwarding bridge: invite links, reading, and sending for skills.

The first third-party API bridge. It stands up a small HTTP API that a skill
body calls instead of talking to `simplex-chat` directly, and it owns its **own**
SimpleX daemon/profile — separate from the command gateway's. That separation is
the point: the gateway takes commands; this bridge forwards the user's messaging
UX (show/create the contact link, read the next message, compose a send).

It buffers every inbound direct message (no allowlist — the user wants to see
who reached the bot through its invite link), and it never runs inside the
gateway process.
"""

from __future__ import annotations

import threading

from semif_agent.simplex_ws import SimplexDaemon

from .base import BridgeInfo, BridgeService
from .inbox import MessagingInbox


class SimplexBridge(BridgeService):
    name = "simplex"
    INFO = BridgeInfo(
        name="simplex",
        service="simplex",
        description=(
            "Read and send SimpleX messages on the user's behalf, and show or "
            "create the contact link others use to reach this agent."
        ),
        url_config_var="simplex_bridge_url",
        endpoints=(
            "GET /health -> 200 {\"ok\": true, \"platform\": \"simplex\"} — the "
            "bridge is up",
            "GET /contacts -> 200 {\"contacts\": [{\"id\", \"display_name\"}]} — "
            "contacts known to the daemon (learned from inbound messages and "
            "refreshed from the daemon's contact list)",
            "GET /inbox -> 200 {\"messages\": [{\"id\", \"contact_id\", "
            "\"display_name\", \"text\", \"received_at\"}]} — peek buffered "
            "inbound messages; does not consume",
            "GET /inbox/next?contact=<id> -> 200 {\"message\": {...}|null} — pop "
            "the oldest unread message (optionally from one contact); null means "
            "none buffered. The bridge owns the read cursor",
            "GET /address -> 200 {\"short_link\", \"full_link\", \"created\"} — "
            "the agent's contact link; creates it on first call. 503 when the "
            "daemon is not connected, 502 when the lookup fails",
            "POST /send {\"recipient\": \"<id|display_name>\", \"text\": \"...\"} "
            "-> 200 {\"ok\": true, \"contact_id\": \"<id>\"}; 400 {\"error\": "
            "\"...\"} on a missing/invalid field or an unknown recipient",
            "any request -> 401 {\"error\": \"unauthorized\"} when the token is "
            "configured and the auth header is missing or wrong",
        ),
        config_vars=(
            "simplex_bridge_url",
            "simplex_default_contact",
            "simplex_bridge_token",
        ),
        config_var_docs=(
            (
                "simplex_bridge_url",
                "Base URL of the local SimpleX forwarding bridge "
                "(e.g. http://127.0.0.1:5227); never hardcode it in the body.",
            ),
            (
                "simplex_default_contact",
                "Optional default SimpleX contact (contact id or display name) "
                "used when the request does not already make the conversation "
                "clear; leave blank to always choose among senders.",
            ),
            (
                "simplex_bridge_token",
                "Shared secret for the bridge, if one is configured; sent as the "
                "X-Semif-Token header. Leave blank when the bridge requires no "
                "auth.",
            ),
        ),
        auth_header="X-Semif-Token",
        auth_config_var="simplex_bridge_token",
    )

    def __init__(self, config: dict | None = None, trace=None, daemon=None):
        super().__init__(config, trace)
        cfg = config or {}
        self.inbox = MessagingInbox(int(cfg.get("max_inbox", 100)))
        self.daemon = daemon or SimplexDaemon(
            cfg.get("ws_url", "ws://127.0.0.1:5228"),
            user_id=int(cfg.get("user_id", 1)),
            auto_accept=bool(cfg.get("auto_accept", True)),
            reconnect=cfg.get("reconnect", {}) or {},
            on_connected=self._refresh_on_connect,
            trace=trace,
            name=self.name,
        )
        self._daemon_thread: threading.Thread | None = None

    # ---- lifecycle ----

    def check_requirements(self) -> tuple[bool, str | None]:
        return self.daemon.check_requirements()

    def start(self) -> int:
        port = super().start()
        self._daemon_thread = threading.Thread(
            target=self.daemon.run,
            args=(self._on_message,),
            name="bridge-simplex-daemon",
            daemon=True,
        )
        self._daemon_thread.start()
        return port

    def _stop_transport(self) -> None:
        self.daemon.close()

    def _on_message(self, message: dict) -> None:
        self.inbox.record(message)

    async def _refresh_on_connect(self) -> None:
        """Prime the contact list from the daemon on (re)connect.

        Runs on the daemon's event loop, so it awaits `contacts()` directly
        rather than the thread-safe wrapper. A failure is traced, not fatal.
        """
        contacts = await self.daemon.contacts()
        self.inbox.merge_contacts(contacts)
        self._event("contacts_refreshed", count=len(contacts))

    # ---- routes ----

    def handle_get(self, path: str, query: dict) -> tuple[int, dict]:
        if path == "/health":
            return 200, {"ok": True, "platform": self.name}
        if path == "/contacts":
            return 200, {"contacts": self.refresh_contacts()}
        if path == "/inbox":
            return 200, {"messages": self.inbox.peek()}
        if path == "/inbox/next":
            contact = (query.get("contact") or [None])[0]
            return 200, {"message": self.inbox.pop(contact)}
        if path == "/address":
            return self._address()
        return 404, {"error": "not found"}

    def handle_post(self, path: str, payload: dict) -> tuple[int, dict]:
        if path != "/send":
            return 404, {"error": "not found"}
        recipient = payload.get("recipient")
        text = payload.get("text")
        if not isinstance(recipient, str) or not recipient.strip():
            return 400, {"error": "recipient is required"}
        if not isinstance(text, str) or not text.strip():
            return 400, {"error": "text is required"}
        try:
            contact_id = self.send(recipient, text)
        except ValueError as exc:
            return 400, {"error": str(exc)}
        return 200, {"ok": True, "contact_id": contact_id}

    # ---- actions ----

    def refresh_contacts(self) -> list[dict]:
        """Best-effort refresh of the daemon's contact list into the inbox.

        Never raises: when the daemon is not connected (or is slow) it returns
        whatever contacts are already cached, so `/contacts` and `/send` stay
        available. The read cursor and inbound-learned names are preserved.
        """
        try:
            contacts = self.daemon.request_contacts()
        except Exception as exc:  # RuntimeError, TimeoutError, daemon error
            self._event("contacts_refresh_failed", message=str(exc)[:200])
            return self.inbox.contacts()
        self.inbox.merge_contacts(contacts)
        self._event("contacts_refreshed", count=len(contacts))
        return self.inbox.contacts()

    def _address(self) -> tuple[int, dict]:
        try:
            link = self.daemon.request_address()
        except RuntimeError as exc:
            return 503, {"error": f"address lookup is not available: {exc}"[:300]}
        except Exception as exc:  # TimeoutError, daemon error
            return 502, {"error": f"address lookup failed: {exc}"[:300]}
        if not isinstance(link, dict):
            return 502, {"error": "address lookup returned no link"}
        return 200, link

    def send(self, recipient: str, text: str) -> str:
        """Resolve a recipient and enqueue the message. Returns the contact id."""
        contact_id = self.resolve(recipient)
        if contact_id is None:
            raise ValueError(f"unknown SimpleX recipient: {recipient!r}")
        self.daemon.enqueue(contact_id, text)
        return contact_id

    def resolve(self, recipient: str) -> str | None:
        """A numeric id (with or without `@`) or a known display name.

        On a display-name miss, refresh the daemon's contact list once and
        retry: a contact the bridge has never received from is still addressable
        as long as the daemon knows them.
        """
        recipient = str(recipient or "").strip()
        if not recipient:
            return None
        bare = recipient.lstrip("@")
        if bare.isdigit():
            return bare
        match = self._match_contact(recipient)
        if match is None:
            self.refresh_contacts()
            match = self._match_contact(recipient)
        return match

    def _match_contact(self, recipient: str) -> str | None:
        for contact in self.inbox.contacts():
            if contact["display_name"] == recipient or contact["id"] == recipient:
                return contact["id"]
        return None
