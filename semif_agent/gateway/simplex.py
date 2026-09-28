"""SimpleX command-gateway adapter.

The gateway is the agent's **command** surface: allowlisted contacts DM the bot
and their text is fed to the scheduler, whose reply is sent back. All the
messaging *UX* surface (invite links, reading a contact's messages, composing on
the user's behalf) lives in separate bridge services, never here — see
AGENTS.md "gateway isolation".

This adapter is the policy layer over the neutral `SimplexDaemon` transport in
`semif_agent.simplex_ws`: it applies the allowlist, batches rapid messages, and
hands authorized text to the scheduler. It must not grow address/inbox/send
endpoints; those belong to a bridge.
"""

from __future__ import annotations

import asyncio
import queue
import threading

from semif_agent.simplex_ws import SimplexDaemon, send_text_command

from .base import GatewayAdapter, InboundMessage, OutboundMessage


class SimplexAdapter(GatewayAdapter):
    name = "simplex"

    def __init__(self, cfg: dict | None = None, trace=None):
        cfg = cfg or {}
        self.allowed_users = [str(u) for u in cfg.get("allowed_users", []) or []]
        self.allow_all_users = bool(cfg.get("allow_all_users", False))
        self.batch_delay = float(cfg.get("text_batch_delay", 0.8))
        self.daemon = SimplexDaemon(
            cfg.get("ws_url", "ws://127.0.0.1:5225"),
            user_id=int(cfg.get("user_id", 1)),
            auto_accept=bool(cfg.get("auto_accept", True)),
            reconnect=cfg.get("reconnect", {}) or {},
            trace=trace,
            name=self.name,
        )
        self.ws_url = self.daemon.ws_url
        self._buffers: dict[str, str] = {}
        self._buffered: dict[str, InboundMessage] = {}
        self._flush_tasks: dict[str, asyncio.Task] = {}
        self._on_inbound = None

    # ---- requirements ----

    def check_requirements(self) -> tuple[bool, str | None]:
        return self.daemon.check_requirements()

    # ---- authorization ----

    def is_authorized(self, contact_id: str, display_name: str | None) -> bool:
        if self.allow_all_users:
            return True
        candidates = {contact_id}
        if display_name:
            candidates.add(display_name)
        if not self.allowed_users:
            return False
        return any(user in candidates for user in self.allowed_users)

    # ---- transport ----

    def run(
        self,
        on_inbound,
        outbound: "queue.Queue[OutboundMessage | None]",
    ) -> None:
        self._on_inbound = on_inbound

        def _pump() -> None:
            # Translate the service's OutboundMessage queue onto the daemon's
            # (chat_id, text) queue without letting scheduler work touch the
            # daemon's event loop.
            while True:
                message = outbound.get()
                if message is None:
                    self.daemon.enqueue(None)
                    return
                self.daemon.enqueue(message.chat_id, message.text)

        threading.Thread(target=_pump, name="simplex-outbound-pump", daemon=True).start()
        self.daemon.run(self._on_item)

    def close(self) -> None:
        self.daemon.close()

    def send_command(self, chat_id: str, text: str) -> str:
        return send_text_command(chat_id, text)

    # ---- inbound ----

    def _on_item(self, message: dict) -> None:
        contact_id = str(message.get("contact_id") or "")
        display_name = message.get("display_name")
        if not self.is_authorized(contact_id, display_name):
            self.daemon._event(
                "gateway_denied", contact_id=contact_id, display_name=display_name
            )
            return
        inbound = InboundMessage(
            text=message.get("text") or "",
            chat_id=contact_id,
            chat_type="dm",
            contact_id=contact_id,
            display_name=display_name,
            raw=message.get("raw") or {},
        )
        self._schedule_flush(inbound, self._on_inbound)

    def _schedule_flush(self, message: InboundMessage, on_inbound) -> None:
        chat = message.chat_id
        existing = self._buffers.get(chat)
        self._buffers[chat] = f"{existing}\n{message.text}" if existing else message.text
        self._buffered[chat] = message
        task = self._flush_tasks.get(chat)
        if task is not None:
            task.cancel()
        self._flush_tasks[chat] = asyncio.ensure_future(self._flush(chat, on_inbound))

    async def _flush(self, chat: str, on_inbound) -> None:
        try:
            await asyncio.sleep(self.batch_delay)
        except asyncio.CancelledError:
            return
        text = self._buffers.pop(chat, "")
        message = self._buffered.pop(chat, None)
        self._flush_tasks.pop(chat, None)
        if message is None or not text:
            return
        message.text = text
        self.daemon._event(
            "gateway_message",
            contact_id=message.contact_id,
            display_name=message.display_name,
            chars=len(text),
        )
        await asyncio.to_thread(on_inbound, message)
