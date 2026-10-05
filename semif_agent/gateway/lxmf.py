"""LXMF (Reticulum) command-gateway adapter.

The gateway is the agent's **command** surface: allowlisted contacts send LXMF
messages to the bot and their text is fed to the scheduler, whose reply is sent
back. All the messaging *UX* surface (reading a contact's messages, composing on
the user's behalf, showing a contact link) would live in a separate bridge
service, never here — see AGENTS.md "gateway isolation".

This adapter is the policy layer over the neutral `LxmfDaemon` transport in
`semif_agent.lxmf_transport`: it applies the allowlist, batches rapid messages,
and hands authorized text to the scheduler. Unlike the SimpleX adapter there is
no external daemon and no event loop — Reticulum runs in-process on its own
threads, so batching uses a per-chat `threading.Timer`.
"""

from __future__ import annotations

import os
import queue
import threading

from semif_agent.lxmf_transport import LxmfDaemon

from .base import GatewayAdapter, InboundMessage, OutboundMessage


class LxmfAdapter(GatewayAdapter):
    name = "lxmf"

    def __init__(self, cfg: dict | None = None, trace=None):
        cfg = cfg or {}
        self.allowed_users = [str(u) for u in cfg.get("allowed_users", []) or []]
        self.allow_all_users = bool(cfg.get("allow_all_users", False))
        self.batch_delay = float(cfg.get("text_batch_delay", 0.8))
        config_dir = cfg.get("config_dir") or os.path.join(
            ".runtime", "lxmf", "reticulum"
        )
        storage_path = cfg.get("storage_path") or os.path.join(
            ".runtime", "lxmf", "router"
        )
        self.daemon = LxmfDaemon(
            config_dir=config_dir,
            storage_path=storage_path,
            display_name=cfg.get("display_name", "semif"),
            announce_interval=float(cfg.get("announce_interval", 3600)),
            stamp_cost=cfg.get("stamp_cost"),
            desired_method=cfg.get("desired_method", "direct"),
            propagation_node=cfg.get("propagation_node", ""),
            trace=trace,
            name=self.name,
        )
        self.config_dir = config_dir
        self.storage_path = storage_path
        self._buffers: dict[str, str] = {}
        self._buffered: dict[str, InboundMessage] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._on_inbound = None

    # ---- requirements ----

    def check_requirements(self) -> tuple[bool, str | None]:
        return self.daemon.check_requirements()

    # ---- authorization ----

    def is_authorized(self, source_hash: str, display_name: str | None) -> bool:
        if self.allow_all_users:
            return True
        candidates = {str(source_hash)}
        if display_name:
            candidates.add(str(display_name))
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
            # Reticulum callback threads.
            while True:
                message = outbound.get()
                if message is None:
                    self.daemon.enqueue(None)
                    return
                self.daemon.enqueue(message.chat_id, message.text)

        threading.Thread(
            target=_pump, name="lxmf-outbound-pump", daemon=True
        ).start()
        self.daemon.run(self._on_item)

    def close(self) -> None:
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for timer in timers:
            timer.cancel()
        self.daemon.close()

    # ---- inbound ----

    def _on_item(self, message: dict) -> None:
        source_hash = str(message.get("source_hash") or "")
        if not source_hash:
            return
        display_name = message.get("display_name")
        if not self.is_authorized(source_hash, display_name):
            self.daemon._event(
                "gateway_denied",
                contact_id=source_hash,
                display_name=display_name,
            )
            return
        inbound = InboundMessage(
            text=message.get("content") or "",
            chat_id=source_hash,
            chat_type="dm",
            contact_id=source_hash,
            display_name=display_name,
            raw=message.get("raw") or {},
        )
        self._schedule_flush(inbound)

    def _schedule_flush(self, message: InboundMessage) -> None:
        chat = message.chat_id
        with self._lock:
            existing = self._buffers.get(chat)
            self._buffers[chat] = (
                f"{existing}\n{message.text}" if existing else message.text
            )
            self._buffered[chat] = message
            timer = self._timers.get(chat)
            if timer is not None:
                timer.cancel()
            timer = threading.Timer(self.batch_delay, self._flush, args=(chat,))
            timer.daemon = True
            self._timers[chat] = timer
            timer.start()

    def _flush(self, chat: str) -> None:
        with self._lock:
            text = self._buffers.pop(chat, "")
            message = self._buffered.pop(chat, None)
            self._timers.pop(chat, None)
        if message is None or not text:
            return
        message.text = text
        self.daemon._event(
            "gateway_message",
            contact_id=message.contact_id,
            display_name=message.display_name,
            chars=len(text),
        )
        if self._on_inbound is not None:
            self._on_inbound(message)
