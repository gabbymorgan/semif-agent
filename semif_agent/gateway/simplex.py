"""SimpleX Chat gateway adapter.

Connects to the local `simplex-chat` daemon (`simplex-chat -p 5225`) over its
JSON WebSocket API, receives DM text from allowlisted contacts, and sends
replies with the structured `/_send` command. Groups, attachments, reactions,
and typing are deliberately out of scope for the first cut.

`websockets` is imported lazily inside the functions that need it so this
module (and the gateway package) stays importable on a machine without the
dependency; `check_requirements()` gates the gateway off with an install hint.
"""

from __future__ import annotations

import asyncio
import json
import queue
import random
import threading
from typing import Any

from .base import GatewayAdapter, InboundMessage, OutboundMessage


class SimplexAdapter(GatewayAdapter):
    name = "simplex"

    def __init__(self, cfg: dict | None = None, trace=None):
        cfg = cfg or {}
        self.ws_url = str(cfg.get("ws_url", "ws://127.0.0.1:5225"))
        self.allowed_users = [str(u) for u in cfg.get("allowed_users", []) or []]
        self.allow_all_users = bool(cfg.get("allow_all_users", False))
        self.auto_accept = bool(cfg.get("auto_accept", True))
        self.batch_delay = float(cfg.get("text_batch_delay", 0.8))
        reconnect = cfg.get("reconnect", {}) or {}
        self.reconnect_initial = float(reconnect.get("initial", 1.0))
        self.reconnect_max = float(reconnect.get("max", 60.0))
        self.reconnect_jitter = float(reconnect.get("jitter", 0.2))
        self.trace = trace
        self._corr = 0
        self._buffers: dict[str, str] = {}
        self._buffered: dict[str, InboundMessage] = {}
        self._flush_tasks: dict[str, asyncio.Task] = {}
        self._stop = threading.Event()

    # ---- requirements ----

    def check_requirements(self) -> tuple[bool, str | None]:
        if not self.ws_url:
            return False, "gateway.simplex.ws_url is required"
        try:
            import websockets  # noqa: F401
        except ImportError:
            return False, "pip install websockets"
        return True, None

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
        asyncio.run(self._serve(on_inbound, outbound))

    def close(self) -> None:
        self._stop.set()

    async def _serve(self, on_inbound, outbound: "queue.Queue") -> None:
        import websockets

        backoff = self.reconnect_initial
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.ws_url, max_size=None) as ws:
                    backoff = self.reconnect_initial
                    self._event("gateway_connected", url=self.ws_url)
                    sender = asyncio.create_task(self._outbound_loop(ws, outbound))
                    try:
                        async for raw in ws:
                            await self._consume(raw, on_inbound, ws)
                    finally:
                        sender.cancel()
            except Exception as exc:  # reconnect on any transport failure
                self._event("gateway_error", message=str(exc)[:300])
            delay = min(backoff, self.reconnect_max)
            delay *= 1 + random.random() * self.reconnect_jitter
            try:
                await asyncio.wait_for(self._stop_wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            backoff = min(max(delay, self.reconnect_initial), self.reconnect_max)

    async def _stop_wait(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.2)

    async def _outbound_loop(self, ws, outbound: "queue.Queue") -> None:
        loop = asyncio.get_running_loop()
        while True:
            message = await loop.run_in_executor(None, outbound.get)
            if message is None:
                return
            try:
                await self._send(ws, message)
            except Exception as exc:
                self._event("gateway_error", phase="send", message=str(exc)[:300])

    async def _send(self, ws, message: OutboundMessage) -> None:
        command = self.send_command(message.chat_id, message.text)
        self._corr += 1
        await ws.send(json.dumps({"corrId": f"sf-{self._corr}", "cmd": command}))
        self._event("gateway_sent", chat_id=message.chat_id, chars=len(message.text))

    def send_command(self, chat_id: str, text: str) -> str:
        """Structured APISendMessages command for a direct contact.

        The CLI shortcut `@<id> <text>` is silently rejected over WebSocket
        (it resolves the ref as a display name), so always use the JSON form.
        """
        composed = [{"msgContent": {"type": "text", "text": text}}]
        return f"/_send @{chat_id} json {json.dumps(composed)}"

    # ---- inbound ----

    async def _consume(self, raw: Any, on_inbound, ws) -> None:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return
        resp = data.get("resp") or {}
        kind = resp.get("type")
        if kind == "receivedContactRequest":
            await self._maybe_accept(ws, resp)
            return
        if kind != "newChatItems":
            return
        for item in resp.get("chatItems", []) or []:
            message = self._parse_chat_item(item)
            if message is not None:
                self._schedule_flush(message, on_inbound)

    def _parse_chat_item(self, item: dict) -> InboundMessage | None:
        chat_info = item.get("chatInfo") or {}
        if chat_info.get("type") != "direct":
            return None
        chat_item = item.get("chatItem") or {}
        direction = (chat_item.get("chatDir") or {}).get("type", "")
        if direction and "Snd" in direction:
            return None  # own echo
        content = chat_item.get("content") or {}
        if content.get("type") != "text":
            return None
        text = str(content.get("text", "")).strip()
        if not text:
            return None
        contact = chat_info.get("contact") or {}
        contact_id = str(contact.get("contactId") or chat_info.get("chatId") or "")
        display_name = contact.get("displayName") or contact.get("localDisplayName")
        if not contact_id:
            return None
        if not self.is_authorized(contact_id, display_name):
            self._event("gateway_denied", contact_id=contact_id, display_name=display_name)
            return None
        return InboundMessage(
            text=text,
            chat_id=contact_id,
            chat_type="dm",
            contact_id=contact_id,
            display_name=display_name,
            raw=item,
        )

    async def _maybe_accept(self, ws, resp: dict) -> None:
        if not self.auto_accept:
            return
        req = resp.get("contactRequest") or {}
        # v7 names the field contactRequestId_ (the trailing underscore is the
        # generated-API optional marker) and /_accept takes the *request* id;
        # older events exposed contactId. Fall back so both shapes work.
        req_id = req.get("contactRequestId")
        if req_id is None:
            req_id = req.get("contactId_") or req.get("contactId")
        if req_id is None:
            return
        self._corr += 1
        await ws.send(
            json.dumps({"corrId": f"sf-{self._corr}", "cmd": f"/_accept {req_id}"})
        )
        contact_id = req.get("contactId_") or req.get("contactId")
        self._event(
            "gateway_accepted",
            contact_request_id=str(req_id),
            contact_id=str(contact_id) if contact_id is not None else "",
        )

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
        self._event(
            "gateway_message",
            contact_id=message.contact_id,
            display_name=message.display_name,
            chars=len(text),
        )
        await asyncio.to_thread(on_inbound, message)

    # ---- tracing ----

    def _event(self, kind: str, **fields) -> None:
        if self.trace is not None:
            self.trace.append(kind, "?", **fields)
