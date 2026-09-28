"""Low-level SimpleX Chat daemon WebSocket protocol (neutral transport).

This module owns the wire protocol only: connecting to a local `simplex-chat`
daemon, parsing received events into plain message dicts, running correlated
request/response round-trips (e.g. the contact-address lookup), accepting
contact requests, and draining an outbound send queue. It knows nothing about
the SemIf scheduler, the command gateway, or the forwarding bridge — both
front ends build on top of it.

Keeping this layer neutral is deliberate (see AGENTS.md "gateway isolation"):
the command gateway and the third-party bridge services have different jobs,
and neither inherits the other's surface by sharing this module.

`websockets` is imported lazily inside the functions that need it so importing
this module stays stdlib-only on a websocket-free machine.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import queue
import random
import threading
from typing import Any, Callable

#: One received direct text message, normalized across the v7 wire shapes.
InboundDict = dict[str, Any]
#: Callback invoked in the event loop for each received direct text message.
MessageHandler = Callable[[InboundDict], None]


def contact_link(resp: dict) -> dict:
    """Pull `{connShortLink, connFullLink}` out of a daemon address response.

    `/_show_address` nests it as `contactLink.connLinkContact`; `/_address`
    returns it directly as `connLinkContact`. Tolerate both (and a flat
    `contactLink`) so callers read the real daemon shape rather than calling
    `/_address` again on an existing link and reporting nothing.
    """
    for candidate in (
        (resp.get("contactLink") or {}).get("connLinkContact"),
        resp.get("connLinkContact"),
        resp.get("contactLink"),
        resp,
    ):
        if isinstance(candidate, dict) and (
            candidate.get("connShortLink") or candidate.get("connFullLink")
        ):
            return candidate
    return {}


def parse_direct_text_item(item: dict) -> InboundDict | None:
    """Normalize one `newChatItems` entry into a message dict, or None.

    Returns `{text, contact_id, display_name, raw}` for a received direct text
    message. Drops group chats, own echoes, non-text content, and items with no
    resolvable contact id. Authorization is the caller's job — this layer is
    deliberately policy-free.
    """
    chat_info = item.get("chatInfo") or {}
    if chat_info.get("type") != "direct":
        return None
    chat_item = item.get("chatItem") or {}
    direction = (chat_item.get("chatDir") or {}).get("type", "")
    if direction and "Snd" in direction:
        return None  # own echo
    content = chat_item.get("content") or {}
    # simplex-chat v7 tags the content union by constructor: a received text
    # message is {"type": "rcvMsgContent", "msgContent": {"type": "text",
    # "text": ...}}. The daemon's DB/Aeson encoding nests under the
    # constructor key instead ({"rcvMsgContent": {"msgContent": ...}}), so
    # accept both shapes.
    if content.get("type") == "rcvMsgContent":
        msg_content = content.get("msgContent") or {}
    elif isinstance(content.get("rcvMsgContent"), dict):
        msg_content = content["rcvMsgContent"].get("msgContent") or {}
    else:
        return None
    if msg_content.get("type") != "text":
        return None
    text = str(msg_content.get("text", "")).strip()
    if not text:
        return None
    contact = chat_info.get("contact") or {}
    contact_id = str(contact.get("contactId") or chat_info.get("chatId") or "")
    if not contact_id:
        return None
    profile = contact.get("profile") or {}
    # v7's Contact has `displayName: null`; the real peer name lives in
    # `profile.displayName`, and `localDisplayName` is auto-suffixed on
    # collisions (e.g. a second contact becomes `pepper_1`).
    display_name = (
        profile.get("displayName")
        or contact.get("displayName")
        or contact.get("localDisplayName")
    )
    return {
        "text": text,
        "contact_id": contact_id,
        "display_name": display_name,
        "raw": item,
    }


def send_text_command(chat_id: str, text: str) -> str:
    """Structured APISendMessages command for a direct contact.

    The CLI shortcut `@<id> <text>` is silently rejected over WebSocket (it
    resolves the ref as a display name), so always use the JSON form.
    """
    composed = [{"msgContent": {"type": "text", "text": text}}]
    return f"/_send @{chat_id} json {json.dumps(composed)}"


def accept_command(resp: dict) -> str | None:
    """The `/_accept` command for a `receivedContactRequest`, or None.

    v7 names the field contactRequestId_ (the trailing underscore is the
    generated-API optional marker) and /_accept takes the *request* id; older
    events exposed contactId. Fall back so both shapes work.
    """
    req = resp.get("contactRequest") or {}
    req_id = req.get("contactRequestId")
    if req_id is None:
        req_id = req.get("contactId_") or req.get("contactId")
    if req_id is None:
        return None
    return f"/_accept {req_id}"


class SimplexDaemon:
    """One WebSocket connection to a `simplex-chat` daemon.

    `run(on_message)` blocks, reconnecting on transport failure, delivering each
    received direct text message to `on_message` in the event loop and draining
    `enqueue()`d sends from an internal queue. `request_address()` is safe to
    call from another thread (the bridge's HTTP handler). Beware: the same
    daemon/profile must not be shared by two processes — each front end points
    at its own daemon.
    """

    def __init__(
        self,
        ws_url: str,
        *,
        user_id: int = 1,
        auto_accept: bool = True,
        reconnect: dict | None = None,
        trace=None,
        name: str = "simplex",
    ):
        self.ws_url = str(ws_url)
        self.user_id = int(user_id)
        self.auto_accept = bool(auto_accept)
        settings = reconnect or {}
        self.reconnect_initial = float(settings.get("initial", 1.0))
        self.reconnect_max = float(settings.get("max", 60.0))
        self.reconnect_jitter = float(settings.get("jitter", 0.2))
        self.trace = trace
        self.name = name
        self._corr = 0
        self._pending: dict[str, asyncio.Future] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self._outbound: "queue.Queue[tuple[str, str] | None]" = queue.Queue()
        self._stop = threading.Event()
        self._on_message: MessageHandler | None = None

    # ---- requirements ----

    def check_requirements(self) -> tuple[bool, str | None]:
        if not self.ws_url:
            return False, "simplex ws_url is required"
        try:
            import websockets  # noqa: F401
        except ImportError:
            return False, "pip install websockets"
        return True, None

    # ---- lifecycle ----

    def run(self, on_message: MessageHandler) -> None:
        asyncio.run(self._serve(on_message))

    def enqueue(self, chat_id: str | None, text: str = "") -> None:
        """Queue an outbound send. `chat_id=None` is the stop sentinel."""
        if chat_id is None:
            self._outbound.put(None)
        else:
            self._outbound.put((str(chat_id), text))

    def close(self) -> None:
        self._stop.set()
        self._outbound.put(None)

    async def _serve(self, on_message: MessageHandler) -> None:
        import websockets

        self._loop = asyncio.get_running_loop()
        self._on_message = on_message
        backoff = self.reconnect_initial
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.ws_url, max_size=None) as ws:
                    self._ws = ws
                    backoff = self.reconnect_initial
                    self._event("gateway_connected", url=self.ws_url)
                    sender = asyncio.create_task(self._outbound_loop(ws))
                    try:
                        async for raw in ws:
                            await self._consume(raw)
                    finally:
                        sender.cancel()
            except Exception as exc:  # reconnect on any transport failure
                self._event("gateway_error", message=str(exc)[:300])
            finally:
                self._ws = None
                self._fail_pending("simplex gateway disconnected")
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

    async def _outbound_loop(self, ws) -> None:
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, self._outbound.get)
            if item is None:
                return
            chat_id, text = item
            try:
                self._corr += 1
                await ws.send(
                    json.dumps(
                        {
                            "corrId": f"sf-{self._corr}",
                            "cmd": send_text_command(chat_id, text),
                        }
                    )
                )
                self._event("gateway_sent", chat_id=chat_id, chars=len(text))
            except Exception as exc:
                self._event("gateway_error", phase="send", message=str(exc)[:300])

    # ---- request/response + address ----

    async def _roundtrip(self, command: str, timeout: float) -> dict:
        """Send a command and await its correlated `resp` from the event loop."""
        ws = self._ws
        if ws is None or self._loop is None:
            raise RuntimeError("simplex gateway is not connected")
        self._corr += 1
        corr = f"req-{self._corr}"
        future = self._loop.create_future()
        self._pending[corr] = future
        await ws.send(json.dumps({"corrId": corr, "cmd": command}))
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending.pop(corr, None)

    def _fail_pending(self, reason: str) -> None:
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(RuntimeError(reason))

    async def address(self, timeout: float = 20.0) -> dict:
        """Show the daemon user's contact address, creating it on first call.

        `/_show_address <userId>` returns an existing link; when there is none,
        `/_address <userId>` creates one. Returns
        `{short_link, full_link, created}`.
        """
        resp = await self._roundtrip(f"/_show_address {self.user_id}", timeout)
        link: dict = {}
        created = False
        if resp.get("type") == "userContactLink":
            link = contact_link(resp)
        if not (link.get("connShortLink") or link.get("connFullLink")):
            resp = await self._roundtrip(f"/_address {self.user_id}", timeout)
            link = contact_link(resp)
            created = True
        return {
            "short_link": link.get("connShortLink") or "",
            "full_link": link.get("connFullLink") or "",
            "created": created,
        }

    def request_address(self, timeout: float = 20.0) -> dict:
        """Thread-safe address lookup for callers off the event loop.

        Raises `RuntimeError` when no live connection exists and `TimeoutError`
        when the daemon does not answer in time — never a fabricated link.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            raise RuntimeError("simplex gateway is not connected")
        future = asyncio.run_coroutine_threadsafe(self.address(timeout=timeout), loop)
        try:
            return future.result(timeout=timeout + 5.0)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError("simplex address request timed out") from exc

    # ---- inbound ----

    async def _consume(self, raw: Any) -> None:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return
        corr_id = data.get("corrId")
        if corr_id is not None:
            # A correlated command response (round-trip): resolve the waiting
            # requester. Fire-and-forget commands (`_send`, `_accept`) never
            # register a future, so their responses are dropped here.
            future = self._pending.pop(corr_id, None)
            if future is not None and not future.done():
                future.set_result(data.get("resp") or {})
            return
        resp = data.get("resp") or {}
        kind = resp.get("type")
        if kind == "receivedContactRequest":
            await self._maybe_accept(resp)
            return
        if kind != "newChatItems":
            return
        for item in resp.get("chatItems", []) or []:
            message = parse_direct_text_item(item)
            if message is not None and self._on_message is not None:
                self._on_message(message)

    async def _maybe_accept(self, resp: dict) -> None:
        if not self.auto_accept:
            return
        command = accept_command(resp)
        if command is None:
            return
        self._corr += 1
        await self._ws.send(json.dumps({"corrId": f"sf-{self._corr}", "cmd": command}))
        req = resp.get("contactRequest") or {}
        contact_id = req.get("contactId_") or req.get("contactId")
        self._event(
            "gateway_accepted",
            contact_request_id=command.split()[-1],
            contact_id=str(contact_id) if contact_id is not None else "",
        )

    # ---- tracing ----

    def _event(self, kind: str, **fields) -> None:
        if self.trace is not None:
            self.trace.append(kind, "?", **fields)
