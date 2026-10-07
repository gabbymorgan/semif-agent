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
from typing import Any, Awaitable, Callable

#: One received direct text message, normalized across the v7 wire shapes.
InboundDict = dict[str, Any]
#: Callback invoked in the event loop for each received direct text message.
MessageHandler = Callable[[InboundDict], None]
#: Async callback invoked in the event loop once a connection is established.
ConnectedHandler = Callable[[], Awaitable[None]]


def contact_entry(contact: dict) -> dict | None:
    """Normalize one daemon `Contact` into a contact dict, or None.

    The single source of truth for contact identity across the wire: the same
    shape appears in `newChatItems` items and in the `/_contacts` list. v7's
    `Contact.displayName` is null and the real peer name lives in
    `profile.displayName`; `localDisplayName` is the daemon's **unique** local
    name, auto-suffixed on collisions (e.g. a second contact becomes
    `pepper_1`). We keep both: `display_name` is the peer's profile name (what
    a human recognizes, but it can collide) and `local_name` is the daemon's
    unique name, so two peers with the same profile name stay distinguishable.
    When the daemon reports connection state (the `/_contacts` list does),
    `connected` and `auth_errors` are included so callers can prefer a live
    peer over a dead one.
    """
    contact_id = str(contact.get("contactId") or contact.get("chatId") or "")
    if not contact_id:
        return None
    profile = contact.get("profile") or {}
    local_name = (
        contact.get("localDisplayName")
        or contact.get("displayName")
        or profile.get("displayName")
        or ""
    )
    display_name = (
        profile.get("displayName")
        or contact.get("displayName")
        or local_name
        or ""
    )
    entry = {"id": contact_id, "display_name": display_name, "local_name": local_name}
    active = contact.get("activeConn") or {}
    conn = active.get("connStatus") or {}
    status = contact.get("contactStatus")
    if status is not None or conn:
        entry["connected"] = status == "active" and conn.get("type") == "ready"
    if isinstance(active.get("authErrCounter"), int):
        entry["auth_errors"] = active["authErrCounter"]
    return entry


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


def _text_from_content(content: dict) -> str | None:
    """Extract the text of a text message from a `CIContent` union, or None.

    Handles the same two shapes as `parse_direct_text_item`: the API form
    (`{"type": "rcvMsgContent", "msgContent": {...}}`) and the DB/Aeson form
    (`{"rcvMsgContent": {"msgContent": {...}}}`), for both received and sent
    messages. Non-text content (files, calls, events) yields None.
    """
    for key in ("rcvMsgContent", "sndMsgContent"):
        if content.get("type") == key:
            msg_content = content.get("msgContent") or {}
        elif isinstance(content.get(key), dict):
            msg_content = content[key].get("msgContent") or {}
        else:
            continue
        if msg_content.get("type") == "text":
            return str(msg_content.get("text", "")).strip()
    return None


def parse_chat_item(item: dict) -> dict | None:
    """Normalize one `ChatItem` into a history entry, or None.

    Returns `{item_id, text, status, unread, direction, sent_at}`. `status` is
    the raw `meta.itemStatus.type` (`rcvNew` / `rcvRead` / `snd*`), and `unread`
    is True only for an unread received message. Items without an id are
    dropped rather than fabricated.
    """
    meta = item.get("meta") or {}
    item_id = meta.get("itemId")
    if item_id is None:
        return None
    text = _text_from_content(item.get("content") or {})
    if not text:
        return None
    status_obj = meta.get("itemStatus") or {}
    status = status_obj.get("type") or ""
    entry = {
        "item_id": str(item_id),
        "text": text,
        "status": status,
        "unread": status == "rcvNew",
        "direction": (item.get("chatDir") or {}).get("type") or "",
        "sent_at": meta.get("itemTs") or "",
    }
    # A failed send carries the agent-level reason (`{"type":"auth"}` etc.);
    # surface it so a caller can report *why* a message did not go out.
    agent_error = status_obj.get("agentError") or {}
    if agent_error:
        entry["error"] = agent_error.get("type") or str(agent_error)
    return entry


def parse_chat(achat: dict) -> dict | None:
    """Normalize one `AChat` into a chat summary, or None for a non-direct chat.

    Returns `{contact_id, display_name, unread_count, min_unread_item_id,
    unread, messages}` from `chatInfo` + `chatStats` + `chatItems`. A group or
    local chat (no direct contact) yields None.
    """
    info = achat.get("chatInfo") or {}
    if info.get("type") != "direct":
        return None
    entry = contact_entry(info.get("contact") or {})
    if entry is None:
        return None
    stats = achat.get("chatStats") or {}
    try:
        unread_count = int(stats.get("unreadCount") or 0)
    except (TypeError, ValueError):
        unread_count = 0
    messages: list[dict] = []
    for item in achat.get("chatItems") or []:
        parsed = parse_chat_item(item)
        if parsed is not None:
            messages.append(parsed)
    min_unread = stats.get("minUnreadItemId")
    return {
        "contact_id": entry["id"],
        "display_name": entry["display_name"],
        "unread_count": unread_count,
        "min_unread_item_id": str(min_unread) if min_unread is not None else "",
        "unread": bool(stats.get("unreadChat")) or unread_count > 0,
        "messages": messages,
    }


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
    contact = dict(chat_info.get("contact") or {})
    if not contact.get("contactId") and not contact.get("chatId"):
        contact["chatId"] = chat_info.get("chatId")
    entry = contact_entry(contact)
    if entry is None:
        return None
    return {
        "text": text,
        "contact_id": entry["id"],
        "display_name": entry["display_name"],
        "local_name": entry.get("local_name", ""),
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


def parse_send_response(resp: dict) -> dict:
    """Normalize a `/_send` response into `{item_id, status, error}`.

    A successful send comes back as `newChatItems` carrying the freshly created
    `chatItem` (its `meta.itemId` and initial `meta.itemStatus`, usually
    `sndNew`). A rejected send comes back as `chatCmdError`. `item_id` is None
    on any rejection so the caller can tell "no message was created" from
    "created but later failed at the agent layer".
    """
    if not isinstance(resp, dict):
        return {"item_id": None, "status": "", "error": "empty daemon response"}
    if resp.get("type") == "chatCmdError" or resp.get("error"):
        detail = resp.get("chatError") or resp.get("error") or "chat command failed"
        return {"item_id": None, "status": "", "error": str(detail)[:300]}
    for item in resp.get("chatItems") or []:
        meta = (item.get("chatItem") or {}).get("meta") or {}
        item_id = meta.get("itemId")
        if item_id is None:
            continue
        status = (meta.get("itemStatus") or {}).get("type") or ""
        error = ((meta.get("itemStatus") or {}).get("agentError") or {}).get("type") or ""
        return {"item_id": str(item_id), "status": status, "error": error}
    return {"item_id": None, "status": "", "error": "daemon returned no message id"}


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
        on_connected: ConnectedHandler | None = None,
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
        self.on_connected = on_connected
        self.trace = trace
        self.name = name
        self._corr = 0
        self._pending: dict[str, asyncio.Future] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self._outbound: "queue.Queue[tuple[str, str] | None]" = queue.Queue()
        self._stop = threading.Event()
        self._on_message: MessageHandler | None = None
        self._active_user_id: int | None = None

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
                    # The on_connected hook may issue correlated commands (e.g.
                    # the startup contact-address lookup), so run it
                    # CONCURRENTLY with the read loop: awaiting it inline would
                    # block the loop that resolves its own response and the
                    # request would time out.
                    hook = asyncio.create_task(self._notify_connected())
                    try:
                        async for raw in ws:
                            await self._consume(raw)
                    finally:
                        hook.cancel()
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

    async def _notify_connected(self) -> None:
        """Run the optional `on_connected` hook; a hook failure must not drop
        the socket (it is traced and the connection stays up)."""
        if self.on_connected is None:
            return
        try:
            await self.on_connected()
        except Exception as exc:
            self._event(
                "gateway_error",
                phase="on_connected",
                message=str(exc)[:300],
            )

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

    # ---- send ----

    async def send(self, chat_id: str, text: str, timeout: float = 20.0) -> dict:
        """Send a direct text message and return the daemon's correlated reply.

        Unlike `enqueue` (fire-and-forget, used by the command gateway), this
        awaits the `/_send` response so the caller sees whether the message was
        created and its initial status. The daemon reports agent-level failures
        (auth, quota) asynchronously, so a caller that needs delivery truth must
        still poll `chat_history` for the item's terminal status.
        """
        return await self._roundtrip(send_text_command(chat_id, text), timeout)

    def request_send(self, chat_id: str, text: str, timeout: float = 20.0) -> dict:
        """Thread-safe correlated send for callers off the event loop.

        Raises `RuntimeError` when no live connection exists and `TimeoutError`
        when the daemon does not answer in time — never a fabricated success.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            raise RuntimeError("simplex gateway is not connected")
        future = asyncio.run_coroutine_threadsafe(
            self.send(chat_id, text, timeout=timeout), loop
        )
        try:
            return future.result(timeout=timeout + 5.0)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError("simplex send request timed out") from exc

    # ---- contacts ----

    async def active_user(self, timeout: float = 20.0) -> int:
        """Resolve the active user id via `/user`, caching it after first use.

        Falls back to the configured `user_id` when the daemon does not report
        one, so a profile with a single user keeps working even on a shape drift.
        """
        if self._active_user_id is not None:
            return self._active_user_id
        resp = await self._roundtrip("/user", timeout)
        user = resp.get("user") or {}
        user_id = user.get("userId")
        self._active_user_id = int(user_id) if user_id is not None else self.user_id
        return self._active_user_id

    async def contacts(self, timeout: float = 20.0) -> list[dict]:
        """The daemon's contact list as `[{"id", "display_name"}]`.

        `/_contacts <userId>` returns `contactsList` with a `Contact` array.
        A `chatCmdError` or malformed reply yields an empty list rather than a
        fabricated contact.
        """
        user_id = await self.active_user(timeout)
        resp = await self._roundtrip(f"/_contacts {user_id}", timeout)
        contacts: list[dict] = []
        for contact in resp.get("contacts") or []:
            entry = contact_entry(contact)
            if entry is not None:
                contacts.append(entry)
        return contacts

    def request_contacts(self, timeout: float = 20.0) -> list[dict]:
        """Thread-safe contact-list lookup for callers off the event loop.

        Raises `RuntimeError` when no live connection exists and `TimeoutError`
        when the daemon does not answer in time — never a fabricated list.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            raise RuntimeError("simplex gateway is not connected")
        future = asyncio.run_coroutine_threadsafe(self.contacts(timeout=timeout), loop)
        try:
            return future.result(timeout=timeout + 5.0)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError("simplex contacts request timed out") from exc

    # ---- chats / history ----

    async def chats(
        self, unread_only: bool = True, count: int = 20, timeout: float = 20.0
    ) -> list[dict]:
        """Chat previews from the daemon, optionally only those with unread.

        `/_get chats <userId> count=<n> {"type":"filters","favorite":false,
        "unread":<bool>}` returns `apiChats`. This reads the daemon's persistent
        state, unlike the bridge's in-memory receive buffer. A `chatCmdError`
        or malformed reply yields an empty list rather than fabricated chats.
        """
        user_id = await self.active_user(timeout)
        query = {"type": "filters", "favorite": False, "unread": bool(unread_only)}
        command = f"/_get chats {user_id} count={max(1, int(count))} {json.dumps(query)}"
        resp = await self._roundtrip(command, timeout)
        if resp.get("type") != "apiChats":
            return []
        chats: list[dict] = []
        for achat in resp.get("chats") or []:
            parsed = parse_chat(achat)
            if parsed is not None:
                chats.append(parsed)
        return chats

    async def chat_history(
        self, contact_id: str, count: int = 20, timeout: float = 20.0
    ) -> list[dict]:
        """Recent messages of one direct chat, newest-aware, from the daemon.

        `/_get chat @<contactId> count=<n>` returns `apiChat`; each message
        carries `meta.itemStatus` (`rcvNew`/`rcvRead`), so callers can tell
        unread from read. A malformed reply yields an empty list.
        """
        chat_ref = str(contact_id).strip().lstrip("@")
        command = f"/_get chat @{chat_ref} count={max(1, int(count))}"
        resp = await self._roundtrip(command, timeout)
        if resp.get("type") != "apiChat":
            return []
        chat = resp.get("chat")
        parsed = parse_chat(chat) if isinstance(chat, dict) else None
        return parsed["messages"] if parsed is not None else []

    def request_chats(
        self, unread_only: bool = True, count: int = 20, timeout: float = 20.0
    ) -> list[dict]:
        """Thread-safe chat-preview lookup for callers off the event loop.

        Raises `RuntimeError` when no live connection exists and `TimeoutError`
        when the daemon does not answer in time — never a fabricated list.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            raise RuntimeError("simplex gateway is not connected")
        future = asyncio.run_coroutine_threadsafe(
            self.chats(unread_only=unread_only, count=count, timeout=timeout), loop
        )
        try:
            return future.result(timeout=timeout + 5.0)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError("simplex chats request timed out") from exc

    def request_chat_history(
        self, contact_id: str, count: int = 20, timeout: float = 20.0
    ) -> list[dict]:
        """Thread-safe per-chat history lookup for callers off the event loop.

        Raises `RuntimeError` when no live connection exists and `TimeoutError`
        when the daemon does not answer in time — never a fabricated list.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            raise RuntimeError("simplex gateway is not connected")
        future = asyncio.run_coroutine_threadsafe(
            self.chat_history(contact_id, count=count, timeout=timeout), loop
        )
        try:
            return future.result(timeout=timeout + 5.0)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError("simplex chat history request timed out") from exc

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
