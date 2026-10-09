"""Read the next unread SimpleX message through the forwarding bridge.

Real integration: the standalone SimpleX forwarding bridge
(`simplex_bridge_url`) buffers every inbound DM and forwards outbound sends.
`act` peeks that inbox, finds which conversations have unread messages, and
resolves which one to read with a SemIf choice over them (state = the user
query): a strong winner (probability >= `CONTACT_WINNER_TAU`) is used, otherwise
the configured default contact is used; with no configured default (or a default
with no unread message) the weak winner stands. It then pops and reports the
oldest unread message for that conversation.

When the live buffer is empty it falls back to the daemon's persistent unread
(`/unread` + `/history`), so a message that arrived before the bridge started is
still reported. After reporting a message, that single item is marked read on
the daemon through the bridge (`POST /read`), so the next call advances to the
next unread message instead of re-reporting the same one.

The bridge owns the read cursor, so this skill keeps no state of its own. It
never opens a WebSocket to the `simplex-chat` daemon — the bridge is the only
messaging surface it uses.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from semif_agent.decisions import DecisionRequest, Option
from semif_agent.skills import ActionResult

INTEGRATION = {
    "service": "simplex",
    "transport": "http",
    "config_vars": [
        "simplex_bridge_url",
        "simplex_bridge_token",
        "simplex_default_contact",
    ],
}

CONTRACT = {
    "simplex_bridge_url": "Base URL of the standalone SimpleX forwarding bridge, e.g. http://127.0.0.1:5227 (no trailing slash needed).",
    "simplex_bridge_token": "Shared secret for the bridge, if one is configured; sent as the X-Semif-Token header. Leave blank when the bridge requires no auth.",
    "simplex_default_contact": "Default SimpleX contact (contact id or display name) used when the request does not strongly point at a conversation; leave blank to always use the strongest candidate.",
}

TIMEOUT_SECONDS = 20
# A SemIf probability at or above this counts as a "strong winner" for the
# conversation; below it the configured default contact is used instead.
CONTACT_WINNER_TAU = 0.6


class BridgeError(Exception):
    """The SimpleX forwarding bridge is unreachable or returned an error."""


# ---- transport ----


def _base(ctx):
    url = str(ctx.config.get("simplex_bridge_url", "") or "").strip().rstrip("/")
    if not url:
        raise BridgeError("simplex_bridge_url is not configured")
    return url


def _headers(ctx):
    token = str(ctx.config.get("simplex_bridge_token", "") or "").strip()
    return {"X-Semif-Token": token} if token else {}


def _get(ctx, path):
    request = urllib.request.Request(
        _base(ctx) + path, headers=_headers(ctx)
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()[:200]
        raise BridgeError(f"GET {path}: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise BridgeError(f"GET {path}: {exc}") from exc


def _post(ctx, path, payload):
    data = json.dumps(payload).encode("utf-8")
    headers = {**_headers(ctx), "Content-Type": "application/json"}
    request = urllib.request.Request(
        _base(ctx) + path, data=data, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()[:200]
        raise BridgeError(f"POST {path}: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise BridgeError(f"POST {path}: {exc}") from exc


def _mark_read(ctx, contact, item_id):
    """Mark one just-reported item read via the bridge.

    Returns None on success, or a short honest reason string on failure. The
    message was still reported, so a failed consume is surfaced as a note rather
    than fabricated into a read failure.
    """
    item = str(item_id or "").strip()
    if not item:
        return None
    try:
        payload = _post(ctx, "/read", {"contact": contact, "item_ids": [item]})
    except BridgeError as exc:
        return str(exc)
    if not payload.get("ok"):
        return payload.get("error") or "the bridge did not mark it read"
    return None


# ---- conversation resolution ----


def _senders(messages):
    """Distinct conversations with buffered messages, in arrival order."""
    senders = []
    seen = set()
    for message in messages:
        contact = str(message.get("contact_id") or "")
        if not contact or contact in seen:
            continue
        seen.add(contact)
        senders.append(
            {"id": contact, "display_name": message.get("display_name") or contact}
        )
    return senders


def _is_healthy(sender):
    """A conversation is healthy when its connection is not known-bad.

    `connected` is absent when the source did not report it (treat as unknown);
    `auth_errors` counts failed sends on the connection and marks a stale peer.
    """
    if sender.get("connected") is False:
        return False
    try:
        return int(sender.get("auth_errors") or 0) == 0
    except (TypeError, ValueError):
        return True


def _default_sender(ctx, senders):
    """The configured default contact, if it has unread messages."""
    default = str(ctx.config.get("simplex_default_contact", "") or "").strip()
    if not default:
        return None
    for sender in senders:
        if sender["id"] == default or sender["display_name"] == default:
            return sender
    return None


def _resolve_contact(ctx, request, senders, decisions):
    """Resolve which conversation to read, mirroring nextcloud.create_event.

    A strong SemIf winner is used; a weak winner falls back to the configured
    default contact, and with no configured default the weak winner stands.
    """
    default = _default_sender(ctx, senders)
    if len(senders) == 1:
        return senders[0]["id"]

    decision = DecisionRequest(
        state=request.text,
        question="Which conversation should I read the next SimpleX message from?",
        options=[Option(sender["id"], sender["display_name"]) for sender in senders],
    )
    result = ctx.engine.call(decision)
    decisions.append((decision, result))
    winner = result.selected
    if result.prob(winner) >= CONTACT_WINNER_TAU:
        return winner
    if default is not None:
        return default["id"]
    return winner


# ---- skill phases ----


def act(ctx, request):
    try:
        payload = _get(ctx, "/inbox")
    except BridgeError as exc:
        return ActionResult(action_log=str(exc), new_state=request.text)
    senders = _senders(payload.get("messages") or [])
    if senders:
        return _read_live(ctx, request, senders)
    # The live buffer only holds messages that arrived while the bridge was
    # connected; fall back to the daemon's persistent unread so a message that
    # landed before the bridge started is not silently missed.
    return _read_persistent(ctx, request)


def _read_live(ctx, request, senders):
    decisions = []
    contact = _resolve_contact(ctx, request, senders, decisions)
    try:
        payload = _get(
            ctx, f"/inbox/next?contact={urllib.parse.quote(contact)}"
        )
    except BridgeError as exc:
        return ActionResult(
            action_log=f"failed reading {contact!r}: {exc}",
            new_state=request.text,
            decisions=decisions,
        )
    message = payload.get("message")
    if not message:
        return ActionResult(
            action_log=f"no unread message from {contact!r}.",
            new_state="no unread SimpleX messages",
            decisions=decisions,
        )
    read_note = _mark_read(ctx, contact, message.get("item_id"))
    return _report(
        message.get("display_name") or contact,
        message.get("text"),
        decisions,
        read_note,
    )


def _read_persistent(ctx, request):
    try:
        payload = _get(ctx, "/unread")
    except BridgeError as exc:
        return ActionResult(
            action_log=f"no unread SimpleX messages ({exc}).",
            new_state="no unread SimpleX messages",
        )
    chats = [c for c in (payload.get("chats") or []) if c.get("unread")]
    if not chats:
        return ActionResult(
            action_log="no unread SimpleX messages.",
            new_state="no unread SimpleX messages",
        )
    senders = [
        {
            "id": str(chat.get("contact_id") or ""),
            "display_name": chat.get("display_name") or str(chat.get("contact_id") or ""),
            "connected": chat.get("connected"),
            "auth_errors": chat.get("auth_errors"),
        }
        for chat in chats
        if chat.get("contact_id")
    ]
    decisions = []
    healthy = [s for s in senders if _is_healthy(s)]
    if len(healthy) == 1:
        # Exactly one conversation is on a healthy connection (the rest are
        # stale duplicates): read it without a coin-flip decision.
        contact = healthy[0]["id"]
    else:
        contact = _resolve_contact(ctx, request, senders, decisions)
    try:
        history = _get(
            ctx, f"/history?contact={urllib.parse.quote(contact)}&count=50"
        )
    except BridgeError as exc:
        return ActionResult(
            action_log=f"failed reading {contact!r}: {exc}",
            new_state=request.text,
            decisions=decisions,
        )
    messages = [m for m in (history.get("messages") or []) if m.get("unread")]
    if not messages:
        # Fall back to the previewed unread messages from /unread.
        chat = next(
            (c for c in chats if str(c.get("contact_id")) == str(contact)), {}
        )
        messages = [m for m in (chat.get("messages") or []) if m.get("unread")]
    if not messages:
        return ActionResult(
            action_log=f"no unread message from {contact!r}.",
            new_state="no unread SimpleX messages",
            decisions=decisions,
        )
    sender = next(
        (s["display_name"] for s in senders if s["id"] == str(contact)), str(contact)
    )
    read_note = _mark_read(ctx, contact, messages[0].get("item_id"))
    return _report(sender, messages[0].get("text"), decisions, read_note)


def _report(sender, body, decisions, read_note=None):
    report = f"SimpleX message from {sender}: {str(body or '')}"
    if read_note:
        report += f" (could not mark it read: {read_note})"
    return ActionResult(action_log=report, new_state=report, decisions=decisions)
