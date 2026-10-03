"""Read the next unread SimpleX message through the forwarding bridge.

Real integration: the standalone SimpleX forwarding bridge
(`simplex_bridge_url`) buffers every inbound DM and forwards outbound sends.
`act` peeks that inbox, finds which conversations have unread messages, and
resolves which one to read with a SemIf choice over them (state = the user
query): a strong winner (probability >= `CONTACT_WINNER_TAU`) is used, otherwise
the configured default contact is used; with no configured default (or a default
with no unread message) the weak winner stands. It then pops and reports the
oldest unread message for that conversation.

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
    """Resolve which conversation to read, mirroring calendar.create_event.

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
        return ActionResult(action_log=f"simplex.next_message: {exc}", new_state=request.text)
    senders = _senders(payload.get("messages") or [])
    if not senders:
        return ActionResult(
            action_log="simplex.next_message: no unread SimpleX messages.",
            new_state="no unread SimpleX messages",
        )
    decisions = []
    contact = _resolve_contact(ctx, request, senders, decisions)
    try:
        payload = _get(
            ctx, f"/inbox/next?contact={urllib.parse.quote(contact)}"
        )
    except BridgeError as exc:
        return ActionResult(
            action_log=f"simplex.next_message failed reading {contact!r}: {exc}",
            new_state=request.text,
            decisions=decisions,
        )
    message = payload.get("message")
    if not message:
        return ActionResult(
            action_log=f"simplex.next_message: no unread message from {contact!r}.",
            new_state="no unread SimpleX messages",
            decisions=decisions,
        )
    sender = message.get("display_name") or contact
    body = str(message.get("text") or "")
    report = f"SimpleX message from {sender}: {body}"
    return ActionResult(action_log=report, new_state=report, decisions=decisions)
