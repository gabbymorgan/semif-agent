"""Read the next unread SimpleX message bridge.

Real integration: the standalone SimpleX forwarding bridge
(`simplex_bridge_url`) buffers every inbound DM and forwards outbound sends.
`predict` peeks that inbox, finds which conversations have unread messages, and
resolves which one to read with a SemIf sub-decision when more than one does
(honoring a configured default contact); `act` pops and reports the oldest
unread message for that conversation.

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
from semif_agent.skills import ActionResult, Prediction

INTEGRATION = {
    "service": "simplex",
    "transport": "http",
    "config_vars": [
        "simplex_bridge_url",
        "simplex_default_contact",
        "simplex_bridge_token",
    ],
}


TIMEOUT_SECONDS = 20


class BridgeError(Exception):
    """The SimpleX forwarding bridge is unreachable or returned an error."""


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


def _match_default(senders, default):
    default = str(default or "").strip()
    if not default:
        return None
    for sender in senders:
        if sender["id"] == default or sender["display_name"] == default:
            return sender["id"]
    return None


def predict(ctx, request):
    try:
        payload = _get(ctx, "/inbox")
    except BridgeError as exc:
        return Prediction(text=f"bridge error: {exc}")
    senders = _senders(payload.get("messages") or [])
    if not senders:
        return Prediction(text="settled: no unread messages")
    chosen = _match_default(senders, ctx.config.get("simplex_default_contact", ""))
    if chosen is not None:
        return Prediction(text=f"contact: {chosen}")
    if len(senders) == 1:
        return Prediction(text=f"contact: {senders[0]['id']}")
    decision = DecisionRequest(
        state=request.text,
        question="Which conversation should I read the next SimpleX message from?",
        options=[Option(sender["id"], sender["display_name"]) for sender in senders],
    )
    result = ctx.engine.call(decision)
    return Prediction(
        text=f"contact: {result.selected}",
        decisions=[(decision, result)],
    )


def act(ctx, request, prediction):
    text = prediction.text if prediction else ""
    if text.startswith("bridge error:"):
        return ActionResult(action_log=text, new_state=request.text)
    if text == "settled: no unread messages":
        return ActionResult(
            action_log="simplex.next_message: no unread SimpleX messages.",
            new_state="no unread SimpleX messages",
        )
    contact = text.removeprefix("contact: ").strip()
    if not contact:
        return ActionResult(
            action_log=(
                "simplex.next_message aborted: no conversation resolved "
                f"({text or 'no prediction'})."
            ),
            new_state=request.text,
        )
    try:
        payload = _get(
            ctx, f"/inbox/next?contact={urllib.parse.quote(contact)}"
        )
    except BridgeError as exc:
        return ActionResult(
            action_log=f"simplex.next_message failed reading {contact!r}: {exc}",
            new_state=request.text,
        )
    message = payload.get("message")
    if not message:
        return ActionResult(
            action_log=f"simplex.next_message: no unread message from {contact!r}.",
            new_state="no unread SimpleX messages",
        )
    sender = message.get("display_name") or contact
    body = str(message.get("text") or "")
    report = f"SimpleX message from {sender}: {body}"
    return ActionResult(action_log=report, new_state=report)
