"""Send an Outlook email on the user's behalf, after a confirmation.

Real integration: the local Outlook bridge (Microsoft Graph) to resolve the
recipient and send, and the local LLM bridge to turn the request into a
`{recipient, subject, body}` draft. Connection comes from `ctx.config`
(`outlook_bridge_url`, `outlook_bridge_token`, `llm_bridge_url`,
`llm_bridge_token`); the body never speaks Graph, OAuth, or the model's protocol
directly.

Because sending is irreversible, the body **asks for confirmation before it
sends**: it composes the message, resolves the recipient (an explicit address is
used as-is; otherwise the contacts are searched and, when several match, a SemIf
choice picks one), then pauses with "Send ...? (yes/no)". A `yes` sends for real;
anything else cancels. A missing recipient is asked for.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from semif_agent.decisions import DecisionRequest, Option
from semif_agent.skills import ActionResult

INTEGRATION = {
    "service": "outlook",
    "transport": "http",
    "config_vars": [
        "outlook_bridge_url",
        "outlook_bridge_token",
        "llm_bridge_url",
        "llm_bridge_token",
    ],
}

CONTRACT = {
    "outlook_bridge_url": "Base URL of the local Outlook bridge, e.g. http://127.0.0.1:5231 (no trailing slash needed).",
    "outlook_bridge_token": "Shared secret for the Outlook bridge, if one is configured; sent as the X-Semif-Token header. Leave blank when the bridge requires no auth.",
    "llm_bridge_url": "Base URL of the local LLM bridge, e.g. http://127.0.0.1:5229 (no trailing slash needed); used to compose the message.",
    "llm_bridge_token": "Shared secret for the LLM bridge, if one is configured; sent as the X-Semif-Token header. Leave blank when the bridge requires no auth.",
}

TIMEOUT_SECONDS = 30
_YES = {"y", "yes", "send", "ok", "okay", "sure", "confirm"}


class OutlookBridgeError(Exception):
    """A real failure talking to the local Outlook or LLM bridge."""


def _request(ctx, service, path, *, method="GET", params=None, payload=None):
    if service == "llm":
        base = str(ctx.config.get("llm_bridge_url", "") or "").strip().rstrip("/")
        token = str(ctx.config.get("llm_bridge_token", "") or "").strip()
    else:
        base = str(ctx.config.get("outlook_bridge_url", "") or "").strip().rstrip("/")
        token = str(ctx.config.get("outlook_bridge_token", "") or "").strip()
    if not base:
        raise OutlookBridgeError(f"the {service} bridge URL is not configured")
    url = base + path
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-Semif-Token"] = token
    http_request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(http_request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()[:200]
        raise OutlookBridgeError(f"{method} {path}: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise OutlookBridgeError(f"{method} {path}: {exc}") from exc


def _compose(ctx, request):
    """Ask the LLM bridge for the recipient, subject, and body."""
    messages = [
        {
            "role": "system",
            "content": (
                "Turn the user's request into an email. Reply with ONLY a JSON "
                'object: {"recipient": "<name or email address>", '
                '"subject": "<short subject>", "body": "<the message>"}.'
            ),
        },
        {"role": "user", "content": request.text},
    ]
    data = _request(ctx, "llm", "/chat", method="POST", payload={"messages": messages})
    text = str(data.get("text") or "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise OutlookBridgeError("the LLM bridge returned no JSON object")
    try:
        draft = json.loads(text[start : end + 1])
    except ValueError as exc:
        raise OutlookBridgeError(f"the LLM bridge returned invalid JSON: {exc}") from exc
    if not isinstance(draft, dict):
        raise OutlookBridgeError("the LLM bridge returned JSON that is not an object")
    return {
        "recipient": str(draft.get("recipient") or "").strip(),
        "subject": str(draft.get("subject") or "").strip(),
        "body": str(draft.get("body") or "").strip(),
    }


def _resolve_address(ctx, decisions, recipient):
    """An email address for `recipient`: used directly, or chosen from contacts."""
    if "@" in recipient:
        return recipient
    data = _request(ctx, "outlook", "/contacts", params={"query": recipient})
    contacts = [c for c in (data.get("contacts") or []) if c.get("emails")]
    if not contacts:
        return None
    if len(contacts) == 1:
        return str(contacts[0]["emails"][0])
    options = [
        Option(str(contact["emails"][0]), str(contact.get("display_name") or contact["emails"][0]))
        for contact in contacts
    ]
    decision = DecisionRequest(
        state=recipient,
        question="Which contact should the message be sent to?",
        options=options,
    )
    result = ctx.engine.call(decision)
    decisions.append((decision, result))
    return str(result.selected)


def _ask(request, question):
    return ActionResult(action_log="waiting for input", new_state=request.text, needs_input=question)


def act(ctx, request):
    meta = request.meta
    awaiting = meta.get("send_message_awaiting")
    decisions = []

    # Resume: the user answered the recipient/address/confirmation question.
    if request.user_input is not None and awaiting:
        if awaiting == "confirm":
            ready = meta.get("send_message_ready") or {}
            meta.pop("send_message_awaiting", None)
            if request.user_input.strip().lower() not in _YES:
                return ActionResult(
                    action_log="Cancelled; the message was not sent.",
                    new_state="cancelled",
                )
            try:
                result = _request(
                    ctx,
                    "outlook",
                    "/mail/send",
                    method="POST",
                    payload={"to": [ready.get("to", "")], "subject": ready.get("subject", ""), "body": ready.get("body", "")},
                )
            except OutlookBridgeError as exc:
                return ActionResult(action_log=str(exc), new_state=request.text)
            report = f"Sent {ready.get('subject')!r} to {ready.get('to')}."
            meta.pop("send_message_ready", None)
            return ActionResult(action_log=report, new_state=report)
        meta.setdefault("send_message_answers", {})[awaiting] = request.user_input.strip()
        meta.pop("send_message_awaiting", None)

    answers = meta.get("send_message_answers") or {}

    draft = meta.get("send_message_composed")
    if draft is None:
        try:
            draft = _compose(ctx, request)
        except OutlookBridgeError as exc:
            return ActionResult(action_log=str(exc), new_state=request.text)
        meta["send_message_composed"] = draft

    recipient = answers.get("recipient") or draft["recipient"]
    if not recipient:
        meta["send_message_awaiting"] = "recipient"
        return _ask(request, "Who should I send the message to?")

    address = answers.get("address")
    if not address:
        try:
            address = _resolve_address(ctx, decisions, recipient)
        except OutlookBridgeError as exc:
            return ActionResult(action_log=str(exc), new_state=request.text)
    if not address:
        meta["send_message_awaiting"] = "address"
        return _ask(request, f"What's the email address for {recipient!r}?")

    meta["send_message_ready"] = {"to": address, "subject": draft["subject"], "body": draft["body"]}
    meta["send_message_awaiting"] = "confirm"
    action = _ask(request, f"Send {draft['subject']!r} to {address}? (yes/no)")
    action.decisions = decisions
    return action
