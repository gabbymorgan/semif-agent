"""Send a text message to a SimpleX contact through the local SimpleX bridge.

Real integration: the simplex bridge over HTTP. The bridge owns the SimpleX
daemon connection and its credentials; this body only calls the bridge's HTTP
API with the base URL and token resolved from `ctx.config`.
"""

from __future__ import annotations

import json
import re
import urllib.error
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
        "llm_bridge_url",
        "llm_bridge_token",
    ],
}

CONTRACT = {
    "simplex_bridge_url": (
        "Base URL of the local SimpleX forwarding bridge that fronts the "
        "user's simplex daemon, e.g. http://127.0.0.1:5227 (no trailing slash "
        "needed)."
    ),
    "simplex_bridge_token": (
        "Shared secret for the SimpleX bridge, if one is configured; sent as "
        "the X-Semif-Token header. Leave blank when the bridge requires no auth."
    ),
    "simplex_default_contact": (
        "Default SimpleX contact (contact id or display name) used when the "
        "request does not clearly name a recipient; leave blank to always use "
        "the strongest candidate."
    ),
    "llm_bridge_url": (
        "Base URL of the local LLM bridge, e.g. http://127.0.0.1:5229 (no "
        "trailing slash needed); used to write the message text when the "
        "request does not state it."
    ),
    "llm_bridge_token": (
        "Shared secret for the LLM bridge, if one is configured; sent as the "
        "X-Semif-Token header. Leave blank when the bridge requires no auth."
    ),
}

# A contact sub-decision winner is used only when its conditional probability
# clears this bar; otherwise the configured default contact wins.
CONFIDENCE_THRESHOLD = 0.5

_HTTP_TIMEOUT = 20

# Ordered: the literal message text the user dictated wins over anything else.
_MESSAGE_PATTERNS = (
    r"\b(?:saying|say|that says|which says|that reads|reads)\b\s*[:\-]?\s*(.+)$",
    r'["\u201c](.+?)["\u201d]',
    r"\b(?:message|text)\s*(?:is|:)\s*(.+)$",
)


# --------------------------------------------------------------------------
# bridge transport
# --------------------------------------------------------------------------


def _headers(token: str, has_body: bool) -> dict:
    headers = {"Accept": "application/json"}
    if has_body:
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-Semif-Token"] = token
    return headers


def _http_json(url: str, token: str, payload=None):
    """GET the url, or POST the payload as JSON, and decode the JSON reply."""
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=_headers(token, data is not None),
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
        raw = response.read().decode("utf-8")
    if not raw.strip():
        return {}
    return json.loads(raw)


def _error_detail(exc) -> str:
    """Best-effort human-readable detail from an HTTP error response."""
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:
        raw = ""
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("error"):
            return str(parsed["error"])
        return raw.strip()
    return str(getattr(exc, "reason", "") or exc)


# --------------------------------------------------------------------------
# contacts
# --------------------------------------------------------------------------


def _contact_id(contact) -> str:
    return str(contact.get("id") or contact.get("display_name") or "").strip()


def _display(contact) -> str:
    return str(contact.get("display_name") or contact.get("id") or "").strip()


def _label(contact) -> str:
    """The unique name for a contact: the daemon's local name when known.

    Two peers can share a profile display name; the local name is unique, so it
    is the right label for a SemIf option and the right thing to match on.
    """
    return str(
        contact.get("local_name") or _display(contact) or contact.get("id") or ""
    ).strip()


def _is_healthy(contact) -> bool:
    """A contact is healthy when its connection is not known-bad.

    `connected` is absent when the source did not report it (treat as unknown,
    i.e. not disqualifying); `auth_errors` counts failed sends on the connection
    and marks a stale/duplicate peer.
    """
    if contact.get("connected") is False:
        return False
    try:
        return int(contact.get("auth_errors") or 0) == 0
    except (TypeError, ValueError):
        return True


def _mentions(query: str, name: str) -> bool:
    if not name:
        return False
    pattern = r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)"
    return re.search(pattern, query.lower()) is not None


def _match_contact(contacts, ref: str):
    """Resolve an id or display name against the bridge's contact list."""
    ref = (ref or "").strip()
    if not ref:
        return None
    lowered = ref.lower()
    for contact in contacts:
        if _contact_id(contact).lower() == lowered:
            return contact
    exact = [
        c
        for c in contacts
        if _label(c).lower() == lowered or _display(c).lower() == lowered
    ]
    if exact:
        return _preferred(exact)
    matches = [
        c
        for c in contacts
        if lowered in _display(c).lower() or lowered in _label(c).lower()
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return _preferred(matches)
    return None


def _preferred(candidates):
    """Prefer a healthy candidate (the others are likely stale duplicates)."""
    healthy = [c for c in candidates if _is_healthy(c)]
    return (healthy or candidates)[0]


def _find_by_id(contacts, contact_id):
    wanted = str(contact_id or "")
    for contact in contacts:
        if _contact_id(contact) == wanted:
            return contact
    return None


def _confidence(result, option_id) -> float:
    prob = getattr(result, "prob", None)
    if callable(prob):
        try:
            return float(prob(option_id))
        except Exception:
            return 0.0
    probs = getattr(result, "probs", None)
    if isinstance(probs, dict):
        try:
            return float(probs.get(option_id, 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


# --------------------------------------------------------------------------
# message text
# --------------------------------------------------------------------------


def _message_from_query(query: str) -> str:
    """Pull the literal message text out of the user's request, if stated."""
    for pattern in _MESSAGE_PATTERNS:
        match = re.search(pattern, query, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip().strip("\"'\u201c\u201d").strip()
            if text:
                return text
    return ""


def _llm_message_text(ctx, query: str):
    """Ask the LLM bridge to write the message body. Returns (text, reason)."""
    base_url = str(ctx.config.get("llm_bridge_url", "") or "").strip().rstrip("/")
    if not base_url:
        return None, "no llm bridge is configured"
    token = str(ctx.config.get("llm_bridge_token", "") or "")
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write the exact text of a chat message a user wants to "
                    "send. Reply with the message text only: no quotes, no "
                    "preamble, no explanation."
                ),
            },
            {"role": "user", "content": query},
        ],
        "max_tokens": 256,
    }
    try:
        reply = _http_json(f"{base_url}/chat", token, payload)
    except urllib.error.HTTPError as exc:
        return None, f"the llm bridge returned HTTP {exc.code} {_error_detail(exc)}"
    except (urllib.error.URLError, ValueError, OSError) as exc:
        return None, f"could not reach the llm bridge: {exc}"
    if not isinstance(reply, dict):
        return None, "the llm bridge returned an unexpected reply"
    text = str(reply.get("text", "")).strip().strip('"').strip()
    if not text:
        return None, "the llm bridge returned no text"
    return text, ""


# --------------------------------------------------------------------------
# action
# --------------------------------------------------------------------------


def act(ctx, request) -> ActionResult:
    """Send the requested message to the requested SimpleX contact."""
    base_url = str(ctx.config["simplex_bridge_url"]).strip().rstrip("/")
    token = str(ctx.config.get("simplex_bridge_token", "") or "")
    default_ref = str(ctx.config.get("simplex_default_contact", "") or "").strip()

    query = (request.text or "").strip()
    user_input = str(getattr(request, "user_input", "") or "").strip()
    decisions = []

    # --- the bridge's real contact list ------------------------------------
    try:
        payload = _http_json(f"{base_url}/contacts", token)
    except urllib.error.HTTPError as exc:
        return ActionResult(
            action_log=(
                f"the simplex bridge at {base_url} returned HTTP {exc.code} "
                f"while listing contacts: {_error_detail(exc)}"
            ),
            new_state=request.text,
        )
    except (urllib.error.URLError, ValueError, OSError) as exc:
        return ActionResult(
            action_log=(
                f"could not reach the simplex bridge at {base_url} to list "
                f"contacts: {exc}"
            ),
            new_state=request.text,
        )

    raw_contacts = payload.get("contacts") if isinstance(payload, dict) else None
    contacts = (
        [c for c in raw_contacts if isinstance(c, dict)]
        if isinstance(raw_contacts, list)
        else []
    )

    # --- resolve the recipient --------------------------------------------
    default_contact = _match_contact(contacts, default_ref)
    named = [
        c
        for c in contacts
        if _mentions(query, _display(c)) or _mentions(query, _label(c))
    ]

    chosen = None
    used_input_for_contact = False

    if named or default_contact is not None:
        candidates = list(named)
        if default_contact is not None and all(
            _contact_id(c) != _contact_id(default_contact) for c in candidates
        ):
            candidates.append(default_contact)

        healthy = [c for c in candidates if _is_healthy(c)]
        if len(candidates) == 1:
            chosen = candidates[0]
        elif len(healthy) == 1:
            # Exactly one candidate's connection is healthy (the rest are stale
            # duplicates): pick it deterministically rather than coin-flipping.
            chosen = healthy[0]
        else:
            decision = DecisionRequest(
                state=query,
                question="Which SimpleX contact should the message be sent to?",
                options=[
                    Option(_contact_id(c), _label(c) or _contact_id(c))
                    for c in candidates
                ],
            )
            result = ctx.engine.call(decision)
            decisions.append((decision, result))
            selected_id = getattr(result, "selected", None)
            selected = _find_by_id(candidates, selected_id)
            if (
                selected is not None
                and _confidence(result, selected_id) >= CONFIDENCE_THRESHOLD
            ):
                chosen = selected
            elif default_contact is not None:
                chosen = default_contact
            else:
                chosen = selected if selected is not None else candidates[0]
    elif user_input:
        # The human answered the question asked on the previous pass.
        chosen = _match_contact(contacts, user_input) or {
            "id": user_input,
            "display_name": user_input,
        }
        used_input_for_contact = True

    if chosen is None:
        known = ", ".join(_display(c) for c in contacts if _display(c)) or "none"
        return ActionResult(
            action_log=(
                "the request does not identify a SimpleX contact to message; "
                f"the bridge knows these contacts: {known}"
            ),
            new_state=request.text,
            needs_input="Which SimpleX contact should I message?",
            decisions=decisions,
        )

    recipient = _contact_id(chosen) or _display(chosen)
    display_name = _display(chosen) or recipient

    # --- resolve the message text -----------------------------------------
    message_text = _message_from_query(query)
    if not message_text and user_input and not used_input_for_contact:
        message_text = user_input
    if not message_text:
        written, reason = _llm_message_text(ctx, query)
        if written:
            message_text = written
        else:
            return ActionResult(
                action_log=(
                    f"the request does not state what the message to "
                    f"{display_name} should say and no message text could be "
                    f"written ({reason})"
                ),
                new_state=request.text,
                needs_input=f"What should the message to {display_name} say?",
                decisions=decisions,
            )

    # --- send it through the bridge ---------------------------------------
    try:
        response = _http_json(
            f"{base_url}/send", token, {"recipient": recipient, "text": message_text}
        )
    except urllib.error.HTTPError as exc:
        return ActionResult(
            action_log=(
                f"the simplex bridge refused to send to {display_name}: "
                f"HTTP {exc.code} {_error_detail(exc)}"
            ),
            new_state=request.text,
            decisions=decisions,
        )
    except (urllib.error.URLError, ValueError, OSError) as exc:
        return ActionResult(
            action_log=(
                f"could not reach the simplex bridge at {base_url} to send the "
                f"message to {display_name}: {exc}"
            ),
            new_state=request.text,
            decisions=decisions,
        )

    if not isinstance(response, dict):
        return ActionResult(
            action_log=(
                f"the simplex bridge returned an unexpected reply while sending "
                f"to {display_name}"
            ),
            new_state=request.text,
            decisions=decisions,
        )

    contact_id = str(response.get("contact_id") or recipient)
    status = str(response.get("status") or "")
    if not response.get("ok"):
        detail = str(response.get("error") or "").strip() or json.dumps(response)
        return ActionResult(
            action_log=(
                f"the message to {display_name} (contact {contact_id}) did not "
                f"send (status {status or 'unknown'}: {detail})"
            ),
            new_state=json.dumps(
                {
                    "sent": False,
                    "contact_id": contact_id,
                    "display_name": display_name,
                    "status": status,
                    "error": detail,
                    "text": message_text,
                }
            ),
            decisions=decisions,
        )

    return ActionResult(
        action_log=(
            f"sent the message {message_text!r} to {display_name} "
            f"(contact {contact_id}, status {status or 'ok'}) through the "
            f"simplex bridge"
        ),
        new_state=json.dumps(
            {
                "sent": True,
                "contact_id": contact_id,
                "display_name": display_name,
                "status": status,
                "text": message_text,
            }
        ),
        decisions=decisions,
    )
