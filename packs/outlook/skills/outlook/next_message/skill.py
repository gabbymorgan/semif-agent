"""Read the latest messages from the user's Outlook inbox.

Real integration: the local Outlook bridge (Microsoft Graph). Connection comes
from `ctx.config` (`outlook_bridge_url`, `outlook_bridge_token`); the body never
speaks Graph or OAuth directly. It asks the bridge for the newest inbox messages
and reports them, or reports the real failure honestly.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from semif_agent.skills import ActionResult

INTEGRATION = {
    "service": "outlook",
    "transport": "http",
    "config_vars": ["outlook_bridge_url", "outlook_bridge_token"],
}

CONTRACT = {
    "outlook_bridge_url": "Base URL of the local Outlook bridge, e.g. http://127.0.0.1:5231 (no trailing slash needed).",
    "outlook_bridge_token": "Shared secret for the Outlook bridge, if one is configured; sent as the X-Semif-Token header. Leave blank when the bridge requires no auth.",
}

TIMEOUT_SECONDS = 30
DEFAULT_COUNT = 5


class OutlookBridgeError(Exception):
    """A real failure talking to the local Outlook bridge."""


def _get(ctx, path, params=None):
    base = str(ctx.config.get("outlook_bridge_url", "") or "").strip().rstrip("/")
    if not base:
        raise OutlookBridgeError("outlook_bridge_url is not configured")
    token = str(ctx.config.get("outlook_bridge_token", "") or "").strip()
    url = base + path
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    headers = {"Accept": "application/json"}
    if token:
        headers["X-Semif-Token"] = token
    http_request = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(http_request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()[:200]
        raise OutlookBridgeError(f"GET {path}: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise OutlookBridgeError(f"GET {path}: {exc}") from exc


def act(ctx, request):
    try:
        data = _get(ctx, "/mail/messages", {"folder": "inbox", "count": DEFAULT_COUNT})
    except OutlookBridgeError as exc:
        return ActionResult(action_log=str(exc), new_state=request.text)

    messages = data.get("messages") or []
    if not messages:
        return ActionResult(action_log="The inbox is empty.", new_state="The inbox is empty.")

    lines = []
    for message in messages[:DEFAULT_COUNT]:
        sender = message.get("from") or "unknown sender"
        subject = message.get("subject") or "(no subject)"
        received = message.get("received") or ""
        flag = " (unread)" if not message.get("is_read") else ""
        lines.append(f"- {sender}: {subject}{flag} [{received}]")
    report = "Recent inbox messages:\n" + "\n".join(lines)
    return ActionResult(action_log=report, new_state=report)
