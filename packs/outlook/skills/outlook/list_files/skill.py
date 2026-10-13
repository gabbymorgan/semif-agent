"""List the user's OneDrive files.

Real integration: the local Outlook bridge (Microsoft Graph). Connection comes
from `ctx.config` (`outlook_bridge_url`, `outlook_bridge_token`); the body never
speaks Graph or OAuth directly. It lists a OneDrive folder (root by default) and
reports the entries, or the real failure honestly.
"""

from __future__ import annotations

import json
import re
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
_STOPWORDS = re.compile(
    r"\b(list|show|what'?s|whats|in|my|the|onedrive|files|file|folder|drive|of|me)\b",
    re.I,
)


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


def _folder(text):
    cleaned = re.sub(r"\s+", " ", _STOPWORDS.sub(" ", text or "")).strip()
    return cleaned or "/"


def act(ctx, request):
    path = _folder(request.text)
    try:
        data = _get(ctx, "/files", {"path": path})
    except OutlookBridgeError as exc:
        return ActionResult(action_log=str(exc), new_state=request.text)

    entries = data.get("entries") or []
    if not entries:
        return ActionResult(
            action_log=f"{data.get('path') or path} is empty.",
            new_state=f"{data.get('path') or path} is empty.",
        )

    lines = []
    for entry in entries[:20]:
        name = entry.get("name") or "(unnamed)"
        kind = "folder" if entry.get("is_folder") else "file"
        size = entry.get("size")
        detail = f"{kind}"
        if kind == "file" and size is not None:
            detail += f", {size} bytes"
        lines.append(f"- {name} ({detail})")
    report = f"Files in {data.get('path') or path}:\n" + "\n".join(lines)
    return ActionResult(action_log=report, new_state=report)
