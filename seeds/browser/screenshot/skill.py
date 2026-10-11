"""Take a screenshot of a web page through the browser bridge.

Real integration: the local browser bridge (`browser_bridge_url`,
`browser_bridge_token`); the body never drives a browser directly. If the
request names a URL it opens that page first, otherwise it captures the page the
browser is already on, and reports the real path of the saved PNG. A bridge
failure is reported honestly — never a fabricated path.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

from semif_agent.skills import ActionResult

INTEGRATION = {
    "service": "browser",
    "transport": "http",
    "config_vars": ["browser_bridge_url", "browser_bridge_token"],
}

CONTRACT = {
    "browser_bridge_url": "Base URL of the local browser bridge, e.g. http://127.0.0.1:5232 (no trailing slash needed).",
    "browser_bridge_token": "Shared secret for the browser bridge, if one is configured; sent as the X-Semif-Token header. Leave blank when the bridge requires no auth.",
}

TIMEOUT_SECONDS = 60
_URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.IGNORECASE)


class BrowserBridgeError(Exception):
    """A real failure talking to the local browser bridge."""


def _request(ctx, path, payload=None, params=None):
    base = str(ctx.config.get("browser_bridge_url", "") or "").strip().rstrip("/")
    if not base:
        raise BrowserBridgeError("browser_bridge_url is not configured")
    token = str(ctx.config.get("browser_bridge_token", "") or "").strip()
    url = base + path
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    headers = {"Accept": "application/json"}
    if token:
        headers["X-Semif-Token"] = token
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    http_request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(http_request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()[:200]
        raise BrowserBridgeError(f"{method} {path}: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise BrowserBridgeError(f"{method} {path}: {exc}") from exc


def _resolve_url(request):
    source = request.user_input if request.user_input is not None else request.text
    match = _URL_RE.search(source or "")
    return match.group(0) if match else None


def act(ctx, request):
    url = _resolve_url(request)
    try:
        if url:
            _request(ctx, "/goto", {"url": url})
        shot = _request(ctx, "/screenshot", params={"full_page": "true"})
    except BrowserBridgeError as exc:
        return ActionResult(action_log=str(exc), new_state=request.text)

    path = shot.get("path") or ""
    title = shot.get("title") or ""
    page_url = shot.get("url") or url or "the current page"
    size = shot.get("bytes")
    where = f"{title} ({page_url})" if title else str(page_url)
    detail = f"Saved a screenshot of {where} to {path}"
    if isinstance(size, int):
        detail += f" ({size} bytes)"
    return ActionResult(action_log=detail, new_state=detail)
