"""Open a web page in the browser bridge and report what is on it.

Real integration: the local browser bridge (`browser_bridge_url`,
`browser_bridge_token`); the body never drives a browser directly. It opens the
URL from the request, then reads the page's text through the bridge. The URL is
taken from the request, or asked for if the request does not contain one. A
missing URL or a bridge failure is reported honestly — never a fabricated page.
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
EXCERPT_CHARS = 1200
_URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.IGNORECASE)


class BrowserBridgeError(Exception):
    """A real failure talking to the local browser bridge."""


def _headers(ctx):
    token = str(ctx.config.get("browser_bridge_token", "") or "").strip()
    headers = {"Accept": "application/json"}
    if token:
        headers["X-Semif-Token"] = token
    return headers


def _base(ctx):
    base = str(ctx.config.get("browser_bridge_url", "") or "").strip().rstrip("/")
    if not base:
        raise BrowserBridgeError("browser_bridge_url is not configured")
    return base


def _request(ctx, path, payload=None, params=None):
    url = _base(ctx) + path
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    headers = _headers(ctx)
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
    """The URL to open, or None. Uses the answer after a needs_input pause."""
    if request.user_input is not None:
        answer = (request.user_input or "").strip()
        if not answer:
            return None
        match = _URL_RE.search(answer)
        return match.group(0) if match else answer
    match = _URL_RE.search(request.text or "")
    return match.group(0) if match else None


def act(ctx, request):
    url = _resolve_url(request)
    if not url:
        return ActionResult(
            action_log="no web address in the request",
            new_state=request.text,
            needs_input="Which web address (URL) should I open?",
        )
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url

    try:
        opened = _request(ctx, "/goto", {"url": url})
        page = _request(ctx, "/content", params={"format": "text", "max_chars": EXCERPT_CHARS})
    except BrowserBridgeError as exc:
        return ActionResult(action_log=str(exc), new_state=request.text)

    title = opened.get("title") or page.get("title") or url
    final_url = page.get("url") or opened.get("url") or url
    status = opened.get("status")
    body = (page.get("content") or "").strip()
    excerpt = body[:EXCERPT_CHARS]
    if len(body) > EXCERPT_CHARS or page.get("truncated"):
        excerpt = excerpt.rstrip() + " …"

    summary = f"Opened {title} ({final_url})"
    if status is not None:
        summary += f" [HTTP {status}]"
    if excerpt:
        report = f"{summary}:\n{excerpt}"
    else:
        report = f"{summary}: the page has no readable text."
    return ActionResult(action_log=report, new_state=report)
