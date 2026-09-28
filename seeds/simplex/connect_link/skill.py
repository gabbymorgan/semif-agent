"""Show the SimpleX contact link others use to connect to this agent.

Real integration: the running messenger gateway serves a local HTTP bridge
(`messaging_bridge_url`). The gateway's own connection to the `simplex-chat`
daemon is the only thing that can read (or create) the bot's *user contact
address*; this body asks the bridge for it and reports the link honestly. It
never opens a WebSocket to the daemon and never invents a link: if the gateway
is not running or has no address, it says so.

`GET /address` shows the existing link and creates one on first call, returning
`{short_link, full_link, created}`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from semif_agent.skills import ActionResult, Prediction

INTEGRATION = {
    "service": "simplex",
    "transport": "http",
    "config_vars": ["messaging_bridge_url"],
}

TIMEOUT_SECONDS = 30


class BridgeError(Exception):
    """The local messaging bridge is unreachable or returned an error."""


def _base(ctx):
    url = str(ctx.config.get("messaging_bridge_url", "") or "").strip().rstrip("/")
    if not url:
        raise BridgeError("messaging_bridge_url is not configured")
    return url


def _get(ctx, path):
    request = urllib.request.Request(_base(ctx) + path)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()[:200]
        raise BridgeError(f"GET {path}: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise BridgeError(f"GET {path}: {exc}") from exc


def predict(ctx, request):
    try:
        payload = _get(ctx, "/health")
    except BridgeError as exc:
        return Prediction(text=f"bridge error: {exc}")
    if not payload.get("ok"):
        return Prediction(text="bridge error: gateway bridge is not healthy")
    return Prediction(text="ready: messenger bridge reachable")


def act(ctx, request, prediction):
    text = prediction.text if prediction else ""
    if text.startswith("bridge error:"):
        return ActionResult(action_log=text, new_state=request.text)
    try:
        payload = _get(ctx, "/address")
    except BridgeError as exc:
        return ActionResult(
            action_log=f"simplex.connect_link failed: {exc}",
            new_state=request.text,
        )
    short = str(payload.get("short_link") or "").strip()
    full = str(payload.get("full_link") or "").strip()
    if not (short or full):
        return ActionResult(
            action_log=(
                "simplex.connect_link: the gateway returned no SimpleX "
                "contact link; the bot has no address yet."
            ),
            new_state=request.text,
        )
    verb = "created" if payload.get("created") else "current"
    lines = [f"SimpleX contact link ({verb}): {short or full}"]
    if short and full:
        lines.append(f"Full link: {full}")
    lines.append(
        "Share this link with someone so they can add the agent and message it."
    )
    report = "\n".join(lines)
    return ActionResult(action_log=report, new_state=report)
