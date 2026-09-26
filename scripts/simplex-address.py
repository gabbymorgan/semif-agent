#!/usr/bin/env python3
"""Print (creating if needed) the SimpleX bot's contact address.

The gateway talks to a local `simplex-chat` daemon running as a bot
(`--create-bot-display-name`, see simplex_chat in config.json). To connect a
phone/client to the bot you need the bot's *user* contact address, not a chat
relay address; this queries the daemon's WebSocket API, creating the address on
first run, and prints the short link (and full link). Requires the daemon to be
running.

Usage:
    .runtime/venv/bin/python scripts/simplex-address.py [--config config.json]
                                                        [--ws-url ws://...]
                                                        [--user-id 1]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


def _ws_url(config_path: str) -> str:
    try:
        with open(config_path) as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return "ws://127.0.0.1:5226"
    return str(cfg.get("gateway", {}).get("simplex", {}).get("ws_url") or "ws://127.0.0.1:5226")


async def _resolve(ws_url: str, user_id: int) -> dict:
    try:
        import websockets
    except ImportError:
        raise SystemExit("the `websockets` package is required (pip install websockets)")
    async with websockets.connect(ws_url, max_size=None) as ws:

        async def cmd(command: str) -> dict:
            await ws.send(json.dumps({"corrId": "addr", "cmd": command}))
            raw = await asyncio.wait_for(ws.recv(), timeout=20)
            return json.loads(raw).get("resp", {}) or {}

        resp = await cmd(f"/_show_address {user_id}")
        if resp.get("type") == "userContactLink":
            return resp.get("contactLink", {}) or {}
        # No address yet -> create one (APICreateMyAddress).
        resp = await cmd(f"/_address {user_id}")
        if resp.get("type") == "userContactLinkCreated":
            return {"connLinkContact": resp.get("connLinkContact", {}) or {}}
        raise SystemExit(f"could not read or create address: {json.dumps(resp)[:300]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--ws-url", default=None)
    ap.add_argument("--user-id", type=int, default=1)
    args = ap.parse_args()

    ws_url = args.ws_url or _ws_url(args.config)
    try:
        link = asyncio.run(_resolve(ws_url, args.user_id))
    except OSError as exc:
        print(f"cannot reach simplex-chat at {ws_url}: {exc}", file=sys.stderr)
        return 1
    except asyncio.TimeoutError:
        print(f"simplex-chat at {ws_url} did not answer", file=sys.stderr)
        return 1

    conn = link.get("connLinkContact", {}) or link
    short = conn.get("connShortLink")
    full = conn.get("connFullLink")
    if short:
        print(short)
    if full:
        print(full)
    if not (short or full):
        print("no contact address found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
