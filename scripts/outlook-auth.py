#!/usr/bin/env python3
"""Authorize the Outlook bridge with Microsoft (OAuth2 device code flow).

The bridge reads the user's Microsoft account through Microsoft Graph; that
needs an access token, obtained once here with the **device code flow** (a public
client, no secret):

    1. register an Entra ID app (see below) and put its client id in
       `config.json` under `bridges.outlook.client_id`;
    2. run this script;
    3. open the printed URL on any device, enter the printed code, and sign in;
    4. the resulting tokens are saved (0600) to `.runtime/outlook/token.json`,
       which the bridge loads and refreshes automatically.

One-time app registration (Entra ID / Azure AD):

    - "App registrations" -> "New registration".
    - Supported account types: include personal Microsoft accounts if the user
      has an @outlook.com/@hotmail.com account (e.g. "Accounts in any
      organizational directory and personal Microsoft accounts").
    - Authentication -> "Allow public client flows" = Yes (no redirect URI is
      needed for device code).
    - No API permissions need adding by hand: the script requests the delegated
      scopes it needs and the user consents during sign-in.

Usage (from the checkout):

    .runtime/venv/bin/python scripts/outlook-auth.py [--config config.json]
                                                     [--client-id ID]
                                                     [--tenant common]
                                                     [--token-path PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from semif_agent.bridges.outlook_client import (  # noqa: E402
    DEFAULT_TENANT,
    OutlookError,
    run_device_code_flow,
    save_token,
)

DEFAULT_TOKEN_PATH = str(REPO_ROOT / ".runtime" / "outlook" / "token.json")


def _block(config_path: str) -> dict:
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return {}
    block = (config.get("bridges", {}) or {}).get("outlook", {}) or {}
    return block if isinstance(block, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--client-id", default=None, help="override bridges.outlook.client_id")
    parser.add_argument("--tenant", default=None, help="override bridges.outlook.tenant (default common)")
    parser.add_argument("--token-path", default=None, help="where to save the tokens")
    args = parser.parse_args(argv)

    block = _block(args.config)
    client_id = str(args.client_id or block.get("client_id") or "").strip()
    tenant = str(args.tenant or block.get("tenant") or DEFAULT_TENANT).strip() or DEFAULT_TENANT
    token_path = str(args.token_path or block.get("token_path") or DEFAULT_TOKEN_PATH)
    scopes = block.get("scopes") or None

    if not client_id:
        print(
            "no Microsoft client id configured: set bridges.outlook.client_id in "
            f"{args.config} (or pass --client-id)",
            file=sys.stderr,
        )
        return 1

    try:
        token = run_device_code_flow(client_id, tenant, scopes)
    except OutlookError as exc:
        print(f"authorization failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return 1

    save_token(token_path, token)
    print(f"saved Outlook authorization to {token_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
