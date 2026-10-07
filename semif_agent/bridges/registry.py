"""The bridge catalog: discovery, prompt description, and standalone runner.

Every known bridge class is listed in `CATALOG`. `known_infos()` exposes their
static metadata, `describe_bridges()` renders the catalog block injected into
the code-generation prompts, and `run_bridges()` starts the enabled ones as a
standalone process (`python -m semif_agent.cli bridge`). Adding a bridge means
adding its class here and a `config.example.json` block; the gateway is never
touched.
"""

from __future__ import annotations

import threading

from .base import BridgeInfo, BridgeService
from .llm import LLMBridge
from .nextcloud import NextcloudBridge
from .simplex import SimplexBridge

#: Known bridge services. Order is presentation order.
CATALOG: tuple[type[BridgeService], ...] = (SimplexBridge, LLMBridge, NextcloudBridge)


def _class_for(name: str) -> type[BridgeService] | None:
    for bridge_class in CATALOG:
        if bridge_class.INFO.name == name:
            return bridge_class
    return None


def known_infos() -> list[BridgeInfo]:
    return [bridge_class.INFO for bridge_class in CATALOG]


def describe_bridges() -> str:
    """A prompt block describing every available bridge service.

    Injected into the skill-body, retry, regen, elicitation, and testgen
    prompts so codegen knows which real services it may build a skill against,
    how to reach and authenticate to each one, and which `ctx.config` variables
    carry its URL, token, and other values. This is the single source of bridge
    specifics — CODEGEN.md/TESTGEN.md carry only the generic pattern. Static: no
    bridge needs to be running or enabled.
    """
    lines = [
        "Available bridge services (call these over HTTP with urllib.request; "
        "never speak a service's native protocol directly; read every value "
        "from ctx.config):",
    ]
    for info in known_infos():
        lines.append(f"- {info.name} (service: {info.service}): {info.description}")
        lines.append(f"    base URL config var: {info.url_config_var}")
        if info.auth_header and info.auth_config_var:
            lines.append(
                f"    auth: send the value of {info.auth_config_var} in the "
                f"{info.auth_header} header when it is set"
            )
        docs = dict(info.config_var_docs)
        if info.config_vars:
            lines.append("    config vars:")
            for name in info.config_vars:
                description = docs.get(name)
                suffix = f" — {description}" if description else ""
                lines.append(f"      - {name}{suffix}")
        if info.endpoints:
            lines.append("    endpoints:")
            for endpoint in info.endpoints:
                lines.append(f"      - {endpoint}")
    return "\n".join(lines) + "\n"


def build_bridge(
    config: dict, name: str, trace=None, llm=None
) -> BridgeService:
    """Instantiate one bridge from the top-level `bridges.<name>` config block.

    `llm` is the scheduler's configured language-model client, handed to the LLM
    bridge so it reuses the top-level `llm` endpoint/model instead of duplicating
    that config. The Nextcloud bridge gets the top-level `nextcloud_*` values as a
    fallback, so the same account the calendar seeds use is configured once. Other
    bridges ignore both.
    """
    bridge_class = _class_for(name)
    if bridge_class is None:
        raise KeyError(f"unknown bridge: {name!r}")
    block = (config.get("bridges", {}) or {}).get(name, {}) or {}
    if bridge_class is LLMBridge:
        return LLMBridge(block, trace=trace, client=llm)
    if bridge_class is NextcloudBridge:
        return NextcloudBridge(
            block,
            trace=trace,
            fallback={
                "url": config.get("nextcloud_url"),
                "username": config.get("nextcloud_username"),
                "app_password": config.get("nextcloud_app_password"),
                "default_calendar": config.get("nextcloud_default_calendar"),
                "default_addressbook": config.get("nextcloud_default_addressbook"),
            },
        )
    return bridge_class(block, trace=trace)


def run_bridges(
    config: dict,
    names: list[str] | None = None,
    trace=None,
    llm=None,
) -> int:
    """Run the selected (+ enabled) bridges until interrupted.

    With explicit `names`, a bridge runs even if disabled is not set; with no
    names, only `enabled` bridges run. Returns a process exit code. `llm` is the
    scheduler's configured language-model client (see `build_bridge`).
    """
    blocks = config.get("bridges", {}) or {}
    selected: list[BridgeService] = []
    for bridge_class in CATALOG:
        name = bridge_class.INFO.name
        if names and name not in names:
            continue
        block = blocks.get(name, {}) or {}
        if not names and not block.get("enabled", False):
            continue
        bridge = build_bridge(config, name, trace=trace, llm=llm)
        ok, hint = bridge.check_requirements()
        if not ok:
            print(f"bridge {name} unavailable: {hint}")
            for running in selected:
                running.stop()
            return 1
        port = bridge.start()
        selected.append(bridge)
        print(
            f"bridge {name} listening on http://{bridge.host}:{port} "
            f"(skills use the {bridge_class.INFO.url_config_var} config var)"
        )
    if not selected:
        print("no bridge services enabled (set bridges.<name>.enabled)")
        return 1

    stop = threading.Event()
    try:
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        for bridge in selected:
            bridge.stop()
    return 0
