"""Third-party API bridges: a secure local layer between skills and services.

A **bridge service** is a standalone process that exposes one third-party system
(SimpleX, the local language model, Nextcloud, Microsoft Graph / Outlook, a real
web browser via Playwright) to codegen-authored skill bodies over a local,
token-guarded HTTP API. A skill body can call a plain HTTP API directly; a
bridge is a deliberate choice for a system that needs state (especially auth
state), a native or heavy protocol, or one sanitized surface. When a bridge
fronts a service, a generated body never speaks that system's native protocol
(WebSocket, SMTP, CalDAV, an OpenAI-compatible chat API, a CLI's private flags)
directly: it calls a small, documented, uniform HTTP surface instead, and the
bridge owns the messy integration details and credentials.

Bridges **ship with their pack**: the core owns only the shared contract
(``bridges/base.py``) and the discovery registry (``bridges/registry.py``); a
bridge implementation lives in ``packs/<name>/bridge/`` and is imported by
convention from the pack that declares it. The one exception is the shared,
pack-less LLM bridge (``bridges/llm.py``), which many packs call.

Two boundaries matter:

- **Bridges are not the gateway.** The command gateway only takes commands and
  replies; it must never grow read/send/invite endpoints. Bridges run as their
  own processes (`python -m semif_agent.cli bridge`) against their own service
  daemons/profiles. This isolation is non-negotiable (see AGENTS.md).
- **The catalog is visible to codegen.** `describe_bridges()` is injected into
  every code-generation prompt (body, retry, regen, testgen) and is
  the single source of bridge specifics: the model learns which real services it
  may build against, each one's base-URL config var, auth header/token var,
  config-var docs, and its endpoints with request/response/error shapes. CODEGEN.md
  and TESTGEN.md carry only the generic pattern.
"""

from .base import BridgeInfo, BridgeService, checkout_root
from .inbox import MessagingInbox
from .registry import (
    build_bridge,
    derived_config_vars,
    describe_bridges,
    discover_bridge_classes,
    known_infos,
    load_pack_bridge_class,
    load_pack_bridge_module,
    load_pack_bridge_package,
    run_bridges,
    set_bridge_packs,
)

__all__ = [
    "BridgeInfo",
    "BridgeService",
    "MessagingInbox",
    "build_bridge",
    "checkout_root",
    "derived_config_vars",
    "describe_bridges",
    "discover_bridge_classes",
    "known_infos",
    "load_pack_bridge_class",
    "load_pack_bridge_module",
    "load_pack_bridge_package",
    "run_bridges",
    "set_bridge_packs",
]
