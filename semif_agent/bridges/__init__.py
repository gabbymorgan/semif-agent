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

from .base import BridgeInfo, BridgeService
from .inbox import MessagingInbox
from .registry import (
    build_bridge,
    derived_config_vars,
    describe_bridges,
    known_infos,
    run_bridges,
)

__all__ = [
    "BridgeInfo",
    "BridgeService",
    "MessagingInbox",
    "build_bridge",
    "derived_config_vars",
    "describe_bridges",
    "known_infos",
    "run_bridges",
]
