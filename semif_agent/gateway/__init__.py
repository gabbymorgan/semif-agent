"""Messaging gateway: external messenger intake/reply for the SemIf agent.

The gateway is the agent's **command** surface and nothing else. A
`GatewayAdapter` owns a transport (SimpleX first), and `GatewayService` maps
inbound text onto the scheduler's existing gate/score/queue/dispatch pipeline
and routes results, questions, and repairs back to the originating chat.

Messaging *UX* — invite links, reading a contact's messages, composing sends —
does not live here; it belongs to the standalone bridge services in
`semif_agent.bridges`. Keeping the command surface free of that functionality
is a hard boundary (see AGENTS.md "gateway isolation").
"""

from .base import GatewayAdapter, InboundMessage, OutboundMessage
from .service import GatewayService

__all__ = [
    "GatewayAdapter",
    "InboundMessage",
    "OutboundMessage",
    "GatewayService",
]

