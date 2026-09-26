"""Messaging gateway: external messenger intake/reply for the SemIf agent.

A `GatewayAdapter` owns a transport (SimpleX first), and `GatewayService`
maps inbound text onto the scheduler's existing gate/score/queue/dispatch
pipeline and routes results, questions, and repairs back to the originating
chat. See `base.py` for the contract.
"""

from .base import GatewayAdapter, InboundMessage, OutboundMessage
from .service import GatewayService

__all__ = ["GatewayAdapter", "InboundMessage", "OutboundMessage", "GatewayService"]
