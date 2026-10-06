"""The messenger command-gateway adapter contract.

An adapter owns one command transport (SimpleX WebSocket, and later Telegram,
Signal, ...). It delivers inbound text as `InboundMessage` and drains outbound
replies from a stdlib queue. Everything about the SemIf scheduler — gating,
scoring, queueing, run ownership, questions, repairs — lives in `GatewayService`;
an adapter must not know it exists. Neither knows anything about messaging UX
(invite links, reading, composing): that is the bridge services' job.

The transport is intentionally a plain `queue.Queue`: the adapter's event loop
puts an outbound message into the queue, the service puts replies into it. This
keeps scheduler work (which is synchronous and can block on the decision engine)
off the adapter's event loop.
"""

from __future__ import annotations

import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class InboundMessage:
    """One (possibly batched) inbound text message from an external chat."""

    text: str
    chat_id: str
    chat_type: str = "dm"  # dm | group
    contact_id: str | None = None
    display_name: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutboundMessage:
    """One reply destined for an external chat."""

    chat_id: str
    text: str


#: Callback the adapter invokes for each authorized inbound message.
InboundHandler = Callable[[InboundMessage], None]


class GatewayAdapter(ABC):
    """Transport contract for a messenger platform."""

    #: short platform name used in request sources (`"simplex:<chat_id>"`)
    name: str = "gateway"

    #: Spoken front ends (the voice gateway) set this so the service speaks only
    #: the skill's result line, not the scheduler's bookkeeping (queue/urgency
    #: status) and not the `<skill>: ok —` wrapper. A per-platform config
    #: `result_only` overrides it. Plain chat adapters leave it False.
    result_only: bool = False

    @abstractmethod
    def check_requirements(self) -> tuple[bool, str | None]:
        """Return `(ok, install_hint)`. Must never raise on a missing dep."""

    @abstractmethod
    def run(self, on_inbound: InboundHandler, outbound: "queue.Queue[OutboundMessage | None]") -> None:
        """Block, delivering inbound messages and sending queued outbound ones.

        A `None` sentinel in `outbound` asks `run` to return.
        """

    def close(self) -> None:
        """Release transport resources. Best-effort; must not raise."""
