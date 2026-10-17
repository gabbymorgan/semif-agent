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
    """One reply destined for an external chat.

    `text` is the reply. `stream` optionally carries a long-form read-aloud
    request (a `semif_agent.speech.ReadAloud`) instead of a plain line: the
    voice front end streams it chunk-by-chunk with text-to-speech until the user
    stops it. A platform that does not stream leaves it None and speaks `text`.
    """

    chat_id: str
    text: str
    stream: Any = None


#: Callback the adapter invokes for each authorized inbound message.
InboundHandler = Callable[[InboundMessage], None]


#: Words/phrases that abort a long-form read-aloud (and a spoken reply) when
#: heard/typed. The match is on the whole normalized utterance (so "stop the
#: timer" stays a real command) with a first-word fallback only for a short
#: utterance with trailing filler ("stop please", "cancel that"). Shared by the
#: voice and LXMF adapters, which both stream read-aloud text.
_STOP_PHRASES = {
    "stop",
    "stop it",
    "stop talking",
    "stop please",
    "cancel",
    "cancel that",
    "cancel please",
    "quiet",
    "quiet down",
    "be quiet",
    "silence",
    "hush",
    "shut up",
    "never mind",
    "nevermind",
    "enough",
}
_STOP_FIRST_WORDS = {"stop", "cancel", "quiet", "silence", "hush", "shut", "enough"}
_STOP_FILLER = {"please", "now", "it", "that", "talking", "down", "up", "already"}

#: Characters stripped from a spoken utterance's edges before matching.
_STOP_PUNCT = " \t\r\n.,!?;:\"'()[]{}"


def is_stop_command(text: str) -> bool:
    """Whether an utterance means "stop talking" (vs a new command).

    Deterministic vocabulary, no model: a bare "stop"/"cancel" (or a short
    variant like "stop please") aborts a spoken reply / long-form read. Anything
    longer is a real command and is submitted as a new request.
    """
    cleaned = (text or "").strip().lower().strip(_STOP_PUNCT)
    if not cleaned:
        return False
    if cleaned in _STOP_PHRASES:
        return True
    words = cleaned.split()
    if len(words) > 3:
        return False
    first = words[0].strip(_STOP_PUNCT)
    if first not in _STOP_FIRST_WORDS:
        return False
    return all(word.strip(_STOP_PUNCT) in _STOP_FILLER for word in words[1:])


class GatewayAdapter(ABC):
    """Transport contract for a messenger platform."""

    #: short platform name used in request sources (`"simplex:<chat_id>"`)
    name: str = "gateway"

    #: Front ends that should send only the skill's result line set this, so the
    #: service drops the scheduler's bookkeeping (queue/urgency status) and the
    #: `<skill>: ok —` wrapper. The voice gateway and the LXMF gateway set it; a
    #: per-platform config `result_only` overrides it either way.
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
