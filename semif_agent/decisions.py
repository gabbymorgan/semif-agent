"""The SemIf decision contract shared across the agent.

A decision is a typed question over a state with declared options; SemIf returns
probabilities conditional on exactly the supplied options. These are not
calibrated confidence values, so callers treat them as conditional scores.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Option:
    id: str
    description: str


@dataclass
class DecisionRequest:
    """One SemIf decision: state + question + typed options."""

    state: str
    question: str
    options: list[Option]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_semif_row(self) -> dict:
        return {
            "id": self.id,
            "state": self.state,
            "question": self.question,
            "options": [{"id": o.id, "description": o.description} for o in self.options],
        }


@dataclass
class DecisionResult:
    """The outcome of one SemIf call: probabilities aligned to option ids."""

    request: DecisionRequest
    option_ids: list[str]
    probabilities: list[float]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def probs(self) -> dict[str, float]:
        return dict(zip(self.option_ids, self.probabilities))

    def prob(self, option_id: str) -> float:
        index = self.option_ids.index(option_id)
        return self.probabilities[index]

    @property
    def selected(self) -> str:
        return max(self.probs, key=self.probs.get)


@dataclass
class Request:
    """An incoming input to the agent, before it is gated/scored."""

    text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = "typed"
    meta: dict[str, Any] = field(default_factory=dict)
    received_at: float = field(default_factory=time.time)
    priority: float = 0.5
    reentries: int = 0
    resume: dict[str, Any] = field(default_factory=dict)

    def copy_for_requeue(self) -> "Request":
        return Request(
            text=self.text,
            id=self.id,
            source=self.source,
            meta=dict(self.meta),
            received_at=self.received_at,
            priority=self.priority,
            reentries=self.reentries + 1,
            resume=dict(self.resume),
        )
