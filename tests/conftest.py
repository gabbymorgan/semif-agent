"""Shared test fixtures.

`ScriptedEngine` is a deterministic in-process stand-in for the SemIf decision
engine used ONLY by pure-stdlib unit tests of scheduling mechanics (pausing,
resuming, routing, authoring). The real engine is the decision-maker in
production and in the integration tests; this double exists so a mechanics test
can drive the loop without loading a GGUF locally, exactly as a
loopback HTTP server stands in for a live service elsewhere.

It is not a mock of SemIf scoring: it answers each `DecisionRequest` with a
chosen option, so a test can assert what the *mechanics* did with that answer.
"""

from __future__ import annotations

from semif_agent.decisions import DecisionResult


class ScriptedEngine:
    """Answers decisions deterministically from a per-phase option policy.

    `default` is the option id returned for any decision not covered by
    `choices`. `choices` maps a substring of the decision `question` (or the
    option ids) to the option id to select; the first matching key wins.
    """

    def __init__(self, choices: dict[str, str] | None = None, default: str | None = None):
        self.choices = dict(choices or {})
        self.default = default
        self.calls: list = []

    def call(self, request):
        self.calls.append(request)
        option_ids = [option.id for option in request.options]
        selected = None
        haystack = f"{request.question} {' '.join(option_ids)}"
        for key, option_id in self.choices.items():
            if key in haystack:
                selected = option_id
                break
        if selected is None:
            selected = self.default or option_ids[0]
        if selected not in option_ids:
            selected = option_ids[0]
        return DecisionResult(
            request=request,
            option_ids=option_ids,
            probabilities=[1.0 if o == selected else 0.0 for o in option_ids],
        )
