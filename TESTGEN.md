# TESTGEN.md — the contract for skill test artifacts

This file is the authoritative spec for the **testing** side of a skill. It is
fed verbatim (alongside the finished `skill.py`) to the model that produces the
test artifact. The runnable body contract lives in `SKILL.md`; this file is
deliberately separate so the body writer never embeds test data and the test
writer never worries about body semantics.

A finished skill leaf is a folder
`data/skills/<category>/<name>/` containing (among others) two artifacts
produced here:

| File              | Meaning                                                            |
| ----------------- | ------------------------------------------------------------------ |
| `contract.json`   | The skill's data contract: what the runner must provide.           |
| `skill.test.py`   | A stdlib-only test, runnable as `python skill.test.py`.            |

## contract.json

A **single JSON object**. Each key is the name of a variable the skill needs;
each value is a plain-language, semantic description of the expected values for
that variable (what it is, what good values look like).

This contract exists for exactly two consumers:

1. **User input** — the config step asks the human for these variables when
   they are not already configured.
2. **SemIf** — the config search and the record-vs-ask decision route on the
   variable names.

Therefore the contract must be flat and semantic. Do NOT put type declarations,
validation logic, or nested data structures in it — the skill body and the test
implement whatever type safety and validation they need on their own. The test
writer consumes `skill.py` in context and keeps the artifact compatible, so the
contract never has to carry that burden. Convoluted structure here is
pointless: it only confuses the human and the decision model.

Rules:

- Top level is exactly one JSON object; no arrays, no nesting.
- Each key is the snake_case name of a variable the body reads from
  `ctx.config`.
- Each value is a non-empty string describing the expected values in plain
  language (e.g. `"The email address the message is sent from."`).
- Derive the keys by reading the finished `skill.py`: every `ctx.config[...]`
  access that carries operational data is a contract key. Do not invent keys
  the body does not use.
- Do not include keys for purely internal/derived values.

## skill.test.py

A stdlib-only Python script that:

- Runs as `python skill.test.py` from the skill folder (its cwd is the folder,
  so the `skill` module is importable at `import skill`).
- Embeds its fixture data **inline** as Python literals — no external files, no
  network, no writes outside the folder.
- Exercises the skill's `predict` and `act` against those fixtures — call them
  directly with a real-ish `ActionContext` (a fake `engine` is acceptable
  **in the test only**; the test is the exception to the no-mocking rule).
- Exits 0 on success and non-zero on failure (use `assert`/`sys.exit(1)` with a
  message). No third-party imports.
- Covers at least the happy path and one edge case (missing input, unclear
  request, unreachable target).

## Acceptance criteria

The test artifact is accepted only if:

1. `contract.json` is a single flat object of snake_case keys to semantic
   descriptions, and every `ctx.config[...]` operational access in `skill.py`
   has a matching key.
2. `skill.test.py` parses as Python, imports only the stdlib and the agent
   package, embeds its fixtures inline (no external files), and runs to exit 0
   when executed from the skill folder.

Produce the two artifacts in this order: first `contract.json` (derived from
`skill.py`), then `skill.test.py` (derived from the same `skill.py` plus the
contract, with fixture data embedded inline).

## Worked example

A body that reads `tracking_id` from `ctx.config` and checks a delivery status.
The test embeds its fixtures inline, imports the `skill` module (its cwd is the
skill folder), and runs as `python skill.test.py`:

```python
import json
import sys

from semif_agent.skills import ActionContext, DecisionResult, Request

FIXTURES = {
    "tracking_id": "AB123XYZ",
    "status": "delivered",
}


class FakeEngine:
    def call(self, decision):
        return DecisionResult(
            state=decision.state,
            question=decision.question,
            options=decision.options,
            predicted_probs=[1.0] + [0.0] * (len(decision.options) - 1),
            selected=decision.options[0].id,
        )


def main():
    ctx = ActionContext(engine=FakeEngine(), config=dict(FIXTURES))
    req = Request("track my drone delivery in real time")
    skill = __import__("skill")
    pred = skill.predict(ctx, req)
    assert pred.text, "predict must produce a forecast"
    result = skill.act(ctx, req, pred)
    assert result.action_log, "act must log what it did"
    print(result.action_log)
    print("ok")


if __name__ == "__main__":
    main()
```