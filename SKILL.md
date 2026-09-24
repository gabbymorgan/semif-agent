# SKILL.md — the contract for new skills

This file is the authoritative spec for what a new skill is and how its code
body must be written. It is fed verbatim to the code-generation model so every
generated skill is consistent, and it is read by humans who want to know what a
good skill looks like.

## What a skill is

A skill is a **leaf** in the agent's skill tree, reached by a chain of SemIf
decisions (category -> skill). It is one specific, single-purpose action the
agent can take — never a broad bucket (that is a category's job). It runs the
standard skill loop: observe -> predict -> act -> observe -> assess.

A skill is two things:

1. **A manifest** — the registry entry that makes it navigable and describes
   what it does.
2. **A code body** — a runnable Python module implementing the `predict` and
   `act` phases.

## Manifest schema

Registered in `data/categories.json` (and mirrored in the running tree). Fields:

| Field                | Meaning                                                              |
| -------------------- | -------------------------------------------------------------------- |
| `name`               | `category.skill` dotted snake_case id. Lowercase letters, digits, `_` and `.` only (`[a-z0-9_]+(?:\.[a-z0-9_]+)*`). Single purpose. |
| `category`           | The top-level category bucket the skill lives under.                 |
| `description`        | One to two sentences: what the skill does, for navigation.           |
| `allowed_inputs`     | What the skill accepts / needs as input context.                     |
| `actions`            | The concrete actions the skill takes.                                |
| `cost_budget`        | Relative budget for one run; a run that exceeds it fails fast.       |
| `decision_log_ref`   | Reference to the decision rows this skill logged during a run.       |

Only `name` and `description` are required for a stub; the rest fill in as the
skill is exercised.

## Code body contract

The generated module is persisted to `data/skills/<category>/<name>.py` and
imported at runtime. It must satisfy **all** of the following:

### Required functions

```python
def predict(ctx, request) -> Prediction:
    """Forecast + make any SemIf sub-decisions. Return the prediction."""

def act(ctx, request, prediction) -> ActionResult:
    """Execute the action. Return the result + new state."""
```

- `ctx` is an `ActionContext` with `ctx.engine` (the real SemIf engine) and
  `ctx.config` (the agent config dict).
- `request` is the `Request` being handled.
- `Prediction(text: str, decisions: list)` and
  `ActionResult(action_log: str, new_state: str)` are imported from
  `semif_agent.skills`; return those exact types. `decisions` carries any
  `(DecisionRequest, DecisionResult)` pairs made during predict so they are
  logged as training rows.

### Rules (hard requirements)

- **Stdlib only.** No third-party imports, no files outside the project. The
  core agent is pure-stdlib and runs on the thin dev box.
- **No mocking.** Sub-decisions use the real engine: build a
  `DecisionRequest(state, question, options=[Option(id, description), ...])`
  and call `ctx.engine.call(decision)`; return it inside `Prediction.decisions`.
- **Never swallow the request.** If the skill cannot act, return an
  `ActionResult` with a short `action_log` explaining why and set `new_state`
  back to `request.text`.
- **Write files under configured data dirs only** (e.g. `ctx.config["drafts"]`),
  never anywhere else on disk.
- **Fail fast on budget.** Keep the work small; do not loop or retry in code.
- **Names match the manifest.** The module is imported as its manifest name;
  the functions are `predict` and `act` exactly.

## Conventions

- Single purpose, single file, single module.
- Avoid duplicating an existing skill in the same category.
- `predict` resolves ambiguity (arguments, recipients, targets) with SemIf
  sub-decisions, mirroring how `email.compose` resolves its recipient.
- `act` performs the concrete action and writes a human-readable `action_log`
  that the self-assessment LLM can judge.

## Acceptance criteria

A generated skill is accepted only if:

1. It compiles (`compile(..., "exec")` succeeds) and defines both `predict`
   and `act`.
2. Its `name` matches the manifest regex and its `category` is given.
3. Its body imports nothing outside the stdlib and the agent package.
4. It uses `ctx.engine` (never mocks) and returns proper `Prediction` /
   `ActionResult` types.
5. It is single-purpose and does not duplicate an existing category leaf.

## Worked example

`email.compose` resolves its recipient with a SemIf sub-decision, then writes a
draft file:

```python
def predict(ctx, request):
    contacts = read_contacts(ctx.config)          # config-driven, local data
    decision = DecisionRequest(
        state=f"{request.text} [current process: none]",
        question="Which contact is the intended recipient?",
        options=[Option(c["name"], c.get("description", "")) for c in contacts]
        + [Option("none", "None of the listed contacts.")],
    )
    result = ctx.engine.call(decision)
    return Prediction(text=f"recipient is {result.selected}",
                      decisions=[(decision, result)])

def act(ctx, request, prediction):
    recipient = prediction.text.removeprefix("recipient is ")
    drafts = Path(ctx.config.get("drafts", "data/drafts"))
    drafts.mkdir(parents=True, exist_ok=True)
    target = drafts / f"{request.id}.txt"
    target.write_text(f"To: {recipient}\nBody: {request.text}\n")
    return ActionResult(
        action_log=f"email.compose: wrote draft {target} for {recipient!r}.",
        new_state=f"Draft written to {target.name} for {recipient}.",
    )
```

Write skill bodies in this shape: resolve ambiguity in `predict` via
`ctx.engine`, do the work in `act`, keep both stdlib-only, and return the
proper types.