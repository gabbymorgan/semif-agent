# SKILL.md — the contract for new skills

This file is the authoritative spec for what a new skill is and how its code
body must be written. It is fed verbatim to the code-generation model so every
generated skill is consistent, and it is read by humans who want to know what a
good skill looks like.

The mocking/testing side of a skill is NOT specified here — it has its own
contract, `TESTGEN.md`. This file is about the runnable body only.

## What a skill is

Semif-agent is an end-user application: an everyday person asks it to do a task,
and it does that task for real against their actual service. A skill is the
thing that does it.

A skill is a **leaf** in the agent's skill tree, reached by a chain of SemIf
decisions (category -> skill). It is one specific, single-purpose action the
agent can take — never a broad bucket (that is a category's job). It runs the
standard skill loop: observe -> predict -> act -> observe -> assess.

A skill is three things:

1. **A manifest** — the registry entry that makes it navigable and describes
   what it does.
2. **A code body** — a runnable Python module implementing the `predict` and
   `act` phases against a real service.
3. **An integration declaration** — the `INTEGRATION` module constant saying
   which real system the body talks to and how.

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

The generated module is persisted to `data/skills/<category>/<name>/skill.py`
and imported at runtime. It must satisfy **all** of the following:

### Required functions

```python
def predict(ctx, request) -> Prediction:
    """Forecast + make any SemIf sub-decisions. Return the prediction."""

def act(ctx, request, prediction) -> ActionResult:
    """Execute the action for real. Return the result + new state."""
```

- `ctx` is an `ActionContext` with `ctx.engine` (the real SemIf engine) and
  `ctx.config` (the merged agent config, including this skill's own config).
- `request` is the `Request` being handled.
- `Prediction(text: str, decisions: list)` and
  `ActionResult(action_log: str, new_state: str, needs_input: str | None = None)`
  are imported from
  `semif_agent.skills`; return those exact types. `decisions` carries any
  `(DecisionRequest, DecisionResult)` pairs made during predict so they are
  logged as training rows. `needs_input` carries a question for the human; see
  the rules below.

### Integration declaration

Every body defines a module-level `INTEGRATION` dict: the real system this
skill acts on. Flat, string-valued, and honest — it must match what the body
actually does.

```python
INTEGRATION = {
    "service": "nextcloud_calendar",
    "transport": "caldav",
    "config_vars": ["nextcloud_url", "nextcloud_username", "nextcloud_app_password"],
}
```

- `service` — short snake_case id of the real system (e.g. `gmail`,
  `nextcloud_calendar`, `simplex`, `local_files`).
- `transport` — exactly one of `http`, `caldav`, `imap`, `smtp`, `pop`,
  `subprocess`, `file`, `compute`
  (`http|caldav|imap|smtp|pop|subprocess|file|compute`). `compute` is only for
  skills that perform no external action at all (pure analysis/formatting); an
  action skill must not use it.
- `config_vars` — the `ctx.config` keys the body reads for this integration
  (the same names the data contract collects). Empty list allowed for `compute`.

The auto-run test is a hermetic mechanics check and never exercises the live
service, so this declaration is how the agent knows what the skill was built to
talk to. Declaring a transport the body does not use is a broken skill.

### Rules (hard requirements)

- **Perform the real action.** If the skill's purpose involves an external
  system, `act` must perform the real operation against the user's configured
  service — using the endpoint, account, credential, and CLI-path values the
  runner provides through `ctx.config`. Never simulate success, never return a
  canned result as if the action happened, and never silently downgrade to "a
  draft you can review": draft-only is allowed only when the requirements say
  so. If the service is unreachable, auth fails, or a tool is missing, report
  the real failure in `action_log` / `new_state` (status code, exception, or
  message) — the agent will offer to fix or learn from it.
- **Stdlib only, transports included.** No third-party imports, no files
  outside the project. Use stdlib transports for real integrations:
  `urllib.request` / `http.client` for HTTP, `imaplib` / `smtplib` / `poplib`
  for mail, `subprocess` to invoke a local CLI at the absolute path given in
  `ctx.config` (check it exists first), file I/O under configured data dirs
  for file integrations. Do not assume a tool, library, or config value
  exists; fail honestly when it does not.
- **No mocking.** Sub-decisions use the real engine: build a
  `DecisionRequest(state, question, options=[Option(id, description), ...])`
  and call `ctx.engine.call(decision)`; return it inside `Prediction.decisions`.
- **Never swallow the request.** If the skill cannot act, return an
  `ActionResult` with a short `action_log` explaining why and set `new_state`
  back to `request.text`.
- **Data comes from the runner, never from you.** A skill body is executed
  across many requests and owns no working data. Every operational value is
  provided by the runner through `ctx.config` under a clear snake_case name —
  endpoints, accounts, credentials, CLI paths, identifiers. Request data from
  the runner; never embed or fabricate working values, never invent mock data,
  and never ask the human for operational data. Whether a value is remembered
  across runs (config) or collected fresh each fire (input) is decided by the
  config step at first fire — treat every variable the same here: read it from
  `ctx.config`. Choose names a reader can extract into a contract (e.g.
  `sender_address`, `tracking_id`, `nextcloud_url`).
- **No test fixtures at runtime.** A body must never contain a hardcoded
  localhost/loopback address, a test port, or fixture data. Which environment
  it talks to is `ctx.config`'s decision, never the code's.
- **Request input for clarification when requirements are unclear from the
  prompt.** If the human's intent is ambiguous, do not guess: return an
  `ActionResult(action_log="...", new_state=request.text, needs_input="<question>")`.
  The run pauses and the human answers on `request.user_input`; `act` is then
  called again with the *same* prediction — check `request.user_input` on the
  resume pass to finish the run. This refines the product goal and requirements
  only — never operational data (the runner supplies that).
- **Write files under configured data dirs only** (e.g. the directory named by
  a config variable such as `ctx.config["output_dir"]`), never anywhere else on
  disk.
- **Fail fast on budget.** Keep the work small; do not loop or retry in code.
- **Names match the manifest.** The module is imported as its manifest name;
  the functions are `predict` and `act` exactly.

## Messaging through the gateway bridge

The agent can read and send SimpleX messages on the user's behalf. A skill
never opens a WebSocket to the `simplex-chat` daemon: the running messenger
gateway exposes a local HTTP bridge, and that is the only messaging surface a
body touches. Declare it like any other HTTP integration:

```python
INTEGRATION = {
    "service": "simplex",
    "transport": "http",
    "config_vars": ["messaging_bridge_url", "simplex_default_contact"],
}
```

`messaging_bridge_url` is the bridge base URL (e.g. `http://127.0.0.1:5227`),
supplied by the runner through `ctx.config`. Call it with `urllib.request`:

| Request                                | Purpose                                                            |
| -------------------------------------- | ------------------------------------------------------------------ |
| `GET  <base>/health`                   | `{"ok": true}` — the bridge is up.                                 |
| `GET  <base>/contacts`                 | `{"contacts": [{"id", "display_name"}]}` — known contacts.         |
| `GET  <base>/inbox`                    | Peek buffered inbound messages; does not consume.                  |
| `GET  <base>/inbox/next?contact=<id>`  | Pop the oldest unread message (optionally from one contact).       |
| `POST <base>/send`                     | `{"recipient": "<id\|display_name>", "text": "..."}` — send.        |

`GET /inbox` returns `{"messages": [{"id", "contact_id", "display_name", "text",
"received_at"}]}`; `GET /inbox/next` returns `{"message": {...} | null}`, where
`null` means nothing is buffered. `POST /send` returns
`{"ok": true, "contact_id": "<id>"}`. The bridge owns the read cursor, so a
read skill needs no state of its own.

Rules:

- **Recipient selection is a SemIf decision.** When more than one conversation
  is relevant (e.g. several have buffered messages), resolve which one with a
  `ctx.engine.call(...)` sub-decision over the contacts — mirroring how
  `calendar.next_event` picks a calendar. Use `simplex_default_contact` only as
  the configured fallback when the request does not already make it clear.
- **Sending requires user intent.** Send only because the request (or the
  requirements) asks for it. Never broadcast, never message a contact the user
  did not name or confirm, and never fabricate a message body as a working
  value.
- **Report real failures.** A bridge error, an unknown recipient, or an empty
  inbox is reported honestly in `action_log` / `new_state`; never claim a
  message was sent or read when it was not.
- The bridge is local-only, but its URL still comes from `ctx.config` — never
  hardcode a host, port, or bridge address in the body.

## Conventions

- Single purpose, single file, single module.
- Avoid duplicating an existing skill in the same category.
- `predict` resolves ambiguity (arguments, recipients, targets) with SemIf
  sub-decisions, mirroring how `calendar.next_event` resolves which calendar
  to read.
- `act` performs the concrete real action and writes a human-readable
  `action_log` that the self-assessment LLM can judge. Include what the service
  actually returned.
- Act like a developer eliciting requirements from a product owner: when the
  prompt leaves a behavioral or integration choice open, prefer a clarifying
  `needs_input` over guessing — a clarifying question is cheaper than a wrong
  body.

## Acceptance criteria

A generated skill is accepted only if:

1. It compiles (`compile(..., "exec")` succeeds) and defines both `predict`
   and `act`.
2. Its `name` matches the manifest regex and its `category` is given.
3. Its body imports nothing outside the stdlib and the agent package.
4. It uses `ctx.engine` (never mocks) and returns proper `Prediction` /
   `ActionResult` types.
5. It is single-purpose and does not duplicate an existing category leaf.
6. Every operational value it needs is read from `ctx.config` under a clear
   snake_case name — nothing is embedded, fabricated, or asked of the human.
7. If its purpose is an external action, `act` performs the real operation via
   the configured service/transport (no simulated success, no draft-by-default)
   and reports real failures honestly.
8. `INTEGRATION` is present, flat, string-valued, uses the transport vocabulary,
   and is consistent with the body's `ctx.config` reads and behavior.
9. If it is a messaging skill, it uses the gateway bridge over HTTP (never the
   daemon WebSocket), resolves recipients with a SemIf sub-decision, and sends
   only on explicit user intent.

## Worked example

A skill that checks a service's health over HTTP — for real, with the endpoint
and token supplied by the runner:

```python
import json
import urllib.error
import urllib.request

from semif_agent.skills import ActionResult, Prediction

INTEGRATION = {
    "service": "status_page",
    "transport": "http",
    "config_vars": ["service_url", "service_token"],
}

def predict(ctx, request):
    return Prediction(text=f"check {ctx.config['service_url']}", decisions=[])

def act(ctx, request, prediction):
    url = ctx.config["service_url"]
    token = ctx.config.get("service_token", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            status = response.status
            payload = json.loads(response.read().decode("utf-8"))
        return ActionResult(
            action_log=f"status_page: {url} is {payload.get('status', 'unknown')} (HTTP {status}).",
            new_state=f"{url} status: {payload.get('status', 'unknown')}",
        )
    except urllib.error.HTTPError as exc:
        return ActionResult(
            action_log=f"status_page: {url} returned HTTP {exc.code}: {exc.reason}.",
            new_state=f"status check failed: HTTP {exc.code}",
        )
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return ActionResult(
            action_log=f"status_page: could not reach {url}: {exc}.",
            new_state=request.text,
        )
```

Write skill bodies in this shape: declare `INTEGRATION`, resolve ambiguity in
`predict` via `ctx.engine`, perform the real operation in `act`, keep both
stdlib-only, read data from `ctx.config`, and return the proper types.
