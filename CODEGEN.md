# CODEGEN.md — the contract for new skills

This file is the authoritative spec for what a new skill is and how its code
body must be written.

The mocking/testing side of a skill is NOT specified here — it has its own
contract, `TESTGEN.md`. This file is about the runnable body only.

## What a skill is

Semif-agent is an end-user application: an everyday person asks it to do a task,
and it does that task using a combination of user-provided data and a catalog of basic services known as bridges. A skill is the thing that performs one of those tasks: it is the leaf in the skill tree that the agent navigates to when it has resolved a request to a single, concrete action.

A skill is three things:

1. **A manifest** — the registry entry that makes it navigable and describes
   what it does.
2. **A code body** — a runnable Python module implementing a single `act`
   phase against a real service.
3. **A declaration** — the `INTEGRATION` constant (which real system the body
   talks to and how) and the `CONTRACT` constant (the values the runner must
   provide).

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

### Required function

```python
def act(ctx, request) -> ActionResult:
    """Perform the real action and return the result + new state."""
```

- `ctx` is an `ActionContext` with `ctx.engine` (the real SemIf engine) and
  `ctx.config` (the resolved values for this skill, including its own config).
- `request` is the `Request` being handled.
- `ActionResult(action_log: str, new_state: str, needs_input: str | None = None,
  decisions: list = [])` is imported from `semif_agent.skills`; return that exact
  type. `decisions` carries any `(DecisionRequest, DecisionResult)` pairs made
  during `act` so they are logged as training rows. `needs_input` carries a
  question for the human; see the rules below.

There is no `predict` phase. Resolution and the action happen in `act`.

### Data contract

Every **required** operational value the body needs is declared in a
module-level `CONTRACT` dict: a flat map of snake_case variable name to a
plain-language, semantic description of the expected value.

```python
CONTRACT = {
    "nextcloud_url": "Base URL of the user's Nextcloud, e.g. https://cloud.example.org.",
    "nextcloud_username": "Nextcloud login name whose calendar is read.",
    "nextcloud_app_password": "Nextcloud app password for that login.",
}
```

- The runner resolves each contract variable from the agent's tiered config
  (global -> category -> skill) and asks the human for any it cannot resolve.
  The resolved values arrive in `ctx.config` under the same names.
- Every key in `CONTRACT` must be read from `ctx.config` in the body. Do not
  declare a variable you do not read.
- **Optional** values with a safe default are NOT contract keys: read them with
  `ctx.config.get("<name>", <default>)`. Declaring an optional value would make
  the runner ask the human for it.
- Do not put type declarations, validation logic, or nested structures in
  `CONTRACT`; the body implements whatever type safety it needs. It is for the
  human and SemIf only.

### Turning the request into arguments

`request.text` is the user's query — the thing the human actually asked for. A
skill that turns queries into outputs must **read `request.text`** and derive
from it the action's **per-request arguments**: the recipient, the subject or
message body, the date or time range, the search terms, the target, the filter,
the filename. Do not ignore the query and do a fixed thing.

There are exactly two lanes of values, and they must not blur:

- **Per-request arguments** come from `request.text`. They change on every
  request. Never declare them in `CONTRACT`, never ask the human for them, and
  never embed them as constants.
- **Operational values** — endpoints, accounts, credentials, CLI paths, and
  stable defaults — come from `CONTRACT` -> `ctx.config`. See "Data contract".

When the query names a target that maps onto a set the service exposes
(contacts, calendars, folders, files), resolve it with a SemIf sub-decision over
the real candidate set, not by hand — see "Candidate selection". When a required
argument is missing or genuinely ambiguous and no candidate set can resolve it,
return `needs_input` (below); do not invent a value.

```python
# The recipient is a per-request argument: read it from the query, then resolve
# it against the real contact list with a SemIf sub-decision.
matches = [c for c in contacts if c["display_name"].lower() in request.text.lower()]
if len(matches) == 1:
    recipient = matches[0]["id"]
elif matches:
    decision = DecisionRequest(
        state=request.text,
        question="Which contact should the message go to?",
        options=[Option(c["id"], c["display_name"]) for c in matches],
    )
    result = ctx.engine.call(decision)
    decisions.append((decision, result))
    recipient = result.selected
else:
    return ActionResult(
        action_log="no contact matched the request",
        new_state=request.text,
        needs_input="Which contact should I message?",
    )
```

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
- `config_vars` — the `ctx.config` keys the body reads for this integration.
  Empty list allowed for `compute`.

The auto-run test is a hermetic mechanics check and never exercises the live
service, so this declaration is how the agent knows what the skill was built to
talk to. Declaring a transport the body does not use is a broken skill.

### Candidate selection

When the body has a **set of candidates to choose from** — contacts,
conversations, calendars, folders, targets — it must employ a relevant SemIf
query over them:

```python
decision = DecisionRequest(
    state=request.text,
    question="Which calendar is the intended one?",
    options=[Option(c["name"], c["label"]) for c in calendars],
)
result = ctx.engine.call(decision)
...
return ActionResult(..., decisions=[(decision, result)])
```

- Use the real engine (`ctx.engine.call`); never mock, never guess, never pick
  by hand when more than one candidate fits.
- Return every `(DecisionRequest, DecisionResult)` pair on
  `ActionResult.decisions` so the choice is logged with the run outcome.
- Include any configured default (a value read from `ctx.config`) as one of the
  options, so the request can override the stored default.

### Producing the result (output)

A skill exists to turn the query into an **output the human wants**. Both
strings on `ActionResult` carry that output and must reflect what really
happened:

- `action_log` is the human-readable record of the run. The `assess:outcome`
  SemIf decision judges the run from it (together with the goal and the resolved
  inputs), and the deterministic run summary is built from it. Write what the
  service actually returned: the message that was read, the id/reference of the
  thing that was created, the status code and error text of a failure.
- `new_state` is the run's resulting observation — the state the agent carries
  forward. For a read/query skill it is the retrieved data itself (the message
  body, the event, the value); for an action skill it is a short report of what
  was done (recipient, id, timestamp).

Rules:

- **Return the real result.** A read/query skill returns the data it fetched; an
  action skill reports the service's real response. A generic "done" that omits
  the result is a broken output.
- **Never an empty or placeholder output.** If there is genuinely nothing (empty
  inbox, no matching event), say so explicitly — that is a real result.
- **Never claim success you did not observe.** If the action failed, put the
  real failure in `action_log` and set `new_state` back to `request.text` (see
  the hard requirements).

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
  and call `ctx.engine.call(decision)`; return it inside
  `ActionResult.decisions`.
- **Never swallow the request.** If the skill cannot act, return an
  `ActionResult` with a short `action_log` explaining why and set `new_state`
  back to `request.text`.
- **Data comes from the runner, never from you.** A skill body is executed
  across many requests and owns no working data. Every required operational
  value is declared in `CONTRACT` and provided by the runner through
  `ctx.config` under a clear snake_case name — endpoints, accounts,
  credentials, CLI paths, identifiers; never embed or fabricate working values,
  never invent mock data, and never ask the human for operational data.
  Optional values with a safe default are read with `ctx.config.get(...)`.
- **No test fixtures at runtime.** A body must never contain a hardcoded
  localhost/loopback address, a test port, or fixture data. Which environment
  it talks to is `ctx.config`'s decision, never the code's.
- **Request input for clarification when requirements are unclear from the
  prompt.** If the human's intent is ambiguous, do not guess: return an
  `ActionResult(action_log="...", new_state=request.text, needs_input="<question>")`.
  The run pauses and the human answers on `request.user_input`; `act` is then
  called again — check `request.user_input` on the resume pass to finish the
  run. This refines the product goal and requirements only — never operational
  data (the runner supplies that).
- **Write files under configured data dirs only** (e.g. the directory named by
  a config variable such as `ctx.config["output_dir"]`), never anywhere else on
  disk.
- **No work at import time.** The module is imported when the skill loads and
  again by its test; keep module scope to constants and function definitions. Do
  not call the network, read files, or touch `ctx` at import.
- **`act` is single-phase and re-runs from the top.** After a `needs_input`
  pause the human's answer arrives on `request.user_input` and `act` is invoked
  again from the beginning — re-derive the query and config, consume
  `request.user_input`, and do not re-ask a question you already have an answer
  for.
- **Fail fast on budget.** Keep the work small; do not loop or retry in code.
- **Names match the manifest.** The module is imported as its manifest name;
  the function is `act` exactly.

## Bridge services

Some real systems are reached through a **bridge service**: a standalone local
process that stands up a small, token-guarded HTTP API in front of the system
and owns its native protocol and credentials. The concrete catalog of bridges —
which services exist, their base-URL config var, auth, config vars, and the
exact endpoints with their request/response/error shapes — is injected into the
prompt at authoring time (render it with
`semif_agent.bridges.describe_bridges()`); read it rather than guessing. This
section is only the generic pattern; the catalog is the source of truth for the
bridge-specific details.

A body must never speak a service's native protocol directly — no WebSocket to
`simplex-chat`, no direct daemon access. It calls the bridge's HTTP API with the
base URL from the bridge's config variable, exactly like any other HTTP
integration:

```python
INTEGRATION = {
    "service": "<the bridge's service id, from the catalog>",
    "transport": "http",
    "config_vars": ["<base URL config var>", "<any other config var it reads>"],
}
```

Rules:

- **Read every value from `ctx.config`.** The base URL, any token, and any
  default (recipient/contact/target) come from the config vars the catalog
  lists. Never hardcode a host, port, or bridge address in the body.
- **Send the auth header when the catalog lists one.** If the catalog names an
  auth config var and header, read the var and send it in that header when it is
  set; omit the header when the value is blank.
- **Resolve the recipient/target with a SemIf sub-decision.** When more than one
  conversation or target is relevant, resolve which one with a
  `ctx.engine.call(...)` sub-decision over the catalog's list endpoint (e.g.
  contacts or buffered senders), mirroring how `calendar.next_event` picks a
  calendar. Use the catalog's default config var as the configured option when
  the request does not already make it clear.
- **Sending requires user intent.** Send only because the request (or the
  requirements) asks for it. Never broadcast, never message a contact the user
  did not name or confirm, and never fabricate a message body as a working
  value.
- **Report real failures.** A bridge error, an unknown recipient, or an empty
  inbox is reported honestly in `action_log` / `new_state` — use the error
  statuses the catalog documents. Never claim a message was sent or read when it
  was not.
- **The bridge owns its own state.** Read cursors and similar state live in the
  bridge, not in the body; the body keeps no working state of its own.

## Conventions

- Single purpose, single file, single module.
- Avoid duplicating an existing skill in the same category.
- `act` resolves ambiguity (arguments, recipients, targets) with SemIf
  sub-decisions, mirroring how `calendar.next_event` resolves which calendar
  to read.
- `act` performs the concrete real action and writes a human-readable
  `action_log` that the SemIf assessment step can judge. Include what the
  service actually returned.
- Act like a developer eliciting requirements from a product owner: when the
  prompt leaves a behavioral or integration choice open, prefer a clarifying
  `needs_input` over guessing — a clarifying question is cheaper than a wrong
  body.

## Acceptance criteria

A generated skill is accepted only if:

1. It compiles (`compile(..., "exec")` succeeds) and defines `act`.
2. Its `name` matches the manifest regex and its `category` is given.
3. Its body imports nothing outside the stdlib and the agent package.
4. It uses `ctx.engine` (never mocks) and returns proper `ActionResult` types.
5. It is single-purpose and does not duplicate an existing category leaf.
6. It derives its per-request arguments from `request.text` — resolving a target
   against a real candidate set with a SemIf sub-decision when one exists —
   rather than ignoring the query or hardcoding values.
7. `CONTRACT` is present, flat, and every key is read from `ctx.config`;
   optional values are read with `.get`. No per-request argument is a contract
   key.
8. Every required operational value it needs is declared in `CONTRACT` — nothing
   is embedded, fabricated, or asked of the human.
9. It returns the real result in `new_state` (and a truthful `action_log`) — the
   fetched data or the service's real response, never an empty or placeholder
   output.
10. If its purpose is an external action, `act` performs the real operation via
    the configured service/transport (no simulated success, no draft-by-default)
    and reports real failures honestly.
11. `INTEGRATION` is present, flat, string-valued, uses the transport vocabulary,
    and is consistent with the body's `ctx.config` reads and behavior.
12. It does no network, file, or engine work at import time.
13. If it is a messaging skill, it uses a bridge service over HTTP (never a
    service's native protocol), resolves recipients with a SemIf sub-decision,
    and sends only on explicit user intent.
