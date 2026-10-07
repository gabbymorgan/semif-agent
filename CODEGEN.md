# CODEGEN.md — the contract for new skills

This file is the authoritative spec for what a new skill is and how its code
body must be written.

The mocking/testing side of a skill is NOT specified here — it has its own
contract, `TESTGEN.md`. This file is about the runnable body only.

## What a skill is

Semif-agent is an end-user application: an everyday person asks it to do a task,
and it does that task using a combination of native python modules, user-provided data and a catalog of basic services known as bridges. A skill is the thing that performs one of those tasks: it is the leaf in the skill tree that the agent navigates to when it has resolved a request to a single, concrete action. Navigation is performed by an llm-based decision engine known as SemIf.

A skill is three things:

1. **A registry entry** — its navigable `name` + `description`, recorded in
   `data/categories.json` and mirrored in the running tree.
2. **A code body** — a runnable Python module implementing a single `act`
   phase against a real service.
3. **A declaration** — the `INTEGRATION` constant (which real system the body
   talks to and how) and the `CONTRACT` constant (the values the runner must
   provide).

The committed seed `seeds/calendar/create_event/` is the **authoritative worked
example** of all three. Read its `manifest.json`, its `skill.py` (which declares
`INTEGRATION` and `CONTRACT`), and its `skill.test.py` alongside this file; a new
body should look and behave like it.

## The decision engine: SemIf

A skill body turns a request into an action, and along the way it must decide
things: which calendar the query means, whether the evidence is enough, what to
call the event. Those decisions are made by **SemIf** — the agent's decision
engine — not by text generation. This section explains what SemIf is, what it
can do, and when a body should reach for it instead of plain code or an LLM.

### What SemIf is

SemIf ("semantic ifs", formerly OpenJev) is a small, pinned, local language
model whose single job is to answer a **typed, runtime-defined decision**
without generating any text. You give it three things:

- a **state** — the unstructured evidence, usually `request.text` (sometimes the
  query plus a list of candidates);
- a **question** — the criterion in plain language ("Which calendar is referred
  to in the user query?");
- **options** — 2–16 named, described alternatives (`work`, `personal`, ...).

It performs one forward pass and reads the probabilities of the declared answer
tokens directly. It returns a probability for each option — no sentence, no
JSON, no decoding loop, nothing to parse. In effect it is a semantic `if`:
software asks a yes/no/which question in natural language and gets back a typed,
thresholdable number. In this agent the `state` is always a string
(`request.text`, or the query plus a list of candidates); the broader SemIf
project also accepts a nonempty JSON object or array as state, but a body here
must pass a string.

The whole point is that the **question and options arrive with the request**.
Routing, selection, and assessment are not baked into the model or the prompt;
they are supplied at call time. You never fine-tune SemIf to add a new decision —
you just describe a new question.

That is the difference from a chat model: a chat model answers by generating text
that your code immediately parses back into an `if`. SemIf skips the text. On the
reference hardware, reading 21 binary decisions directly took about 1.0 s and
zero output tokens; the same decisions as a generated JSON array took about
5.3 s and 111 tokens. It is decision-native and cheap enough to embed in ordinary
code.

### SemIf's feature set

SemIf (<https://github.com/TheoLeeCJ/SemIf>) is a broader project than
the one call this agent makes. Its full feature set:

- **Runtime-defined decisions** — state, criterion, and option descriptions are
  supplied per call; no retraining, no per-decision prompt engineering.
- **Decision-native readout** — one forward pass over the declared option
  logits, softmaxed over the allowed answer tokens. No token is sampled, so
  there is no generation, no JSON repair, and no decoding loop to get stuck.
- **Typed probabilities** — a probability per option, conditional on exactly the
  options you supplied. 2–16 described options per decision.
- **Structured state** — the state can be a string or a nonempty JSON object or
  array; direct modes preserve it as structured JSON rather than flattening it.
- **Shared-state awareness** — one long state can be prefetched once and branched
  across many criteria: serial prefix reuse (~10.8 decisions/s) and parallel
  suffix branches (~20 decisions/s) instead of re-scoring the state per decision.
  Best when every decision shares the same state.
- **Multiple backends** — CUDA/PyTorch, Apple Silicon (MLX and PyTorch/MPS), and
  a CPU llama.cpp path over a local GGUF. This agent pins the llama.cpp CPU
  backend.
- **Reranker mode** — a second readout that scores each candidate as a yes/no
  relevance proposition and softmaxes the log-odds across options; strongest for
  ranking/retrieval rather than categorical decisions.
- **Auditability** — every row carries the option scores, timing, the exact model
  revision, and a `prompt_sha256`, so a decision can be reproduced and compared.
- **Calibration** — optional per-workload temperature scaling fitted on labeled
  decisions, which tightens expected calibration error (e.g. WANLI 0.208 → 0.069
  out of fold). Calibration does not change the selected option.
- **Batch scoring CLI** (`semif-score`) — reads a JSONL of decisions and writes a
  JSONL of typed scores, for offline evaluation.
- **Open, auditable evidence** — owned fixtures, committed runners, raw results,
  and known failure modes.

Two caveats shape how the agent uses it: the probabilities are **conditional on
the options you supplied** — not calibrated operational confidence — so treat
them as conditional scores and threshold them rather than believing them; and a
forced typed output can still be semantically wrong. Quality depends on the
model: the pinned 4B reaches ~0.81 balanced accuracy on the project's authored
decision set (a 27B EXL3 bridge reaches ~0.96).

### What a skill body can use

A body does not speak SemIf's wire protocol. It reaches the engine through
**one** interface: the typed decision call.

```python
from semif_agent.decisions import DecisionRequest, Option

decision = DecisionRequest(
    state=request.text,
    question="Which calendar is referred to in the user query?",
    options=[Option(c["name"], c["label"]) for c in calendars],
)
result = ctx.engine.call(decision)
result.selected      # option id with the highest probability
result.prob("work")  # that option's probability
result.probs         # {option_id: probability}
```

The engine is always real (`ctx.engine`); there is no mock path. The runner
serializes calls onto one model, so keep decisions few. Return every
`(DecisionRequest, DecisionResult)` pair on `ActionResult.decisions` so the choice
is logged as a training row with the run outcome.

The shared-state, reranker, batch, and calibration features above are
engine/runtime concerns, not body-level tools: a body makes direct typed
decisions and lets the runner own the model.

Note that the runner's own **assessment** of whether a run succeeded
(`assess:outcome`) is itself a SemIf decision over the body's `action_log`, the
goal, and the resolved inputs. The body does not self-assess; it reports
truthfully and lets SemIf judge.

### Choosing between deterministic code, SemIf, and LLM text

Every value a body produces comes from one of three sources. Pick deliberately;
mixing them up is the most common way a body goes wrong.

| Source | Use it for | Examples |
| ------ | ---------- | -------- |
| **Deterministic code** | Exact, mechanical mappings that are right every time | parsing a date/time or number, formatting output, computing a duration, building a URL, comparing ids, iterating a fetched list, reading a status code, escaping text |
| **SemIf** (`ctx.engine.call`) | A judgment among a known, enumerated set of options, when the right one is semantic rather than an exact match | which calendar/contact/conversation/folder the query means; whether the evidence supports / contradicts / is insufficient; which target to use when several fit |
| **LLM text** (the LLM bridge) | Novel natural-language content that is not chosen from a set | an event title, a subject line, a message body, a summary, a description, a reformulation |

Rules:

- **Deterministic code is the default.** If a rule can decide it, write the rule.
  Never spend a model call on something a regex, a dict lookup, or arithmetic
  settles.
- **Use SemIf for choices, not for generation.** SemIf returns a probability over
  the options you gave it; it cannot write a title or a sentence. Give it the
  real candidate set (fetched from the service), include any configured default as
  one of the options, and threshold the winner: use it when its probability clears
  the skill's confidence threshold, otherwise fall back to the configured default
  (see `calendar.create_event` and `simplex.next_message`).
- **Use the LLM bridge for text, not for routing or assessment.** Generated text
  goes through the LLM bridge — a generic `POST /chat` whose base URL, token,
  and endpoint shapes are in the bridge catalog injected below — never by
  speaking a model's protocol directly. Keep the model extractive and bounded
  (see how `calendar.create_event` asks only for a title and description), and
  report a real failure when the bridge is down rather than guessing the text.
- **Never blur the boundaries.** SemIf never generates; an LLM never routes,
  selects, or assesses; deterministic code never invents a value it cannot
  observe. A request input is resolved by SemIf, not by substring matching by
  hand; a title is generated by the LLM, not assembled by string concatenation
  when it should be written; a date is parsed by code, not asked of a model.

## The registry entry

The body writer does **not** author a manifest. The `llm` provider authors the
leaf's `name` and `description`, and the scheduler records them in
`data/categories.json` (mirrored in the running tree). You are told the chosen
`name` and `description` in the prompt; use them, and never invent or rename the
skill.

The registry is a map of category -> bucket:

```json
{
  "<category>": {
    "description": "<one to two sentences: the bucket's purpose>",
    "skills": [
      {
        "name": "<leaf_name>",
        "description": "<what this leaf does, for navigation>",
        "request_text": "<the originating request, optional>",
        "requirements": {"<question>": "<answer>"}
      }
    ]
  }
}
```

- A leaf `name` is a single snake_case token (`[a-z0-9_]+(?:\.[a-z0-9_]+)*`),
  not dotted with the category; the category is the key it lives under, and the
  runner joins them as `category.name` only when it builds the run summary.
- `request_text` and `requirements` are recorded by the scheduler so a stub can
  be re-driven later; the body never sets them.

## Code body contract

The generated module is persisted to `data/skills/<category>/<name>/skill.py`
and imported at runtime. It must satisfy **all** of the following.

**Import everything you use.** The module is not handed `ActionResult`, the
engine types, or any stdlib module for free — the runner imports your file
as-is. A body that calls `urllib.request.urlopen`, `json.loads`, `re.match`, or
returns `ActionResult` **without importing them raises `NameError` the moment it
runs**. Start every body with its imports (see the complete example below).

### Complete minimal example

This is the whole shape of a body — imports first, then `INTEGRATION`,
`CONTRACT`, and `act`. Copy this structure.

```python
"""Check whether a configured HTTP service is reachable.

Real integration: the service over HTTP. Operational values come from
`ctx.config` (declared in CONTRACT).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from semif_agent.skills import ActionResult

INTEGRATION = {
    "service": "example_service",
    "transport": "http",
    "config_vars": ["example_base_url"],
}

CONTRACT = {
    "example_base_url": "Base URL of the service, e.g. https://service.example.org.",
}


def act(ctx, request) -> ActionResult:
    """Perform the real action and return the result + new state."""
    base_url = ctx.config["example_base_url"]
    try:
        with urllib.request.urlopen(f"{base_url}/status", timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError) as exc:
        return ActionResult(
            action_log=f"could not reach {base_url}: {exc}",
            new_state=request.text,
        )
    return ActionResult(
        action_log=f"{base_url} returned {payload}",
        new_state=json.dumps(payload),
    )
```

That example checks a fixed configured endpoint, so it takes no per-request
argument; a skill that acts on the query reads `request.text` and derives its
arguments from it — see "Turning the request into arguments".

### Required function

```python
from semif_agent.skills import ActionResult

def act(ctx, request) -> ActionResult:
    """Perform the real action and return the result + new state."""
```

- `ctx` is an `ActionContext` with `ctx.engine` (the real SemIf engine),
  `ctx.config` (the resolved values for this skill, including its own config),
  and `ctx.timers` (the in-process timer/alarm service — use it to schedule a
  notification, as the `time.set_timer` and `time.set_alarm` seeds do).
  `ctx.admin` exists for the built-in housekeeping meta skills only; an ordinary
  body must never touch it.
- `request` is the `Request` being handled.
- `ActionResult(action_log: str, new_state: str, needs_input: str | None = None,
  decisions: list = [])` is imported from `semif_agent.skills`; return that exact
  type. `decisions` carries any `(DecisionRequest, DecisionResult)` pairs made
  during `act` so they are logged as training rows. `needs_input` carries a
  question for the human; see the rules below.

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
# DecisionRequest/Option come from semif_agent.decisions (not .skills).
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
- `config_vars` — the `ctx.config` keys the body reads for this integration. In
  the committed seeds this is exactly the body's `CONTRACT` key set; an empty
  list is allowed for `compute`. Declaring a variable the body never reads is a
  mismatch the fidelity check flags.

The auto-run test is a hermetic mechanics check and never exercises the live
service, so this declaration is how the agent knows what the skill was built to
talk to. Declaring a transport the body does not use is a broken skill.

### Candidate selection

When the body has a **set of candidates to choose from** — contacts,
conversations, calendars, folders, targets — it must employ a relevant SemIf
query over them. The decision types come from the agent package, not
`semif_agent.skills`:

```python
from semif_agent.decisions import DecisionRequest, Option

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
- When the winner is weak (its probability is below the skill's confidence
  threshold), fall back to the configured default; with no configured default,
  keep the weak winner. See how `calendar.create_event` resolves its calendar.

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

- **Do not prefix the skill name to `action_log` or `new_state`.** The runner
  wraps the result in the deterministic summary (`category.name: ok — <detail>`)
  and carries the skill ref structurally, so a body writes only the result text
  — never `"<category>.<name>: ..."`.
- **Return the real result.** A read/query skill returns the data it fetched; an
  action skill reports the service's real response. A generic "done" that omits
  the result is a broken output.
- **Never an empty or placeholder output.** If there is genuinely nothing (empty
  inbox, no matching event), say so explicitly — that is a real result, and a
  **successful** run: the `assess:outcome` step treats a definitive "nothing
  found / nothing to do" result as success. Phrase it as the answer the user
  asked for ("checked the inbox: no unread messages"), not as an absence or an
  error, so the assessment does not read a healthy empty result as a failure.
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
  (both imported from `semif_agent.decisions`) and call
  `ctx.engine.call(decision)`; return it inside
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
  for. `request.meta` is a scratch dict carried on the same request across the
  pause, so a body can cache a computed value or an already-collected answer
  there and reuse it on the resume pass instead of recomputing or re-asking —
  see how `calendar.create_event` caches the extracted title/description and the
  answers it collected.
- **Fail fast on budget.** Keep the work small; do not loop or retry in code.
- **The function is `act` exactly.** The module is imported under its registered
  leaf name; the entry point is a module-level `act(ctx, request)`.

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
  contacts or buffered senders), mirroring how `calendar.create_event` picks a
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
  sub-decisions, mirroring how `calendar.create_event` resolves which calendar
  to write to.
- `act` performs the concrete real action and writes a human-readable
  `action_log` that the SemIf assessment step can judge. Include what the
  service actually returned.
- Act like a developer eliciting requirements from a product owner: when the
  prompt leaves a behavioral or integration choice open, prefer a clarifying
  `needs_input` over guessing — a clarifying question is cheaper than a wrong
  body.

## Acceptance criteria

A generated skill is accepted only if:

1. It parses as Python (`ast.parse` succeeds), defines a module-level `act`
   function, and imports every name it uses (`ActionResult` from
   `semif_agent.skills`; `DecisionRequest`/`Option` from `semif_agent.decisions`
   only when it actually builds a sub-decision; each stdlib module it calls — no
   `NameError` at run time).
2. Its `name` matches the leaf-name regex and its `category` is given.
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

Items 1 and 7 are hard checks in the parser (`parse_skill_body`) and reject the
body outright. The rest are not statically rejected: a missing or inconsistent
`INTEGRATION` (#11) is inferred and the body is accepted badged `unverified`, a
third-party import (#3) only fails when the module is actually imported, and
import-time work (#12) only shows up when it runs. The fidelity gate rewrites a
body once for these, then accepts it — so treat every item as required, not
optional.
