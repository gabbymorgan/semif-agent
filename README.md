# semif-agent

A self-hosted agent whose **entire control flow is one decision model**. You
ask it to do something in plain language ("check my email", "message Sam on
SimpleX", "put lunch with Dana on my calendar") and it either chooses and executes a known skill in seconds or learns a new one in minutes.

The routing, gating, scoring, assessment and fidelity decisions are all made by
**SemIf** — a small local model that reads option logits directly from a single
forward pass. An LLM is used only to generate what cannot be achieved deterministically or through classification - for example, a skill's title, description, or code body. There are no simulated successes with mocked data and fake endpoint.  There is chain-of-thought infodump to give the illusion of confidence in a failed outcome. There is just a decision tree leading to reusable code snippets, plugged into a deliberately planned infrastructure of basic services. If intelligence is "on tap," then semif-agent is the catchment system.

A successful release of this could be defined by a single question: "Can I use this for most of the things I use my phone for besides scrolling?" The idea of using a single chat interface operating on minimal back-end resources to replace the diversity of features currently accomplished by a smartphone is audacious for sure. Many have tried and failed. But as technology progresses, it is always worth revisiting old problems with fresh eyes, and we are confident that the end result will be something that a substantial base of end users would enjoy.


---

## Table of contents

- [Why this exists (the reasoning)](#why-this-exists-the-reasoning)
- [How it works](#how-it-works)
  - [The scheduler: choose, score, queue, dispatch](#the-scheduler-choose-score-queue-dispatch)
  - [The skill tree and navigation](#the-skill-tree-and-navigation)
  - [The skill run loop](#the-skill-run-loop)
  - [Authoring a new skill](#authoring-a-new-skill)
  - [What each model sees](#what-each-model-sees)
  - [Data contracts and tiered config](#data-contracts-and-tiered-config)
  - [Repairing a failed run](#repairing-a-failed-run)
  - [Dream: the learning signal](#dream-the-learning-signal)
- [Features](#features)
- [Repository layout](#repository-layout)
- [Installation](#installation)
  - [Option A: one-command provisioning (`bootstrap.sh`)](#option-a-one-command-provisioning-bootstrapsh)
  - [Option B: core only (stdlib)](#option-b-core-only-stdlib)
  - [Prerequisites: ollama models](#prerequisites-ollama-models)
  - [Configuration](#configuration)
- [Running the agent](#running-the-agent)
  - [REPL](#repl)
  - [Scripted mode](#scripted-mode)
  - [Dashboard](#dashboard)
  - [Messenger gateway (SimpleX)](#messenger-gateway-simplex)
  - [Bridge services](#bridge-services)
  - [Dream report](#dream-report)
- [Skills in detail](#skills-in-detail)
- [Testing](#testing)
- [Roadmap / status](#roadmap--status)

---

## Why this exists

The design premise is that **language models are good at generating text and bad
at being a reliable control plane**. Repeatedly asking an LLM "should I interrupt
this process? how urgent is this? which skill handles this?" produces slow,
non-deterministic, un-loggable behavior. So control flow is pulled out of the
model entirely:

- **Semantic ifs, not text generation.** SemIf is a local rebuild of Jev's
  interface pattern. One forward pass reads typed option logits directly from a
  model; it returns per-option probabilities conditional on exactly the option
  set you supplied. There is no answer sentence and no decoding loop. It is fast
  (~0.9 s/decision on CPU), deterministic, cheap, and — crucially — **loggable**.
  Probabilities are conditional on the option set, not absolute confidence, so
  the agent treats them as conditional scores and calibrates per workload.
- **Local by construction.** The target is a desktop machine; data stays
  in-network. The decision model is pinned to an exact revision, so every swap is
  auditable via prompt hashes.
- **The decision engine is always real.** There is no mock engine and no degraded
  mode. If SemIf cannot be loaded, the app reports it and exits non-zero rather
  than pretending. (Pure unit tests may use a scripted double; the runtime never
  does.)
- **Simulated Agile patterns.** A skill is a real integration into the active tool chain for end user satisfaction. It takes existing mail
  protocols, SimpleX, CalDAV/WebDAV, configured local CLIs, HTTP APIs and your supplied data to produce actions that best fit your query. In the absence of a an existing skill, the skill generation process is activated. Skill generation has four major phases - definition, refinement, coding, testing (unit and integration). Each step in this process has deterministic and probabilistic handling optimized to achieve the stated goal of the user in the context of the platform's core mission.
- **Nothing is gated out up front.** Every input is dispatched and scored.
  Inputs that are not requests fall through navigation into the closed `response`
  category — a hardcoded tree of canned replies (`greeting`, `thanks`,
  `acknowledge`, `farewell`, `affirm`, and the catchall `clarify`). A request no
  skill covers is **not** a canned reply: the actionability guard
  (`navigate:actionability`) sends it to skill authoring instead, so
  `response` is reached only when the input asks for nothing.
  `response` never goes through codegen and its runs are never assessed (a canned
  line has no side effect to succeed or fail). It never offers generated text. By design, you will never be having a "conversation" with an LLM through this platform.
- **Everything is a training row.** Every SemIf decision is logged as a labeled
  row; the `dream` pass computes the prediction-vs-observation cost
  (cross-entropy / NLL + ECE) that later drives fine-tuning. This means that the platform should improve with use, much like SOTA agentic harnesses, but through reinforcement learning and procedurally generated context augmentations (as opposed to markdown files that grow, compress, and prune their way through a meandering and inherently context-limited self-improvement loop).


---

## How it works

### The scheduler: choose, score, queue, dispatch

Implemented in `semif_agent/scheduler.py` and `semif_agent/queue.py`:

```
intake (typed input / messenger gateway / scripted)
  └─ choice   "should this interrupt the current process?"   (SemIf)
       ├─ yes → preempt current, requeue it with state preserved
       └─ no  → score   "how urgent?"  critical/high/medium/low   (SemIf)
                 └─ urgency priority queue (weight desc, then FIFO)
```

- The **queue** is a max-heap by urgency weight, FIFO within equal weight, with
  upward ageing so low-priority items cannot starve (`queue.py`).
- A **skill run paused for input** (`needs_input`) keeps the current slot busy;
  an answer routes straight back to the pending run, bypassing
  choice/score/navigation.
- A high-priority input can preempt the running process, which re-queues with its
  state preserved.
- Exactly one execution slot. (Concurrency, when it comes, is SemIf's
  shared-state mode for parallel *decisions* — not parallel process execution.)

### The skill tree and navigation

Skills are leaves: categories → skills → actions. Navigation is a chain of SemIf
choices, one per level (`navigate` in `semif_agent/skills.py`), and **every
choice is logged** (`navigate:category`, `navigate:leaf`, `navigate:response`).

- **Two-stage at both levels; the guards are the create doors.**
- **Category.** The softmax offers every existing category **with its
  description** plus a `create_category` branch, and the winner is confirmed by
  the **scope guard** (`confirm_category_fit`, phase `navigate:category_scope`):
  a name-anchored SemIf check at `navigation.category_tau` asking whether that
  category's scope covers the request. Confirmed → descend; rejected (or a
  `create_category` win) → author a new category. Descriptions are load-bearing:
  with bare names the model sent "book a flight to japan" to `simplex` (0.49 vs
  `create_category` 0.12); with descriptions it picks `create_category` 0.87.
  An empty tree short-circuits straight to create.
- **Leaf.** The leaf softmax picks only among the existing skills (a create
  option diluted its probability and let a crowded tree drift into create on a
  weak plurality). The picked skill then goes through the **intent guard**
  (`confirm_skill_fit`, phase `navigate:intent`): an action comparison at
  `navigation.intent_tau` asking whether the skill performs the same action the
  request asks for. Same → run it; different → author a new leaf in that
  category. An empty (or single-skill) category skips the softmax and goes
  straight to the guard.
- Both guards are deliberately permissive; with real descriptions the in-scope
  cases score 0.9–1.0, so a borderline over-cover routes into a plausible
  category and the leaf guard catches it.

### The skill run loop

Every skill (`semif_agent/skill.py`) follows the same loop:

```
observe baseline → act → observe outcome → assess
```

- `act(ctx, request)` is the single body phase. It reads resolved values from
  `ctx.config` and performs the real action via stdlib transports, returning an
  `ActionResult(action_log, new_state, needs_input, decisions)`. Any SemIf
  **sub-decisions** it makes (e.g. which contact is "girlfriend", resolved over
  a live candidate set) are returned on `ActionResult.decisions` and logged as
  training rows with the run outcome.
- `assess` is a **SemIf decision**, not generation:
  - `assess:outcome` — success/failure at `tau`; its state includes the resolved
    inputs (secrets redacted) so a wrong resolution is visible.
  - `assess:requeue` — on failure, `complete` vs `retry` (bounded by
    `max_reentries`).
- The run summary is **deterministic** (same inputs ⇒ same string):
  `f"{category}.{name}: {'ok'|'failed'} — {action_log or new_state}"`.
- A skill can pause for human input two ways: `act` returns `needs_input` (act
  re-runs on resume, with `request.user_input` set), or — for a contract
  variable the runner cannot satisfy — a **pre-act** pause collects config
  before act runs.

### Authoring a new skill

When navigation chooses a create branch, the scheduler queues the draft to a
single-slot `llm` worker and the body write to a separate single-slot codegen
worker. It is **asynchronous**, so the gate stays free while either model thinks:

1. **Stub.** The small `llm` provider (`generate_skill`) authors a title +
   description on its own worker. The stub is registered and hot-merged into the
   running tree as
   soon as it lands; a `skill_writing` trace event is emitted.
2. **Elicitation** (`generate_elicitation`, default on): implementation
   questions — which service/account, how to connect, where the credential
   comes from, what success looks like. Every front end (REPL, dashboard,
   gateway) defers to a question queue and the worker waits
   (`codegen.elicitation.wait_timeout`).
3. **Codegen body** (`generate_skill_body`): a larger OpenAI-compatible model
   (`codegen`, your configured code-capable model) writes a single `act` against `CODEGEN.md`,
   reading every operational value from `ctx.config` and declaring `INTEGRATION`
   (service / transport / config_vars) plus its own flat `CONTRACT`
   (variable → description; every key must be read from `ctx.config`).
4. **Fidelity gate** (`authoring:fidelity` SemIf accept/reconsider decision +
   static `integration_findings`): is the action real, or simulated? One
   corrective regen with the raw evidence bundle, then accept-with-badge
   (`unverified`) rather than hard-fail.
5. **Contract** (`parse_contract`): read from the body's own `CONTRACT`
   constant — a flat map of snake_case variable name → semantic description, for
   user input and SemIf only; `contract.json` is a persisted mirror.
6. **Test** (`generate_skill_tests`): a shared-context call produces
   `skill.test.py`, a hermetic mechanics test (inline fixtures, a loopback
   `http.server` for HTTP bodies, no external network).
7. **Auto-run** (`run_skill_test`): executes the test as a subprocess. On
   failure a 2-option SemIf decision (`codegen_regen`) picks code/test to
   regenerate (the contract rides in the code), bounded by
   `codegen.test_max_attempts`.

On success the body materializes to `data/skills/<category>/<name>/`, is
hot-merged, and **the original request is re-queued** at its scored weight and
re-runs navigation onto the new leaf. On failure the leaf stays a restartable
stub. A whole new category runs the same chain deterministically:
`create_category` → `create_skill` → async body → re-dispatch.

A rejected/simulated body is traced (`fidelity_review`), badged `unverified`,
and never blocks authoring. Codegen failures (including timeouts) surface as a
graceful stub and are retryable with `restart <category> <skill>`.

### What each model sees

Two kinds of model calls exist, and they are injected with **different things**.

A **SemIf decision** carries only three fields — `state`, `question`, and
`options` (each with an `id` + `description`) — and
`semif_phase1.llamacpp_backend.score` reads the option logits from that one
forward pass. There is no hidden system prompt and no growing conversation:
what you see in the table below is the entire input. Probabilities are
conditional on exactly the supplied options.

A **generative call** (`llm` for titles/descriptions, `codegen` for bodies and
tests) is a built OpenAI-style `messages` list. Its context is assembled from
the request, the tree, the elicitation answers, the runtime **bridge catalog**
(`describe_bridges()`), and — for bodies and tests — the contract files
(`CODEGEN.md`, `TESTGEN.md`).

#### SemIf decisions

| Phase | File | State injected | Options |
| --- | --- | --- | --- |
| `interrupt:choice` | `scheduler.py:279` | request text + `[current process: <skill>]` | `interrupt` / `defer` |
| `priority:score` | `scheduler.py:296` | request text (+ `[current process: …]` when busy) | critical / high / medium / low |
| `navigate:category` | `skills.py:682` | request text | every category name (+ `create_category`) |
| `navigate:leaf` | `skills.py:736` | request text + current category | every skill name + description (+ `create_skill`) |
| `navigate:response` | `skills.py:634` | request text + category | canned reply names + descriptions (catchall last) |
| `navigate:intent` | `skills.py:811` | `Requested action: <text>` + `Action of the existing skill: <description>` | `same` / `different` |
| `assess:outcome` | `skill.py:257` | skill label + goal + action log | `success` / `failure` |
| `assess:requeue` | `skill.py:279` | skill label + goal + action log + `outcome: failed` | `complete` / `retry` |
| `config:search` | `scheduler.py:1477` | skill + variable name + its semantic description | each candidate config key (+ `ask`) |
| `config:record` | `skill.py:112` | skill + `variable = value` | `record` / `ask_again` |
| `authoring:fidelity` | `scheduler.py:1311` | skill, request, description, `INTEGRATION` JSON, static findings, first 4000 chars of body | `accept` / `reconsider` |
| `repair:choice` | `scheduler.py:658` | skill, request, integration, truncated failure | `retry` / `repair_skill` / `ask_user` / `no_repair` |
| degeneration watchdog¹ | `cli.py:128` | `[codegen <model>]` + last 2000 chars of the stream | `continue` / `stop` |
| `codegen_regen`¹ | `cli.py:165` | `[testgen <model>]` + last 1200 chars of the test error | `regen_code` / `regen_test` / `regen_contract` |

¹ Trace-only: logged to `runs.jsonl`, never to `decisions.jsonl`.

#### Generative calls

| Call | Client | System context | User context | File |
| --- | --- | --- | --- | --- |
| `generate_category` | `llm` | authoring instruction: propose one broad category, JSON-only | request text + full tree summary | `skills.py:896` |
| `generate_skill` | `llm` | authoring instruction: propose one specific skill, JSON-only | request + category + existing skill names | `skills.py:947` |
| `generate_skill_body` | `codegen` | full **CODEGEN.md** | request, category, name, description, existing skills + tree summary, elicitation answers, integration hint, **bridge catalog**, `BODY_DIRECTIVES` | `codegen.py:156` |
| body retry | `codegen` | full **CODEGEN.md** | rejection reason, request, category, name, description, elicitation answers, integration hint, **bridge catalog**, `BODY_DIRECTIVES` | `codegen.py:474` |
| `generate_elicitation` | `codegen` | elicitation instructions + example questions + anti-patterns + `max_questions` | request, category, name, description, existing skills + tree summary, **bridge catalog** | `codegen.py:580` |
| `generate_data_contract` | `codegen` | full **TESTGEN.md** | skill, request, full body code, **bridge catalog** + the contract ask | `codegen.py:833` |
| `generate_skill_tests` | `codegen` | full **TESTGEN.md** | skill, request, full body code, **bridge catalog**; continued with the accepted **contract JSON** as an assistant turn + the test-gen ask | `codegen.py:942` |
| `regenerate_skill_body` | `codegen` | full **CODEGEN.md** | correction header, request, description, **raw evidence bundle** (rendered verbatim), previous body, integration hint, **bridge catalog**, `BODY_DIRECTIVES` | `codegen.py:1046` |

The **bridge catalog** is `describe_bridges()` (`bridges/registry.py:33`): each
known bridge's name, service, description, base-URL config var, auth header +
token var, documented config vars, and endpoints. It is the single runtime
source of bridge specifics — `CODEGEN.md` and `TESTGEN.md` carry only the generic
pattern — so a bridge can be added without touching either contract file.

The **raw evidence bundle** handed to a corrective regen carries the request,
description, requirements, the observed failure fields (error, action log, new
state, summary) and the previous body, rendered as-is with no prose diagnosis.

### Data contracts and tiered config

Generated bodies own no data. At runtime the runner merges:

```
global config → category config → skill config → per-fire answers  =  ctx.config
```

- After the contract lands, a SemIf `choice` per variable auto-populates the
  skill config from the whole cascade (global → category → skill config)
  (`config:search`).
- Variables still unresolved pause the run **before act** and are asked of
  the human one at a time.
- Each answer gets a SemIf `record-as-config vs ask-again-each-fire` choice
  (`config:record`); recorded values persist to the skill `config.json`.
- On successive firings only the unresolved variables are asked.

A finished skill folder is:

```
data/skills/<category>/<name>/
  skill.py        # runnable body
  skill.test.py   # hermetic mechanics test
  contract.json   # variable name -> semantic description
  config.json     # recorded values (secrets live here, gitignored)
```

Each body also records the `CODEGEN.md` revision it was written against
(`contract_ref` / `contract_dirty` on the `skill_writing` trace).

### Repairing a failed run

A failed real run logs a SemIf `repair:choice` decision offering:

```
retry | repair_skill | ask_user | no_repair
```

It is surfaced as a `repair_offered` trace, in the REPL (`repairs`,
`repair <id> [action]`) and in the dashboard. Executing is user-confirmed
(a codegen write is slow): `retry` re-queues, `repair_skill` rewrites the body
with the observed failure (bounded by `codegen.repair.max_attempts`), and
`ask_user` posts a repair question. A run is never silently auto-repaired.

### Dream: the learning signal

`semif_agent/dream.py` replays `decisions.jsonl` and computes, per row, the NLL
of the observed outcome under the predicted distribution, plus weighted
cross-entropy, accuracy and binned ECE. Human relabels (`relabel`, or
`POST /api/relabel`) are weighted 3×. This is the training signal for the v2
fine-tuning loop — it is not yet the fine-tune itself.

---

## Features

- **SemIf-only decisions** — choice, scoring, navigation, assessment, fidelity,
  config search/record, repair choice, regeneration choice, and a codegen
  degeneration watchdog all route through one local decision model.
- **No handle/ignore gate** — every input is dispatched; non-tasks fall through
  to the closed `response` canned tree.
- **Real integrations by construction** — bodies must perform the real action,
  read data from `ctx.config`, declare `INTEGRATION`, and report real failures.
- **Async skill authoring** — elicitation → body → fidelity → contract → test →
  auto-run, with a SemIf regeneration ladder on test failure. Gate stays free.
- **Tiered config + first-fire collection** — global/category/skill config with
  a `record-vs-ask` decision per answer.
- **Runtime repair loop** — retry / repair / ask / no-op, user-confirmed.
- **Decision logging + dream cost** — every decision is a labeled training row
  with prompt hashes, tokens and timing.
- **Run lifecycle tracing** — `runs.jsonl` keyed by `run_id`, powering a
  Redux-DevTools-style dashboard.
- **Messenger command gateway (SimpleX)** — take commands and reply over DM.
- **Standalone bridge services** — a token-guarded localhost HTTP layer in front
  of third-party systems, so generated bodies never speak native protocols.
- **Seeds** — shipped real starter skills: `calendar.create_event` (the
  authoritative worked example), `calendar.next_event` (Nextcloud CalDAV), and
  `simplex.next_message` / `simplex.connect_link`.
- **Fatal-on-engine-loss** — the engine is always real; its absence exits the app.
- **Stdlib-only core** — SemIf/llama.cpp are lazy imports, so the package stays
  importable and testable without the heavy engine installed.

---

## Repository layout

```
semif_agent/
  cli.py            argparse: run / dream / skills / status / relabel /
                    dashboard / gateway / bridge
  scheduler.py      no up-front gate; choice -> score -> queue -> dispatch;
                    preempt + requeue; needs_input pauses; async single-slot
                    skill authoring worker; fidelity gate; repair loop
  skill.py          observe -> act -> observe -> assess; deterministic
                    summary; pre-act contract collection; resolved-input state
  skills.py         tree + registry, canned `response` tree, navigation,
                    create gates, intent guard, tiered config resolution
  codegen.py        OpenAI-compatible client; body/elicitation/test generation;
                    parse/validate; CONTRACT + INTEGRATION extraction; test runner
  provider.py       shared OpenAI-compatible transport (SSE/budget/idle/watchdog)
  engine.py         SemIfEngine -> semif_phase1.llamacpp_backend (lazy import)
  llm.py            small OpenAI-compatible provider; authors new title + description
  log.py            decisions.jsonl rows {state, question, options, probs, ...}
  trace.py          runs.jsonl lifecycle events keyed by run_id
  dream.py          NLL / weighted CE / accuracy / ECE cost report
  decisions.py      DecisionRequest / Option / DecisionResult / Request dataclasses
  queue.py          urgency max-heap (weight desc, FIFO, ageing)
  dashboard.py      stdlib HTTP + JSON API + static UI
  static/           dashboard frontend (html/js/css)
  simplex_ws.py     neutral SimpleX daemon protocol (shared by gateway + bridge)
  gateway/          messenger COMMAND intake/reply (SimpleX first)
  bridges/          standalone third-party API bridges (SimpleX + the LLM bridge)
seeds/              committed starter skills (calendar, simplex)
scripts/            bootstrap.sh, simplex-address.py, systemd/*.in
requirements/       staging.txt — the pinned engine deps
tests/              stdlib unit tests + tests/integration (real engine + LLM)
CODEGEN.md            the contract fed to the skill-body codegen model
TESTGEN.md          the contract fed to the test/contract codegen model
AGENTS.md           the operational guide (read this before touching code)
config.example.json per-machine config template (seeded to config.json once)
pins.json           committed pins: engine/GGUF/tokenizer/simplex-chat refs
local/              gitignored: this deployment's notes (hosts, addresses, models)
```

---

## Installation

The agent runs anywhere pure-Python runs, but a *working* agent needs the SemIf
decision engine, a pinned GGUF, and an OpenAI-compatible endpoint for codegen.
`scripts/bootstrap.sh` provisions all of it inside a gitignored `.runtime/` tree
in the checkout.

### Prerequisites: ollama models

Two OpenAI-compatible endpoints (usually ollama), which may be on different
machines:

- **`codegen`** — a large, code-capable model that writes skill bodies. This is
  slow and ideally on a beefier host.
- **`llm`** — a small, fast model that authors the title + description of a
  newly created category/skill. Deliberately its own endpoint/model, so it can
  stay local even when `codegen` is remote; it uses the same provider transport
  (`provider.py`) and an unreachable endpoint is graceful (the request is not
  re-dispatched), never fatal.

The **decision engine is separate** and runs via llama.cpp CPU (or Vulkan if your
build enables it), not ollama.

### Option A: one-command provisioning (`bootstrap.sh`)

On a fresh Ubuntu machine, from a checkout of this repo:

```sh
scripts/bootstrap.sh
```

With no `config.json` yet, the script seeds one from `config.example.json`; edit
it to set `llm.model`/`codegen.model` (and the endpoints). Flags override those
values for the run only — `--llm-url`, `--codegen-url`, `--llm-model`,
`--codegen-model` — and are never written back to `config.json`.

The script is idempotent (every stage no-ops on existing state) and:

1. installs system prereqs + enables linger;
2. generates/uses `~/.ssh/id_ed25519` and clones + pins the SemIf engine;
3. creates `.runtime/venv` and installs the pinned deps from
   `requirements/staging.txt` (installing `semif-phase1` with `--no-deps`, since
   the llama.cpp path never imports torch);
4. downloads and sha256-verifies the pinned GGUF into `.runtime/models/`;
5. pre-fetches the HF tokenizer into `.runtime/hf/`;
6. downloads and sha256-verifies the pinned `simplex-chat` binary;
7. seeds `config.json` from `config.example.json` **only if it does not already
   exist** — it never rewrites `config.json` (only a human edits that file);
8. renders and enables systemd **user** units:
   `semif-simplex` (bot daemon), `semif-gateway` (agent gateway),
   `semif-simplex-forward` (the bridge's own daemon), `semif-bridge`;
9. prints the bot's SimpleX contact address.

Flags (`--llm-url`, `--codegen-url`, `--llm-model`, `--codegen-model`,
`--copy-data SRC`) override the values read from `config.json` for that run only.
Run `scripts/bootstrap.sh -h` for the full list.

It installs **no ollama**; it expects one for `llm` and pulls `llm.model`,
warning (never auto-pulling) if the remote codegen host is missing its model.
Pinned external refs (engine commit, GGUF url+sha256, HF tokenizer revision,
simplex-chat version/url+sha256) live in the committed `pins.json`, not in
`config.json`.

### Option B: core only (stdlib)

The core is dependency-free, so unit tests run anywhere:

```sh
python3 -m pytest tests/ -q --ignore=tests/integration
```

For a full install on a machine you manage, follow the same steps bootstrap
performs (venv, `semif-phase1 --no-deps`, `requirements/staging.txt`, GGUF,
tokenizer cache), then create `config.json` from the template and adjust paths:

```sh
cp config.example.json config.json
```

### Configuration

`config.json` is **gitignored, per-machine, and required** — the agent refuses
to start without it. `scripts/bootstrap.sh` seeds it from `config.example.json`
once; afterwards only a human edits it (nothing writes `config.json`). Pinned
external refs live in the committed `pins.json`, so `config.json` holds only
per-machine settings:

```sh
cp config.example.json config.json   # if not already created by bootstrap
```

Key blocks:

| Block | What it controls |
| --- | --- |
| `engine` | per-machine engine settings: backend, context, threads, optional GGUF path override (refs come from `pins.json`) |
| `llm` | small model endpoint that authors new category/skill title + description |
| `codegen` | skill-body model endpoint, timeouts, sampler, elicitation, fidelity, repair, test, degeneration watchdog, token budget |
| `navigation` | two-stage guards: leaf `intent_tau`, category `category_tau` |
| `tau` | decision threshold; `max_reentries` requeue bound |
| `queue` | `max_size`, `age_rate` |
| `skill_bodies` / `skill_seeds` / `log` / `trace` / `category_registry` | runtime paths (anchored to the checkout) |
| `gateway.simplex` | command gateway: ws_url, allowlist, batching |
| `bridges.simplex` | forwarding bridge: host/port/token/ws_url |
| `simplex_bridge_url` / `simplex_bridge_token` | what skill bodies call |
| `bridges.llm` | LLM bridge: host/port/token (model comes from `llm`) |
| `llm_bridge_url` / `llm_bridge_token` | what skill bodies call for generation |
| `simplex_chat` | gateway/forward ports and bot display names (binary refs come from `pins.json`) |
| `dashboard` | bind host/port |

`llm.model` and `codegen.model` are **required** (no default): set each to a
model its endpoint serves. A missing model fails fast at startup with a clear
config error rather than sending a request to a model that isn't there.

Runtime artifacts (`data/decisions.jsonl`, `data/runs.jsonl`,
`data/categories.json`, `data/skills/`, `data/drafts/`) are gitignored too.

> **Codegen timeout:** do not cap codegen `max_tokens`. A reasoning codegen
> model can write for 25–45 min; `codegen.timeout` defaults to 1200 s and a
> per-machine config may raise it to 3600 s.

---

## Running the agent

All subcommands go through the CLI:

```sh
python -m semif_agent.cli run              # interactive REPL
python -m semif_agent.cli run --script inputs.jsonl
python -m semif_agent.cli dream            # cost report
python -m semif_agent.cli skills           # print the skill tree
python -m semif_agent.cli status           # current process + queue
python -m semif_agent.cli relabel <id> <outcome>
python -m semif_agent.cli dashboard [--port 8765] [--host 0.0.0.0] [--replay]
python -m semif_agent.cli gateway [--platform simplex,lxmf|all] [--dashboard]
python -m semif_agent.cli bridge [--name simplex]
```

On a provisioned host, use the venv and set `HF_HOME` to the checkout's cache:

```sh
REPO=<your-checkout>
HF_HOME="$REPO/.runtime/hf" "$REPO/.runtime/venv/bin/python" -m semif_agent.cli run
```

### REPL

Type a request, or one of the meta-commands:

```
busy <text>   mark a current process (so preemption can be demonstrated)
idle          clear the current process
status        current process + queue
skills        skill tree
dream         cost report
relabel <id> <outcome>
restart <category> <skill>
repairs       list pending repair offers
repair <offer-id> [retry|repair_skill|ask_user|no_repair]
quit
```

Authoring questions and repair offers are printed after each turn; answer them
inline.

### Scripted mode

`--script file.jsonl` submits each `{"text": ..., "source": ...}` row, then
drains the queue.

### Dashboard

A stdlib HTTP server serving a Redux-DevTools-style inspector: static skill tree,
run flow with decisions/events and dream costs, status, and write endpoints.

```
GET  /api/trace /api/tree /api/dream /api/status /api/questions /api/repairs
POST /api/submit /api/answer /api/questions /api/restart /api/repair /api/relabel
```

Live mode warms the engine; `--replay` reads the logs without loading SemIf
(submit degrades to a JSON error). Binds `127.0.0.1:8765` by default.

### Messenger gateway (SimpleX + LXMF)

`cli gateway` runs one or more command transports in a single process that
shares one scheduler/skill store. `--platform` takes a comma-separated list or
`all` and defaults to `simplex`:

```sh
python -m semif_agent.cli gateway [--platform simplex,lxmf|all] [--dashboard]
```

Each platform feeds authorized DM text through the normal gate/score/queue/
dispatch pipeline. Results, authoring questions and repair offers go back to the
originating chat. Config lives under `gateway.<platform>` in `config.json`;
`enabled` defaults to false. Both transports are **default-deny** — put allowed
identities in `gateway.<platform>.allowed_users` (or set `allow_all_users` for
dev) — and **all adapters share one process** (never run two gateway processes
over one skill store). On a provisioned host `semif-gateway.service` runs
`gateway --platform all`, so it survives logout/reboot.

**SimpleX.** Connects to the local `simplex-chat` daemon over its JSON WebSocket
API (`gateway.simplex.ws_url`, which must match `simplex_chat.port`). Requires
the optional `websockets` package (lazy import; the gateway refuses to start
without it). `allowed_users` entries match a numeric contactId or a display
name. The daemon must run in **bot mode** (`simplex-chat -p PORT
--create-bot-display-name NAME`), not `--headless`/`--relay` — in v7 those mean
"chat relay" and yield a relay address, not a user contact address. Add the
bot's contact link to reach it: `scripts/simplex-address.py` shows/creates it
(bootstrap prints it), and the gateway prints it once on connect.

**LXMF (Reticulum).** Runs **in-process — there is no external daemon**;
`pip install lxmf` provides the transport (optional, lazy import; the gateway
refuses to start with an install hint otherwise). The bot's reachable address is
its LXMF delivery destination hash (32 hex chars), printed once on connect; the
identity is persisted under `gateway.lxmf.storage_path` (default
`.runtime/lxmf/router`) so the address is stable across restarts. `allowed_users`
matches the peer's LXMF address (32 hex) or a best-effort display name.
`desired_method` is `direct` (reliable link) or `opportunistic` (single packet),
and an optional `propagation_node` enables store-and-forward. Add the printed
address as a contact in your LXMF client and DM it.

> **Gateway isolation is non-negotiable.** The gateway takes commands and sends
> replies — nothing else. It never reads history, shows or creates invite links,
> or composes messages. That is messaging UX and lives in the bridges.

### Bridge services

`cli bridge` runs standalone bridges: each is a process that stands up a small,
token-guarded localhost HTTP API in front of one third-party system and owns its
own daemon/profile. Generated skill bodies call them like any HTTP service (base
URL from a config var), so they never speak a native protocol directly.

The first bridge is **SimpleX** (`semif_agent/bridges/simplex.py`), with its own
simplex-chat daemon separate from the gateway's. Its HTTP surface:

```
GET  /health
GET  /contacts
GET  /inbox            # peek buffered inbound DMs
GET  /inbox/next?contact=<id>   # pop oldest unread
GET  /unread           # daemon's persistent unread chats (read-only; 503/502)
GET  /history?contact=<id>&count=<n>   # recent messages for one chat (read-only; 503/502)
GET  /address          # show/create the contact link (503 not connected, 502 failure)
POST /send {"recipient": "<id|display_name>", "text": "..."}
```

`/unread` and `/history` read the daemon's own state (`/_get chats` / `/_get chat`),
so they see messages that arrived while the bridge was down — unlike the live
`/inbox` receive buffer. Both are read-only: v7 has no mark-read command, so
acking is a client-side read receipt, not an API call.

`simplex_bridge_url` (top-level config) is what skills call; the bridge catalog
is injected into every codegen prompt via `describe_bridges()`, so it is the
single source of bridge specifics. Adding a bridge = a class + a config block +
a `CATALOG` entry.

The **LLM bridge** (`semif_agent/bridges/llm.py`) is the second: a generic

```
POST /chat {"messages": [{"role","content"}, ...], "max_tokens"?: int} -> {"text": "..."}
```

in front of the agent's language model, so a body can ask for short generated
text (e.g. extract an event title + description) over HTTP instead of speaking
the OpenAI-compatible protocol itself. It does not own the model connection:
`cli bridge` threads the scheduler's already-configured `llm` client into it, so
the endpoint/model/sampler live only in the top-level `llm` block; `bridges.llm`
carries just the local listener. Skills call it via `llm_bridge_url`
(+ optional `llm_bridge_token`); with no client it returns `502`, never a
fabricated reply. It starts in the same `semif-bridge.service` process as the
other enabled bridges.

### Dream report

```sh
HF_HOME=... python -m semif_agent.cli dream
```

Prints row counts, weighted cross-entropy, accuracy, ECE and the highest-cost
rows.

---

## Skills in detail

A skill is one specific, single-purpose action reachable by category → skill
navigation. It is three things:

1. a **manifest** (navigable description — `name`, `category`, `description`,
   plus optional `allowed_inputs`, `actions`, `cost_budget`, `decision_log_ref`);
2. a **code body** (a single `act`);
3. an **`INTEGRATION` declaration** (service / transport / config_vars) and a
   flat **`CONTRACT`** (the operational values the runner must provide).

`CODEGEN.md` is the authoritative body contract and is fed verbatim to the codegen
model. It requires: perform the real action; stdlib transports only
(`urllib`/`http.client`, `imaplib`/`smtplib`/`poplib`, `subprocess`, file I/O);
no mocking; data from `ctx.config` (never embedded or fabricated); request
clarification via `needs_input` when requirements are unclear; write files only
under configured data dirs; fail fast; and declare a consistent `INTEGRATION`.

`TESTGEN.md` is the separate contract for tests. A generated test is a
**hermetic mechanics check** — inline fixtures, a loopback `http.server` for HTTP
bodies, no external network, no `mock_data.json`. Passing it proves the code
runs, not that the live integration works. That is what the real run and the
repair loop are for.

**Seeds** (`seeds/`) ship real starter integrations in the exact generated
folder format, plus a `manifest.json`. `merge_seed_store` loads them into every
tree at startup; recorded config is written to the runtime store, so the
committed seed never holds secrets.

- `calendar.create_event` — create an event on Nextcloud CalDAV (the
  authoritative worked example); resolves the target calendar with a SemIf
  choice over the owned calendars (strong winner, else the configured default);
  config vars `nextcloud_url`, `nextcloud_username`, `nextcloud_app_password`,
  `nextcloud_default_calendar`, plus the LLM bridge (`llm_bridge_url`,
  `llm_bridge_token`).
- `calendar.next_event` — next upcoming event from Nextcloud CalDAV (recurring
  events expanded server-side); config vars `nextcloud_url`,
  `nextcloud_username`, `nextcloud_app_password`, `nextcloud_default_calendar`.
- `simplex.next_message` — read the next unread message via the forwarding
  bridge; config vars `simplex_bridge_url`, `simplex_bridge_token`,
  `simplex_default_contact`.
- `simplex.connect_link` — show/create the forwarding bot's contact link.

---

## Testing

```sh
# anywhere, fast, stdlib only
python3 -m pytest tests/ -q --ignore=tests/integration

# where real SemIf + a real ollama are available (run from the checkout root)
./.runtime/venv/bin/python -m pytest tests/integration -q -s
```

Unit tests use the `ScriptedEngine` double (`tests/conftest.py`) to drive
scheduling mechanics without a GGUF; data-structure tests cover queue ordering,
dream cost and contract serialization. Integration tests exercise the real
engine + LLM end to end.

**Fixtures are not the runtime.** Skill tests use real local endpoints (a
loopback server, never a mock); a real run is the only proof an integration
works.

---

## Roadmap / status

**Done.** Core loop, urgency queue, skill tree, real SemIf assessment, decision
logging, `dream` cost pass, REPL + JSONL CLI; live `create_category` /
`create_skill` with the async authoring pipeline (elicitation → body → fidelity →
contract → test → auto-run → re-dispatch); tiered config + first-fire
collection; runtime repair loop; the closed `response` canned tree; real
integrations with `INTEGRATION` declarations; the SimpleX command gateway and the
standalone SimpleX bridge + seeds (contact-list refresh, daemon-backed message
history / true unread).

**Next.** Real fine-tuning from grounded decision rows ("dreaming") with
validation + pinned-revision swap; queue persistence; more intake sources
(events/timers); dashboard run-requeue cross-linking; per-decision thresholds and
calibration; more bridges; a `simplex.send_message` seed. See `AGENTS.md` for the
detailed backlog and known gotchas.
