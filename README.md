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
  - [The scheduler: queue and dispatch](#the-scheduler-queue-and-dispatch)
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
  - [Messenger gateway (SimpleX + LXMF)](#messenger-gateway-simplex--lxmf)
  - [Voice gateway](#voice-gateway)
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
- **Nothing is gated out up front.** Every input is dispatched.
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

### The scheduler: queue and dispatch

Implemented in `semif_agent/scheduler.py` and `semif_agent/queue.py`:

```
intake (typed input / messenger gateway / scripted)
  ├─ idle  → dispatch immediately
  └─ busy  → FIFO queue (arrival order, no priority)
```

- The **queue** is a strict first-in-first-out list of pending requests with a
  maximum depth (`queue.py`). There is no urgency scoring and no preemption.
- A **skill run paused for input** (`needs_input`) keeps the current slot busy;
  an answer routes straight back to the pending run, bypassing the queue and
  navigation.
- The running process always finishes before the next queued request is
  dispatched.
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
  (`confirm_skill_fit`, phase `navigate:intent`): at `navigation.intent_tau` it
  decides what the skill does for the request, given the steps already taken
  (the run ledger is part of its state). `same` (it performs the next action the
  request needs) or `part` (the request needs more than one action and this
  skill performs one required part) → run it; `different` (it does no action the
  request asks for) → author a new leaf in that category. The guard only decides
  whether the skill contributes; whether more work remains is `assess:outcome`'s
  call. An empty (or single-skill) category skips the softmax and goes straight
  to the guard.
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
  - `assess:outcome` — is there any *additional work* to satisfy the request?
    Three outcomes at threshold precedence `failed > done > continue`: `done`
    (request satisfied), `continue` (this step worked but more work is needed →
    the scheduler routes the logical next step with a bounded run ledger),
    `failed` (the step errored → the repair path). Its state includes the skill
    action, the goal, the resolved inputs (secrets redacted), the bounded action
    log, and the steps already taken.
- The run summary is **deterministic** (same inputs ⇒ same string):
  `f"{category}.{name}: {'ok'|'failed'} — {action_log or new_state}"`.
- Every skill's output is bounded to one **speakable budget** (`chain.result_chars`,
  ~300 chars): the result string and its run-ledger entry. A long product belongs
  in a file I/O side effect, not the result (see `CODEGEN.md`).
- A skill can pause for human input two ways: `act` returns `needs_input` (act
  re-runs on resume, with `request.user_input` set), or — for a contract
  variable the runner cannot satisfy — a **pre-act** pause collects config
  before act runs.

### Authoring a new skill

When navigation chooses a create branch, the scheduler queues the draft to a
single-slot `llm` worker and the body write to a separate single-slot codegen
worker. It is **asynchronous**, so the gate stays free while either model thinks:

1. **Spec.** The small `llm` provider (`generate_skill_spec`) authors a title +
   description **and a de-identified, reusable goal** (plus the generic
   service/transport) on its own worker. The stub is registered and hot-merged
   into the running tree as soon as it lands; a `skill_writing` trace event is
   emitted. The goal is the *only* task data codegen is ever shown.
2. **Codegen body** (`generate_skill_body`): a larger OpenAI-compatible model
   (`codegen`, your configured code-capable model) writes a single `act` against `CODEGEN.md`,
   reading every operational value from `ctx.config` and declaring `INTEGRATION`
   (service / transport / config_vars) plus its own flat `CONTRACT`
   (variable → description; every key must be read from `ctx.config`). **Privacy:**
   codegen is the primarily-cloud model, so it receives only the de-identified
   spec — never the user's request, question answers, or run output.
4. **Fidelity gate** (`authoring:fidelity` SemIf accept/reconsider decision +
   static `integration_findings`): does the body contain a concrete,
   side-effecting operation that really performs its goal? The state carries the
   **whole** body (never a prefix) plus a deterministic operation inventory and
   the de-identified goal — never the raw request — and asks the positive
   question rather than asking the model to prove a negative. Because a decision
   ingests the body whole, generated artifacts are size-bounded
   (`max_body_chars`/`max_test_chars`, derived from the engine's context). One
   corrective regen with the de-identified evidence bundle (inventory + findings
   included), then accept with a trace rather than hard-fail.
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
hot-merged, and **the original request is re-queued** and re-runs navigation
onto the new leaf. On failure the leaf stays a restartable stub. A whole new
category runs the same chain deterministically:
`create_category` → `create_skill` → async body → re-dispatch.

A rejected/simulated body is traced (`fidelity_review`) and never blocks
authoring. Codegen failures (including timeouts) surface as a graceful stub and
are retryable with `restart <category> <skill>`.

### What each model sees

Two kinds of model calls exist, and they are injected with **different things**.

A **SemIf decision** carries only three fields — `state`, `question`, and
`options` (each with an `id` + `description`) — and
`semif_phase1.llamacpp_backend.score` reads the option logits from that one
forward pass. There is no hidden system prompt and no growing conversation:
what you see in the table below is the entire input. Probabilities are
conditional on exactly the supplied options.

A **generative call** (`llm` for titles/descriptions/goals, `codegen` for bodies
and tests) is a built OpenAI-style `messages` list. The trusted local `llm`
context is assembled from the request and the tree; the **codegen** context is
assembled only from the de-identified spec (goal + generic service/transport),
the tree, the runtime **bridge catalog** (`describe_bridges()`), and — for bodies
and tests — the contract files (`CODEGEN.md`, `TESTGEN.md`). Codegen never
receives the user's request, question answers, or run output.

#### SemIf decisions

| Phase | File | State injected | Options |
| --- | --- | --- | --- |
| `navigate:category` | `skills.py:682` | request text | every category name (+ `create_category`) |
| `navigate:leaf` | `skills.py:736` | request text + current category | every skill name + description (+ `create_skill`) |
| `navigate:response` | `skills.py:634` | request text + category | canned reply names + descriptions (catchall last) |
| `navigate:intent` | `skills.py:811` | `Requested action: <text>` + `Action of the existing skill: <description>` | `same` / `different` |
| `assess:outcome` | `skill.py:311` | skill label + goal + resolved inputs + bounded action log + steps already taken | `done` / `continue` / `failed` |
| `config:search` | `scheduler.py:1477` | skill + variable name + its semantic description | each candidate config key (+ `ask`) |
| `config:record` | `skill.py:112` | skill + `variable = value` | `record` / `ask_again` |
| `authoring:fidelity` | `scheduler.py:1836` | skill, goal, description, `INTEGRATION` JSON, static findings, operation inventory, **whole** body (no raw request) | `accept` / `reconsider` |
| `repair:choice` | `scheduler.py:658` | skill, request, integration, truncated failure | `retry` / `repair_skill` / `ask_user` / `no_repair` |
| degeneration watchdog¹ | `cli.py:128` | `[codegen <model>]` + last 2000 chars of the stream | `continue` / `stop` |
| `codegen_regen`¹ | `cli.py:165` | `[testgen <model>]` + last 1200 chars of the test error | `regen_code` / `regen_test` / `regen_contract` |

¹ Trace-only: logged to `runs.jsonl`, never to `decisions.jsonl`.

#### Generative calls

| Call | Client | System context | User context | File |
| --- | --- | --- | --- | --- |
| `generate_category` | `llm` | authoring instruction: propose one broad category, JSON-only | request text + full tree summary | `skills.py:896` |
| `generate_skill_spec` | `llm` | authoring instruction: propose one specific skill + a **de-identified goal** and generic service/transport, JSON-only | request + category + existing skill names | `skills.py:947` |
| `generate_skill_body` | `codegen` | full **CODEGEN.md** | **de-identified spec** (goal + description), category, name, existing skills + tree summary, integration hint, **bridge catalog**, `BODY_DIRECTIVES` | `codegen.py:156` |
| body retry | `codegen` | full **CODEGEN.md** | rejection reason, de-identified spec (goal + description), integration hint, **bridge catalog**, `BODY_DIRECTIVES` | `codegen.py:474` |
| `generate_skill_tests` | `codegen` | full **TESTGEN.md** | skill, de-identified goal, full body code, **bridge catalog**; continued with the accepted **contract JSON** as an assistant turn + the test-gen ask | `codegen.py:942` |
| `regenerate_skill_body` | `codegen` | full **CODEGEN.md** | correction header, de-identified goal + description, **de-identified evidence bundle** (rendered verbatim), previous body, integration hint, **bridge catalog**, `BODY_DIRECTIVES` | `codegen.py:1046` |

The **bridge catalog** is `describe_bridges()` (`bridges/registry.py`): each
known bridge's name, service, description, base-URL config var, auth header +
token var, documented config vars, and endpoints. It is the single runtime
source of bridge specifics — `CODEGEN.md` and `TESTGEN.md` carry only the generic
pattern — so a bridge can be added without touching either contract file (a
bridge ships with its pack; the registry discovers it by convention).

The **evidence bundle** handed to a corrective regen is de-identified: a generic
failure kind, the integration, the goal, and the previous body, rendered as-is
with no prose diagnosis. It carries no request text and no run output.

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
- **Async skill authoring** — spec → body → fidelity → contract → test →
  auto-run, with a SemIf regeneration ladder on test failure. Gate stays free.
- **Tiered config + first-fire collection** — global/category/skill config with
  a `record-vs-ask` decision per answer.
- **Runtime repair loop** — retry / repair / ask / no-op, user-confirmed.
- **Decision logging + dream cost** — every decision is a labeled training row
  with prompt hashes, tokens and timing.
- **Run lifecycle tracing** — `runs.jsonl` keyed by `run_id`, powering a
  Redux-DevTools-style dashboard.
- **Messenger command gateway (SimpleX + LXMF)** — take commands and reply over DM,
  including Sideband push-to-talk **voice messages** on LXMF (local transcription,
  and voice replies synthesized with Piper).
- **Voice gateway** — a local microphone front end: wake word + speech-to-text +
  text-to-speech, all local (openWakeWord / faster-whisper / Piper).
- **Standalone bridge services** — an optional token-guarded localhost HTTP
  layer for third-party systems that need state/auth, a native protocol, or a
  sanitized surface; a plain HTTP API is called directly.
- **Seeds** — shipped real starter skills: `nextcloud.create_event` (the
  authoritative worked example), `nextcloud.next_event` (the user's Nextcloud
  calendar, through the bridge), `nextcloud.create_task` (Nextcloud task list),
  `simplex.next_message` / `simplex.send_message` / `simplex.connect_link`,
  `calculator.calculate` (two operands, one SemIf-chosen operator, spoken answer),
  and `skills.list_categories` / `skills.list_skills` (report the agent's own
  skill categories, or the skills inside a named category).
- **Fatal-on-engine-loss** — the engine is always real; its absence exits the app.
- **Stdlib-only core** — SemIf/llama.cpp are lazy imports, so the package stays
  importable and testable without the heavy engine installed.

---

## Repository layout

```
semif_agent/
  cli.py            argparse: run / dream / skills / status / relabel /
                    dashboard / gateway / bridge
  scheduler.py      no up-front gate; FIFO queue -> dispatch;
                    needs_input pauses; async single-slot
                    skill authoring worker; fidelity gate; repair loop
  skill.py          observe -> act -> observe -> assess; deterministic
                    summary; pre-act contract collection; resolved-input state
  skills.py         tree + registry, canned `response` tree, navigation,
                    create gates, intent guard, tiered config resolution
  codegen.py        OpenAI-compatible client; body/test generation;
                    parse/validate; CONTRACT + INTEGRATION extraction; test runner
  provider.py       shared OpenAI-compatible transport (SSE/budget/idle/watchdog);
                    SemIfEngineClient generates text from the in-process engine
  engine.py         SemIfEngine -> semif_phase1.llamacpp_backend (lazy import);
                    call() scores, generate() samples text
  llm.py            small provider; authors new title + description
                    (ollama / opencode / the loaded engine via `semif`)
  log.py            decisions.jsonl rows {state, question, options, probs, ...}
  trace.py          runs.jsonl lifecycle events keyed by run_id
  dream.py          NLL / weighted CE / accuracy / ECE cost report
  decisions.py      DecisionRequest / Option / DecisionResult / Request dataclasses
  queue.py          bounded FIFO request queue
  dashboard.py      stdlib HTTP + JSON API + static UI
  static/           dashboard frontend (html/js/css)
  simplex_ws.py     neutral SimpleX daemon protocol (shared by gateway + bridge)
  voice_transport.py neutral voice transport: mic capture, wake word, VAD,
                    speech-to-text (faster-whisper), text-to-speech (Piper)
  gateway/          messenger COMMAND intake/reply (SimpleX, LXMF, voice)
  bridges/          bridge contract + discovery (base.py, registry.py, inbox.py,
                    llm.py); the concrete bridges ship in their packs
  config.py         layered config: index includes + !secret + schema/example
  packs.py          SkillPack + PackRegistry (bundles skills by concern/bridge)
packs/              skill packs, each cloned from its own repo (pack.<name>)
                    into this gitignored dir — each ships its own bridge/ — the
                    pack list lives in the semif-packs index repo (see PACKGEN.md)
scripts/            bootstrap.sh, config_merge.py, voice-*.py, systemd/*.in
requirements/       core.txt (test runner), engine.txt (local decision engine),
                    transports.txt (websockets/lxmf), voice.txt (microphone
                    stack), transcribe.txt (faster-whisper, LXMF voice
                    messages), tts.txt (piper-tts + PyAV, LXMF voice replies),
                    codec2.txt (pycodec2, LXMF Codec2 PTT);
                    staging.txt = union of the non-hardware files;
                    a pack's own deps live with the pack at
                    packs/<name>/requirements.txt
tests/              stdlib unit tests + tests/integration (real engine + LLM)
CODEGEN.md            the contract fed to the skill-body codegen model
TESTGEN.md          the contract fed to the test/contract codegen model
PACKGEN.md          the contract for authoring a skill pack (packs/<name>/)
AGENTS.md           the operational guide (read this before touching code)
config.json         per-machine config INDEX ({"include": [...]}); seeded once
config/             split config fragments (core, gateway, bridges, packs/*)
secrets.json        gitignored credentials, referenced as "!secret <key>"
config.example.json GENERATED per-machine config reference (from the schemas)
pins.json           committed pins: engine/GGUF/tokenizer/simplex-chat refs
local/              gitignored: this deployment's notes (hosts, addresses, models)
```

---

## Installation

The agent runs anywhere pure-Python runs, but a *working* agent needs the SemIf
decision engine (local or remote). `scripts/bootstrap.sh` provisions what this
machine needs inside a gitignored `.runtime/` tree in the checkout — and only
what it needs, based on the answers to a short first-run interview.

### Prerequisites: the decision engine, and optional models

The **decision engine** routes every request. It runs one of two ways, chosen in
the interview:

- **local** — the pinned SemIf 4B GGUF via llama.cpp (no network). This is the
  default and the whole minimum path.
- **remote** — a Winnow `/v1/systemone` decision server (`engine.provider ==
  "winnow"`). The local engine clone, GGUF, and tokenizer are then skipped.

The local engine is CPU by default; to offload it to a GPU, set
`engine.gpu_layers` in `config.json` and run
`scripts/bootstrap.sh --gpu vulkan|rocm|cuda` (a GPU build is opt-in because it
is slower on CPU).

**Skill authoring is opt-in** (`new_skill_creation`: `deny` by default). When it
is off, no `llm`/`codegen` endpoint is needed at all — the agent only runs the
shipped packs. When it is on, two OpenAI-compatible endpoints are required
(usually ollama, possibly on different machines):

- **`llm`** — a small, fast model that authors the title + description of a
  newly created category/skill. Its `provider` can be `"ollama"` (default),
  `"opencode"` (the hosted Console API), or `"semif"` — the already-loaded
  decision engine model generates in-process, with no endpoint and no
  `llm.model` at all.
- **`codegen`** — a large, code-capable model that writes skill bodies. This is
  slow and ideally on a beefier host.

### Option A: one-command provisioning (`bootstrap.sh`)

On a fresh Ubuntu machine, from a checkout of this repo:

```sh
scripts/bootstrap.sh
```

On a fresh machine the script **asks a few questions** and writes this machine's
config from the answers:

1. **Decision engine** — local SemIf 4B, or a remote Winnow server (URL/model/key).
2. **Skill authoring** — `deny` (default), `ask`, or `allow`; if enabled, the
   `llm` + `codegen` endpoints and credentials.
3. **Capabilities** — the skill packs to enable, offered from the pack index
   repo (`semif-packs`): SimpleX, browser, Nextcloud, Outlook, Gutendex (the
   `essentials` pack is always on). Only the packs you pick are cloned.
4. **Gateways** — how you talk to the agent: SimpleX, LXMF, voice (or none —
   REPL + dashboard only).
5. **Timezone** — an IANA name for timers and reported times.

It then installs **only what those answers require**. Both ends of the spectrum
are reachable by the end of a run:

- **minimum** — local SemIf 4B + `essentials` (canned replies, time/timers,
  calculator, skill listing). No account, no network, no gateway, no heavy
  downloads.
- **maximum** — every pack, a remote Winnow engine, remote `llm`/`codegen`
  endpoints, and all gateways.

Non-interactive runs: `--yes` takes the minimum defaults; `--answers FILE`
replays a saved answer set (JSON matching the `setup` fields). The script is
idempotent — on a rerun the interview is **skipped** and the existing config is
used; pass `--reconfigure` to re-run it. `--voice` / `--browser` additionally
install those optional stacks even if the plan did not pick them.

The script:

1. installs system prereqs + enables linger;
2. clones/updates the skill packs, then runs the setup interview and writes
   `config/` + `secrets.json` (first run or `--reconfigure`);
3. generates/uses `~/.ssh/id_ed25519` and, for a local engine, clones + pins the
   SemIf engine;
4. creates `.runtime/venv` and installs only the needed dep files:
   `requirements/core.txt` always, `engine.txt` for a local engine,
   `transports.txt` when a SimpleX/LXMF transport is selected (`voice.txt` for
   the voice stack, `transcribe.txt` when `gateway.lxmf.voice` is enabled —
   plus `tts.txt` when `voice.reply_with_voice` is on, and `codec2.txt` + system
   `libcodec2` best-effort for non-HQ Codec2 PTT — and each selected pack's own
   `packs/<name>/requirements.txt` — e.g. the browser pack's Playwright deps);
5. for a local engine, downloads/sha256-verifies the GGUF into `.runtime/models/`
   and pre-fetches the HF tokenizer into `.runtime/hf/`;
6. downloads/sha256-verifies the pinned `simplex-chat` binary **only** when a
   SimpleX gateway or the simplex pack is selected;
7. renders and enables only the systemd **user** units the answers need
   (`semif-simplex` bot daemon, `semif-gateway` agent, `semif-simplex-forward`
   bridge daemon, `semif-bridge`), retiring any a reconfigure dropped;
8. prints the SimpleX contact address(es) when those services are active.

Run `scripts/bootstrap.sh -h` for the full list. It installs **no ollama**; it
pulls `llm.model` from the target host's ollama when configured, and never
auto-pulls `codegen`. Pinned external refs (engine commit, GGUF url+sha256, HF
tokenizer revision, simplex-chat version/url+sha256) live in the committed
`pins.json`, not in `config.json`.

### Option B: core only (stdlib)

The core is dependency-free, so unit tests run anywhere:

```sh
python3 -m pytest tests/ -q --ignore=tests/integration
```

For a full install on a machine you manage, follow the same steps bootstrap
performs (venv, `semif-phase1 --no-deps`, the `requirements/` files, GGUF,
tokenizer cache), then generate the split config:

```sh
python -m semif_agent.cli setup --interview    # or --yes / --answers FILE
python -m semif_agent.cli config-init          # schema defaults, no interview
```

### Configuration

The per-machine config is **gitignored, per-machine, and required** — the agent
refuses to start without it. `config.json` is a thin **index**
(`{"include": ["config/core.json", "config/gateway.json", ...]}`); the fragments
under `config/` hold the values and are deep-merged in order. Credentials live in
the gitignored `secrets.json` and are referenced from any fragment as
`"!secret <key>"` (there is no `api_key_env`). `scripts/bootstrap.sh` runs
`setup` to generate the split config from your answers; afterwards only a human
edits it (`--reconfigure` re-runs the interview).

```sh
python -m semif_agent.cli setup --interview    # generate config/ from answers
python -m semif_agent.cli config-init          # seed schema defaults (no interview)
python -m semif_agent.cli check-config         # validate against the schemas
```

The committed `config.example.json` is **generated** from the core schema
(`semif_agent/config.py`) and each pack's `config` schema; regenerate it after
changing a schema:

```sh
python -m semif_agent.cli check-config --write-example config.example.json
```

Pinned external refs live in the committed `pins.json`, so the config holds only
per-machine settings.

Key blocks:

| Block | What it controls |
| --- | --- |
| `engine` | per-machine engine settings: backend, context, threads, optional GGUF path override (refs come from `pins.json`) |
| `llm` | small model endpoint that authors new category/skill title + description |
| `codegen` | skill-body model endpoint, timeouts, sampler, approval gate, fidelity, repair, test, degeneration watchdog, token budget |
| `navigation` | two-stage guards: leaf `intent_tau`, category `category_tau` |
| `tau` | decision threshold; `max_reentries` requeue bound |
| `queue` | `max_size` |
| `skill_bodies` / `skill_packs` / `log` / `trace` / `category_registry` | runtime paths (anchored to the checkout) |
| `gateway.simplex` | command gateway: ws_url, allowlist, batching |
| `gateway.lxmf` | LXMF gateway: allowlist, Reticulum paths, `result_only` replies, Sideband voice transcription (`voice.enabled` + `voice.stt` model) and voice replies (`voice.reply_with_voice` + `voice.tts` voice) |
| `gateway.voice` | voice gateway: wake/STT/TTS engines + models, audio devices, VAD |
| `bridges.simplex` | forwarding bridge: host/port/token/ws_url |
| `bridges.llm` | LLM bridge: host/port/token (model comes from `llm`) |
| `bridges.nextcloud` | Nextcloud bridge: host/port/token + connection (url/username/app_password/defaults) |
| `bridges.outlook` | Outlook bridge: host/port/token + Microsoft Graph connection (client_id/tenant/defaults) |
| `bridges.browser` | Browser bridge: host/port/token + Playwright settings (engine/headless/profile dirs) |
| `simplex_chat` | gateway/forward ports and bot display names (binary refs come from `pins.json`) |
| `dashboard` | bind host/port |

Bridge skills read `simplex_bridge_url` / `llm_bridge_url` /
`nextcloud_bridge_url` / `outlook_bridge_url` / `browser_bridge_url` (+ their
`_token`) from
`ctx.config`; each is **derived** at startup from its `bridges.<name>`
host/port/token, so there is no duplicate top-level key to keep in sync. A bridge
with no block (or no port) leaves the variable unresolved and the operator is
asked for it.

`llm.model` and `codegen.model` are **required only when authoring is enabled**
(`new_skill_creation` is `ask`/`allow`): set each to a model its endpoint serves.
An empty `codegen.model` means "no authoring path" — the minimum config leaves it
empty and the agent simply runs the shipped packs. With authoring on, a missing
model fails fast at startup with a clear config error rather than sending a
request to a model that isn't there.

Runtime artifacts (`data/decisions.jsonl`, `data/runs.jsonl`,
`data/categories.json`, `data/skills/`, `data/drafts/`) are gitignored too.

> **Codegen timeout:** do not cap codegen `max_tokens`. A reasoning codegen
> model can write for 25–45 min; `codegen.timeout` defaults to 1200 s and a
> per-machine config may raise it to 3600 s.

---

## Running the agent

All subcommands go through the CLI:

```sh
python -m semif_agent.cli setup --interview   # generate config/ from answers
python -m semif_agent.cli run              # interactive REPL
python -m semif_agent.cli run --script inputs.jsonl
python -m semif_agent.cli dream            # cost report
python -m semif_agent.cli skills           # print the skill tree
python -m semif_agent.cli status           # current process + queue
python -m semif_agent.cli relabel <id> <outcome>
python -m semif_agent.cli dashboard [--port 8765] [--host 0.0.0.0] [--replay]
python -m semif_agent.cli gateway [--platform simplex,lxmf,voice|all] [--dashboard]
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
busy <text>   mark a current process (so the queue can be demonstrated)
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
python -m semif_agent.cli gateway [--platform simplex,lxmf,voice|all] [--dashboard]
```

Each platform feeds authorized DM text through the normal queue/dispatch
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
bot's contact link to reach it: `packs/simplex/scripts/simplex-address.py` shows/creates it
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

**Voice commands (Sideband push-to-talk).** Set
`gateway.lxmf.voice.enabled: true` and a Sideband PTT voice message is
transcribed into a command, just like typed text. HQ PTT (Opus/OGG) is decoded
and transcribed by faster-whisper, which needs only `requirements/transcribe.txt`
(installed by the install plan — no microphone, no voice stack). Non-HQ Codec2
PTT is decoded by the optional `pycodec2` bindings (`requirements/codec2.txt`,
installed best-effort alongside the system `libcodec2`); without it a Codec2
message is reported back rather than dropped. The audio is transient and never
leaves the machine. The model is `gateway.lxmf.voice.stt.model` (default
`base.en`).

**Voice replies.** With `gateway.lxmf.voice.reply_with_voice: true` (the
default), a chat that last messaged by voice is answered with a synthesized
voice message: the reply text is spoken with Piper and sent back as a Sideband
HQ push-to-talk message, with the text also carried as the message content so a
text-only client still sees it. A later text message from that chat reverts it to
text replies. This needs only `piper-tts` + PyAV (`requirements/tts.txt`,
installed by the install plan — still no microphone); a missing stack or a
synthesis failure falls back to a text reply rather than dropping it. The voice
is `gateway.lxmf.voice.tts.voice` (default `en_US-lessac-medium`).

**Result-only replies.** The LXMF gateway sends only the skill's result line:
the scheduler's queue/urgency bookkeeping is dropped and the `<skill>: ok —`
wrapper is stripped, so a DM is the answer, not the internals. Override with
`gateway.lxmf.result_only: false` for the full scheduler output.

> **Gateway isolation is non-negotiable.** The gateway takes commands and sends
> replies — nothing else. It never reads history, shows or creates invite links,
> or composes messages. That is messaging UX and lives in the bridges.

### Voice gateway

`gateway --platform voice` turns the machine's microphone into a command front
end: an always-on **wake word** (openWakeWord), **speech-to-text**
(faster-whisper), and **text-to-speech** (Piper), all local — no cloud. A spoken
command is fed through the same queue/dispatch pipeline as a typed or
messaged one, and every reply is spoken back. It is a normal gateway adapter, so
`needs_input` questions, repair offers, and approvals are spoken and answered by
voice too.

```sh
python -m semif_agent.cli gateway --platform voice
```

- **Requirements.** The voice stack is optional and lazy-imported; install it
  with `scripts/bootstrap.sh --voice` (or
  `.runtime/venv/bin/pip install -r requirements/voice.txt`) plus the system
  PortAudio lib and ALSA mixer tools (`apt install libportaudio2 alsa-utils`).
  The gateway refuses to start with an install hint if the stack is missing.
  PortAudio has no volume control — use `alsamixer`/`amixer` (then
  `sudo alsactl store`) to unmute and set the mic/speaker levels.
- **Models.** `gateway.voice.wake.model` is an openWakeWord model (default
  `hey_computer`, spoken phrase "Hey Computer"), `stt.model` a faster-whisper
  size (default `base.en`), and
  `tts.voice` a Piper voice (default `en_US-lessac-medium`). Bootstrap downloads
  the official openWakeWord models into the venv's package dir (under
  `.runtime/venv`) and the Piper voice into `.runtime/voice/tts`; the
  faster-whisper model lands in the `.runtime/hf` cache. openWakeWord ships no
  "computer" model, so the default is a **pinned community model**
  (openWakeWord-compatible, sha256-checked in `pins.json`) that
  `scripts/voice-models.py` fetches into `.runtime/voice/wake`, where the
  detector resolves it via `wake.model_dir`. Set `wake.model` to an official
  name (e.g. `alexa`) or a path to switch. Because the community model's scores
  run lower than the official models, the seeded `wake.threshold` is `0.2` (the
  official models score ~0.99 and use `0.5`).
- **Behavior.** Half-duplex — capture is discarded while a reply is speaking, so
  the agent never transcribes itself (no barge-in yet). The wake word is required
  before every command and is held while the scheduler is still running the
  current request, so a second command cannot start until the first reaches a
  result. A short beep plays when
  the wake word fires (`gateway.voice.cue`), so you know you're being heard; a
  short multi-note "bling" (`cue.submit_notes`, each entry
  `[frequency_hz, duration_ms]`, spaced by `cue.note_gap_ms`) plays once the
  utterance is transcribed. (Set `cue.submit_notes` to `[]` and
  `cue.submit_frequency > 0` for the older single submit beep.) After the wake
  word, capture waits `listen_timeout_s` for the command to begin, so a pause before speaking
  (e.g. while the cue beep plays) does not truncate it. `max_speak_chars`
  truncates long
  replies before speaking. `result_only` (default true for voice) speaks only the
  skill's result line — scheduler bookkeeping is dropped and the
  `<skill>: ok —` wrapper is stripped — so the spoken output is the answer, not
  the scheduler's internals; set it false to speak the full scheduler output.
  There is no allowlist: the mic is local, so physical access is the
  authorization.
- **Run mode.** Foreground, in the user's audio session — a headless systemd
  user service does not share the user's audio session, so voice is not rendered
  as a unit.

### Bridge services

`cli bridge` runs standalone bridges: each is a process that stands up a small,
token-guarded localhost HTTP API in front of one third-party system (and, where
the system needs one, owns its own daemon/profile). A bridge is a deliberate
choice for a system that needs state (especially auth state), a native or heavy
protocol, or one sanitized surface — a skill may call a plain HTTP API directly.
When a bridge fronts a service, generated skill bodies call it like any HTTP
service (base URL from a config var) and never speak the native protocol.

The first bridge is **SimpleX** (`packs/simplex/bridge/simplex.py`), with its own
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

Skills call the forwarding bridge at the address derived from `bridges.simplex`
(`simplex_bridge_url`); the bridge catalog
is injected into every codegen prompt via `describe_bridges()`, so it is the
single source of bridge specifics. Adding a bridge = a pack that ships its
`bridge/<name>.py` + a config block; the core registry discovers it by convention.

The **LLM bridge** (`semif_agent/bridges/llm.py`) is the shared, pack-less one: a
generic

```
POST /chat {"messages": [{"role","content"}, ...], "max_tokens"?: int} -> {"text": "..."}
```

in front of the agent's language model, so a body can ask for short generated
text (e.g. extract an event title + description) over HTTP instead of speaking
the OpenAI-compatible protocol itself. It does not own the model connection:
`cli bridge` threads the scheduler's already-configured `llm` client into it, so
the endpoint/model/sampler live only in the top-level `llm` block; `bridges.llm`
carries just the local listener. Skills call it at the address derived from
`bridges.llm` (`llm_bridge_url`, + optional `llm_bridge_token`); with no client
it returns `502`, never a fabricated reply. It starts in the same
`semif-bridge.service` process as the other enabled bridges.

The **Nextcloud bridge** (`packs/nextcloud/bridge/nextcloud.py`) is the third: it
wraps the account's native protocols — WebDAV files, CalDAV calendar events and
tasks, CardDAV contacts, and the OCS Notes API — behind one JSON surface, so a
body never speaks DAV/OCS directly. It has **no service daemon** (Nextcloud is
plain HTTPS) and is stdlib-only (`urllib` + `xml.etree`), so it starts instantly
in the same `semif-bridge` process. The connection (base URL, username, app
password) is configured in `bridges.nextcloud`; `NextcloudBridge.from_config`
also accepts the top-level `nextcloud_url` / `nextcloud_username` /
`nextcloud_app_password` / `nextcloud_default_calendar` as a fallback for a
hand-edited config, with a block value winning. Skills call it at the address
derived from `bridges.nextcloud` (`nextcloud_bridge_url`, + optional
`nextcloud_bridge_token`). Its HTTP surface:

```
GET  /health
GET  /user                            # account info (OCS)
GET  /capabilities                    # server capabilities (OCS)
GET  /files?path=<dir>                # list a folder (WebDAV)
GET  /files/stat?path=<path>
GET  /files/read?path=<path>          # text, or base64 for binary
GET  /files/search?query=<text>&path=<dir>
POST /files/write   {"path","content","encoding"?,"overwrite"?}
POST /files/mkdir | /files/delete | /files/move | /files/copy
GET  /calendars                       # includes "default" (configured default name)
GET  /calendars/events?calendar=<name>&start=<iso>&end=<iso>   # CalDAV VEVENT; recurrences expand with a window
POST /calendars/events | /calendars/events/update | /calendars/events/delete
GET  /tasks?calendar=<name>           # CalDAV VTODO
POST /tasks | /tasks/update | /tasks/complete | /tasks/delete
GET  /addressbooks
GET  /contacts?addressbook=<name>&query=<text>                 # CardDAV
POST /contacts | /contacts/update | /contacts/delete
GET  /notes | /notes/get?id=<id>      # Notes app (OCS)
POST /notes | /notes/update | /notes/delete
```

A missing connection is `503`; a real Nextcloud failure (transport, HTTP error,
or a bad reply) is `502` — never a fabricated success. Calendar and addressbook
defaults resolve from `bridges.nextcloud` when a request names none and the
account has several.

The **Outlook bridge** (`packs/outlook/bridge/outlook.py`) wraps the user's
Microsoft account (Outlook / Microsoft 365) through **Microsoft Graph**: mail,
calendar events, Microsoft To Do tasks, contacts, and OneDrive files. Like
Nextcloud it has **no service daemon** (Graph is plain HTTPS) and is stdlib-only,
so it starts instantly in the same `semif-bridge` process. Authorization is the
OAuth2 **device code flow**: register an Entra ID public-client app, put its
`client_id` in `bridges.outlook`, and run `packs/outlook/scripts/outlook-auth.py` once — the
tokens are stored (0600) under the checkout's `.runtime/outlook/token.json` and
refreshed automatically. The bridge starts without a token; an unauthenticated
route returns `503` telling the operator to run that helper. Skills call it at the
address derived from `bridges.outlook` (`outlook_bridge_url`, + optional
`outlook_bridge_token`). Its HTTP surface:

```
GET  /health
GET  /user                                          # the signed-in account
GET  /mail/folders
GET  /mail/messages?folder=<name|id>&count=<n>&search=<text>&unread=<bool>
GET  /mail/messages/get?id=<id>
POST /mail/send   {"to":[...],"cc"?,"subject"?,"body"?,"content_type"?}
POST /mail/draft  {"to"?,"cc"?,"subject"?,"body"?,"content_type"?}
GET  /calendars
GET  /calendars/events?calendar=<name|id>&start=<iso>&end=<iso>   # recurrences expand
POST /calendars/events | /calendars/events/update | /calendars/events/delete
GET  /tasklists                                     # Microsoft To Do
GET  /tasks?list=<name|id>
POST /tasks | /tasks/update | /tasks/complete | /tasks/delete
GET  /contacts?query=<text>&count=<n>
GET  /files?path=<dir>                              # OneDrive
GET  /files/read?path=<path>                        # text, or base64 for binary
POST /files/write   {"path","content","encoding"?,"overwrite"?}
```

A missing `client_id` or an unauthorized bridge is `503`; a real Graph failure
(transport, HTTP error, or a bad reply) is `502` — never a fabricated success.
Seven `packs/outlook/skills/outlook/*` skills ship (`next_message`, `send_message`,
`next_event`, `create_event`, `create_task`, `list_contacts`, `list_files`);
`send_message` composes the message through the LLM bridge and asks for
confirmation before it sends.

The **browser bridge** (`packs/browser/bridge/browser.py`) is real browser
automation via **Playwright**: open a URL, read a page's text or links, take a
screenshot, and click/fill/press/select to complete a flow. It is the first
bridge with a heavy optional dependency, so it is **disabled by default** and
installed with `scripts/bootstrap.sh --browser` (Playwright + Chromium land under
`.runtime/browser`; the bridge sets `PLAYWRIGHT_BROWSERS_PATH` there at launch).
If `playwright install` fails — notably on a host with an IPv6 address but no
global IPv6 route, where Playwright's downloader prefers IPv6 and hangs —
bootstrap falls back to `packs/browser/scripts/playwright-browsers.py`, which downloads the
same pinned artifacts over IPv4. To skip the managed download altogether, set
`bridges.browser.channel` (a Playwright channel such as `chrome`/`msedge`) or
`bridges.browser.executable_path` (a browser binary) to use an already-installed
browser.
Unlike the others it owns a browser and keeps a **persistent profile**
(`.runtime/browser/profile`) so a login you do once survives across calls and
restarts — the browser analog of the SimpleX bridge owning its own daemon
(`POST /reset` with `purge=true` wipes it). A dedicated event-loop thread runs
Playwright's async API and serializes whole operations, because the sync API is
bound to its creating thread and the bridge serves each request on a new one.
`check_requirements()` never blocks startup, so a host without Playwright still
runs every other bridge; a missing package/browser is a `503` install hint on the
call, and a real navigation/timeout/selector failure is a `502`. Skills call it
at the address derived from `bridges.browser` (`browser_bridge_url`, + optional
`browser_bridge_token`). Its HTTP surface:

```
GET  /health
GET  /state                                    # current url + title
GET  /content?format=text|html&max_chars=<n>    # whole page's rendered text
GET  /text?selector=<css>                       # text of matching elements
GET  /links?query=<text>&limit=<n>              # anchors (label + href)
GET  /screenshot?full_page=<bool>&selector=<css>&inline=<bool>
POST /goto    {"url","wait_until"?,"timeout"?}
POST /click   {"selector"?|"text"?,"index"?,"timeout"?}
POST /fill    {"selector","value","submit"?,"timeout"?}
POST /press   {"key","selector"?,"timeout"?}
POST /select  {"selector","value"?|"label"?,"timeout"?}
POST /wait    {"selector"?|"text"?,"timeout"?}
POST /back | /forward {"timeout"?}
POST /reset   {"purge"?}                        # close; purge deletes the profile
```

Three `packs/browser/skills/browser/*` skills ship (`open_page`, `read_page`, `screenshot`).

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

1. a **registry entry** (navigable description — `name` + `description`,
   authored by the small `llm` provider and recorded in `data/categories.json`);
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

**Packs** ship real starter integrations in the exact generated folder format,
plus a `manifest.json` and a `pack.json` manifest binding the concern to its
categories, bridge, and config schema. Each pack lives in its **own repository**
(`pack.<name>`, listed in the `semif-packs` index repo); `scripts/bootstrap.sh` clones them into
the gitignored `packs/<name>/`, and `merge_packs` loads every enabled pack into
the tree at startup. Recorded config is written to the runtime store, so a pack
repo never holds secrets. `PACKGEN.md` is the authoritative contract for
authoring a new pack (and a new bridge to back one).

- `nextcloud.create_event` — create an event on the user's Nextcloud calendar
  through the Nextcloud bridge (`GET /calendars`, `POST /calendars/events`);
  resolves the target calendar with a SemIf choice over the owned calendars
  (strong winner, else the configured default); config vars
  `nextcloud_bridge_url`, `nextcloud_bridge_token`, plus the LLM bridge
  (`llm_bridge_url`, `llm_bridge_token`).
- `nextcloud.next_event` — next upcoming event through the Nextcloud bridge
  (`GET /calendars`, `GET /calendars/events`; recurring events expanded
  server-side); config vars `nextcloud_bridge_url`, `nextcloud_bridge_token`.
- `nextcloud.create_task` — add a task to a Nextcloud task list through the
  Nextcloud bridge (`GET /tasklists`, `POST /tasks`); config vars
  `nextcloud_bridge_url`, `nextcloud_bridge_token`, `llm_bridge_url`,
  `llm_bridge_token`.
- `simplex.next_message` — read the next unread message via the forwarding
  bridge; config vars `simplex_bridge_url`, `simplex_bridge_token`,
  `simplex_default_contact`.
- `simplex.send_message` — send a message to a contact resolved over the
  bridge's contact list (a SemIf choice, defaulting to the configured contact);
  config vars `simplex_bridge_url`, `simplex_bridge_token`,
  `simplex_default_contact`, plus the LLM bridge (`llm_bridge_url`,
  `llm_bridge_token`) to write the text when the request states none.
- `simplex.connect_link` — show/create the forwarding bot's contact link.
- `calculator.calculate` — evaluate a bounded arithmetic request (two operands,
  one operator) and speak the answer in words; the operator is a SemIf choice
  over addition, subtraction, multiplication, division, exponents, squares,
  cubes, square roots, cube roots, fractions, and percentages, and SemIf picks
  the operands when the request writes more numbers than the operation needs
  (pure local compute, no config).
- `skills.list_categories` — list the categories of skills the agent has
  (pure local compute, no config; reads the read-only `ctx.catalog`, not the
  housekeeping-only `ctx.admin`).
- `skills.list_skills` — list the skills inside a category named in the request
  ("list skills in nextcloud"), asking for one when it is missing and reporting
  an unknown category honestly (pure local compute, no config).

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
dream cost and contract serialization. `tests/test_voice_gateway.py` drives the
voice loop with injected fake engines (no hardware, network, or heavy packages).
Integration tests exercise the real engine + LLM end to end.

**Fixtures are not the runtime.** Skill tests use real local endpoints (a
loopback server, never a mock); a real run is the only proof an integration
works.

---

## Roadmap / status

**Done.** Core loop, FIFO queue, skill tree, real SemIf assessment, decision
logging, `dream` cost pass, REPL + JSONL CLI; live `create_category` /
`create_skill` with the async authoring pipeline (spec → body → fidelity →
contract → test → auto-run → re-dispatch); tiered config + first-fire
collection; runtime repair loop; the closed `response` canned tree; real
integrations with `INTEGRATION` declarations; the SimpleX command gateway and the
standalone SimpleX bridge + pack skills (contact-list refresh, daemon-backed message
history / true unread); the LXMF gateway; and the voice gateway (wake word +
speech-to-text + text-to-speech).

**Next.** Real fine-tuning from grounded decision rows ("dreaming") with
validation + pinned-revision swap; queue persistence; more intake sources
(events/timers); dashboard run-requeue cross-linking; per-decision thresholds and
calibration; more bridges; a `simplex.send_message` seed. See `AGENTS.md` for the
detailed backlog and known gotchas.
