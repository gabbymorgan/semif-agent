# AGENTS.md

Guidance for working on the semif agent. Read this before touching code.

## Product premise (read first)

Semif-agent is an **end-user application**. An everyday person asks it to do a
basic task in plain language, and it does it:

1. The user asks for a task ("check my email", "message Sam on SimpleX", "put
   lunch with Dana on my Nextcloud calendar").
2. If the skill tree already knows how, the agent does it.
3. If it does not, the agent asks the user a few questions about what they want
   and how it should connect, generates a new skill, and then does it.
4. It performs the task for real, against the user's actual service. If the
   real attempt fails, it says so and offers to fix/learn. It never pretends a
   task succeeded.

Skills are therefore **real integrations**, not simulations: mail protocols,
SimpleX, CalDAV/WebDAV, configured local CLIs, APIs — using the user's own
accounts. A skill that returns a canned result, fabricates working values, or
silently downgrades to "draft you can review" is broken, regardless of what its
test says. Connection details, accounts, and credentials are asked once and
recorded in tiered config; bodies read them from `ctx.config`.

Two boundaries must never blur:

- **SemIf routes, gates, scores, assesses, and automates; an LLM only
  generates.** Skills are not chat responses. Assessment and fidelity are SemIf
  decisions, never model prose.
- **"No mocking" means the decision engine is always real.** Skill tests are
  hermetic mechanics checks with fixtures; passing one is not proof the real
  integration works. Only a real run against the user's service is.

When requirements or connection details are unclear, the agent asks the user.
Guessing, inventing data, or shipping a toy is always the wrong answer.

## What this is

A local desktop CLI agent whose entire control flow is a single decision model
(SemIf). Inputs are scored for urgency, queued, and dispatched through a
skill tree. Nothing is gated out up front: every input is dispatched and inputs
that are not requests fall through navigation into the closed `response` tree of
canned replies (a request no skill covers authors a skill instead — the
actionability guard). Every SemIf decision is logged as a labeled training row; the
`dream` pass computes the prediction-vs-observation cost (cross-entropy / NLL +
ECE) that later drives fine-tuning. See "Product premise" above for what the
agent is for; this section is the machinery.

The design spec is `IDEA.md`. The key point: **semantic ifs, not text
generation, do the routing.** SemIf returns probabilities conditional on the
supplied options; an LLM is used only for generation.

**Gateway isolation is non-negotiable.** The command gateway (`gateway/`) exists
only to take commands and send replies. Everything that is messaging *UX* —
invite links, reading a contact's messages, composing/sending on the user's
behalf — belongs to the standalone bridge services (`semif_agent.bridges`),
which run as their own processes against their own service daemons/profiles.
Never widen the gateway's surface for convenience: a new read/send/address
capability is a new bridge, not a gateway feature.

## Architecture map

```
cli.py          argparse: run (REPL / --script), dream, skills, status, relabel,
                dashboard, gateway (command intake), bridge (standalone
                third-party API bridges)
scheduler.py    no up-front gate: every input is dispatched
                -> choice(tau) -> score -> queue; preempt + requeue;
                a skill run paused for input (needs_input) keeps `current`
                busy; `answer` routes straight to the pending run, bypassing
                gate/score/navigation; draft authoring (title+description by the
                small `llm` provider) AND skill-body authoring (codegen) are ASYNC
                single-slot workers: a create_category/create_skill request is
                queued, the gate stays free, the category job chains into its
                skill job, and the original request is re-queued and re-runs the
                new leaf when the body lands; an optional human approval gate
                (`creation_approval`, off by default) blocks the draft worker
                after the proposal is authored and before anything is
                registered — one prompt per request, denial/timeout aborts
                creation, surfaced via the approval queue; the body worker runs the full
                pipeline: elicitation ->
                codegen body (declares its own flat `CONTRACT`) -> fidelity
                gate -> test -> auto-run test (SemIf regen ladder on failure,
                code/test); config
                search auto-populates the skill config from the full config
                cascade; elicitation asks implementation questions one at a
                time (all front ends defer via the question queue, `answer_timeout`
                per question); a
                failed real run triggers a logged SemIf repair choice
                (retry/repair_skill/ask_user/no_repair) surfaced to the user
queue.py        urgency max-heap (desc weight, FIFO seq), age pulls toward 1.0
skills.py    tree + registry (hardcoded built-ins: the closed `response` canned
                tree only), navigation = SemIf choices per level (logged;
                the actionability guard runs first, at the top of navigation,
                to arbitrate the response/create boundary), create_category
                and create_skill author + register stubs via the small `llm`
                provider (separate from codegen); SkillStore persists one folder per skill
                (skill.py, skill.test.py, contract.json, config.json) and
                materialize_skill / merge_skill_store
                hot-load runnable skills from data/skills/;
                merge_seed_store loads committed starter skills from seeds/ with
                their recorded config read from data/skills/;
                ActionResult.needs_input pauses a run for human input;
                Skill.integration / integration_source read the body's
                INTEGRATION declaration (or infer it); Skill.contract is read
                from the body's `CONTRACT` constant (contract.json is a derived
                mirror);
                resolve_skill_config / unresolved_variables drive the tiered
                config merge (global -> category -> skill) + pre-act
                contract collection
skill.py        loop: observe -> act -> observe -> assess; the body is a single
                act(ctx, request) phase that returns ActionResult (its own
                SemIf sub-decisions ride on ActionResult.decisions and are
                logged with the run outcome, phase `act`);
                assess is a SemIf decision (`assess:outcome` success/failure at
                tau; on failure `assess:requeue` complete/retry) and the run
                summary is deterministic (no generation), built from
                category.skill + ok/failed + action_log; the assess state carries
                the resolved inputs (secrets redacted);
                a run paused for input is resumed by re-invoking act with the
                answer on request.user_input; a
                contract variable the runner cannot satisfy pauses BEFORE
                act (pre_act), collects it, and re-runs the full path
timers.py       in-process timer/alarm service (TimerService) owned by the
                scheduler and exposed on ActionContext.timers: a single
                background thread fires scheduled timers on the host clock,
                records a `timer_fired` trace event, and queues the
                notification for the front end that set it (the REPL prints it,
                the gateway routes it to the originating chat, the dashboard
                shows pending/fired). Real local compute (`compute` transport),
                per-process, and deliberately NOT persisted across restarts.
engine.py       SemIfEngine -> semif_phase1.llamacpp_backend (lazy import)
codegen.py      CodegenClient (OpenAI-compatible) writes real-integration skill
                bodies against CODEGEN.md (real actions via stdlib transports,
                data from the runner via ctx.config, never embedded; bodies
                declare INTEGRATION service/transport/config_vars, and declare
                their own flat `CONTRACT` {var: description});
                parse/validate (compile + act + CONTRACT keys all read from
                ctx.config + every referenced name imported/bound, via symtable)
                + parse_contract / parse_integration /
                infer_integration / integration_findings; the bridge catalog
                (describe_bridges()) is injected into the body, retry, regen,
                elicitation, and testgen prompts — it is the single source of
                bridge specifics (service, base-URL var, auth header/token var,
                config-var docs, endpoints with error shapes), while CODEGEN.md/
                TESTGEN.md carry only the generic pattern; the advisory
                elicitation integration hint rides on every body/retry/regen
                prompt; elicitation questions
                + integration hint; TESTGEN.md drives the shared-context test
                call producing skill.test.py (hermetic mechanics test: inline
                fixtures, loopback http.server for HTTP bodies, no external
                network, no mock_data.json); run_skill_test executes
                the test as a subprocess
llm.py          small OpenAI-compatible provider (LLMClient) that authors a
                new category/skill title + description — its own endpoint/model,
                deliberate separate from codegen; provider logic (SSE/budget/idle)
                is shared from provider.py; `_parse_json` is also borrowed by the
                authoring parsers
provider.py    shared OpenAI-compatible chat transport (OpenAICompatClient:
                SSE stream, token budget, idle watchdog, degeneration hook,
                optional `api_key` bearer auth + `extra_headers`/`user_agent`,
                and `query_context` to skip the ollama /api/show probe)
                subclassed by llm.LLMClient and codegen.CodegenClient
console.py      OpenCode Console provider: OpenCodeConsoleClient + the
                ConsoleLLMClient / ConsoleCodegenClient endpoint subclasses.
                Hosted, authenticated OpenAI-compatible Chat Completions API
                (https://opencode.ai/inference/openai/v1); bearer key, skips the
                ollama probe, always sends a User-Agent (the gateway 403s
                urllib's default Python-urllib/<ver>), and forces
                disable_thinking off (the gateway 400s reasoning_effort).
                Selected via the llm/codegen `provider` key
                ("opencode" vs "ollama").
log.py          decisions.jsonl rows {state, question, options, predicted_probs,
                selected, observed_outcome, label_source}
trace.py        runs.jsonl lifecycle events keyed by run_id (submit/queued/
                preempted/assessed/...); decisions reference run_id in extra
dream.py        NLL of observed outcome per row; weighted CE, accuracy, ECE
decisions.py    contract dataclasses (Option, DecisionRequest, DecisionResult,
                Request)
dashboard.py    stdlib http.server + JSON API (tree/trace/dream/status/
                questions/repairs + POST submit/answer/questions/repair/restart/
                relabel); static/ frontend served at /
gateway/        messenger COMMAND intake/reply — and nothing else. base.py:
                GatewayAdapter contract + Inbound/OutboundMessage. service.py:
                GatewayService maps chats onto the single-slot scheduler
                (submit_request, owner maps, pending-run ownership, queue
                drain, authoring questions/repairs routed back to the origin
                chat); it is platform-aware and fronts one or more adapters at
                once, keyed by platform name (source `<platform>:<chat>`, the
                single pending-run guard shared across platforms). simplex.py:
                SimplexAdapter — command simplex-chat daemon over its JSON
                WebSocket API (lazy `websockets`, allowlist, batching,
                structured `/_send`). lxmf.py: LxmfAdapter — LXMF/Reticulum
                (lazy `lxmf`), in-process, allowlist + batching. voice.py:
                VoiceAdapter — the local microphone front end (wake word +
                speech-to-text + text-to-speech) over the neutral
                `semif_agent/voice_transport.py`; no remote peer and no
                allowlist (physical mic access is the authorization), a single
                configured `chat_id`, replies spoken back. Run with
                `python -m semif_agent.cli gateway [--platform
                simplex,lxmf,voice|all] [--dashboard]`; config under
                `gateway.simplex` / `gateway.lxmf` / `gateway.voice`. The
                gateway MUST NOT read history, show/create invite links, or
                compose messages — see "gateway isolation" below.
bridges/        standalone third-party API bridges (SimpleX first, plus the
                LLM bridge). base.py:
                BridgeInfo + BridgeService (shared localhost JSON HTTP layer,
                `X-Semif-Token` guard). inbox.py: MessagingInbox (bounded FIFO +
                contacts learned from inbound DMs and the daemon contact list,
                bridge-owned read cursor). simplex.py: SimplexBridge —
                owns its OWN simplex-chat daemon/profile (separate from the
                gateway's), buffers every inbound DM, refreshes the daemon
                contact list on (re)connect + on demand, serves invite-link/
                read/send plus daemon-backed unread + per-chat history.
                llm.py: LLMBridge — a generic `POST /chat` in front of the
                scheduler's configured `llm` client (reused, not re-configured),
                so bodies get short generated text (e.g. an event title +
                description) over HTTP.
                registry.py: CATALOG, describe_bridges() (catalog injected
                into the codegen prompts), run_bridges() (threads the scheduler's
                `llm` client into the LLM bridge). Run with `python -m
                semif_agent.cli bridge [--name NAME]`; config under `bridges`.
simplex_ws.py   neutral SimpleX daemon protocol shared by the command gateway
                adapter and the bridge (parse direct text across v7 shapes,
                structured `/_send`, corrId→Future round-trips, contact-address
                `/_show_address`/`/_address`, contact-request accept, contact
                list + chat previews/history). Knows nothing about the
                scheduler or either front end.
lxmf_transport.py
                neutral LXMF (Reticulum) transport: one in-process Reticulum
                instance + LXMF router, persisted delivery identity, announce
                loop, inbound normalization, outbound send queue. Lazy `RNS`/
                `LXMF` imports keep the module stdlib-only. Knows nothing about
                the scheduler or the gateway.
voice_transport.py
                neutral voice transport: a 16 kHz mono capture loop with an
                always-on wake word (openWakeWord), VAD endpointing
                (webrtcvad, energy fallback), speech-to-text (faster-whisper)
                and text-to-speech (Piper) over `sounddevice`; half-duplex
                (capture discarded while speaking) with a follow-up window so a
                reply can be answered without the wake word. Every heavy dep is
                lazy-imported, keeping the module stdlib-only; the engine
                interfaces are plain classes so tests inject fakes. Knows
                nothing about the scheduler or the gateway.
```

## Run / verify

Pure stdlib unit tests run anywhere:
`python3 -m pytest tests/ -q --ignore=tests/integration`. Integration tests need
a real SemIf engine + a real LLM endpoint; run them on a provisioned host (see
"### Provisioning (`scripts/bootstrap.sh`)").

## Local deployment notes

This repo is generic: it names no particular host, address, path, or model.
**Every deployment-specific value lives in the gitignored `local/` folder** —
see `local/ENVIRONMENT.md` for this deployment's hostnames, addresses, SSH keys,
paths, and model choices. If that file is absent, this is a fresh clone and the
values are yours to choose.

## Git / sync

- Push/pull from your git remote — never rsync/tar the code.
- **`config.json` is gitignored, per-machine, and required** (hosts use
  different LLM/codegen endpoints and models). `scripts/bootstrap.sh` seeds it
  from `config.example.json` once (only when absent) and **nothing writes it
  afterwards — only a human edits `config.json`**. `data/decisions.jsonl`,
  `data/runs.jsonl`, and `data/drafts/` are runtime artifacts and gitignored too.
- **`pins.json` is committed and code-owned** — the single home for every
  external ref (SemIf commit, GGUF url+sha256, HF tokenizer revision,
  simplex-chat version/url+sha256). These are bumped by a git change and picked
  up on the next bootstrap run; they are deliberately NOT in `config.json` (a
  ref replaced on every update is not per-machine user config).
- Decision rows logged before the `run_id` threading landed show up under
  run_id `"?"` in the dashboard — that's expected, not a bug.

## Roadmap

### v1 (done)
Core loop, urgency queue, skill tree, skill loop with real SemIf assessment
(decision, not generation), decision logging, `dream` cost pass, REPL + JSONL
CLI, unit tests (24) + integration tests (2).

### v2
- Real fine-tuning from `decisions.jsonl` at a regular interval ("dreaming"):
  accumulate labeled rows, compute cost, fine-tune the decision model, CI/CD
  validate (accuracy/ECE on a held-out slice, prompt-hash regression), swap the
  pinned model revision. GPU offload: train on a beefier GPU; the running agent
  keeps a frozen inference revision until a swap validates.
- `create_skill` branch: live, mirroring `create_category`. The small `llm`
  provider (a dedicated endpoint/model, separate from `codegen`; production may
  point both at the same host) proposes a specific skill title + description for
  the chosen category; the stub is
  persisted to `data/categories.json` (under that category's `skills` list) and
  merged into the running tree as a leaf. Since Sep 2026 the leaf also gets a
  real runnable body via the async authoring pipeline: a larger
  OpenAI-compatible model (`codegen`, your configured code-capable model) writes an `act`-only
  body plus its own flat `CONTRACT` against `CODEGEN.md`, then a test
  (against `TESTGEN.md`) is generated and auto-run before the leaf
  is declared ready — all persisted to a folder in `data/skills/` and
  hot-loaded. Authoring is **asynchronous**: the draft is authored by a
  single-slot `llm` worker, the stub is created and the gate
  freed immediately, the body write runs on a separate single-slot codegen
  worker, and
  once the body lands the request that prompted creation is re-queued and re-runs
  navigation onto the new leaf (an empty/in-progress leaf can be restarted via
  `restart <category> <skill>` or the dashboard). A request that prompted a whole
  new category runs the same chain deterministically: `create_category` →
  `create_skill` → async body → re-dispatch. An unreachable `llm` endpoint is
  graceful (traced `draft_failed`, user notified, request not re-dispatched);
  only SemIf availability is fatal. Authoring is still a single pass —
  validating/reusing written bodies across runs is future work.
  **Navigation is two-stage at both levels; the guards are the create doors**
  (Oct 2026). Category level: the softmax offers every existing category **with
  its description** plus `create_category`, and the winner is confirmed by
  `confirm_category_fit` (phase `navigate:category_scope`, P(covers) >=
  `navigation.category_tau`); a rejected winner or a `create_category` win
  authors a new category. Descriptions are load-bearing — with bare names the
  model sent "book a flight to japan" to `simplex` (0.49 vs create_category
  0.12); with descriptions it picks `create_category` 0.87. `Skill` carries
  `category_description`, seeded by `CATEGORY_DESCRIPTIONS` for the built-ins
  (response/calendar/simplex) and read from the registry for authored
  categories; `category_descriptions(tree)` collapses it per category so an
  empty bucket is never offered bare. The old category threshold gate
  (`create_tau`/`create_margin`) is **retired**. Leaf level: the softmax
  contains **only existing skills** — `create_skill` was removed because sharing
  a softmax with the real skills diluted its probability and let a crowded tree
  drift into create on a weak plurality (real runs scored `create_skill`
  0.585–0.803 on requests a seed skill clearly matched). The reuse-vs-create
  decision is the **intent guard** (`confirm_skill_fit`, phase
  `navigate:intent`) at `navigation.intent_tau`. The **actionability guard**
  (`confirm_non_action`, phase `navigate:actionability`) at
  `navigation.action_tau` (default 0.5) runs at the **top** of navigation,
  before category selection: it decides whether the input is an actionable
  request at all, so the closed `response` tree is reached **only** for
  non-action input (chit-chat, a statement, or a fragment too vague/incomplete
  to act on, e.g. "what") and a request can never be canned. Non-action goes
  straight to the `response` tree and never reaches the category softmax — a
  vague fragment lands on the catchall `response.clarify`; a request proceeds to
  the category softmax, which now offers only the **real** categories (the
  canned `response` tree is never an option) plus `create_category`. Deciding
  this first — rather than only when the softmax happens to propose
  `response`/`create_category` — is what keeps a non-request out of a real
  category: a single category description cannot draw that line ("1+1=2" a
  statement and "what is 255 * 12?" a request both read as math to the
  softmax), and the scope guard is deliberately permissive. A request no skill
  covers (e.g. "what is 255 * 12?") authors instead of getting a canned reply,
  while a stray statement (e.g. "1+1=2") or a vague fragment (e.g. "what") still
  gets one. All three taus are
  separate from the top-level `tau`, so tuning reuse-vs-create or actionability
  does not move assessment/fidelity. The guards are deliberately permissive
  (in-scope scores 0.9–1.0): a borderline over-cover routes into a plausible
  category and the leaf guard catches it. Empty and single-skill categories
  skip their softmax.
- **No handle/ignore gate; the `response` tree is the non-request catchall**
  (Sep 2026; actionability guard Oct 2026): the top-level `_contains_request`
  handle/ignore gate is gone. Every input is scored and dispatched; input that
  is not a request falls through navigation into the closed `response` category,
  a hardcoded tree of canned replies (`response.greeting`, `response.thanks`,
  `response.acknowledge`, `response.farewell`, `response.affirm`, and the
  catchall `response.clarify` — "Could you try being more specific?"). A
  **request** is never answered from here, even one no skill covers: the
  actionability guard (`confirm_non_action`, phase `navigate:actionability`)
  runs at the top of navigation and sends a request to authoring (this is what
  keeps "what is 255 * 12?" out of the tree; the old
  `response.unable` canned refusal is gone — an unsupported request authors a
  skill, it is not refused). A fragment too vague/incomplete to act on (e.g.
  "what") is non-action, so it lands on the catchall `response.clarify` rather
  than authoring a redundant clarify skill. `response` is a `CANNED_CATEGORIES` member:
  `navigate` never offers `create_skill` there (the intent guard is skipped
  too), `Scheduler._dispatch_skill`/`restart_skill` refuse to author it, and the
  runner skips `assess:outcome`/repair for it — a canned line has no side effect
  to assess. The leaf choice logs phase `navigate:response` and traces
  `response_selected`. `response.reject` (a stub that never ran) is gone.
- **Optional human approval on creation** (Oct 2026): `creation_approval`
  (top-level, default `false`) is a single boolean. When on, the single-slot
  `llm` draft worker blocks after it authors a proposal and before anything is
  registered; the human approves/denies via the REPL prompt, the dashboard
  approvals panel (`GET/POST /api/approvals`), or a gateway chat reply
  (`yes`/`no`). One prompt per creation request — a new category's chained skill
  is covered by the category approval (`DraftAuthor.approved`). Denial or
  timeout aborts creation: nothing is registered, no body is written, and the
  request is not re-dispatched; the wait reuses
  `codegen.elicitation.wait_timeout`. With no deferring front end it denies
  immediately rather than stalling the worker. Traced as
  `creation_approval_requested` / `_approved` / `_denied` / `_timeout` /
  `_skipped`; it is a human veto, not a decision row — the SemIf create doors
  still make and log the routing decision.
- **Real integrations, implementation questions, fidelity + repair** (Sep 2026):
  elicitation is on by default and asks implementation questions (which
  service/account, how to connect, where the credential comes from, what success
  looks like); every front end defers via the question queue
  (`/api/questions`, worker waits `codegen.elicitation.answer_timeout` per
  question, one at a time). Bodies
  must perform the real action via stdlib transports with values from
  `ctx.config` and declare `INTEGRATION` (service/transport/config_vars); a
  fidelity gate (`authoring:fidelity`, a SemIf accept/reconsider decision) plus
  static declaration checks regenerates a simulated body once with the raw
  evidence bundle, then accepts it badged `unverified`. A failed real run logs a
  SemIf repair choice (retry / repair_skill / ask_user / no_repair) surfaced in
  REPL/dashboard; repair writes carry the raw observed-failure bundle and are
  bounded by `codegen.repair.max_attempts`. Tests are hermetic mechanics checks
  (loopback `http.server` for HTTP bodies) and never certify the live
  integration — only a real run does.
- Queue persistence (durable across restarts).
- Event/timer intake sources beyond typed input.
- Concurrency: SemIf shared-state mode (`score_shared` / `SerialPrefixScorer`)
  for parallel decisions; single execution slot remains for processes.
- Dashboard: run-requeue cross-linking (child run references its parent),
  scheduler sim controls (busy/idle/tau) as a first-class panel.
- **Data-contract tiered data lookup on skill creation** (BRAINSTORM item 1):
  the body declares its own flat `CONTRACT` (variable name -> semantic
  description, for user input + SemIf only); `contract.json` is a persisted
  mirror. The config step searches the full cascade — global config → category
  config → skill config — via SemIf `choice` per variable, reusing an existing
  source when one matches (`config:search`, decision-logged) and auto-populating
  the skill `config.json`. Variables with no match are asked of the human AFTER
  code generation, on first fire, via a pre-act `needs_input` pause; each
  answer gets a SemIf `record-as-config vs ask-again-each-fire` choice
  (`config:record`). On successive firings only unresolved variables are asked.
  **Shipped with item 1 (Sep 2026); contract now lives in the body (Sep 2026).**
- **Post-codegen mock-data test** (BRAINSTORM item 1): after the body +
  contract land, a shared-context testgen call produces `skill.test.py` with
  fixture data embedded inline (no separate mock_data.json), and
  `run_skill_test` executes the test as a subprocess in
  the skill folder before the leaf is declared runnable. On failure a 3-option
  SemIf decision (`codegen_regen`, trace-only) picks code/contract/test to
  regenerate, feeding the error + existing files back, bounded by
  `codegen.test_max_attempts` — a generated body that can't execute its own data
  path fails authoring, not the first real request. **Shipped with item 1
  (Sep 2026).**
- **Skill-code inspection**: new skills are labeled as new, and the CLI and
  dashboard gain the ability to inspect a skill's generated code body
  (read-only view of `data/skills/<category>/<name>/` and its trace) so the
  author can audit what was generated.
- **Messenger gateway (SimpleX first, then LXMF)** (Sep 2026; LXMF Oct 2026):
  `python -m semif_agent.cli gateway` runs a dedicated process that connects to
  one or more transports and feeds authorized DM text through the normal
  gate/score/queue/dispatch pipeline; results, authoring questions, and repair
  offers are sent back to the originating chat. `gateway/base.py` is the
  transport contract (`GatewayAdapter`, `InboundMessage`, `OutboundMessage`),
  `gateway/service.py` the platform-aware scheduler glue (single execution slot,
  `(platform, chat)` owner maps, pending-run ownership shared across platforms,
  queue drain), `gateway/simplex.py` the SimpleX adapter (lazy `websockets`,
  default-deny allowlist by contactId/display name, rapid-message batching,
  structured `/_send`), and `gateway/lxmf.py` the LXMF/Reticulum adapter (lazy
  `lxmf`, in-process via the neutral `semif_agent/lxmf_transport.py`, allowlist
  by LXMF address, `threading.Timer` batching). All adapters share one process
  and one scheduler (`--platform simplex,lxmf|all`). DMs only for the first cut;
  groups/attachments/reactions are future work. See "### gateway (messenger
  intake)".
- **Voice gateway (wake word + STT + TTS)** (Oct 2026): `gateway/voice.py`
  (`VoiceAdapter`, platform `voice`) is the local microphone front end over the
  neutral `semif_agent/voice_transport.py`: an always-on openWakeWord wake word,
  `webrtcvad` endpointing, faster-whisper speech-to-text, and Piper
  text-to-speech over `sounddevice`. It is a normal `GatewayAdapter`, so the
  single-session routing, `needs_input` questions, repairs, and approvals all
  work through `GatewayService` unchanged — a spoken command is just text from
  `voice:<chat_id>`, and every reply is spoken. No remote peer and no allowlist
  (physical mic access is the authorization). Half-duplex: capture is discarded
  while a reply is speaking (no barge-in yet); after each reply a
  `follow_up_window_s` accepts the next utterance without the wake word. The
  whole voice stack is optional and lazy-imported (`requirements/voice.txt`,
  pinned separately from the engine set since it is hardware-dependent); the
  gateway refuses to start with an install hint. `bootstrap.sh --voice`
  installs the deps and downloads its models (the openWakeWord models land in
  the venv under `.runtime/venv`, the Piper voice under `.runtime/voice/tts`,
  the faster-whisper model in the `.runtime/hf` cache). See
  "### gateway (messenger intake)".
- **Bridge services (SimpleX first)** (Sep 2026): messaging *UX* — invite links,
  reading incoming messages, composing sends — is decoupled from the command
  gateway. `semif_agent.bridges` is a folder of standalone third-party API
  bridges, each its own process (`cli bridge`) against its own service daemon.
  `SimplexBridge` owns a second simplex-chat profile and serves a token-guarded
  localhost HTTP API; `registry.describe_bridges()` injects the catalog into the
  codegen prompts. Gateway isolation is non-negotiable: the gateway must never
  grow read/send/address surface. See "### bridges (third-party API services)".

### Later / open questions
- Safety/authority: which inputs may interrupt high-stakes processes; is
  interrupt a per-skill permission?
- Calibration: SemIf ships per-workload temperature scaling; adopt it before
  treating probabilities as confidence.
- Enumerate the intake source taxonomy and per-source gating.

## Constraints & gotchas (learned the hard way)

### Environment
- The core is pure stdlib and runs anywhere Python does. Never pip-install heavy
  deps on a thin client; a full install belongs on a provisioned host.
- The SemIf engine runs via llama.cpp **CPU** backend (its llamacpp backend
  forces `n_gpu_layers=0`), so the decision engine does not use a GPU; ollama
  does.
- The SemIf tokenizer is fetched from HF (`Qwen/Qwen3.5-4B` at the pinned
  revision), cached under each checkout's `.runtime/hf` (`HF_HOME`).
- Per-host specs and model choices for this deployment: `local/ENVIRONMENT.md`.

### SemIf install
- New machines use `scripts/bootstrap.sh`, which installs the venv/engine/GGUF/
  cache under the checkout's `.runtime/` — see "### Provisioning
  (`scripts/bootstrap.sh`)" and "Code principles". (A legacy layout that
  installed the engine outside the checkout predates the `.runtime/`
  containment rule; prefer bootstrap.)
- SemIf hard-pins `torch==2.10.0`, `numpy==2.2.6`, etc. The llamacpp path does
  **not** need torch (torch is imported lazily inside `direct.score`). Install
  with `--no-deps` and bring only what's needed — the committed manifest
  `requirements/staging.txt` is the single source of truth:
  `pip install -e <semif-clone> --no-deps`, then
  `CMAKE_BUILD_PARALLEL_LEVEL=6 MAKEFLAGS=-j6 pip install -r requirements/staging.txt`
  (numpy 2.3.5, transformers 5.17.0, tokenizers 0.23.2, huggingface-hub 1.31.0,
  llama-cpp-python 0.3.35, pytest).
- **Expected pip warnings:** pip reports "dependency conflicts" against
  semif-phase1's declared requirements (torch/accelerate/protobuf/sentencepiece
  not installed, numpy 2.2.6 vs 2.3.5). These are informational — the staging
  host runs exactly this set — not a bug; do not "fix" them by installing torch.
- `numpy==2.2.6` has **no cp314 wheel** → pip tries a source build that fails
  without `pkg-config` + `python3-dev`. Use numpy 2.3.5 (has cp314 wheels).
- `llama-cpp-python==0.3.35` builds from source. With all 32 cores it OOM-kills
  gcc (`internal compiler error: Segmentation fault`). Limit parallelism:
  `CMAKE_BUILD_PARALLEL_LEVEL=6 MAKEFLAGS=-j6 pip install llama-cpp-python==0.3.35`.
  Do NOT bump the llama-cpp-python version — SemIf calls specific llama.cpp C
  APIs that change between versions.
- The pinned GGUF: `Qwen3.5-4B-Q4_K_M.gguf` from bartowski (2.8G), downloaded
  into `.runtime/models/` by bootstrap. Load ~34s; score ~0.9s/decision on CPU
  at 8 threads.

### ollama
- `bootstrap.sh` installs **no ollama**: it expects the target host to already
  run one for `llm` (and pulls `llm.model` into it), while `codegen` may be
  remote. Point `--llm-url`/`--codegen-url` at the right daemons.
- If generation hangs with no log output, restart the ollama service — the GPU
  runner can wedge.
- **A local reasoning model needs thinking off for `llm`.** A reasoning model
  emits hidden chain-of-thought and puts the answer in `content`. Through the
  `/v1` compat endpoint the CoT consumes the entire reply budget, so
  `generate_category`/`generate_skill`'s `max_tokens=128` returns
  `finish_reason: length` with `content: ""` → `JSONDecodeError` and a traced
  `draft_failed` (no category/skill is ever authored — the whole create branch
  is silently dead). `LLMClient` sends `reasoning_effort: "none"`
  (`llm.disable_thinking`, **default on**) so the model answers the ~30-token
  JSON directly. Ollama's native `think:false` is **not** plumbed
  through `/v1` (silently ignored), and `/no_think` in the prompt is ignored
  too; only `reasoning_effort` works there. Codegen *wants* the reasoning, so
  the knob is `llm`-only — `CodegenClient` never sets it. Unbounded (no
  `max_tokens`) does not help: a reasoning model can loop on this prompt
  (thousands of reasoning tokens, still empty `content`).

### Provisioning (`scripts/bootstrap.sh`)

- Provision a fresh Ubuntu machine into a running host: clone `semif-agent` to
  a checkout (register its SSH key on your git host first), then run
  `scripts/bootstrap.sh`. With no `config.json` yet it seeds one from
  `config.example.json`; edit that to set `llm.model`/`codegen.model` and the
  endpoints. Flags (`--llm-url`, `--codegen-url`, `--llm-model`,
  `--codegen-model`) override the values read from `config.json` **for that run
  only** and are never written back. `--voice` additionally installs the
  optional voice-gateway stack (`requirements/voice.txt` + system
  `libportaudio2` + `alsa-utils`) and downloads its models into `.runtime/voice/` via
  `scripts/voice-models.py`. The script must be run from a checkout —
  it reads pins from that checkout's `pins.json` and never re-clones the agent
  repo (only the SemIf engine and the simplex-chat binary). Idempotent and
  rerunnable; every stage no-ops on existing state, so it also boots an
  unknown-state machine. It installs **no ollama**: `llm` is the local ollama
  (bootstrap pulls `llm.model` if configured), `codegen` may point at a peer
  host. It WARNs (never auto-pulls) if the remote codegen host is missing its
  model, and WARNs if `llm.model`/`codegen.model` are unset.
- **All installation artifacts live inside the checkout under a gitignored
  `.runtime/`** (`venv/`, `engine/`, `models/`, `hf/`, `bin/simplex-chat`,
  `simplex/`, `systemd/`, and `voice/` for the optional voice models), so an
  end user can find and debug the whole stack in
  one tree. Only operationally-forced artifacts live outside: the SSH key
  (`~/.ssh`) and the real systemd user dir + linger (the rendered units are
  stored in `.runtime/systemd/` and symlinked into `~/.config/systemd/user/`).
  See the "Code principles" containment rule.
- All pins are read from the committed `pins.json`: the `engine` block
  (`semif_repo` public GitHub `TheoLeeCJ/SemIf`, `semif_ref` pinned commit,
  `gguf_url`/`gguf_sha256`, HF tokenizer `source`/`revision`) and the
  `simplex_chat` block (`version`, `bin_url`, `sha256`). Per-machine
  `simplex_chat` ports/display names stay in `config.json`. The python dep pins
  live in `requirements/staging.txt` (committed, one versioned artifact — every
  host provisions from it). The agent reads `engine.source`/`revision` from
  `pins.json` and derives the GGUF path from `engine.gguf_url` under
  `.runtime/models/` (an explicit `engine.gguf` in `config.json` overrides it).
  **Maintenance**: bump the pins in `pins.json` / `requirements/staging.txt`,
  rerun the script, re-run the integration tests. The script never guesses.
- pip prints **expected** "dependency conflict" warnings at install time
  (semif-phase1 declares torch/accelerate/protobuf/sentencepiece/numpy 2.2.6
  that we intentionally do not install — the llama.cpp CPU path doesn't need
  them; numpy 2.3.5 is deliberate, 2.2.6 has no cp314 wheel). Do not "fix" them
  by installing torch.
- Seeds `config.json` from `config.example.json` (only when absent) and **never
  writes it again** — only a human edits `config.json`. `config.example.json`
  ships the provisioned defaults, so a seeded `config.json` already has
  `codegen.timeout: 3600`, the SimpleX gateway enabled
  (`gateway.simplex.enabled = true`, `ws_url` from `simplex_chat.port`) and the
  standalone forwarding bridge enabled (`bridges.simplex.enabled = true`,
  `ws_url` from `simplex_chat.forward_port`, top-level `simplex_bridge_url`) and
  the LLM bridge enabled (`bridges.llm.enabled = true`, top-level
  `llm_bridge_url`; it reuses the `llm` model);
  the operator sets `llm.model`/`llm.base_url` and `codegen.model`/
  `codegen.base_url`. Bootstrap renders/enables four user services:
  `semif-simplex.service` (the command `simplex-chat` bot daemon,
  `--create-bot-display-name` on `simplex_chat.port`), `semif-gateway.service`
  (`.runtime/venv/bin/python -m semif_agent.cli gateway`),
  `semif-simplex-forward.service` (the bridge's **own** daemon/profile on
  `simplex_chat.forward_port`) and `semif-bridge.service`
  (`.runtime/venv/bin/python -m semif_agent.cli bridge`). The gateway allowlist
  (`gateway.simplex.allowed_users`) and fallback (`home_channel`) are
  `config.json` fields the operator edits; with an empty allowlist the gateway
  denies everyone (the safe default until the human adds their contact id).
  `--copy-data SRC` rsyncs a prior host's `data/` for continuity.
  Bootstrap then runs `scripts/simplex-address.py` to create/print the command
  bot's contact address (and the forwarding bot's, via `--ws-url`).
- First run downloads the 2.8G GGUF and builds `llama-cpp-python` from source
  (~10 min on 6 cores); reruns are fast no-ops.
- Per-host run/verify commands and addresses: `local/ENVIRONMENT.md`.

### codegen (skill bodies)
- Skill **bodies** are written by a separate OpenAI-compatible model, configured
  under `codegen` in config.json (your configured code-capable model —
  larger/slower than `llm`). Title + description for new skills come from the
  separate small `llm` provider (its own endpoint/model, shared transport in
  `provider.py`); only the runnable code body uses codegen.
- **Alternate hosted provider: OpenCode Console** (`console.py`). Either
  endpoint can run against the hosted, authenticated OpenAI-compatible Console
  inference API instead of local ollama: set `llm.provider` / `codegen.provider`
  to `"opencode"` (default `"ollama"`), set `model` to a **Chat Completions**
  family id (`kimi-k2.7-code`, `glm-5.3`, `deepseek-v4.1-flash`, `qwen3.8-max`,
  the free models, …), and give `api_key` (literal) or `api_key_env` (env var
  name — preferred). `build_scheduler` maps `"opencode"` → `ConsoleLLMClient` /
  `ConsoleCodegenClient`, which keep the existing error contracts
  (`LLMError` / `CodegenError` / `DegenerationError`) and the same opt-in
  sampler behavior (none sent unless configured). The `OpenAICompatClient`
  transport already speaks the Console's SSE
  shape (`delta.content` + `delta.reasoning_content` + `include_usage` +
  `[DONE]`); the Console subclass adds bearer auth, defaults the base URL to
  `https://opencode.ai/inference/openai/v1`, and sets `query_context=False` so no
  ollama `/api/show` probe is sent. **It always sends a `User-Agent`** — the
  Console gateway returns HTTP 403 for urllib's default `Python-urllib/<ver>`
  (urllib does not send one; it sends its default, which is blocked) — override
  with `user_agent` if desired. The Console `llm` client also forces
  `disable_thinking` off: `LLMClient` sends `reasoning_effort: "none"` (an
  ollama-compat knob) and the Console gateway rejects that parameter with HTTP
  400. Only the Chat Completions family is drop-in;
  Claude / OpenAI-Responses / Gemini models use different wire formats and need
  their own adapters. The Console's **Jev** (`/zen/v1/systemone`) is a
  probability/decision evaluator — do NOT wire it into routing/assessment; SemIf
  alone decides.
- **Do NOT cap `max_tokens`** on the codegen call. A reasoning codegen model
  reasons first and a cap truncates the hidden reasoning, leaving `content`
  empty (`finish_reason: length`) and the body write fails with "skill body is
  empty". Unbounded, it runs to completion (tens of thousands of reasoning
  tokens then the code); the client reads only `content`, so reasoning is
  filtered automatically. The client default timeout is 1200s and a slow codegen
  model can exceed it — the per-machine `config.json` sets `codegen.timeout:
  3600`. Raise `codegen.timeout` in config if a harder prompt needs more.
- Bodies are persisted as one **folder per skill**: `data/skills/<category>/<name>/`
  holding `skill.py`, `skill.test.py`, `contract.json`, and `config.json` (all
  gitignored), loaded back at startup via `importlib`, so
  skills stay runnable and configurable across restarts. The **old single-file
  layout** (`data/skills/<category>/<name>.py`) is **not read** — clean switch,
  no compat shim. `CODEGEN.md` at the repo root is the contract the codegen model
  is prompted with — change it only with intent, it shapes every generated body.
  Bodies declare `INTEGRATION` (service/transport/config_vars) and must perform
  the real action via stdlib transports with values from `ctx.config`.
  Mocking/testing has its **own** contract, `TESTGEN.md`: a **hermetic mechanics
  check** (inline fixtures, loopback `http.server` for HTTP bodies, no external
  network) that never certifies the live integration — only a real run does.
  The body owns no data; everything comes from the runner via `ctx.config`.
- **Seed skills.** `seeds/<category>/<name>/` ships a real starter integration in
  the exact generated-skill folder format (`skill.py`, `contract.json`,
  `skill.test.py`, plus `manifest.json` with the description). `merge_seed_store`
  loads them into every tree at startup; recorded config (credentials collected
  at first fire) is written to the runtime store `data/skills/`, so the committed
  seed never holds secrets, and a generated body with the same name replaces the
  seed. `calendar.create_event` is the authoritative worked example (CalDAV PUT;
  derives the title/date/time from the query, asks for missing fields and
  resumes, resolves the target calendar with a SemIf choice over the owned
  calendars — a strong winner is used, otherwise the configured default);
  `calendar.next_event` is the read counterpart (Nextcloud CalDAV, recurring
  events expanded server-side); `simplex.next_message` (read via the forwarding
  bridge) and `simplex.connect_link` (show/create the forwarding bot's contact
  link) are the messenger seeds; `time.now`, `time.date`, `time.set_timer`, and
  `time.set_alarm` are the local time utilities (`compute` transport, empty
  contract) — they read the host clock and schedule on `ctx.timers`;
  `tests/test_seed_skills.py` keeps them honest.
- **Contract provenance.** Each authored body records the CODEGEN.md revision it
  was written against. `skill_contract_ref()` (`codegen.py`) returns
  `{"ref", "dirty"}`: `ref` is the short git commit sha the contract was read
  under (revive with `git show <ref>:CODEGEN.md`) and `dirty` records whether the
  working-tree contract differed from that commit; both degrade to `None`
  outside a git checkout. Recorded as `contract_ref`/`contract_dirty` on the
  `skill_writing` trace event; the dashboard shows `CODEGEN.md @ <ref>` with a `*`
  when dirty. **Deferred:** this pointer covers only the generic pattern now that
  bridge specifics live in the runtime catalog, not CODEGEN.md — recording a
  catalog revision on the trace is future work.
- **Trust boundary**: generated skill code is executed locally (it is imported
  as a module and its `act` runs in-process; `skill.test.py` runs as a
  subprocess in the skill folder). The provisioned host is the intended target;
  treat the endpoint as trusted.
- Flow in `scheduler._dispatch_skill`: the request is queued to the **single-slot
  `llm` draft worker** (small model authors title+description) → stub registered
  + hot-merged into the tree → the body write is queued to a **separate
  single-slot codegen worker** (the 12G codegen model can't run twice), the gate
  stays free for new input. A category request chains the same way in one worker.
  The codegen worker runs the full authoring pipeline:
  1. **elicitation** (`generate_elicitation`): implementation questions +
     integration hint. Every front end (REPL, dashboard, gateway) sets
     `defer_questions` and the worker posts them to the question queue one at a
     time, waiting `codegen.elicitation.answer_timeout` seconds for each (the
     clock resets on every answer submit; a timeout stops the sequence).
  2. **codegen body** (`generate_skill_body` / `regenerate_skill_body` on
     repair): CODEGEN.md + request + tree + requirements answers; the body is a
     single `act(ctx, request)`, reads every operational value from `ctx.config`,
     performs the real action, and declares `INTEGRATION` and its own flat
     `CONTRACT` (every contract key must be read from `ctx.config`, and every
     name the body references must be imported/bound — both enforced by
     `parse_skill_body`).
  3. **fidelity gate** (`authoring:fidelity` SemIf decision +
     `integration_findings`): a rapid sanity check (accept/reconsider) — is the
     action real or simulated/declared-but-unused? It only triggers a rewrite,
     never a diagnosis. One corrective regen with the raw evidence bundle
     (`codegen.fidelity.max_attempts`), then accept with a trace rather than
     hard-failing authoring.
  4. **contract** (`parse_contract`): read from the body's own flat `CONTRACT`
     constant — a single flat object of snake_case variable name -> semantic
     description, for user input and SemIf only (no types/validation; that lives
     in the code + test). No separate generation call; `materialize_skill`
     rewrites `contract.json` as a mirror.
  5. **test** (`generate_skill_tests`): a shared-context call (seeded with
     TESTGEN.md + `skill.py` + the body's `CONTRACT`) produces `skill.test.py` —
     hermetic mechanics, loopback fixture injects the endpoint through config,
     fixtures inline.
  6. **auto-run test** (`run_skill_test`): subprocess in the skill folder,
     `codegen.test_timeout`; on failure a **2-option SemIf decision**
     (`codegen_regen`, trace-only) picks code/test to regenerate (the contract
     rides in the code), the error + existing files are fed back, bounded by
     `codegen.test_max_attempts`.
  On success the body is materialized (`materialize_skill`), hot-merged, the
  leaf's `writing` flag clears, and the **original request is re-queued** at its
  scored weight and re-runs navigation onto the new leaf (`skill_requeued`). On
  failure (`CodegenError`/`ValueError`) the user is notified, the leaf stays a
  restartable stub, and the request is NOT re-dispatched. A re-dispatched
  request whose skill is still unwritten reports the pending write instead of
  authoring a second skill (`awaiting_skill_body` meta guard). `create_category`
  runs the same chain after authoring the category (`create_category` →
  `create_skill` → async body → re-dispatch). An empty (stub) or in-progress
  leaf picked by navigation is reported gracefully — no silent no-op; restart it
  with the `restart <category> <skill>` REPL command or the dashboard button on
  the skill row / `skill_write_failed` flow node. Codegen failure — including a
  request timeout — is raised as `CodegenError` by the client, never a raw
  `TimeoutError`. The default codegen timeout is 1200s
  (`cli.build_scheduler`); a slow codegen model's writes can run 25–45 min, so
  the per-machine `config.json` sets `codegen.timeout: 3600`. Raise
  `codegen.timeout` in config for harder prompts.
- **Config tiering + first-fire collection.** After the contract lands, a SemIf
  `choice` per variable auto-populates the skill `config.json` from the whole
  cascade — global config → category config (`data/skills/<category>/config.json`)
  → skill config (`codegen.contract_search`, phase `config:search`,
  decision-logged). At runtime the runner merges global -> category -> skill
  config (+ per-fire answers) into `ctx.config`; contract variables it cannot
  satisfy pause the run **before act** (`pre_act`), asking the human one at a
  time. Each answer gets a SemIf `record-as-config vs ask-again-each-fire` choice
  (phase `config:record`); recorded values persist to the skill `config.json`.
  **An empty answer (or the literal "skip") skips that variable for the run**: it
  stays unset in `ctx.config` and is never re-asked, and the run proceeds (or
  asks the next missing variable). An act-driven `needs_input` also accepts an
  empty answer — it is forwarded to `act` as `user_input=""`; a body that then
  fails for lack of the info is assessed/repair-offered normally.
- **Resolved-input observation.** The `assess:outcome` / `assess:requeue` state
  includes the resolved contract values (`resolved inputs:` block) so the
  decision sees the full query → resolution → action path; secret-named
  variables (pass/token/secret/key/credential/auth) are redacted to `***` so
  credentials never reach the decision log.
- **Requirements elicitation (implementation questions, default on).**
  `codegen.elicitation.enabled` asks the product owner how the new skill should
  connect (which service/account, how to connect, where the credential comes
  from, what success looks like, how failure should behave). Every front end
  defers — the single-slot worker posts
  questions to `scheduler.questions` (`GET/POST /api/questions`) one at a time
  and waits up to `codegen.elicitation.answer_timeout` seconds for each (the
  clock resets on every answer submit, so the next question is only shown after
  the current is answered; a timeout stops the sequence), then proceeds with
  whatever answers arrived (`questions_timeout` trace). Answers ride on the draft into
  every body attempt (including retries and regens) and are persisted in the
  registry so a `restart` does not ask again. The prompt keeps an explicit
  anti-pattern block: never config-vs-input cadence questions ("should the
  sender address change?"), never operational data values, never trivia.
- **Fidelity gate.** After the body lands, `authoring:fidelity` — a real SemIf
  accept/reconsider decision (`P(reconsider) >= tau` or any static
  `integration_findings` from the declared transport vs. real calls and declared
  config_vars actually read) decides whether the body really performs the
  action. The gate never diagnoses: it only triggers a rewrite, which is handed
  the raw evidence bundle as-is (request, description, requirements, raw
  INTEGRATION, findings, SemIf verdict+probs, previous body). A rejected body is
  rewritten once (`codegen.fidelity.max_attempts`, reason_kind `fidelity`); a
  body that still fails is accepted (never hard-fail authoring) but traced
  (`fidelity_review`, `performs_real_action=false`) and badged `unverified` in
  the dashboard. A decision row is logged under phase `authoring:fidelity`.
- **Repair loop.** A failed real run (assessment failure or a raised skill
  error) logs a SemIf choice (phase `repair:choice`) offering
  `retry` / `repair_skill` / `ask_user` / `no_repair`, recorded as a
  `repair_offered` trace and surfaced in the REPL (`repairs`, `repair <id>
  [action]`) and the dashboard repair panel. Executing is user-confirmed (a
  codegen write is slow): `retry` requeues the request, `repair_skill` rewrites
  the body with the observed failure (`run_failure` reason_kind) and re-runs it,
  `ask_user` posts a repair question whose answer starts the repair. Repairs are
  bounded by `codegen.repair.max_attempts` per skill; the run itself is never
  auto-repaired silently.
- Engine calls are serialized with a `threading.Lock` inside `SemIfEngine` so
  the codegen worker's degeneration SemIf checks never race main-thread
  navigation/scoring on one llama.cpp context.
- Set `codegen.stream: true` to echo the codegen output as an SSE token stream
  to stdout during body writes — including the chain-of-thought, so a long
  (~30 min) write shows live progress. The client reads reasoning from either
  `reasoning` (ollama) or `reasoning_content` (other OpenAI-compatible
  backends) — do not drop one for the other. Echoing is console-only; the
  returned content is identical either way. Integration tests already force
  streaming; see it with `-s`.
- `codegen.idle_warn` (default 60s) / `codegen.idle_timeout` (default 180s)
  surface a silent stream: a wedged generation prints a warning at `idle_warn`
  seconds with no tokens, then raises `CodegenError` (→ graceful stub) at
  `idle_timeout` — instead of blocking on the 1200s total budget. A streaming
  stall with zero output usually means the ollama ROCm runner wedged;
  `sudo systemctl restart ollama` is the recovery.
- **Token budget + exact accounting.** The context window is auto-detected once
  from ollama `/api/show` (`parameters.num_ctx`, falling back to
  `model_info.<arch>.context_length`, then the `context_window` config, then
  default 100000). `codegen.smart_limit`/`warn_limit`/`max_fill_ratio`/
  `warn_fill_ratio` bound total fill (prompt + output); `max_output` caps
  streamed output (>=1 = absolute tokens, 0<x<1 = fraction of window);
  `chars_per_token` converts chars to the token estimate for the output cap.
  The request always sends `stream_options.include_usage`, so the client logs
  the **exact** `prompt/completion/total` tokens, % of window, and a
  `SMART|WARN|DUMB` zone (absolute thresholds `smart_limit`/`warn_limit`) at
  the end of the stream. Real ollama serves `/api/show`'s `parameters` as
  modelfile **text** (`"num_ctx 100000\n..."`), not a dict, so the parser
  handles both shapes and falls back to `model_info.<arch>.context_length`. The
  total wall-clock `timeout` is enforced while the stream is live (a
  continuously-streaming runaway is cut off), not only during idle gaps. The
  budget is a safety net, not the loop cure: a reasoning codegen model routinely
  exceeds even the relaxed cap before it settles on a body.
- **Sampler params (opt-in).** `temperature`/`top_p`/`presence_penalty`/
  `frequency_penalty` are **not sent by default**: a `None`/absent value is
  omitted from the request so the server's model/Modelfile default applies. The
  provider never guesses a value for the user — set one in the `llm`/`codegen`
  config block (or per `chat` call) only when you know the right value for the
  model. `config.example.json` ships explicit values as a starting point, so a
  provisioned host still sends them. When configured, a high `presence_penalty`
  (not greedy temperature) is the anti-repetition loop cure for a looping codegen model, and
  those four are the only sampler knobs reachable via ollama's OpenAI-compat
  API (`repeat_penalty`/`min_p`/`top_k` are Modelfile-only). There is **no
  escalated preset**: a rejected body retries with a fresh short prompt that
  resets the context to SMART and reuses the same (configured or omitted)
  sampler params; `max_attempts` (default 3) bounds the ladder, then a graceful
  stub. Degeneration does **not** retry: the watchdog raises
  `DegenerationError(CodegenError)`, which propagates straight to the graceful
  stub (retry-on-degeneration is left as an open decision).
- **Degeneration watchdog.** `codegen.degeneration.{enabled, threshold=0.9,
  interval=8000, window=2000, min_chars=4000}`: while the body streams, a
  2-option SemIf decision (continue/stop) runs every `interval` chars against
  the last `window` chars of content+reasoning once `min_chars` have
  accumulated; `P(stop) >= threshold` aborts the write with a graceful stub.
  Recorded as **trace-only** events (kind `codegen`, with probs and
  `stop_prob`) — never in the decision log. Disabled when no engine is
  available (`enabled` defaults true; the runtime engine check is what gates
  it off when no engine is loaded).

### gateway (messenger intake)

- **Run mode.** `python -m semif_agent.cli gateway [--platform simplex,lxmf,voice|all]
  [--dashboard]` builds the normal scheduler (engine lazy) and runs the enabled
  adapters in the foreground. `--platform` takes a comma-separated list or
  `all`; default is `simplex`. **All adapters share one process and one
  scheduler** (never run two gateway processes over one skill store) — each
  gets its own transport thread but they route through a single
  platform-aware `GatewayService`. `--dashboard` co-serves the browser UI from
  a daemon thread. Config lives under `gateway.<platform>` in `config.json`;
  `enabled` defaults false. The voice platform is foreground-only by design (a
  headless systemd user service does not share the user's audio session). On a
  bootstrap-provisioned host
  `scripts/bootstrap.sh` renders and enables the `semif-simplex.service` bot
  daemon (pinned `simplex-chat` in **bot mode**, profile under
  `.runtime/simplex/`, port from `simplex_chat.port`) and the
  `semif-gateway.service` agent process (unit runs `gateway --platform all`),
  so the gateway runs across logout/reboot. The gateway is its own process —
  the REPL and the gateway are independent front ends onto the same on-disk
  logs/registry. **On connect the gateway prints each bot's own address once**
  (the SimpleX contact link via the daemon it already owns,
  `_gateway_address_callback`; the LXMF address via `_lxmf_address_callback`) so
  a human can reach it without running a separate script — the operator front
  end prints it; the command adapter itself gains no address surface.
- **Bot address / v7 relay gotcha.** The daemon must NOT run with `--headless`
  or `--relay`: in SimpleX Chat v7 `--headless` means "chat relay" (requires
  `--relay`) and yields a *relay* address, not a user contact address. Run bot
  mode instead (`simplex-chat -p PORT --create-bot-display-name NAME`). **The
  CLI only pumps WebSocket events while it has a controlling terminal**, so the
  unit wraps it in `script -q -e -c '…' /dev/null` to allocate a PTY: a TTY-less
  process (plain `ExecStart=` under systemd) accepts the socket but never emits
  `receivedContactRequest`/`newChatItems`, so contact requests hang forever.
  `python scripts/simplex-address.py` then shows (creating on first run, via
  `/_address` / `/_show_address <userId>`) the bot's user contact link, which a
  human adds in their SimpleX app to start a DM. `bootstrap.sh` prints it at the
  end of provisioning.
- **Transport contract** (`gateway/base.py`): `GatewayAdapter.run(on_inbound,
  outbound_queue)` blocks, delivering `InboundMessage`s and draining a stdlib
  `queue.Queue[OutboundMessage | None]`. Scheduler work is synchronous and can
  block on the decision engine, so the adapter bridges it off its event loop
  (`asyncio.to_thread`) and puts replies on the queue. A new platform means a
  new adapter subclass plus a `_build_gateway_adapter` branch and a config
  block; the platform-aware service routes it unchanged.
- **SimpleX adapter** (`gateway/simplex.py`): connects to the local
  `simplex-chat` daemon at `gateway.simplex.ws_url` (default port 5226, pinning
  `simplex_chat.port`), XML-JSON WebSocket protocol
  (`{"corrId","cmd"}` → `{"corrId","resp"}` / events). `websockets` is
  **lazy-imported**; `check_requirements()` returns an install hint and the
  gateway refuses to start without it, so the core stays importable on a
  websocket-free machine. Default-deny allowlist matches either a numeric
  `contactId` or a display name (`allowed_users`); `allow_all_users` is the
  dev escape hatch. `auto_accept` answers `receivedContactRequest` with
  `/_accept`. Inbound `newChatItems` are filtered to direct, non-echo
  (`chatDir.type` not `*Snd`), text-only items, buffered per contact and
  flushed after `text_batch_delay`. Outbound uses the **structured** command
  `/_send @<contactId> json [{"msgContent":{"type":"text","text":...}}]` — the
  `@<id> <text>` shortcut is silently rejected over WebSocket. Groups,
  attachments, reactions, and typing are out of scope for this cut. The wire
  protocol itself lives in the neutral `semif_agent/simplex_ws.py`
  (`SimplexDaemon`); this adapter adds only the allowlist and batching on top.
- **LXMF adapter** (`gateway/lxmf.py`): the Reticulum-based command transport.
  Unlike SimpleX there is **no external daemon** — Reticulum and the LXMF router
  run in-process (`semif_agent/lxmf_transport.py`, `LxmfDaemon`). `lxmf` (and
  `rns`) are **lazy-imported**; `check_requirements()` returns a `pip install
  lxmf` hint and the gateway refuses to start without it. The bot's reachable
  address is its LXMF delivery destination hash (32 hex chars); the identity is
  persisted under `gateway.lxmf.storage_path` (default `.runtime/lxmf/router`)
  so the address is stable across restarts. Default-deny allowlist matches the
  peer's LXMF address or a best-effort display name (`allowed_users`);
  `allow_all_users` is the dev escape hatch. Rapid messages are batched per chat
  (`text_batch_delay`) with a `threading.Timer` (no event loop).
  `desired_method` is `direct` (reliable link) or `opportunistic` (single
  packet); an optional `propagation_node` enables store-and-forward. **`RNS.
  Reticulum` and `LXMF.LXMRouter` install process signal handlers, which only
  work on the main thread** — `run_gateway` calls `daemon.prepare()` on the main
  thread before the transport threads start, and `run()` reuses it.
- **Voice adapter** (`gateway/voice.py`): the local microphone front end. It is
  a policy layer over the neutral `semif_agent/voice_transport.py` (`VoiceDaemon`)
  — openWakeWord wake word → `webrtcvad` endpointing → faster-whisper
  speech-to-text, and Piper text-to-speech, all over `sounddevice` at 16 kHz
  mono int16 (capture/playback run at the device's native rate and are resampled
  to/from 16 kHz; a per-block DC removal handles mics with a constant offset).
  There is **no external daemon and no allowlist**: the mic is local,
  so physical access is the authorization; a single configured `chat_id`
  identifies the session and `home_channel` defaults to it so background
  notifications (timers, repair offers) are spoken too. `run(on_inbound,
  outbound)` starts an outbound-pump thread (`OutboundMessage` → `speak`) and
  blocks in the mic loop. **Half-duplex**: capture frames are discarded while a
  reply is speaking, so the agent never transcribes itself (no barge-in yet);
  after a reply a `follow_up_window_s` accepts the next utterance without the
  wake word, so answering a question is conversational. `max_speak_chars`
  truncates long replies before speaking, and a short sine `cue` beep plays on
  wake (optionally again on transcription) so the user knows they were heard.
  Every heavy dep
  (`sounddevice`/`openwakeword`/`faster_whisper`/`piper`/`onnxruntime`) is
  **lazy-imported**; `check_requirements()` returns an install hint and the
  gateway refuses to start without the stack. The engine interfaces are plain
  classes (`WakeWordDetector`/`VadGate`/`Transcriber`/`Synthesizer`/`AudioIO`),
  so tests inject fakes and drive the loop with no hardware. It runs best in the
  user's audio session (foreground `gateway --platform voice`), not a headless
  systemd user service; `bootstrap.sh --voice` installs the deps and downloads
  the models (openWakeWord into the venv's package dir under `.runtime/venv`,
  the Piper voice into `.runtime/voice/tts`, the faster-whisper model into the
  `.runtime/hf` cache).
- **Scheduler glue** (`gateway/service.py`): inbound text →
  `Scheduler.submit_request(text, source=f"{platform}:{chat_id}")`; the returned
  request id maps to the `(platform, chat)` (`owners`), and
  `Scheduler.on_request_requeued` (a scheduler hook, default `None`) copies that
  ownership across an updated request so its completion still routes home. A
  `needs_input` pause sets `pending_owner` from the pending run's source; the
  same chat's next message goes straight to `Scheduler.answer` (no
  score/navigation). A *different* chat — **including a chat on another
  platform** — during that pause is told to wait: the single-slot scheduler must
  not silently abandon the first chat's run, and one service fronts all
  adapters so the guard spans them. A background poll calls
  `Scheduler.run_queue()` (routing each `[run_id] summary` to its owner) and
  surfaces newly posted authoring questions / repair offers (`defer_questions
  = True`; the service routes them by `run_id` → `(platform, chat)`, falling
  back to that platform's `home_channel`); a chat's plain reply answers the
  question or picks the repair action (`retry`/`repair`/`ask`/`no`).
- **Gateway isolation (non-negotiable).** The gateway exists only to take
  commands and send replies. It MUST NOT read a contact's history, show/create
  invite links, or compose messages on the user's behalf: no `observer`, no
  inbound buffering/pull mode, no `/address`, no `/inbox`, no `/send`. Those are
  messaging-UX concerns and live in the standalone bridges
  (`semif_agent.bridges`) against their own daemons/profiles. Letting them bleed
  into the always-connected command bot enlarges its attack surface for nothing.
  If you are tempted to add read/send/address functionality to `gateway/`, add a
  bridge instead.
- **Neutral SimpleX transport** (`semif_agent/simplex_ws.py`): the wire protocol
  both front ends share — parsing a direct text item across the v7 wire shapes,
  the structured `/_send`, `/_accept`, and the corrId→Future round-trip used by
  `/_show_address` / `/_address` (`SimplexDaemon.request_address`, thread-safe
  via `run_coroutine_threadsafe`). It carries no policy: no allowlist, no
  scheduler, no HTTP.
- **Tests.** `tests/test_gateway.py` (stdlib): SimpleX adapter allowlist/auth,
  structured send command, batching (real `asyncio`), LXMF adapter
  allowlist/auth and batching (a `threading.Timer`), cross-platform pending-run
  guard and per-platform reply routing, and `GatewayService` routing against a
  real `Scheduler` (lazy engine, unreachable LLM).
  `tests/test_lxmf_transport.py` covers the neutral LXMF transport's
  missing-dep contract, default paths, and `normalize_message` (no real `lxmf`
  needed). `tests/test_simplex_ws.py` covers the neutral protocol layer (parse
  shapes, accept ids, corrId round-trips, address show/create).
  `tests/test_bridges.py`
  exercises `SimplexBridge` over a real loopback server (peek/pop, recipient
  resolution, outbound routing, token/body validation, `/address` success/503/502,
  daemon close on stop, catalog/`describe_bridges()`).
  `tests/test_seed_skills.py` hermetically tests every
  `seeds/<category>/<name>/` package, including `simplex.next_message` and
  `simplex.connect_link`. `tests/test_voice_gateway.py` drives the voice loop
  with injected fake engines (wake gating, endpointing, half-duplex,
  follow-up window, `speak` truncation/failure, adapter + CLI wiring) — no
  hardware, network, or heavy packages. The live `websockets` transport against
  a real daemon, the live LXMF transport against a real Reticulum network, and
  the live voice pipeline against a real microphone are live-integration
  concerns.
- **Deps.** `websockets` and `lxmf`/`rns` are pinned in
  `requirements/staging.txt` (staging only); both are lazy-imported and the core
  stays stdlib-only. The voice stack is pinned separately in
  `requirements/voice.txt` (hardware-dependent: `sounddevice`, `openwakeword`,
  `faster-whisper`, `piper-tts`, `onnxruntime`, `webrtcvad-wheels`, plus the
  system `libportaudio2` + `alsa-utils`); it is lazy-imported too, and `bootstrap.sh --voice`
  installs it and downloads the models (openWakeWord into the venv package dir,
  the Piper voice into `.runtime/voice/tts`, faster-whisper into `.runtime/hf`).

### bridges (third-party API services)

- **What they are.** A bridge service is a standalone process that stands up a
  simple, secure localhost HTTP layer in front of one third-party system so a
  codegen-authored skill body never speaks that system's native protocol.
  SimpleX is the first; adding a service means adding a class to
  `bridges/registry.py` `CATALOG` and a `config.example.json` block. The catalog
  (`BridgeInfo`: name, service, description, URL config var, other config vars
  with one-line docs, auth header + token config var, and endpoints with
  request/response/error shapes) is injected into every codegen prompt
  (body/retry/regen/elicitation/testgen) via `describe_bridges()` and is the
  single source of bridge specifics — CODEGEN.md/TESTGEN.md carry only the generic
  pattern. The bridge's optional shared secret is mirrored to the top-level
  `simplex_bridge_token` config var (bootstrap syncs it) so bodies can send the
  `X-Semif-Token` header.
- **Run mode.** `python -m semif_agent.cli bridge [--name simplex]` builds the
  scheduler (engine lazy, used only for trace) and runs the selected + enabled
  bridges until interrupted. Each bridge owns its service daemon: `SimplexBridge`
  connects to a **second** simplex-chat daemon/profile (`bridges.simplex.ws_url`,
  default port 5228, pinning `simplex_chat.forward_port`), separate from the
  command gateway's. On a bootstrap-provisioned host, `scripts/bootstrap.sh`
  renders and enables `semif-simplex-forward.service` (the second daemon) and
  `semif-bridge.service` (`cli bridge`) alongside the gateway units.
- **HTTP surface** (`bridges/base.py` + `bridges/simplex.py`): JSON in/out,
  localhost-bound, optional shared-secret `X-Semif-Token` header. `SimplexBridge`
  buffers **every** inbound DM (no allowlist — the user wants to see who reached
  the bot through its invite link) in a bounded `MessagingInbox` that owns the
  read cursor, and serves `GET /health`, `GET /contacts` (the daemon's contact
  list, refreshed on (re)connect and on demand — best-effort, falling back to
  the cached/inbound-learned set when the daemon is unavailable), `GET /inbox`
  (peek), `GET /inbox/next?contact=<id>` (pop oldest), `GET /unread` (the
  daemon's persistent unread chats via `/_get chats` + `ChatListQuery` unread
  filter — `chatStats{unreadCount, minUnreadItemId}` plus each previewed
  `meta.itemStatus`; read-only), `GET /history?contact=<id>&count=<n>` (recent
  messages for one chat via `/_get chat @<id>`; read-only), `GET /address`
  (show/create the forwarding bot's contact link; 503 when the daemon is not
  connected, 502 when the lookup fails), and `POST /send`
  `{"recipient","text"}` (resolves a numeric id or a known display name —
  refreshing the daemon contact list once on a miss — and enqueues on the
  bridge's own daemon). `/unread` and `/history` read the daemon's own state
  (surviving bridge restarts), unlike the live `/inbox` buffer; both are
  read-only because v7 has no mark-read command (acking is a client-side read
  receipt).
- **Skill-facing config.** The top-level `simplex_bridge_url` is the address a
  body calls (the data-contract config search auto-populates it); bootstrap keeps
  it in sync with the bridge port. Skills reach a bridge with the ordinary `http`
  transport, so their hermetic tests are loopback HTTP like any other HTTP body.
  `simplex.next_message` and `simplex.connect_link` are the seeds.
- **LLM bridge** (`bridges/llm.py`). A generic `POST /chat`
  `{"messages": [{"role","content"}, ...], "max_tokens"?: int}` ->
  `{"text": "<model reply>"}` (plus `GET /health`, `400` bad body, `502` model
  error) in front of the agent's language model, so a body can ask for short
  generated text over HTTP instead of speaking the OpenAI-compatible protocol
  itself. It does **not** own the model connection: `run_bridge` passes the
  scheduler's already-configured `llm` client through `run_bridges`/`build_bridge`
  into `LLMBridge`, so the endpoint/model/sampler stay configured once in the
  top-level `llm` block; the `bridges.llm` block carries only the local listener
  (`enabled`/`host`/`port`/`token`). Bodies reach it via the top-level
  `llm_bridge_url` (+ optional `llm_bridge_token`). It runs inside the existing
  `semif-bridge.service` process (all enabled bridges start together); with no
  client it reports `502`, never a fabricated reply. `calendar.create_event` is
  the first consumer (title + description extraction, cached in `request.meta`
  across a `needs_input` resume).

## Bridge backlog (one session per item)

The bridge read path (`simplex.next_message`) and contact-link lookup
(`simplex.connect_link`) are covered. Deferred follow-ups:

- [x] **1. Contact-list refresh from the daemon** (`bridges/simplex.py`,
  `simplex_ws.py`). Shipped: `SimplexDaemon.contacts()` /
  `request_contacts()` query `/_contacts <userId>` (active user from `/user` →
  `activeUser.userId`, cached, falling back to the configured `user_id`) over
  the existing corrId→Future path; `SimplexBridge` merges the result into the
  inbox on (re)connect via an `on_connected` hook and on demand (`GET
  /contacts`, and a send-resolution miss), so `/send` can address a contact it
  has never received from. Best-effort: a disconnected daemon keeps the cached
  set.
- [x] **2. Message history + true unread** (`bridges/simplex.py`,
  `simplex_ws.py`). Shipped: `SimplexDaemon.chats(unread_only, count)` /
  `request_chats()` query `/_get chats <userId> count=<n>
  {"type":"filters","favorite":false,"unread":true}` and parse
  `apiChats` (`AChat.chatStats{unreadCount, minUnreadItemId, unreadChat}` +
  `chatItem.meta.itemStatus` `rcvNew`/`rcvRead`); `chat_history(contact_id,
  count)` / `request_chat_history()` query `/_get chat @<id> count=<n>` and
  parse `apiChat`. `SimplexBridge` exposes `GET /unread` (persistent unread
  chats, summaries + previewed messages) and `GET /history?contact=<id>&count=<n>`
  (recent messages with per-item status), both read-only and degrading like
  `/address` (503 not connected / 502 lookup failure; `/history` 400 without a
  contact). The live `/inbox` buffer is unchanged. Note: v7 has **no mark-read
  command**, so acking is via read receipts, not an API call.
- [ ] **3. `simplex.send_message` seed** (`seeds/simplex/send_message/`). The
  outbound counterpart to `simplex.next_message`: resolve the recipient with a
  SemIf sub-decision over `/contacts`, send only on explicit user intent, and
  report the bridge's `contact_id` / errors honestly.
- [x] **4. LLM bridge** (`bridges/llm.py`). Shipped: `LLMBridge` exposes a
  generic `POST /chat` in front of the scheduler's configured `llm` client
  (threaded through `run_bridges`/`build_bridge`, never re-configured), registered
  in `CATALOG` and seeded under `bridges.llm` + top-level `llm_bridge_url`.
  `calendar.create_event` extracts its title + description through it.
- [ ] **5. More bridges.** Each new third-party API gets its own `bridges/<name>.py`
  + config block + `CATALOG` entry; the codegen prompts pick it up automatically
  through `describe_bridges()`.

### Code principles
- **Contain installation artifacts in the repo.** Everything a machine installs
  to run the agent (python venv, SemIf engine clone, GGUF, HF cache, the
  simplex-chat binary/profile, rendered systemd units) lives inside the checkout
  under a gitignored `.runtime/` tree — never scattered across `$HOME`. An end
  user must be able to find and debug the whole stack with as few steps as
  possible. Only artifacts that operationally cannot live there are outside
  (`~/.ssh/id_ed25519`; the real systemd user dir + linger, which the repo units
  are symlinked into). `scripts/bootstrap.sh` owns this layout; derive paths from
  the checkout, never hardcode `$HOME`.
- **No mocking.** The decision engine is always real SemIf. Pure unit tests
  touch data-structure math only (queue ordering, dream cost, contract
  serialization) or drive scheduling mechanics through the `ScriptedEngine`
  double in `tests/conftest.py`; decisions themselves are verified by
  integration tests on a provisioned host.
- Engine import is **lazy** (`engine.py`) so the rest of the package stays pure
  stdlib and testable without SemIf installed. Keep it that way.
- `DecisionLog.append` labels a row with the *selected* option by default
  (self-consistent, near-zero cost). Real labels come from `relabel` (human,
  weight 3x in `dream`) — failures alone don't produce correct labels.
- Navigation decisions ARE logged (`navigate:category`, `navigate:leaf` in
  skills.py) and therefore count toward dream cost. This is intended per the
  design; don't silently drop them.
- Queue ordering: urgency desc, then FIFO (`seq`). Recency is stored but is NOT
  in the sort key (it's anti-correlated with FIFO). Ageing pulls weights toward
  the max (1.0) so low items catch up; uniform additive boosts do nothing.
- The decision engine is always real and every `EngineUnavailable` is **fatal**:
  `submit` records it (`("fatal", ...)`), the scheduler exposes `fatal`, and the
  CLI prints it and exits non-zero. There is no degraded/soft-error path.
- Keep deps stdlib-only in the core; heavy deps live on the staging venv
  (`.runtime/venv`, provisioned by `scripts/bootstrap.sh`).

## Testing

- `python3 -m pytest tests/ -q --ignore=tests/integration` — anywhere, fast.
  Includes the dashboard API tests (`tests/test_dashboard_api.py`), which spin
  up the stdlib HTTP server on an ephemeral port with the engine never loaded;
  `tests/test_codegen.py` for prompt/parse/validate, integration
  extraction/findings, and body store round-trips; and `tests/test_llm.py` for
  the `_parse_json` helper and the LLMClient provider. `tests/conftest.py``s `ScriptedEngine`
  drives scheduling mechanics (assessment is now a SemIf decision) without a
  GGUF.
- `tests/test_no_env_leaks.py` guards the generic-repo rule: it fails if a
  deployment hostname, address, username, absolute home path, or local model
  name reappears in a tracked file. Deployment specifics belong in the
  gitignored `local/` folder (see "Local deployment notes").
- `tests/integration/` — run only where real SemIf + a real LLM endpoint are
  available (the endpoint may be remote).
- After touching scheduler/skills/codegen/engine, re-run both; the integration
  tests are the only end-to-end verification.
- **Fixtures are not the runtime.** Generated skill tests (and unit tests of the
  codegen client) use real local endpoints (a loopback `http.server`, never a
  mock); skill bodies must call the user's configured service, never a fixture
  address. Passing a skill test proves mechanics only; a real run proves the
  integration.