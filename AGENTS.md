# AGENTS.md

Guidance for working on the semif agent. Read this before touching code.

## Product premise (read first)

Semif-agent is an **end-user application**. An everyday person asks it to do a
basic task in plain language, and it does it:

1. The user asks for a task ("check my email", "message Sam on SimpleX", "put
   lunch with Dana on my Nextcloud calendar").
2. If the skill tree already knows how, the agent does it.
3. If it does not, the agent says it cannot help — unless the host has opted in
   to live skill authoring (the default is **deny**; see "New-skill creation
   mode" below), in which case it asks a few questions and generates a new skill.
   The primary way to extend the agent is the committed **skill packs**
   (`packs/`), not runtime authoring.
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
(SemIf). Inputs are queued FIFO and dispatched through a
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
                third-party API bridges), check-config (validate config/
                against the core + pack schemas), config-init (seed the split
                config/ + secrets.json)
config.py       layered per-machine config: config.json is a thin index
                (`{"include": [...]}`) over config/core.json + config/gateway.json
                + config/bridges.json + config/packs/<name>.json, deep-merged in
                order; `"!secret <key>"` resolves from the gitignored
                secrets.json; the core schema owns defaults and, with each
                pack's `config` schema, GENERATES config.example.json
                (check-config --write-example) and validates a live config
packs.py        SkillPack + PackRegistry: discovers packs/ (a `pack.json`
                manifest + skills/<category>/<leaf>/; each pack is cloned from its own repo — see packs.json), binds a concern to its
                categories, bridge(s), requirements, and config schema;
                merge_packs loads every enabled pack into the tree
scheduler.py    no up-front gate: every input is dispatched
                -> FIFO queue when busy; a skill run paused for input
                (needs_input) keeps `current` busy; `answer` routes straight to
                the pending run, bypassing the queue/navigation; draft authoring
                (title+description by the
                small `llm` provider) AND skill-body authoring (codegen) are ASYNC
                single-slot workers: a create_category/create_skill request is
                queued, the gate stays free, the category job chains into its
                skill job, and the original request is re-queued and re-runs the
                new leaf when the body lands; an optional human approval gate
                (new-skill mode `ask`; see below) blocks the draft worker
                after the proposal is authored and before anything is
                registered — one prompt per request, denial/timeout aborts
                creation, surfaced via the approval queue; the body worker runs the full
                pipeline: codegen body from a **de-identified spec** (goal +
                service/transport; codegen never sees the request, question
                answers, or run output) that declares its own flat
                `CONTRACT` -> fidelity
                gate -> test -> auto-run test (SemIf regen ladder on failure,
                code/test); config
                search auto-populates the skill config from the full config
                cascade; a
                `failed` step triggers a logged SemIf repair choice
                (retry/repair_skill/ask_user/no_repair) surfaced to the user,
                while a `continue` step is re-dispatched into next-step routing
                with a bounded run ledger (the chained multi-step path);
                a completed real-skill run's ledger is also held in memory per
                origin source for `query_recall_window` seconds (0 disables), so a
                follow-up request in the same chat arrives with it as
                `prior_ledger` (a `previous request` context block) and can refer
                to the prior result — kept separate from `run_ledger` so a
                recall never skips the actionability guard or feeds the
                monotonic-progress guard;
                housekeeping meta skills (delete/clear-config/regen/cancel-build)
                operate on the tree via `ctx.admin`; new-skill creation is gated by
                a tri-state mode (`new_skill_creation`: deny/allow/ask, **deny by
                default**) with a
                per-category override in the category config, with
                `housekeeping`/`response` always denied in code; front ends get a
                structured `SchedulerReply` (status, text, run_id, skill_ref,
                result) from `submit_request`/`answer`/`run_queue` so they never
                parse the run id or skill ref back out of the summary
queue.py        bounded FIFO request queue (arrival order, max depth)
skills.py    tree + registry (hardcoded built-ins: the closed `response` canned
                tree plus the `housekeeping` meta skills), navigation = SemIf
                choices per level (logged; the question is "which category/skill
                is the logical next step toward satisfying this request?" for the
                first step AND each continuation, and the run ledger is rendered
                into the state, so a chained step sees what was already done;
                the actionability guard runs first, at the top of navigation, to
                arbitrate the response/create boundary — but ONLY on the first
                step, since a continuation is actionable by construction),
                create_category
                and create_skill author + register stubs via the small `llm`
                provider (separate from codegen); SkillStore persists one folder per skill
                (skill.py, skill.test.py, contract.json, config.json) and
                materialize_skill / merge_skill_store
                hot-load runnable skills from data/skills/;
                merge_packs (semif_agent/packs.py) loads each pack's skills
                from packs/<name>/skills/ (each cloned from its own repo) with their recorded config read from
                data/skills/ (each pack bundles its concern, category scope,
                bridge, and config schema);
                ActionResult.needs_input pauses a run for human input;
                Skill.integration / integration_source read the body's
                INTEGRATION declaration (or infer it); Skill.contract is read
                from the body's `CONTRACT` constant (contract.json is a derived
                mirror);
                resolve_skill_config / unresolved_variables drive the tiered
                config merge (global -> category -> skill) + pre-act
                contract collection; Skill.origin records builtin/seed/generated;
                DeletedSkills (data/deleted_skills.json) tombstones a deleted
                skill so a seed does not reappear on restart; resolve_skill_ref
                parses a free-form request for a category.skill or bare name;
                parse_meta_command recognizes explicit housekeeping commands and
                the meta skills resolve or ask for their target as an input
                variable; compose_state / render_ledger / render_recall render
                the steps already taken (and, for a follow-up within
                `query_recall_window`, the recalled `prior_ledger` as a `previous
                request` block) into the routing/assessment state, and
                last_result returns the last run-ledger (or recalled) outcome a
                chained body can reuse instead of recomputing
skill.py        loop: observe -> act -> observe -> assess; the body is a single
                act(ctx, request) phase that returns ActionResult (its own
                SemIf sub-decisions ride on ActionResult.decisions and are
                logged with the run outcome, phase `act`);
                assess is a SemIf decision (`assess:outcome`: is there any
                additional work to satisfy the request? -> done / continue /
                failed, at threshold precedence failed > done > continue) and the
                run summary is deterministic (no generation), built from
                category.skill + ok/failed + action_log — the body writes only the
                result, the runner adds the `category.name:` ref; the assess state
                carries the skill's declared action, the goal, the resolved inputs
                (secrets redacted), the bounded action log, and the steps already
                taken (the run ledger); `continue` re-dispatches the request into
                next-step routing with a bounded run-ledger entry (capped by
                `max_reentries` + the monotonic-progress guard), `failed` offers a
                repair; a `DETERMINISTIC_CATEGORIES`
                run (housekeeping) skips assessment/repair — it is an internal
                mechanical action with a known result, not an external run;
                every skill's output is truncated to the one shared speakable
                budget (`chain.result_chars`);
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
engine.py       DecisionEngine (the pluggable contract: call/warm/loaded/
                generate) + SemIfEngine, the default local provider ->
                semif_phase1.llamacpp_backend (lazy import); call() scores a
                decision, generate() autoregressively samples text from the
                same loaded model (shares the engine lock, clears the KV cache)
                — backs the `semif` llm provider. Selected per machine by
                `engine.provider` (mirrors llm.provider/codegen.provider).
winnow.py       WinnowEngine -> a remote Winnow `/v1/systemone` typed-decision
                HTTP API (`engine.provider == "winnow"`): maps DecisionRequest
                to a `choice` question (options -> criteria), validates and
                normalizes the returned distribution over exactly those option
                ids, and generate() via /v1/chat/completions. stdlib urllib; an
                internal lock serializes on the single-slot server; every
                failure raises EngineUnavailable (fatal, like the local engine)
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
                and testgen prompts — it is the single source of
                bridge specifics (service, base-URL var, auth header/token var,
                config-var docs, endpoints with error shapes), while CODEGEN.md/
                TESTGEN.md carry only the generic pattern; the advisory
                service/transport integration hint rides on every body/retry/regen
                prompt; TESTGEN.md drives the shared-context test
                call producing skill.test.py (hermetic mechanics test: inline
                fixtures, loopback http.server for HTTP bodies, no external
                network, no mock_data.json); run_skill_test executes
                the test as a subprocess
llm.py          small OpenAI-compatible provider (LLMClient) that authors a
                new category/skill title + description — its own endpoint/model,
                deliberate separate from codegen; provider logic (SSE/budget/idle)
                is shared from provider.py; `_parse_json` is also borrowed by the
                authoring parsers; SemIfLLMClient serves the same `llm` task from
                the loaded SemIf engine model (llm.provider == "semif")
provider.py    shared OpenAI-compatible chat transport (OpenAICompatClient:
                SSE stream, token budget, idle watchdog, degeneration hook,
                optional `api_key` bearer auth + `extra_headers`/`user_agent`,
                and `query_context` to skip the ollama /api/show probe)
                subclassed by llm.LLMClient and codegen.CodegenClient;
                SemIfEngineClient is the non-HTTP sibling that generates text
                from the in-process SemIf engine behind the same `chat(...)`
                surface (llm.provider == "semif")
console.py      OpenCode Console provider: OpenCodeConsoleClient + the
                ConsoleLLMClient / ConsoleCodegenClient endpoint subclasses.
                Hosted, authenticated OpenAI-compatible Chat Completions API
                (https://opencode.ai/inference/openai/v1); bearer key, skips the
                ollama probe, always sends a User-Agent (the gateway 403s
                urllib's default Python-urllib/<ver>), and forces
                disable_thinking off (the gateway 400s reasoning_effort).
                Selected via the llm/codegen `provider` key
                ("opencode" vs "ollama"; `llm` also accepts "semif", the
                in-process engine model).
log.py          decisions.jsonl rows {state, question, options, predicted_probs,
                selected, observed_outcome, label_source}
trace.py        runs.jsonl lifecycle events keyed by run_id (submit/queued/
                dequeued/assessed/...); decisions reference run_id in extra
dream.py        NLL of observed outcome per row; weighted CE, accuracy, ECE
decisions.py    contract dataclasses (Option, DecisionRequest, DecisionResult,
                Request)
dashboard.py    stdlib http.server + JSON API (tree/trace/dream/status/
                questions/repairs + POST submit/answer/questions/repair/restart/
                relabel); static/ frontend served at /; submit/answer drain the
                queue after replying (like the CLI and gateway) so a multi-step
                `continue` chain completes instead of stranding its child
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
                configured `chat_id`, replies spoken back. humanize.py:
                Humanizer — optional LLM cleanup of a result line (generation,
                never a decision; grounded prompt + always falls back to the
                raw text), gated by `gateway.humanize` / a per-platform
                `humanize` override, reusing the scheduler's `llm` client with
                a short per-call timeout. Run with
                `python -m semif_agent.cli gateway [--platform
                simplex,lxmf,voice|all] [--dashboard]`; config under
                `gateway.simplex` / `gateway.lxmf` / `gateway.voice`. The
                gateway MUST NOT read history, show/create invite links, or
                compose messages — see "gateway isolation" below.
bridges/        standalone third-party API bridges. The core keeps only the
                shared contract + discovery: base.py: BridgeInfo +
                BridgeService (shared localhost JSON HTTP layer,
                `X-Semif-Token` guard, `from_config` build hook,
                `checkout_root()` for `.runtime/` anchoring). inbox.py:
                MessagingInbox (bounded FIFO + contacts learned from inbound DMs
                and the daemon contact list, bridge-owned read cursor).
                llm.py: LLMBridge — the one shared, pack-less bridge (a generic
                `POST /chat` in front of the scheduler's configured `llm`
                client, reused not re-configured). registry.py: discovers each
                bridge from its pack's `bridge/` by convention (no hardcoded
                CATALOG), describe_bridges() (catalog injected into the codegen
                prompts), derived_config_vars(), run_bridges() (threads the
                scheduler's `llm` client through each bridge's `from_config`).
                The concrete bridges ship with their packs:
                packs/simplex/bridge/simplex.py (SimplexBridge — owns its OWN
                simplex-chat daemon/profile, separate from the gateway's,
                buffers every inbound DM, refreshes the daemon contact list on
                (re)connect + on demand, serves invite-link/read/send plus
                daemon-backed unread + per-chat history);
                packs/nextcloud/bridge/nextcloud.py + nextcloud_client.py
                (files/WebDAV, calendar events + tasks/CalDAV, contacts/CardDAV,
                Notes/OCS behind one JSON surface; no daemon, stdlib-only;
                `from_config` falls back to the top-level `nextcloud_*` values);
                packs/outlook/bridge/outlook.py + outlook_client.py (Microsoft
                Graph: mail, calendar, To Do, contacts, OneDrive, plus the
                OAuth2 device-code + refresh flow; tokens under
                `.runtime/outlook/token.json`, seeded once with
                `packs/outlook/scripts/outlook-auth.py`);
                packs/browser/bridge/browser.py + browser_client.py (Playwright
                open/read/click/fill/screenshot, owning a persistent browser
                profile; optional heavy dep, lazy so a missing browser is a
                per-call 503 and never blocks the other bridges; runs the async
                API on one dedicated event-loop thread).
                Run with `python -m semif_agent.cli bridge [--name NAME]`;
                config under `bridges`.
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
                (capture discarded while speaking). Every heavy dep is
                lazy-imported, keeping the module stdlib-only; the engine
                interfaces are plain classes so tests inject fakes. Knows
                nothing about the scheduler or the gateway.
voice_train.py
                neutral recording helpers for training a custom Piper voice
                (optional, off-bootstrap): LJSpeech dataset writer
                (`metadata.csv` + `wavs/NNNN.wav`), prompt loading, WAV I/O,
                level math for the mic check, and the silence-endpointing
                capture loop. Stdlib-only (numpy lazy in `resample_pcm16`);
                `scripts/voice-record.py` is the CLI front end and
                `scripts/voice-train.sh` the GPU fine-tune/export pipeline. See
                docs/VOICE-TRAIN.md. Knows nothing about the scheduler.
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
- **`config.json` (a thin index), the `config/` fragments, and `secrets.json` are
  gitignored, per-machine, and required** (hosts use different LLM/codegen
  endpoints and models). `config.json` is an index (`{"include": [...]}`);
  `config/core.json` + `config/gateway.json` + `config/bridges.json` +
  `config/packs/<name>.json` hold the values, and `secrets.json` holds
  credentials referenced as `"!secret <key>"`. `scripts/bootstrap.sh` seeds the
  split config once (via `config-init`, only when `config.json` is absent) and
  **nothing writes it afterwards — only a human edits it**. The committed
  `config.example.json` is GENERATED from the core schema
  (`semif_agent/config.py`) and each pack's `config` schema — regenerate it with
  `python -m semif_agent.cli check-config --write-example config.example.json`
  after changing a schema; validate a live config with `check-config`.
  `data/decisions.jsonl`, `data/runs.jsonl`, and `data/drafts/` are runtime
  artifacts and gitignored too.
- **`pins.json` is committed and code-owned** — the single home for every
  external ref (SemIf commit, GGUF url+sha256, HF tokenizer revision,
  simplex-chat version/url+sha256). These are bumped by a git change and picked
  up on the next bootstrap run; they are deliberately NOT in `config.json` (a
  ref replaced on every update is not per-machine user config).
- Decision rows logged before the `run_id` threading landed show up under
  run_id `"?"` in the dashboard — that's expected, not a bug.

## Roadmap

### v1 (done)
Core loop, FIFO queue, skill tree, skill loop with real SemIf assessment
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
  (response/nextcloud/simplex/time/calculator/browser) and read from the registry
  for authored
  categories; `category_descriptions(tree)` collapses it per category so an
  empty bucket is never offered bare. The old category threshold gate
  (`create_tau`/`create_margin`) is **retired**. Leaf level: the softmax
  contains **only existing skills** — `create_skill` was removed because sharing
  a softmax with the real skills diluted its probability and let a crowded tree
  drift into create on a weak plurality (real runs scored `create_skill`
  0.585–0.803 on requests a seed skill clearly matched). The reuse-vs-create
  decision is the **intent guard** (`confirm_skill_fit`, phase
  `navigate:intent`) at `navigation.intent_tau`: it decides what the softmax
  winner does for the request — `same` (performs the next action the request
  needs), `part` (the request needs more than one action and this skill performs
  one required part), or `different` (author a new leaf). The run ledger is part
  of the guard's state so a later step judges against the *remaining* work, not
  the whole request; the guard only decides whether the skill contributes —
  whether more work remains is `assess:outcome`'s call, so a compound request is
  broken into known skills while the final step still stops cleanly (a forced
  continue on `part` was tried and rejected: the guard labeled the final send
  step `part` 0.85 while assessment correctly said `done` 0.99, and the forced
  continue authored a spurious category). The **actionability guard**
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
  gets one — **unless new-skill creation is denied** (see the tri-state mode
  below), in which case an unservable request gets a plain refusal
  (`skills.UNABLE_MESSAGE`, "I'm sorry, I'm not able to help with that.") and is
  never told that authoring exists. All three taus are
  separate from the top-level `tau`, so tuning reuse-vs-create or actionability
  does not move assessment/fidelity. The guards are deliberately permissive
  (in-scope scores 0.9–1.0): a borderline over-cover routes into a plausible
  category and the leaf guard catches it. Empty and single-skill categories
  skip their softmax.
- **No handle/ignore gate; the `response` tree is the non-request catchall**
  (Sep 2026; actionability guard Oct 2026): the top-level `_contains_request`
  handle/ignore gate is gone. Every input is dispatched; input that
  is not a request falls through navigation into the closed `response` category,
  a hardcoded tree of canned replies (`response.greeting`, `response.thanks`,
  `response.acknowledge`, `response.farewell`, `response.affirm`, and the
  catchall `response.clarify` — "Could you try being more specific?"). A
  **request** is never answered from here, even one no skill covers: the
  actionability guard (`confirm_non_action`, phase `navigate:actionability`)
  runs at the top of navigation and sends a request to authoring (this is what
  keeps "what is 255 * 12?" out of the tree; the old
  `response.unable` canned refusal is gone — an unsupported request authors a
  skill, it is not refused, **except under `new_skill_creation: deny`, where it
  gets the plain `UNABLE_MESSAGE` refusal**). A fragment too vague/incomplete to
  act on (e.g. "what") is non-action, so it lands on the catchall
  `response.clarify` rather than authoring a redundant clarify skill. `response`
  is a `CANNED_CATEGORIES` member:
  `navigate` never offers `create_skill` there (the intent guard is skipped
  too), `Scheduler._dispatch_skill`/`restart_skill` refuse to author it, and the
  runner skips `assess:outcome`/repair for it — a canned line has no side effect
  to assess. The leaf choice logs phase `navigate:response` and traces
  `response_selected`. `response.reject` (a stub that never ran) is gone.
- **New-skill creation mode: deny / allow / ask** (Oct 2026; tri-state Oct 2026;
  deny-by-default Oct 2026): one setting governs whether the agent may author new
  skills/categories, as a three-way mode. **Global default** is the top-level
  `new_skill_creation` — **`"deny"` by default** (the agent leans on the
  shipped packs, the primary way to extend functionality), with `"allow"` an
  explicit opt-in for a host with the resources, or `"ask"`; a category overrides
  it with its own `new_skill_creation` in `data/skills/<category>/config.json`
  (absent = inherit the global). `allow` authors immediately; `deny` blocks authoring
  outright (`Scheduler._blocked_create`, traced `skill_create_blocked` /
  `category_create_blocked`) **and answers the request with a plain refusal**
  (`skills.UNABLE_MESSAGE`) that never names a category, skill, or config — a
  denied deployment gives no hint that authoring exists; `ask` authors only
  after the human approves the proposal. The `ask` path is the old approval
  gate: the single-slot `llm` draft worker blocks after it authors a proposal
  and before anything is registered;
  the human approves/denies via the REPL prompt, the dashboard approvals panel
  (`GET/POST /api/approvals`), or a gateway chat reply (`yes`/`no`). One prompt
  per creation request — a new category's chained skill is covered by the
  category approval (`DraftAuthor.approved`). Denial or timeout aborts creation:
  nothing is registered, no body is written, and the request is not
  re-dispatched; the wait reuses `codegen.approval.wait_timeout`. With no
  deferring front end `ask` denies immediately rather than stalling the worker.
  The mode is resolved by `Scheduler._category_mode` (deny for the hard-locked
  `housekeeping`/`response` categories) and checked in `_dispatch`,
  `_dispatch_skill`, `_author_skill`, `_request_approval`, and `restart_skill`.
  Toggled from the config file or `Scheduler.set_category_mode` (`/mode`,
  `/api/new-skill-mode`; an empty mode clears the override). Traced as
  `category_mode_set`, plus the approval events
  `creation_approval_requested` / `_approved` / `_denied` / `_timeout` /
  `_skipped`; the gate is deterministic (a human veto or a config value), not a
  decision row — the SemIf create doors still make and log the routing decision.
- **Housekeeping meta skills + per-category new-skill modes** (Oct 2026): a
  built-in `housekeeping` category of four meta skills that act on the agent's
  own tree through `ctx.admin` (the scheduler): `delete_skill` (confirm, then
  remove the leaf, registry entry, and folder; durable via a
  `data/deleted_skills.json` tombstone so a deleted seed does not reappear),
  `clear_config` (empty the skill's recorded config), `regen_skill` (ask
  **"What needs to be fixed?"** then rewrite the body through the corrective
  path with `reason_kind="manual"`), and `cancel_build` (confirm, then discard a
  queued/in-flight write and delete the half-built leaf). The target skill is an
  **input variable**: the skill resolves it from the request if named
  ("delete skill nextcloud.foo") and otherwise **asks** for it ("delete skill" →
  "Which skill?"), so the routed request never has to carry a task-like skill
  name. Explicit command forms ("delete skill", "clear config variables for X",
  "regen skill X", "cancel skill build") are recognized deterministically by
  `parse_meta_command` and routed straight to the housekeeping leaf (trace
  `meta_command`): the pinned 4B decision model reads a task-like name
  (`nextcloud.create_event`) as the task itself and misroutes the generic guards,
  so the explicit command is deterministic while natural-language phrasings
  still go through navigation (the actionability guard's `action` option now
  names skill-management commands). They are built-ins
  (`Skill.origin == "builtin"`) reachable by natural language and, for operators,
  by slash-prefixed REPL commands (`/delete`, `/clear-config`, `/regen`,
  `/cancel-build`, `/mode`) and dashboard endpoints/buttons. Their runs are
  deterministic internal actions: `DETERMINISTIC_CATEGORIES` skips
  `assess:outcome`/repair, and the category is hard-denied from authoring. Every
  category carries a **new-skill mode** in its category config
  (`data/skills/<category>/config.json` → `{"new_skill_creation": "allow|deny|ask"}`),
  overriding the global `new_skill_creation`; the mode is a deterministic gate
  (`Scheduler._category_mode` / `_category_locked`) checked in `_dispatch`,
  `_dispatch_skill`, `_author_skill`, and `restart_skill`, and toggled from the
  config file or `Scheduler.set_category_mode` (`/mode`, `/api/new-skill-mode`).
  `housekeeping` and the closed `response` tree are hard-denied in code
  (`HARD_LOCKED_CATEGORIES`/`CANNED_CATEGORIES`) and can never be changed. The
  mode is creation-only: existing skills still run and repair/regen of an
  existing leaf is allowed.
- **Real integrations, de-identified codegen, fidelity + repair** (Sep 2026;
  codegen privacy Oct 2026): the codegen endpoint is the primarily-cloud model,
  so it is shown **only a de-identified spec** — the trusted local `llm` authors a
  reusable `goal` (plus a generic service/transport) with every user-specific
  value stripped; the user's request, question answers, and run output never reach
  codegen. Bodies
  must perform the real action via stdlib transports with values from
  `ctx.config` and declare `INTEGRATION` (service/transport/config_vars); a
  fidelity gate (`authoring:fidelity`, a SemIf accept/reconsider decision over
  the whole body + an operation inventory) plus static declaration checks
  regenerates a simulated body once with the de-identified evidence bundle, then
  accepts it traced (never a hard authoring failure). A failed real run logs a
  SemIf repair choice (retry / repair_skill / ask_user / no_repair) surfaced in
  REPL/dashboard; repair writes carry a de-identified evidence bundle (a generic
  failure kind, the integration, the goal, and the previous body — never the run's
  action log/new state or the request) and are
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
  queue/dispatch pipeline; results, authoring questions, and repair
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
  while a reply is speaking (no barge-in yet); the wake word is required before
  every command and is held while the scheduler's single slot is still running
  the current request (`GatewayService.input_ready`), so a second command cannot
  start until the first reaches a result (or pauses for the user's answer). The
  whole voice stack is optional and lazy-imported (`requirements/voice.txt`,
  pinned separately from the engine set since it is hardware-dependent); the
  gateway refuses to start with an install hint. `bootstrap.sh --voice`
  installs the deps and downloads its models (the openWakeWord models land in
  the venv under `.runtime/venv`, the Piper voice under `.runtime/voice/tts`,
  the faster-whisper model in the `.runtime/hf` cache). See
  "### gateway (messenger intake)".
- **Custom Piper voice (train your own)** (Oct 2026): the TTS voice is
  replaceable with one fine-tuned on the user's own recordings. This is an
  optional, off-bootstrap pipeline (`scripts/voice-record.py` records an
  LJSpeech dataset; `scripts/voice-train.sh` fine-tunes from the checkpoint
  pinned in `pins.json` `voice` and exports ONNX on a GPU host;
  `semif_agent/voice_train.py` holds the neutral recording helpers).
  Deployment needs **no gateway code change**: `cli._build_gateway_adapter`
  anchors `tts.voice_dir` to `.runtime/voice/tts`, so a trained `<name>.onnx` +
  `<name>.onnx.json` dropped there is selected by setting
  `gateway.voice.tts.voice` to `<name>`. Only a real spoken reply proves it.
  The recorder has three modes: interactive (default), `--speak-prompts`
  (TTS reads each line) and `--read-prompts` (prints each line + beeps, no
  speech). Two field gotchas: a mic that re-enumerates (usually on a USB hub)
  drops the stream — reads take a `--read-timeout` so a take ends instead of
  hanging, and prefer a direct port; and prompt/reply audio needs the ALSA
  output unmuted (PortAudio has no volume control). See `docs/VOICE-TRAIN.md`.
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
- **Decision-engine latency.** A single SemIf forward pass over a small prompt
  can take seconds on CPU, and a trivial request pays several of them
  (actionability + category + scope + leaf + intent + assess), so routing and
  assessment dominate end-to-end agent latency — not audio I/O (STT/TTS are
  sub-second) and not the network. Before treating latency as a front-end
  problem, profile the engine: confirm thread utilization, whether the cost is
  prefill vs. per-option scoring, and whether a smaller/faster decision model or
  a shared-state (`score_shared`) path helps. `decision extra.timing`
  (`input_tokens`/`forward_seconds`/`total_seconds`) and the voice latency
  breakdown (`scripts/voice-latency.py`) are the measurement hooks.

## Constraints & gotchas (learned the hard way)

### Decision engine (providers)
- The decision engine is **pluggable**, selected per machine by `engine.provider`
  in config.json — the same provider pattern as `llm`/`codegen`. `"semif"`
  (default) is the local llama.cpp GGUF (`SemIfEngine`); `"winnow"` is a remote
  Winnow `/v1/systemone` typed-decision API (`semif_agent.winnow.WinnowEngine`,
  configured by the nested `engine.winnow` block). `cli.build_engine` picks the
  class; `SemIfEngine`/`WinnowEngine` both satisfy `engine.DecisionEngine`
  (`call`/`warm`/`loaded`/`generate`).
- **The engine is always real and every `EngineUnavailable` is fatal**,
  whichever provider is selected: `WinnowEngine` converts every network, HTTP,
  and malformed/unnormalized-distribution failure into `EngineUnavailable` so
  the scheduler's fatal path (and the codegen watchdog / LLM bridge's graceful
  paths) behave identically to the local engine. There is no fallback model.
- The agent's flat decision (`state` + `question` + typed `options`) maps onto
  Winnow's `choice` question type (`options` -> `criteria` id->description,
  `question` -> `instructions`); the returned distribution is validated against
  exactly the requested option ids and normalized. `generate` goes through the
  same server's `/v1/chat/completions`, so `llm.provider == "semif"` keeps
  working when the decision engine is Winnow.
- Swapping the decision model changes routing/assessment behaviour and the
  decision-log distribution; that is the intended per-machine choice. The
  decision log and `dream` cost are unchanged because `DecisionResult` is the
  same contract.
- Context differs by provider (the local engine's `engine.context_tokens` vs the
  remote server's own `--ctx-size`, mirrored in `engine.winnow.context_tokens`);
  a state that overflows is a fatal `EngineUnavailable`, the same class as the
  local token limit. Every engine exposes its window as `context_tokens` (a
  `DecisionEngine` property), which the scheduler uses to size generated
  artifacts — see "Artifact size budget".

### Environment
- The core is pure stdlib and runs anywhere Python does. Never pip-install heavy
  deps on a thin client; a full install belongs on a provisioned host.
- The SemIf engine runs via llama.cpp. It is **CPU by default**: SemIf's
  llamacpp backend hardcodes `n_gpu_layers=0`. Setting `engine.gpu_layers` in
  config.json offloads to a GPU when llama-cpp-python was built with a GPU
  backend (Vulkan/ROCm/CUDA). The agent wraps the backend's `_cpu_model_params`
  so the knob works without patching the re-cloned engine source, and on the GPU
  path also forces a full cache reset (`llama_memory_clear(mem, True)`): SemIf's
  metadata-only clear leaves stale hybrid-attention state on Vulkan, so repeated
  scores drift otherwise. **A GPU build is opt-in** — llama.cpp's Vulkan backend
  splits the graph across CPU and GPU even at `n_gpu_layers=0`, roughly halving
  CPU throughput, so a CPU-only host must not get one (`bootstrap.sh --gpu
  vulkan|rocm|cuda`, or set `engine.gpu_layers` and rerun, builds it; the default
  stays CPU-only). Offload needs the accelerator **to itself**: the 4B Q4 model
  needs ~2.9 GiB, so a GPU already holding another model (e.g. ollama) makes the
  engine slower than CPU and can abort the process with `vk::DeviceLostError`.
  ollama is a separate GPU user.
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
  llama-cpp-python 0.3.35, pytest). GPU offload is **opt-in**: set
  `engine.gpu_layers` and run `bootstrap.sh --gpu vulkan|rocm|cuda` (with
  `--gpu auto`, detection picks a backend when `engine.gpu_layers` is set),
  which installs the Vulkan build prereqs
  (`libvulkan-dev glslc spirv-headers glslang-tools`) when needed. A CPU-only
  host keeps the default CPU wheel — a GPU build is slower on CPU.
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
  APIs that change between versions. **GPU builds:** pip's wheel cache serves the
  prebuilt CPU wheel even when `CMAKE_ARGS` requests a GPU backend, so force a
  real source build with
  `--force-reinstall --no-cache-dir --no-binary llama-cpp-python` (bootstrap does
  this when the installed build reports no GPU offload).
- The pinned GGUF: `Qwen3.5-4B-Q4_K_M.gguf` from bartowski (2.8G), downloaded
  into `.runtime/models/` by bootstrap. Load ~34s; score ~0.9s/decision on CPU
  at 8 threads.

### configuration (split + secrets)

- **Three layers, mirroring Home Assistant.** Declarative structure is committed
  (`pack.json` manifests + the core schema); per-machine values are the split
  config; secrets are a separate gitignored file. `config.json` is a **thin
  index** (`{"include": ["config/core.json", "config/gateway.json",
  "config/bridges.json", "config/packs/<name>.json", ...]}`); fragments are
  deep-merged in order (later wins). `semif_agent/config.py` owns the loader.
- **Secrets.** A value written as `"!secret <key>"` resolves from the gitignored
  `secrets.json` at load (the analog of HA's `!secret`). A missing key resolves
  to `""` at load (so startup never hard-fails) and is reported by `check-config`.
  **`api_key_env` is gone** — `secrets.json` is the one place credentials live.
- **Schema owns defaults.** `semif_agent/config.py` `CORE_SCHEMA` (core sections)
  plus each pack's `config` schema (bridge blocks) own every default. The
  committed `config.example.json` is GENERATED from them
  (`check-config --write-example config.example.json`); secret keys are emitted
  as `"!secret <name>"`, never a real value. `check-config` validates a live
  config (type mismatches and missing secrets are errors; unknown keys warn).
- **Seeding.** `config-init` writes the split layout + an empty `secrets.json`
  from the schemas (bootstrap calls it once, only when `config.json` is absent);
  afterwards only a human edits it. `scripts/config_merge.py` is the stdlib-only
  merger bootstrap uses before the venv exists.

### ollama
- `bootstrap.sh` installs **no ollama**: it expects the target host to already
  run one for `llm` (and pulls `llm.model` into it), while `codegen` may be
  remote. Point `--llm-url`/`--codegen-url` at the right daemons.
- If generation hangs with no log output, restart the ollama service — the GPU
  runner can wedge.
- **A local reasoning model needs thinking off for `llm`.** A reasoning model
  emits hidden chain-of-thought and puts the answer in `content`. Through the
  `/v1` compat endpoint the CoT consumes the entire reply budget, so
  `generate_category`/`generate_skill_spec`'s small `max_tokens` returns
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
  `scripts/bootstrap.sh`. With no `config.json` yet it seeds the split config
  (`config-init` writes `config/core.json`, `config/gateway.json`,
  `config/bridges.json`, `config/packs/*.json`, the `config.json` index, and an
  empty `secrets.json`); edit those to set `llm.model`/`codegen.model` and the
  endpoints, and put credentials in `secrets.json`. Flags (`--llm-url`,
  `--codegen-url`, `--llm-model`,
  `--codegen-model`) override the values read from the merged config **for that run
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
  `simplex_chat` block (`version`, and an `assets` map of per-arch
  `bin_url`/`sha256`; bootstrap selects by `uname -m`, `arm64` normalized to
  `aarch64`). Per-machine
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
- Seeds the split config (`config-init`: `config/core.json`,
  `config/gateway.json`, `config/bridges.json`, `config/packs/*.json`, the
  `config.json` index, and an empty `secrets.json`) only when `config.json` is
  absent, and **never writes it again** — only a human edits it. The committed
  `config.example.json` is GENERATED from the core + pack schemas and ships the
  provisioned defaults, so a seeded config already has
  `codegen.timeout: 3600`, the SimpleX gateway enabled
  (`gateway.simplex.enabled = true`, `ws_url` from `simplex_chat.port`) and the
  standalone forwarding bridge enabled (`bridges.simplex.enabled = true`,
  `ws_url` from `simplex_chat.forward_port`) and
  the LLM bridge enabled (`bridges.llm.enabled = true`; it reuses the `llm`
  model);
  the operator sets `llm.model`/`llm.base_url` and `codegen.model`/
  `codegen.base_url`, and puts credentials in `secrets.json`. Bootstrap renders/enables four user services:
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
  Bootstrap then runs `packs/simplex/scripts/simplex-address.py` to create/print the command
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
- **Local engine as the `llm` provider: `"semif"`** (`provider.SemIfEngineClient`
  + `llm.SemIfLLMClient`). Set `llm.provider = "semif"` to generate the
  title/description (and any other `llm` text: gateway result cleanup, the LLM
  bridge) from the **already-loaded SemIf decision model** in-process, with no
  ollama endpoint and no `llm.model` — the pinned GGUF is the model. `engine.
  SemIfEngine.generate` autoregressively samples through the backend's low-level
  llama.cpp context (chat template with `enable_thinking=False`, KV cache
  cleared), holding the same lock as `call`, so a reply briefly serializes
  against routing/assessment on the one loaded model. The adapter is a drop-in
  sibling of `OpenAICompatClient` (same `chat(messages, max_tokens=...,
  temperature=..., ...)` surface; HTTP-only args accepted and ignored) and
  raises `LLMError` on any engine failure, so the scheduler's graceful
  `draft_failed` path and the LLM bridge's 502 path are unchanged. Knobs:
  `llm.semif.temperature` (0 = greedy) and `llm.semif.max_tokens`. Only the
  `llm` task supports it (codegen still needs a real code model).
- **Alternate hosted provider: OpenCode Console** (`console.py`). Either
  endpoint can run against the hosted, authenticated OpenAI-compatible Console
  inference API instead of local ollama: set `llm.provider` / `codegen.provider`
  to `"opencode"` (default `"ollama"`), set `model` to a **Chat Completions**
  family id (`kimi-k2.7-code`, `glm-5.3`, `deepseek-v4.1-flash`, `qwen3.8-max`,
  the free models, …), and set `api_key` to `"!secret <key>"` with the key in
  `secrets.json` (a literal `api_key` also works). `build_scheduler` maps `"opencode"` → `ConsoleLLMClient` /
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
- **Skill packs.** Each pack lives in its own repository (`pack.<name>`, listed
  in the committed `packs.json`); `scripts/bootstrap.sh` clones them into the
  gitignored `packs/<name>/`. A pack bundles a concern with its skills: a
  `pack.json` manifest (name, concern, `categories` → descriptions, `bridges`
  referenced by name, `requires`, and a `config` schema) plus
  `skills/<category>/<name>/` in the exact generated-skill folder format
  (`skill.py`, `contract.json`, `skill.test.py`, plus `manifest.json` with the
  description). `semif_agent/packs.py` discovers them (`PackRegistry`, scanning
  `skill_packs` roots for `*/pack.json`; a root with no manifest is one implicit
  pack), and `merge_packs` loads every **enabled** pack into the tree at startup.
  Pack enable is one switch: an explicit `packs.<name>.enabled`, else a bridge
  pack follows `bridges.<bridge>.enabled`, else `enabled_by_default`. Disabling a
  pack drops its categories entirely (including generated bodies authored into
  one) and stops its bridge. Recorded config (credentials collected at first
  fire) is written to the runtime store `data/skills/`, so a pack repo
  never holds secrets, and a generated body with the same name replaces the pack
  skill. The shipped packs: `nextcloud`, `simplex`, `outlook`, `browser` (each
  owning its bridge), and `essentials` (the built-in local utilities `time` +
  `calculator`; no bridge). `nextcloud.create_event` is the authoritative worked
  example (CalDAV PUT; derives the title/date/time from the query, asks for
  missing fields and resumes, resolves the target calendar with a SemIf choice
  over the owned calendars — a strong winner is used, otherwise the configured
  default); `nextcloud.next_event` is the read counterpart (Nextcloud CalDAV,
  recurring events expanded server-side); `nextcloud.create_task`
  adds a task to a Nextcloud task list through the Nextcloud bridge (`GET
  /tasklists` for a VTODO-capable target, `POST /tasks` to create it, the LLM
  bridge to phrase the summary/description); `simplex.next_message` (read via the
  forwarding bridge), `simplex.send_message` (send to a contact resolved over the
  bridge's contact list), and `simplex.connect_link` (show/create the forwarding
  bot's contact link) are the messenger pack skills; `time.now`, `time.date`,
  `time.set_timer`, and `time.set_alarm` are the local time utilities (`compute`
  transport, empty contract) — they read the host clock and schedule on
  `ctx.timers`. Local time is the top-level `timezone` config value (an IANA name
  such as `America/Chicago`; `timers.local_now`/`resolve_timezone` resolve it via
  `zoneinfo`), falling back to the host's own timezone when it is empty or
  unknown; `TimerService` formats due/fired times in that zone too. The nextcloud
  pack skills derive "now" the same way, so a request time is parsed in the
  user's zone rather than the host's; `calculator.calculate` is the
  local-arithmetic skill (`compute` transport, empty contract): bounded to two
  operands and one operator, the operator is a SemIf choice over addition/
  subtraction/multiplication/division/exponents/squares/cubes/square roots/cube
  roots/fractions/percentages, SemIf picks the operands when the request offers
  more numbers than the operator consumes, and the answer is spelled out in words
  for the ear (an inexact result is announced as "approximately ...");
  `tests/test_packs.py` keeps every pack skill honest.
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
  `llm` draft worker** (small model authors title+description+a **de-identified
  goal**) → stub registered + hot-merged into the tree → the body write is queued
  to a **separate single-slot codegen worker** (the 12G codegen model can't run
  twice), the gate stays free for new input. A category request chains the same
  way in one worker. The codegen worker runs the full authoring pipeline:
  1. **codegen body** (`generate_skill_body` / `regenerate_skill_body` on
     repair): CODEGEN.md + the **de-identified spec** (goal + generic
     service/transport) + tree + bridge catalog — never the request, question
     answers, or run output; the body is a
     single `act(ctx, request)`, reads every operational value from `ctx.config`,
     performs the real action, and declares `INTEGRATION` and its own flat
     `CONTRACT` (every contract key must be read from `ctx.config`, and every
     name the body references must be imported/bound — both enforced by
     `parse_skill_body`).
  2. **fidelity gate** (`authoring:fidelity` SemIf decision +
     `integration_findings`): a rapid sanity check (accept/reconsider) — does
     the body contain a concrete, side-effecting operation that really performs
     its goal (vs simulated/declared-but-unused)? The state carries the
     **whole** body plus a deterministic operation inventory and the
     de-identified goal (never the raw request), and asks the positive question;
     its threshold is its own `codegen.fidelity.tau`. It only triggers a
     rewrite, never a diagnosis. One corrective regen with the de-identified
     evidence bundle (`codegen.fidelity.max_attempts`, inventory + findings
     included), then accept with a trace rather than hard-failing authoring.
  3. **contract** (`parse_contract`): read from the body's own flat `CONTRACT`
     constant — a single flat object of snake_case variable name -> semantic
     description, for user input and SemIf only (no types/validation; that lives
     in the code + test). No separate generation call; `materialize_skill`
     rewrites `contract.json` as a mirror.
  4. **test** (`generate_skill_tests`): a shared-context call (seeded with
     TESTGEN.md + `skill.py` + the body's `CONTRACT`) produces `skill.test.py` —
     hermetic mechanics, loopback fixture injects the endpoint through config,
     fixtures inline.
  5. **auto-run test** (`run_skill_test`): subprocess in the skill folder,
     `codegen.test_timeout`; on failure a **2-option SemIf decision**
     (`codegen_regen`, trace-only) picks code/test to regenerate (the contract
     rides in the code), the error + existing files are fed back, bounded by
     `codegen.test_max_attempts`.
  On success the body is materialized (`materialize_skill`), hot-merged, the
  leaf's `writing` flag clears, and the **original request is re-queued** and
  re-runs navigation onto the new leaf (`skill_requeued`). The worker drains it
  with `run_queue_buffered`, so the re-run's reply lands in the scheduler's
  outbox and a front end (`drain_outbox`) routes it to the chat — the worker
  must not swallow the answer to the user's own request. On
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
- **Resolved-input observation.** The `assess:outcome` state includes the
  resolved contract values (`resolved inputs:` block), the bounded action log, and
  the run ledger (steps already taken) so the decision sees the full query →
  resolution → action → chain path; secret-named
  variables (pass/token/secret/key/credential/auth) are redacted to `***` so
  credentials never reach the decision log.
- **Codegen privacy (de-identified spec, no elicitation).** The codegen endpoint
  is the primarily-cloud model, so it is shown **only a de-identified spec**. The
  trusted local `llm` authors a reusable `goal` (plus a generic
  service/transport) from the request, with every user-specific value stripped:
  names, addresses, emails, phone numbers, dates, times, numbers, account names,
  hostnames, credentials, and quoted content are removed, and the body derives
  those from `request.text` at runtime. The user's request, question answers, and
  run output are **never** placed in a codegen prompt (body, retry, regen, or
  testgen); the repair/fidelity evidence bundles are reduced to a generic failure
  kind + integration + goal + previous body. The stub's `goal` is persisted in
  the registry so a restart/regen re-drives codegen without the request.
  Implementation questions were removed with this change: the spec's
  service/transport carries what the questions used to; the only remaining
  question queue is the repair `ask_user` path (local).
- **Fidelity gate.** After the body lands, `authoring:fidelity` — a real SemIf
  accept/reconsider decision (`P(reconsider) >= codegen.fidelity.tau` or any
  static `integration_findings` from the declared transport vs. real calls and
  declared config_vars actually read) decides whether the body really performs
  the action. **The state carries the whole body, never a prefix**, plus a
  deterministic `operation_inventory` (the HTTP verb + path, CLI, mail, file
  write, or timer it performs) and the de-identified goal; the raw request is
  **not** included (it added a remote-engine PII path and no signal — the goal
  is the reusable action). The question is positive ("is there a concrete
  operation that realizes the goal?"), which the model answers far more
  reliably than the old "does it simulate?" on a prefix. It never diagnoses: it
  only triggers a rewrite, which is handed the de-identified evidence bundle
  as-is (goal, description, raw INTEGRATION, findings, operation inventory,
  SemIf verdict+probs, previous body). A rejected body is rewritten once
  (`codegen.fidelity.max_attempts`, reason_kind `fidelity`); a body that still
  fails is accepted (never hard-fail authoring) but traced (`fidelity_review`,
  `performs_real_action=false`). A decision row is logged under phase
  `authoring:fidelity`. (An *undeclared* INTEGRATION is what the dashboard
  badges `unverified` — that is `integration_source`, not the fidelity verdict.)
- **Artifact size budget.** A decision ingests a generated artifact *whole*, so
  each is bounded: `parse_skill_body`/`parse_skill_test` reject an over-budget
  artifact with `ArtifactTooLong` (a hard ceiling, since it is a hard limit),
  which the codegen retry ladder answers with a shorter-artifact prompt. The
  ceiling is derived from the **active engine's context** —
  `engine.context_tokens` (local GGUF) or `engine.winnow.context_tokens`
  (Winnow; must mirror the server's `--ctx-size`) — minus
  `codegen.decision_reserve_tokens` (default 1024), the headroom for everything
  else in the fidelity state (goal, description, integration, findings,
  inventory, question, options, prefill margin), which is known because codegen
  runs immediately before the gate. `codegen.max_body_tokens` /
  `max_test_tokens` (0 = derive; a positive value overrides — lower it to avoid
  a long fidelity evaluation) set each artifact's budget, converted to chars by
  `chars_per_token`. This is why the gate never truncates and a body can never
  exceed what the gate can read.
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
  `python packs/simplex/scripts/simplex-address.py` then shows (creating on first run, via
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
  outbound)` starts an outbound-pump thread (`OutboundMessage` → `speak`) and an
  **inbound-worker thread**, and blocks in the mic loop. Utterances are handed to
  the worker, not run on the audio thread: the scheduler is synchronous and can
  block on the decision engine, so a slow task must not stop the mic being read,
  and the worker traces + speaks a task error instead of letting it kill the
  front end. **Capture-stream watchdog**: `_read_frame` reads with
  `audio.read_timeout_s`; when the stream stalls continuously for
  `audio.stall_timeout_s` (a wedged PortAudio/ALSA callback that never delivers
  another frame) it traces `voice_stream_stalled` and closes/reopens the input
  (`voice_stream_reopened`), so the loop recovers instead of blocking forever.
  **Endpointing tolerates the wake tail**: the wake detector can fire inside the
  wake phrase and the cue beep follows, so the first speech captured is the
  phrase's tail, not the command — until `vad.min_command_ms` of speech is in
  hand the trailing-silence endpoint uses `vad.initial_silence_ms`, so a natural
  pause before the command does not end the capture (then `vad.silence_ms`).
  **Half-duplex**: capture frames are discarded while a
  reply is speaking, so the agent never transcribes itself (no barge-in yet);
  the wake word is required before every command, and is **held while the
  scheduler slot is busy** (`GatewayService.input_ready`): a wake that fires
  while the current request is still running is ignored (traced
  `voice_wake_blocked`), so a second command cannot start until the first
  reaches a result. `max_speak_chars`
  truncates long replies before speaking, and a short sine `cue` beep plays on
  wake (optionally again on transcription) so the user knows they were heard.
  **Result-only output**: `VoiceAdapter.result_only` (default true; per-platform
  config `result_only` overrides) makes `GatewayService` speak just the skill's
  result line — scheduler bookkeeping is dropped and the `<skill>: ok —` wrapper
  is stripped — so the spoken output is the answer, not the scheduler's
  internals. Questions, repair offers, approvals, and timer notifications are
  still spoken.
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
  queue/navigation). The answer-side entry points (`answer`, `answer_question`,
  `resolve_repair`, `answer_approval`) are guarded like `submit_request`: an
  `EngineUnavailable` is fatal-and-reported and any other error returns a reply,
  so a resume/repair failure never propagates into a front end. A *different* chat — **including a chat on another
  platform** — during that pause is told to wait: the single-slot scheduler must
  not silently abandon the first chat's run, and one service fronts all
  adapters so the guard spans them. A background poll drains
  `Scheduler.drain_outbox()` (replies a background path buffered — the codegen
  worker's post-authoring re-run, a repair retry — so an off-gate completion
  still reaches its chat) then `Scheduler.run_queue()` (routing each
  `[run_id] summary` to its owner) and
  surfaces newly posted authoring questions / repair offers (`defer_questions
  = True`; the service routes them by `run_id` → `(platform, chat)`, falling
  back to that platform's `home_channel`); a chat's plain reply answers the
  question or picks the repair action (`retry`/`repair`/`ask`/`no`). The
  approval/repair vocabulary is matched **case- and punctuation-insensitively**
  (`_normalize_action`), with a first-word fallback: speech-to-text emits
  "Approve." / "Yes." / "retry.", and an exact lookup silently dropped those
  into `submit_request`, so the approval timed out and nothing was created.
- **Optional result cleanup** (`gateway/humanize.py`): a gateway-only
  presentation layer that rewrites a completed skill's result line from the
  scheduler's deterministic `<skill>: <ok|failed> — <action_log>` into one
  grounded, friendly sentence. It is **generation, never a decision** — the
  success/failure verdict and routing are already settled by SemIf and the
  runner, and the rewrite is never fed back into assessment, tracing, or the
  decision log (the runner and REPL keep the raw deterministic summary). The
  prompt forbids adding/inferring/dropping facts, and `Humanizer.humanize`
  always falls back to the original text on any error, timeout, empty reply, or
  runaway length, so a cleanup failure never swallows a real result.
  It reuses the scheduler's `llm` client (endpoint/model/sampler configured
  once) with a short per-call `timeout` (a new optional `chat()` argument) so a
  slow cleanup cannot hold the gateway poll loop for the authoring budget. Off
  by default: `gateway.humanize` (`enabled`/`timeout`/`max_tokens`/`max_chars`)
  sets the global default and a per-platform `gateway.<platform>.humanize`
  bool/`{enabled}` overrides it. A **skill's own recorded config**
  (`data/skills/<category>/<name>/config.json` -> `{"humanize": bool}`) wins
  over both, so a skill whose result is already speakable (e.g. `time.now`)
  skips the rewrite while a mechanical one keeps it; the gateway reads the
  skill ref from the structured `SchedulerReply.skill_ref` (the scheduler's
  `run_queue`/`submit_request`/`answer` carry `run_id`, `skill_ref`, and the
  unwrapped `result` alongside the human-facing `text` — no parsing it back out
  of the summary). Only `kind == "result"`
  lines are touched
  (questions, repair/approval notices, and status chatter pass through); like
  `result_only`, an enabled platform unwraps the result detail before cleaning.
  `_resolve_platforms` skips the reserved `humanize` key.
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
  `tests/test_humanize.py` covers the optional result cleanup with an injected
  fake `llm` client (rewrite, quote/whitespace normalization, fallback on
  error/empty/runaway, disabled passthrough, and the global +
  per-platform gateway enable/disable wiring).
  `tests/test_lxmf_transport.py` covers the neutral LXMF transport's
  missing-dep contract, default paths, and `normalize_message` (no real `lxmf`
  needed). `tests/test_simplex_ws.py` covers the neutral protocol layer (parse
  shapes, accept ids, corrId round-trips, address show/create).
  `tests/test_bridges.py`
  covers what stays in the core: the shared LLM bridge (over a real loopback
  server) and the discovery registry (`known_infos`/`describe_bridges`/
  `derived_config_vars`).
  `tests/test_packs.py` hermetically tests every
  `packs/<name>/skills/<category>/<leaf>/` package, including `simplex.next_message`,
  `simplex.send_message`, and
  `simplex.connect_link`, and runs every pack's bridge test
  (`packs/<name>/bridge/test_<bridge>_bridge.py`: the SimpleX bridge over a real
  loopback server — peek/pop, recipient resolution, outbound routing,
  token/body validation, `/address` success/503/502, daemon close on stop — and
  the Nextcloud/Outlook/Browser bridges over a loopback `http.server` with a fake
  service or driver).
  `tests/test_voice_gateway.py` drives the voice loop
  with injected fake engines (wake gating, endpointing, half-duplex,
  `speak` truncation/failure, adapter + CLI wiring) — no
  hardware, network, or heavy packages. `tests/test_voice_record.py` covers the
  recording helpers (`semif_agent/voice_train.py`) — prompt loading, WAV I/O,
  level math, the silence-endpointing loop, and the LJSpeech dataset writer —
  with a scripted fake audio source. The live `websockets` transport against
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

### skill packs (bundles)

- **The contract.** `PACKGEN.md` is the authoritative spec for a new pack —
  `pack.json`, its `config` schema, its bridge reference, and how to author a
  new bridge to back one. It is the sibling of `CODEGEN.md`/`TESTGEN.md`; read it
  before adding a pack (each pack is its own repository).
- **What they are.** A pack bundles a concern/product with the skills that serve
  it, in `packs/<name>/`: a `pack.json` manifest plus
  `skills/<category>/<leaf>/` folders (the exact generated-skill format). Each
  pack is its own repository (`pack.<name>`, listed in `packs.json`), cloned by
  `scripts/bootstrap.sh` into the gitignored `packs/<name>/`. The
  manifest binds the pieces that used to be scattered by convention:
  `concern`/`title`, `categories` (id -> scope description, feeding
  `CATEGORY_DESCRIPTIONS`), `bridges` (names the pack ships under `bridge/` and
  talks to; the core registry discovers them by convention), `requires`
  (optional deps like `playwright`), `enabled_by_default`, and a
  `config` schema. `semif_agent/packs.py` discovers them (`PackRegistry`, which
  scans the `skill_packs` roots for `*/pack.json`; a root with no manifest is one
  implicit pack, so a plain skills directory works) and `merge_packs` loads every
  **enabled** pack at startup.
- **Enable is one switch.** `pack_enabled` resolves: an explicit
  `packs.<name>.enabled`, else a bridge pack follows its
  `bridges.<bridge>.enabled`, else `enabled_by_default`. Disabling a pack drops
  its categories entirely (including any generated body authored into one) and
  stops its bridge — a host never offers a skill whose bridge is off. The switch
  lives in the pack's `config/packs/<name>.json` fragment.
- **Config is pack-owned.** A pack's `config` schema (key -> `type`/`default`/
  `doc`/`secret`) owns its defaults and drives both `config.example.json`
  generation and `check-config` validation. Bridge packs map the schema onto
  `bridges.<name>`; a bridge-less pack (e.g. `essentials`) maps onto
  `packs.<name>`. Shared, skill-less bridges (the LLM bridge) live in
  `semif_agent/config.py` `SHARED_BRIDGE_SCHEMA`.
- **Shipped packs.** `nextcloud`, `simplex`, `outlook`, `browser` (each owning
  its bridge) and `essentials` (the built-in local utilities `time` +
  `calculator`; no bridge) — each in its own repository. Adding a concern means
  adding a pack repo, not editing `skills.py`.

### bridges (third-party API services)

- **What they are.** A bridge service is a standalone process that stands up a
  simple, secure localhost HTTP layer in front of one third-party system,
  warranted when the system needs state (especially auth state), a native or
  heavy protocol, or one sanitized surface; a system with a plain HTTP API needs
  no bridge — its skill calls the API directly. When a bridge fronts a service, a
  codegen-authored skill body calls the bridge and never speaks that system's
  native protocol.
  A bridge **ships with its pack**: the pack declares it in `bridges` and ships
  `packs/<name>/bridge/<name>.py` (+ `bridge/__init__.py` and any sibling client
  module); the core registry imports it by convention — there is no core
  `CATALOG` to edit. The one shared, pack-less bridge is the LLM bridge, whose
  `config` block lives in the core `SHARED_BRIDGE_SCHEMA` (see
  `semif_agent/packs.py`). The catalog
  (`BridgeInfo`: name, service, description, URL config var, other config vars
  with one-line docs, auth header + token config var, and endpoints with
  request/response/error shapes) is injected into every codegen prompt
  (body/retry/regen/testgen) via `describe_bridges()` and is the
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
- **HTTP surface** (`bridges/base.py` + `packs/simplex/bridge/simplex.py`): JSON in/out,
  localhost-bound, optional shared-secret `X-Semif-Token` header. `SimplexBridge`
  buffers **every** inbound DM (no allowlist — the user wants to see who reached
  the bot through its invite link) in a bounded `MessagingInbox` that owns the
  read cursor (each buffered entry keeps the daemon `item_id` so a consumer can
  consume exactly that item), and serves `GET /health`, `GET /contacts` (the
  daemon's contact list, refreshed on (re)connect and on demand — best-effort,
  falling back to the cached/inbound-learned set when the daemon is unavailable),
  `GET /inbox` (peek), `GET /inbox/next?contact=<id>` (pop oldest), `GET /unread`
  (the daemon's persistent unread chats via `/_get chats` + `ChatListQuery`
  unread filter — `chatStats{unreadCount, minUnreadItemId}` plus each previewed
  `meta.itemStatus`), `GET /history?contact=<id>&count=<n>` (recent messages for
  one chat via `/_get chat @<id>`), `GET /address` (show/create the forwarding
  bot's contact link; 503 when the daemon is not connected, 502 when the lookup
  fails), `POST /send` `{"recipient","text"}` (resolves a numeric id or a known
  display name — refreshing the daemon contact list once on a miss — and
  enqueues on the bridge's own daemon), and `POST /read` `{"contact","item_ids"}`
  (marks exactly the listed received items read on the daemon via `/_read chat
  items @<id> <ids>`, so they leave its persistent unread state; `ok` only when
  the daemon accepted it). `/unread` and `/history` read the daemon's own state
  (surviving bridge restarts), unlike the live `/inbox` buffer; `/read` is a real
  local write on the bridge's own profile (v7 `/_read chat` updates the local
  `item_status` only; no wire read receipt).
- **Skill-facing config.** A body calls a bridge at the address derived from that
  bridge's own `bridges.<name>` block (`host`/`port`/`token`) — the
  `simplex_bridge_url` / `llm_bridge_url` / `nextcloud_bridge_url` var (+ `_token`)
  is injected into `ctx.config` by `build_scheduler` via
  `bridges.registry.derived_config_vars`, so there is no duplicate top-level key to
  keep in sync and the data-contract config search finds it and auto-populates the
  skill config. Skills reach a bridge with the ordinary `http`
  transport, so their hermetic tests are loopback HTTP like any other HTTP body.
  `simplex.next_message`, `simplex.send_message`, and `simplex.connect_link` are
  the pack skills.
- **LLM bridge** (`bridges/llm.py`). A generic `POST /chat`
  `{"messages": [{"role","content"}, ...], "max_tokens"?: int}` ->
  `{"text": "<model reply>"}` (plus `GET /health`, `400` bad body, `502` model
  error) in front of the agent's language model, so a body can ask for short
  generated text over HTTP instead of speaking the OpenAI-compatible protocol
  itself. It does **not** own the model connection: `run_bridge` passes the
  scheduler's already-configured `llm` client through `run_bridges`/`from_config`
  into `LLMBridge`, so the endpoint/model/sampler stay configured once in the
  top-level `llm` block; the `bridges.llm` block carries only the local listener
  (`enabled`/`host`/`port`/`token`). Bodies reach it via the derived
  `llm_bridge_url` (+ optional `llm_bridge_token`). It runs inside the existing
  `semif-bridge.service` process (all enabled bridges start together); with no
  client it reports `502`, never a fabricated reply. `nextcloud.create_event` is
  the first consumer (title + description extraction, cached in `request.meta`
  across a `needs_input` resume).
- **Nextcloud bridge** (`packs/nextcloud/bridge/nextcloud.py` +
  `nextcloud_client.py`).
  Wraps the account's native protocols — WebDAV files, CalDAV calendar events and
  tasks (VTODO), CardDAV contacts, and the OCS Notes API — behind one uniform
  JSON surface, so a body never speaks DAV/OCS directly. It has **no service
  daemon** (Nextcloud is plain HTTPS) and is stdlib-only (`urllib` + `xml.etree`
  + `ssl`), so `check_requirements` is always `(True, None)` and it starts
  instantly inside the existing `semif-bridge.service`. HTTP Basic auth uses the
  username + app password. The connection is configured in `bridges.nextcloud`
  (`url`/`username`/`app_password`, plus `default_calendar`/`default_addressbook`
  and `verify_tls`/`timeout`); `NextcloudBridge.from_config` passes the top-level
  `nextcloud_url`/`nextcloud_username`/`nextcloud_app_password`/
  `nextcloud_default_calendar`/`nextcloud_default_addressbook` as a fallback, so
  the same account the nextcloud pack skills collect is configured once and a block value
  wins. Bodies reach it via the derived `nextcloud_bridge_url` (+ optional
  `nextcloud_bridge_token`); the `BridgeInfo` declares only those two config vars.
  Routes (reads `GET`, mutations `POST`): files (`/files`, `/files/stat`,
  `/files/read` text-or-base64, `/files/search`, `/files/write|mkdir|delete|move|copy`),
  calendars (`/calendars` with each collection's `supported-calendar-component-set`
  as `components`, `/tasklists` for VTODO-capable task lists, `/calendars/events`
  list, `POST /calendars/events` + `/update` + `/delete`), tasks (`/tasks` list,
  `POST /tasks` + `/update` + `/complete` + `/delete`; task routes only target a
  VTODO-capable calendar and reject an event-only one), contacts (`/addressbooks`, `/contacts` list/search,
  `POST /contacts` + `/update` + `/delete`), notes (`/notes`, `/notes/get`,
  `POST /notes` + `/update` + `/delete`), and `/user` + `/capabilities` (OCS).
  A missing connection is `503`; a real failure (`NextcloudError` — transport,
  HTTP error, or bad reply) is `502`, never a fabricated success. Event/task
  updates edit the stored iCalendar properties in place (preserving VALARM/
  ATTENDEE), guarded by `If-Match`; creates use `If-None-Match: *`. Calendar/
  addressbook defaults resolve from config when a request names none and the
  account has several. `packs/nextcloud/bridge/test_nextcloud_bridge.py` drives
  the real client over a loopback `http.server` faking Nextcloud's DAV/OCS shapes.
- **Outlook bridge** (`packs/outlook/bridge/outlook.py` + `outlook_client.py`).
  Wraps the user's Microsoft account — Outlook / Microsoft 365 mail, calendar
  events, Microsoft To Do tasks, contacts, and OneDrive files — behind one
  uniform JSON surface over **Microsoft Graph**, so a body never speaks Graph or
  OAuth directly. Like Nextcloud it has **no service daemon** (Graph is plain
  HTTPS) and is stdlib-only (`urllib` + `json` + `ssl` + `base64`), so
  `check_requirements` is always `(True, None)` and it starts instantly inside
  `semif-bridge.service`. Authorization is the OAuth2 **device code flow** (a
  public client, no secret): `packs/outlook/scripts/outlook-auth.py` runs it once and stores
  the access + refresh tokens 0600 under the checkout's
  `.runtime/outlook/token.json` (override with `bridges.outlook.token_path`); the
  client refreshes on expiry/401. The bridge starts without a token — an
  unauthenticated route returns `503` telling the operator to run the helper
  (`OutlookAuthRequired`), a real Graph failure is `502` (`OutlookError`), and a
  missing `client_id` is `503`. The connection is configured in
  `bridges.outlook` (`client_id`, `tenant` default `common`,
  `default_calendar`/`default_task_list`/`default_mail_folder`/`default_timezone`,
  `verify_tls`/`timeout`); it also accepts `graph`/`login` base URLs (a test
  seam/defaulting to the real hosts). Bodies reach it via the derived
  `outlook_bridge_url` (+ optional `outlook_bridge_token`). Routes (reads `GET`,
  mutations `POST`): `/user`, mail (`/mail/folders`, `/mail/messages` list +
  `/mail/messages/get`, `POST /mail/send` immediate + `/mail/draft`), calendar
  (`/calendars`, `/calendars/events` via `calendarView` so recurrences expand,
  `POST /calendars/events` + `/update` + `/delete`), To Do (`/tasklists`,
  `/tasks`, `POST /tasks` + `/update` + `/complete` + `/delete`), contacts
  (`/contacts` search), and OneDrive (`/files` list, `/files/read`
  text-or-base64, `POST /files/write`). Graph's `@odata.nextLink` paging and
  `429`/`Retry-After` backoff are handled in the client.
  `packs/outlook/bridge/test_outlook_bridge.py` drives the real client over a
  loopback `http.server` faking the token endpoint and Graph shapes.
- **Browser bridge** (`packs/browser/bridge/browser.py` + `browser_client.py`).
  Real browser automation via **Playwright** — open a URL, read a page's text or
  links, take a screenshot, and click/fill/press/select to complete a flow —
  behind one uniform JSON surface, so a body never bundles a browser stack. It is
  the first bridge with a heavy optional dependency, so it is **disabled by
  default** and installed with `scripts/bootstrap.sh --browser`
  (`requirements/browser.txt` pins Playwright; the Chromium binary is downloaded
  into the checkout's `.runtime/browser`, and the bridge sets
  `PLAYWRIGHT_BROWSERS_PATH` there at launch). Unlike Nextcloud/Outlook it owns a
  browser and keeps a **persistent profile** (`.runtime/browser/profile`), so a
  login done once survives across calls and restarts — the browser analog of the
  SimpleX bridge owning its own daemon (`POST /reset` with `purge=true` wipes it,
  and `headless:false` shows the window so the operator can sign in).
  `check_requirements()` deliberately **always returns ok**: `run_bridges()`
  aborts *all* bridges when one's requirements fail and this dependency is
  optional, so a missing Playwright/browser must be a per-call `503` (with an
  install hint), never take down simplex/llm/nextcloud. Install is
  `scripts/bootstrap.sh --browser`; when `playwright install` fails (notably on a
  host with an IPv6 address but no global IPv6 route — Playwright's downloader
  forces `autoSelectFamily` + a dual-stack resolver that prefers IPv6 and hangs
  instead of falling back to IPv4), bootstrap falls back to
  `packs/browser/scripts/playwright-browsers.py`, which fetches the same pinned artifacts over
  IPv4 and lays them out where Playwright expects. `bridges.browser.channel` /
  `executable_path` instead launch an already-installed browser (a Playwright
  channel like `chrome`/`msedge`, or a specific binary) and skip the managed
  download entirely. Playwright's sync API is
  bound to its creating thread and the bridge serves each request on a new one,
  so the driver runs the **async** API on one dedicated event-loop thread and
  serializes whole operations behind an `asyncio.Lock`, submitting coroutines
  with `run_coroutine_threadsafe`. A real navigation/timeout/selector failure is a
  `502` (`BrowserError`), a missing package/browser a `503`
  (`BrowserUnavailable`), bad input a `400` — never a fabricated success. Routes
  (reads `GET`, mutations `POST`): `/health`, `/state`, `/content`
  (text|html, bounded), `/text`, `/links`, `/screenshot`
  (`full_page`/`selector`/`inline`, saved under `.runtime/browser/screenshots`),
  `/goto`, `/click` (selector or visible text), `/fill`, `/press`, `/select`,
  `/wait`, `/back`, `/forward`, `/reset`. Bodies reach it via the derived
  `browser_bridge_url` (+ optional `browser_bridge_token`). Three
  `packs/browser/skills/browser/*` skills ship (`open_page`, `read_page`, `screenshot`).
  `packs/browser/bridge/test_browser_bridge.py` exercises the routes over a
  loopback HTTP server with an injected fake driver (no browser, no Playwright
  needed).

## Bridge backlog (one session per item)

The bridge read path (`simplex.next_message`), outbound send
(`simplex.send_message`), and contact-link lookup (`simplex.connect_link`) are
covered. Deferred follow-ups:

- [x] **1. Contact-list refresh from the daemon** (`packs/simplex/bridge/simplex.py`,
  `simplex_ws.py`). Shipped: `SimplexDaemon.contacts()` /
  `request_contacts()` query `/_contacts <userId>` (active user from `/user` →
  `activeUser.userId`, cached, falling back to the configured `user_id`) over
  the existing corrId→Future path; `SimplexBridge` merges the result into the
  inbox on (re)connect via an `on_connected` hook and on demand (`GET
  /contacts`, and a send-resolution miss), so `/send` can address a contact it
  has never received from. Best-effort: a disconnected daemon keeps the cached
  set.
- [x] **2. Message history + true unread** (`packs/simplex/bridge/simplex.py`,
  `simplex_ws.py`). Shipped: `SimplexDaemon.chats(unread_only, count)` /
  `request_chats()` query `/_get chats <userId> count=<n>
  {"type":"filters","favorite":false,"unread":true}` and parse
  `apiChats` (`AChat.chatStats{unreadCount, minUnreadItemId, unreadChat}` +
  `chatItem.meta.itemStatus` `rcvNew`/`rcvRead`); `chat_history(contact_id,
  count)` / `request_chat_history()` query `/_get chat @<id> count=<n>` and
  parse `apiChat`. `SimplexBridge` exposes `GET /unread` (persistent unread
  chats, summaries + previewed messages) and `GET /history?contact=<id>&count=<n>`
  (recent messages with per-item status), degrading like `/address` (503 not
  connected / 502 lookup failure; `/history` 400 without a contact), plus
  `POST /read` `{"contact","item_ids"}` which marks exactly the listed items read
  on the daemon (`/_read chat items @<id> <ids>`), so a reported message is
  consumed and not re-reported. The live `/inbox` buffer is unchanged. (The
  pinned v7.0.3 **does** have `/_read chat @<id>`, `/_read chat items @<id>
  <ids>`, and `/_read user <uid>`; an earlier note claiming v7 had no mark-read
  command was wrong.)
- [x] **3. `simplex.send_message` pack skill** (`packs/simplex/skills/simplex/send_message/`). The
  outbound counterpart to `simplex.next_message`: resolves the recipient with a
  SemIf sub-decision over the bridge's `/contacts` (falling back to the
  configured default contact), writes the message from the request or the LLM
  bridge when no literal text is given, sends via `POST /send`, and reports the
  bridge's `contact_id` / errors honestly.
- [x] **4. LLM bridge** (`bridges/llm.py`). Shipped: `LLMBridge` exposes a
  generic `POST /chat` in front of the scheduler's configured `llm` client
  (threaded through `run_bridges`/`from_config`, never re-configured), the one
  shared core built-in, seeded under `bridges.llm` (its URL is derived from that
  block). `nextcloud.create_event` extracts its title + description through it.
- [x] **5. More bridges.** Each new third-party API gets its own pack
  (`packs/<name>/bridge/<name>.py`) + config block; the core registry discovers it
  by convention and the codegen prompts pick it up automatically
  through `describe_bridges()`. Nextcloud shipped (`packs/nextcloud/bridge/nextcloud.py` +
  `nextcloud_client.py`): files/calendars/tasks/contacts/notes over one
  JSON surface, no daemon, connection from `bridges.nextcloud` with a top-level
  `nextcloud_*` fallback. Future bridges follow the same shape.
- [x] **6. Outlook / Microsoft Graph bridge** (`packs/outlook/bridge/outlook.py` +
  `outlook_client.py`). Shipped: the `OutlookBridge` exposes mail,
  calendar, Microsoft To Do, contacts, and OneDrive over one JSON surface in
  front of Graph; OAuth2 device-code + refresh (`packs/outlook/scripts/outlook-auth.py`, tokens
  under `.runtime/outlook/token.json`); seven `packs/outlook/skills/outlook/*` skills
  (`next_message`, `send_message` confirm-gated, `next_event`, `create_event`,
  `create_task`, `list_contacts`, `list_files`). Deferred: writing/reusing
  Outlook app-registration guidance in the docs, and a mail-send seed that
  attaches files.
- [x] **7. Browser bridge (Playwright)** (`packs/browser/bridge/browser.py` +
  `browser_client.py`). Shipped: `BrowserBridge` exposes open/read
  (text/links/screenshot) and act (goto/click/fill/press/select/wait/back/
  forward/reset) over one JSON surface in front of a real Playwright browser
  with a persistent profile; optional heavy dep installed via
  `requirements/browser.txt` + `bootstrap.sh --browser`, lazy so a missing
  browser is a per-call 503 and never aborts the other bridges; the async API
  runs on one dedicated event-loop thread. Three `packs/browser/skills/browser/*` skills
  (`open_page`, `read_page`, `screenshot`). Deferred: arbitrary JS `evaluate`,
  file upload/download, multiple tabs/contexts, and a per-URL allowlist.

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
- Queue ordering: strict FIFO (arrival order). There is no urgency score and no
  preemption; the running process always finishes before the next queued request
  is dispatched.
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
  extraction/findings, and body store round-trips; `tests/test_housekeeping.py`
  for the housekeeping operations (name resolution, store/registry deletion,
  tombstones, the lock gate, cancel/regen, and the meta-skill act flows); and
  `tests/test_llm.py` for
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