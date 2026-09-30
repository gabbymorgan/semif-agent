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
that are not tasks fall through navigation into the closed `response` tree of
canned replies. Every SemIf decision is logged as a labeled training row; the
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
                new leaf when the body lands; the body worker runs the full
                pipeline: elicitation ->
                codegen body -> fidelity gate -> data contract -> test ->
                auto-run test (3-option SemIf regen ladder on failure); config
                search auto-populates the skill config from global/category
                config; elicitation asks implementation questions (all front ends
                defer via the question queue, `wait_timeout`); a
                failed real run triggers a logged SemIf repair choice
                (retry/repair_skill/ask_user/no_repair) surfaced to the user
queue.py        urgency max-heap (desc weight, FIFO seq), age pulls toward 1.0
skills.py    tree + registry (hardcoded built-ins: the closed `response` canned
                tree only), navigation = SemIf choices per level (logged),
                create_category
                and create_skill author + register stubs via the small `llm`
                provider (separate from codegen); SkillStore persists one folder per skill
                (skill.py, skill.test.py, contract.json, config.json) and
                materialize_skill / merge_skill_store
                hot-load runnable skills from data/skills/;
                merge_seed_store loads committed starter skills from seeds/ with
                their recorded config read from data/skills/;
                ActionResult.needs_input pauses a run for human input;
                Skill.integration / integration_source read the body's
                INTEGRATION declaration (or infer it);
                resolve_skill_config / unresolved_variables drive the tiered
                config merge (global -> category -> skill) + pre-predict
                contract collection
skill.py        loop: observe -> predict -> act -> observe -> assess;
                assess is a SemIf decision (`assess:outcome` success/failure at
                tau; on failure `assess:requeue` complete/retry) and the run
                summary is deterministic (no generation), built from
                category.skill + ok/failed + action_log;
                a run paused for input is resumed by re-invoking act with the
                answer on request.user_input (predict is never re-run); a
                contract variable the runner cannot satisfy pauses BEFORE
                predict (pre_predict), collects it, and re-runs the full path
engine.py       SemIfEngine -> semif_phase1.llamacpp_backend (lazy import)
codegen.py      CodegenClient (OpenAI-compatible) writes real-integration skill
                bodies against SKILL.md (real actions via stdlib transports,
                data from the runner via ctx.config, never embedded; bodies
                declare INTEGRATION service/transport/config_vars);
                parse/validate (compile + predict/act) + parse_integration /
                infer_integration / integration_findings; the bridge catalog
                (describe_bridges()) is injected into the body, retry, regen,
                elicitation, and testgen prompts — it is the single source of
                bridge specifics (service, base-URL var, auth header/token var,
                config-var docs, endpoints with error shapes), while SKILL.md/
                TESTGEN.md carry only the generic pattern; the advisory
                elicitation integration hint rides on every body/retry/regen
                prompt; elicitation questions
                + integration hint; TESTGEN.md drives two shared-context calls
                producing contract.json then skill.test.py (hermetic mechanics
                test: inline fixtures, loopback http.server for HTTP bodies, no
                external network, no mock_data.json); run_skill_test executes
                the test as a subprocess
llm.py          small OpenAI-compatible provider (LLMClient) that authors a
                new category/skill title + description — its own endpoint/model,
                deliberate separate from codegen; provider logic (SSE/budget/idle)
                is shared from provider.py; `_parse_json` is also borrowed by the
                authoring parsers
provider.py    shared OpenAI-compatible chat transport (OpenAICompatClient:
                SSE stream, token budget, idle watchdog, degeneration hook)
                subclassed by llm.LLMClient and codegen.CodegenClient
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
                chat). simplex.py: SimplexAdapter — command simplex-chat daemon
                over its JSON WebSocket API (lazy `websockets`, allowlist,
                batching, structured `/_send`). Run with `python -m
                semif_agent.cli gateway [--dashboard]`; config under
                `gateway.simplex`. The gateway MUST NOT read history, show/create
                invite links, or compose messages — see "gateway isolation"
                below.
bridges/        standalone third-party API bridges (SimpleX first). base.py:
                BridgeInfo + BridgeService (shared localhost JSON HTTP layer,
                `X-Semif-Token` guard). inbox.py: MessagingInbox (bounded FIFO +
                contacts, bridge-owned read cursor). simplex.py: SimplexBridge —
                owns its OWN simplex-chat daemon/profile (separate from the
                gateway's), buffers every inbound DM, serves invite-link/read/
                send. registry.py: CATALOG, describe_bridges() (catalog injected
                into the codegen prompts), run_bridges(). Run with `python -m
                semif_agent.cli bridge [--name NAME]`; config under `bridges`.
simplex_ws.py   neutral SimpleX daemon protocol shared by the command gateway
                adapter and the bridge (parse direct text across v7 shapes,
                structured `/_send`, corrId→Future round-trips, contact-address
                `/_show_address`/`/_address`, contact-request accept). Knows
                nothing about the scheduler or either front end.
```

## Run / verify

Dev machine is a thin client (no GPU, ~1.4G disk): only pure stdlib unit tests
run here (`python3 -m pytest tests/ -q --ignore=tests/integration`).
Integration tests run on the staging machine `jarvis` (see "### jarvis (staging)").

## Git / sync

- Canonical repo lives on Gitea: `git.manyworlds.fit`, **SSH on port 222**
  (`ssh://git@git.manyworlds.fit:222/gabby/semif-agent.git`). Key
  `~/.ssh/id_ed25519` is registered there. The staging machine `jarvis` runs
  from a manual clone at `~/repos/semif-agent` (`scripts/bootstrap.sh` provisions
  everything else and never re-clones the agent repo); the old box working copy
  is also at `~/repos/semif-agent` (its editable install still points at the moved
  `~/semif-agent`, so `import semif_agent` is broken there — moot, guppy runs
  ollama only now); the dev machine at `~/Repos/semif-agent`. Push/pull from
  Gitea — never rsync/tar the code.
- **`config.json` is gitignored and per-machine** (dev and the box use different
  engine/LLM paths). Copy `config.example.json` to `config.json` and edit.
  `data/decisions.jsonl`, `data/runs.jsonl`, and `data/drafts/` are runtime
  artifacts and gitignored too.
- The Gitea instance was rebuilt fresh (Sep 2026) after its git pack transfer
  broke (`sh: bad option '--oneshot'` — a stray system `uploadpack.packObjectsHook`
  killed pack generation; the old AGENTS.md note blaming `authorized_keys` was
  a misdiagnosis). Both the dev key (`shitass@nunya`) and the box key
  (`guppy@semif-agent`) are re-registered; clone/fetch/push all work now. If a
  fresh machine can't pull, the durable fix is on the Gitea host:
  `git config --system --unset-all uploadpack.packObjectsHook`.
- Decision rows logged before the `run_id` threading landed show up under
  run_id `"?"` in the dashboard — that's expected, not a bug.

Machine split (Sep 2026): **guppy** hosts the remote **codegen** model (the
12G `qwen38-iq3s`) only; **jarvis** is the staging/test target (semif-agent +
SemIf engine + deps) and runs its **own local ollama** serving the small
title/description model (`qwen3.5:4b`). Only codegen traffic travels to guppy.
Provision jarvis with `scripts/bootstrap.sh` (see "### jarvis (staging)").

**Connecting to jarvis (`ssh jarvis@192.168.8.130`)**: this is a plain `ssh`
call. The key `~/.ssh/id_ed25519` is passphrase-protected and Linux Mint pops an
askpass dialog for it; the flatpak overlay
(`SSH_AUTH_SOCK=/run/flatpak/ssh-auth`) blocks the user's password locker, so
they cannot paste from it and must type the passphrase by hand. **Warn the user
right before running the command** ("about to ssh to jarvis — be ready to type
your passphrase") so they can enter it, then just run `ssh`. Do **not** burn
tokens on ssh-agent/askpass/fixed-socket workarounds — once the user types the
passphrase at the prompt, plain `ssh` works for the session. The `agent refused
operation` error from the flatpak socket is exactly this prompt, not a broken
key.

Guppy key facts:

- ssh key `~/.ssh/id_ed25519` is passphrase-protected. Load it into an agent at
  a fixed socket before connecting (the default flatpak `SSH_AUTH_SOCK` refuses):
  ```sh
  SOCK=/tmp/opencode/ssh-agent.sock; rm -f "$SOCK"; eval $(ssh-agent -a "$SOCK")
  printf '#!/bin/sh\necho "<PASSPHRASE>"\n' > /tmp/opencode/askpass.sh; chmod 700 /tmp/opencode/askpass.sh
  SSH_ASKPASS=/tmp/opencode/askpass.sh SSH_ASKPASS_REQUIRE=force setsid -w ssh-add ~/.ssh/id_ed25519
  ```
  The agent dies if this machine restarts; redo it each session.
- The agent on guppy is **deprecated** (guppy is ollama-only now). The box
  venv's editable install (`__editable__.semif_agent_0_1_0_finder.py`) still
  maps to `/home/abby/semif-agent`, which was moved to `~/repos/semif-agent`,
  so `import semif_agent` fails there — expected, not a bug. Run the agent on
  **jarvis** instead (below).
- Dashboard: runs as a systemd **user** service on the box
  (`semif-dashboard.service`, linger enabled, binds `0.0.0.0:8765`), so it's up
  after reboots with no manual launch — browser UI at http://192.168.8.181:8765/.
  Manage it with `systemctl --user status/restart semif-dashboard.service`.
  Note the submit/relabel POST endpoints are therefore open to the whole LAN.
  It works in live mode on the box (submit runs the real engine + LLM) and in
  replay mode anywhere (`--replay`; reads decisions.jsonl + runs.jsonl; submit
  degrades to a JSON error without the engine). Relabeling in the UI writes a
  human override (3x weight in dream) via `POST /api/relabel`. On the dev
  machine, replay mode: `cd ~/Repos/semif-agent && python3 -m semif_agent.cli
  dashboard --replay` (binds 127.0.0.1:8765).
- Integration tests (real engine + real LLM) only run on the box:
  `~/semif-venv/bin/python -m pytest tests/integration -q -s`
  They take ~100s (model load ~34s). Run them in the background and poll —
  long-lived ssh sessions get SIGHUP'd and kill the run.

## Roadmap

### v1 (done)
Core loop, urgency queue, skill tree, skill loop with real SemIf assessment
(decision, not generation), decision logging, `dream` cost pass, REPL + JSONL
CLI, unit tests (24) + box integration tests (2).

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
  OpenAI-compatible model (`codegen`, default `qwen38-iq3s`) writes
  `predict`/`act` code against `SKILL.md`, then a contract + test
  (against `TESTGEN.md`) are generated and the test is auto-run before the leaf
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
  **Create-branch confidence gate** (Sep 2026): `create_skill`/`create_category`
  compete in a softmax with the real options, so a weak plurality win is not
  evidence that nothing matches. Navigation fires a create branch only when
  `P(create) >= navigation.create_tau` **and** it leads the best existing option
  by `navigation.create_margin`; otherwise it falls back to the best existing
  option (a genuinely unmatched action still reaches authoring through the
  intent guard) and traces `create_suppressed` with the probs. An empty
  tree/category still short-circuits straight to create.
- **No handle/ignore gate; the `response` tree is the catchall** (Sep 2026):
  the top-level `_contains_request` handle/ignore gate is gone. Every input is
  scored and dispatched; inputs that are not tasks fall through navigation into
  the closed `response` category, a hardcoded tree of canned replies
  (`response.greeting`, `response.thanks`, `response.acknowledge`,
  `response.farewell`, `response.affirm`, `response.unable`, and the catchall
  `response.clarify` — "Could you try being more specific?"). `response` is a
  `CANNED_CATEGORIES` member: `navigate` never offers `create_skill` there (the
  intent guard is skipped too), `Scheduler._dispatch_skill`/`restart_skill`
  refuse to author it, and the runner skips `assess:outcome`/repair for it — a
  canned line has no side effect to assess. The leaf choice logs phase
  `navigate:response` and traces `response_selected`. `response.reject` (a stub
  that never ran) is gone.
- **Real integrations, implementation questions, fidelity + repair** (Sep 2026):
  elicitation is on by default and asks implementation questions (which
  service/account, how to connect, where the credential comes from, what success
  looks like); every front end defers via the question queue
  (`/api/questions`, worker waits `codegen.elicitation.wait_timeout`). Bodies
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
  the contract `contract.json` (single flat object of variable name -> semantic
  description, for user input + SemIf only) lands after the codegen body. The
  config step searches a tiered source of truth — global config → category
  config → skill config — via SemIf `choice` per variable, reusing an existing
  source when one matches (`config:search`, decision-logged) and auto-populating
  the skill `config.json`. Variables with no match are asked of the human AFTER
  code generation, on first fire, via a pre-predict `needs_input` pause; each
  answer gets a SemIf `record-as-config vs ask-again-each-fire` choice
  (`config:record`). On successive firings only unresolved variables are asked.
  **Shipped with item 1 (Sep 2026).**
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
- **Messenger gateway (SimpleX first)** (Sep 2026): `python -m semif_agent.cli
  gateway` runs a dedicated process that connects to the local `simplex-chat`
  daemon over its JSON WebSocket API and feeds authorized DM text through the
  normal gate/score/queue/dispatch pipeline; results, authoring questions, and
  repair offers are sent back to the originating chat. `gateway/base.py` is the
  transport contract (`GatewayAdapter`, `InboundMessage`, `OutboundMessage`),
  `gateway/service.py` the scheduler glue (single execution slot, owner maps,
  pending-run ownership, queue drain), `gateway/simplex.py` the SimpleX adapter
  (lazy `websockets`, default-deny allowlist by contactId/display name,
  rapid-message batching, structured `/_send`). DMs only for the first cut;
  groups/attachments/reactions are future work. See "### gateway (messenger
  intake)".
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
- **Dev box**: Python 3.13, GTX 780M (Kepler, useless), ~1.4G disk free. Never
  pip-install heavy deps here.
- **guppy box**: Python 3.14, AMD RX 6950 XT (gfx1030), 32 cores, 30G RAM,
  passwordless sudo. Now serves **codegen only** (`qwen38-iq3s`). SemIf runs via
  llama.cpp **CPU** backend (its llamacpp backend forces `n_gpu_layers=0`), so
  the GPU is NOT used by the decision engine — it IS used by ollama.
- **jarvis box**: runs the agent and its **own local ollama** serving the small
  title/description model (`qwen3.5:4b`). `config.json` `llm.base_url` points at
  `http://127.0.0.1:11434/v1`; only `codegen.base_url` points at guppy.
- The SemIf tokenizer is fetched from HF (`Qwen/Qwen3.5-4B` at the pinned
  revision), cached under each checkout's `.runtime/hf` (`HF_HOME`).

### SemIf install (box)
- **Legacy layout** (predates the `.runtime/` containment rule; kept for the
  deprecated guppy agent venv). New machines use `scripts/bootstrap.sh`, which
  installs the venv/engine/GGUF/cache under the checkout's `.runtime/` — see
  "### jarvis (staging)" and "Code principles".
- SemIf hard-pins `torch==2.10.0`, `numpy==2.2.6`, etc. The llamacpp path does
  **not** need torch (torch is imported lazily inside `direct.score`). Install
  with `--no-deps` and bring only what's needed — the committed manifest
  `requirements/staging.txt` is the single source of truth:
  `pip install -e ~/semif --no-deps`, then
  `CMAKE_BUILD_PARALLEL_LEVEL=6 MAKEFLAGS=-j6 pip install -r requirements/staging.txt`
  (numpy 2.3.5, transformers 5.17.0, tokenizers 0.23.2, huggingface-hub 1.31.0,
  llama-cpp-python 0.3.35, pytest).
- **Expected pip warnings:** pip reports "dependency conflicts" against
  semif-phase1's declared requirements (torch/accelerate/protobuf/sentencepiece
  not installed, numpy 2.2.6 vs 2.3.5). These are informational — the box runs
  exactly this set — not a bug; do not "fix" them by installing torch.
- `numpy==2.2.6` has **no cp314 wheel** → pip tries a source build that fails
  without `pkg-config` + `python3-dev`. Use numpy 2.3.5 (has cp314 wheels).
- `llama-cpp-python==0.3.35` builds from source. With all 32 cores it OOM-kills
  gcc (`internal compiler error: Segmentation fault`). Limit parallelism:
  `CMAKE_BUILD_PARALLEL_LEVEL=6 MAKEFLAGS=-j6 pip install llama-cpp-python==0.3.35`.
  Do NOT bump the llama-cpp-python version — SemIf calls specific llama.cpp C
  APIs that change between versions.
- The pinned GGUF: `Qwen3.5-4B-Q4_K_M.gguf` from bartowski (2.8G) at
  `~/models/`. Load ~34s; score ~0.9s/decision on CPU at 8 threads.

### ollama
- **guppy = codegen host.** Installed at `/home/abby/ollama/bin/ollama` (not on
  PATH), systemd service `ollama.service`, ROCm backend with
  `HSA_OVERRIDE_GFX_VERSION=10.3.0` and KV cache q4_0 + flash attention. Serves
  `qwen38-iq3s` (12G 27B, the codegen model). Binds `OLLAMA_HOST=0.0.0.0`, so
  jarvis reaches it as a plain remote API at `http://192.168.8.181:11434` (no
  auth — LAN-visible, same exposure as the old dashboard POST endpoints).
- `semif-hermes` / `semif-hermes-v3` are for ANOTHER project (hermes agent) —
  ignore them; they spew "token repeat limit" errors.
- If generation hangs with no log output, restart the service
  (`sudo systemctl restart ollama`) — the ROCm runner can wedge.
- **jarvis = local small model.** Runs its own ollama (`/usr/local/bin/ollama`,
  systemd `ollama.service`, `127.0.0.1:11434`) serving `qwen3.5:4b` for
  new-skill title/description authoring (~3s). `bootstrap.sh` ensures
  that model is pulled; `config.json` `llm.base_url` points here.
- `bootstrap.sh` installs **no ollama**: it expects the target box to already
  run one for `llm` (and pulls `llm.model` into it), while codegen is remote.

### jarvis (staging)

- Provision a fresh Ubuntu machine into a running staging box: clone
  `semif-agent` to a checkout (jarvis keeps it at `~/repos/semif-agent`; the
  script derives all paths from wherever it is run, so any path works — just
  register its SSH key on Gitea first), then
  `scripts/bootstrap.sh --llm-url http://127.0.0.1:11434 --codegen-url http://192.168.8.181:11434`.
  The script must be run from a checkout — it reads pins from that checkout's
  `config.example.json` and never re-clones the agent repo (only the SemIf
  engine and the simplex-chat binary). Idempotent and rerunnable; every stage
  no-ops on existing state, so it also boots an unknown-state machine. It
  installs **no ollama**: `llm` is this box's local ollama (bootstrap pulls
  `llm.model`), only `codegen` points at the peer guppy. It pulls the pinned
  small model into the local ollama and WARNs (never auto-pulls) if the remote
  codegen host is missing.
- **All installation artifacts live inside the checkout under a gitignored
  `.runtime/`** (`venv/`, `engine/`, `models/`, `hf/`, `bin/simplex-chat`,
  `simplex/`, `systemd/`), so an end user can find and debug the whole stack in
  one tree. Only operationally-forced artifacts live outside: the SSH key
  (`~/.ssh`) and the real systemd user dir + linger (the rendered units are
  stored in `.runtime/systemd/` and symlinked into `~/.config/systemd/user/`).
  See the "Code principles" containment rule.
- All pins are read from `config.example.json`: the `engine` block
  (`semif_repo` public GitHub `TheoLeeCJ/SemIf`, `semif_ref` pinned commit,
  `gguf_url`/`gguf_sha256`, HF tokenizer `source`/`revision`) and the
  `simplex_chat` block (`version`, `bin_url`, `sha256`, `port`,
  `display_name`). The python dep pins live in `requirements/staging.txt`
  (committed, one versioned artifact — jarvis, guppy, and any future box all
  provision from it). **Maintenance**: bump the pins in `config.example.json` /
  `requirements/staging.txt`, rerun the script, re-run the integration tests.
  The script never guesses.
- pip prints **expected** "dependency conflict" warnings at install time
  (semif-phase1 declares torch/accelerate/protobuf/sentencepiece/numpy 2.2.6
  that we intentionally do not install — the llama.cpp CPU path doesn't need
  them; numpy 2.3.5 is deliberate, 2.2.6 has no cp314 wheel). Same as the box;
  do not "fix" them by installing torch.
- Generates `config.json` with `llm.base_url` = `--llm-url` (this box, local) and
  `codegen.base_url` = `--codegen-url` (the remote codegen host; `codegen.timeout:
  3600` since it travels the LAN), a repo-relative `.runtime/models` GGUF path,
  and a backup of any prior file. `--threads N` overrides engine threads;
  `--copy-data SRC` rsyncs guppy's `data/` for continuity; `--public-dashboard`
  binds the dashboard to `0.0.0.0`.
  It also enables the SimpleX gateway (`gateway.simplex.enabled = true`,
  `ws_url` from `simplex_chat.port`) and the standalone forwarding bridge
  (`bridges.simplex.enabled = true`, `ws_url` from `simplex_chat.forward_port`,
  top-level `simplex_bridge_url`), rendering/enabling four user services:
  `semif-simplex.service` (the command `simplex-chat` bot daemon,
  `--create-bot-display-name` on `simplex_chat.port`), `semif-gateway.service`
  (`.runtime/venv/bin/python -m semif_agent.cli gateway`),
  `semif-simplex-forward.service` (the bridge's **own** daemon/profile on
  `simplex_chat.forward_port`) and `semif-bridge.service`
  (`.runtime/venv/bin/python -m semif_agent.cli bridge`). `--simplex-allowed-users
  CSV` / `--simplex-home-channel ID` / `--simplex-display-name NAME` populate
  the gateway allowlist/fallback/identity; with an empty allowlist the gateway
  denies everyone (the safe default until the human adds their contact id).
  Bootstrap then runs `scripts/simplex-address.py` to create/print the command
  bot's contact address (and the forwarding bot's, via `--ws-url`).
- Run / verify on jarvis:
  ```sh
  REPO=~/repos/semif-agent && cd "$REPO" && export HF_HOME="$REPO/.runtime/hf"
  "$REPO/.runtime/venv/bin/python" -m semif_agent.cli run              # REPL
  "$REPO/.runtime/venv/bin/python" -m semif_agent.cli dream            # cost report
  "$REPO/.runtime/venv/bin/python" -m pytest tests/integration -q -s   # ~100s, background+poll
  "$REPO/.runtime/venv/bin/python" -m semif_agent.cli dashboard --port 8765
  systemctl --user status semif-simplex semif-gateway semif-simplex-forward semif-bridge   # gateway + bridge stack
  ```
- First run downloads the 2.8G GGUF and builds `llama-cpp-python` from source
  (~10 min on 6 cores); reruns are fast no-ops.

### codegen (skill bodies, box)
- Skill **bodies** are written by a separate OpenAI-compatible model, configured
  under `codegen` in config.json (default model `qwen38-iq3s`, the 12G 27B
  IQ3_S GGUF — huge/slow). Title + description for new skills come from the
  separate small `llm` provider (its own endpoint/model, shared transport in
  `provider.py`); only the runnable code body uses codegen.
- **Do NOT cap `max_tokens`** on the codegen call. qwen38-iq3s reasons first
  and a cap truncates the hidden reasoning, leaving `content` empty
  (`finish_reason: length`) and the body write fails with "skill body is
  empty". Unbounded, it runs to completion in ~25–45 min with the card
  sampler (~28–70k tokens of reasoning then the code, ~25–40 tok/s); the
  client reads only `content`, so reasoning is filtered automatically. The
  client default timeout is 1200s and qwen38-iq3s routinely exceeds it —
  the box `config.json` sets `codegen.timeout: 3600`. Raise `codegen.timeout`
  in config if a harder prompt needs more.
- Bodies are persisted as one **folder per skill**: `data/skills/<category>/<name>/`
  holding `skill.py`, `skill.test.py`, `contract.json`, and `config.json` (all
  gitignored), loaded back at startup via `importlib`, so
  skills stay runnable and configurable across restarts. The **old single-file
  layout** (`data/skills/<category>/<name>.py`) is **not read** — clean switch,
  no compat shim. `SKILL.md` at the repo root is the contract the codegen model
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
  seed. `calendar.next_event` (Nextcloud CalDAV, recurring events expanded
  server-side) is the reference seed; `simplex.next_message` (read via the
  forwarding bridge) and `simplex.connect_link` (show/create the forwarding
  bot's contact link) are the messenger seeds; `tests/test_seed_skills.py` keeps
  them honest.
- **Contract provenance.** Each authored body records the SKILL.md revision it
  was written against. `skill_contract_ref()` (`codegen.py`) returns
  `{"ref", "dirty"}`: `ref` is the short git commit sha the contract was read
  under (revive with `git show <ref>:SKILL.md`) and `dirty` records whether the
  working-tree contract differed from that commit; both degrade to `None`
  outside a git checkout. Recorded as `contract_ref`/`contract_dirty` on the
  `skill_writing` trace event; the dashboard shows `SKILL.md @ <ref>` with a `*`
  when dirty. **Deferred:** this pointer covers only the generic pattern now that
  bridge specifics live in the runtime catalog, not SKILL.md — recording a
  catalog revision on the trace is future work.
- **Trust boundary**: generated skill code is executed locally (it is imported
  as a module and its `predict`/`act` run in-process; `skill.test.py` runs as a
  subprocess in the skill folder). The box is the intended target; treat the
  endpoint as trusted.
- Flow in `scheduler._dispatch_skill`: the request is queued to the **single-slot
  `llm` draft worker** (small model authors title+description) → stub registered
  + hot-merged into the tree → the body write is queued to a **separate
  single-slot codegen worker** (the 12G codegen model can't run twice), the gate
  stays free for new input. A category request chains the same way in one worker.
  The codegen worker runs the full authoring pipeline:
  1. **elicitation** (`generate_elicitation`): implementation questions +
     integration hint. Every front end (REPL, dashboard, gateway) sets
     `defer_questions` and the worker posts to the question queue and waits
     (`codegen.elicitation.wait_timeout`) for answers.
  2. **codegen body** (`generate_skill_body` / `regenerate_skill_body` on
     repair): SKILL.md + request + tree + requirements answers; the body reads
     every operational value from `ctx.config`, performs the real action, and
     declares `INTEGRATION`.
  3. **fidelity gate** (`authoring:fidelity` SemIf decision +
     `integration_findings`): a rapid sanity check (accept/reconsider) — is the
     action real or simulated/declared-but-unused? It only triggers a rewrite,
     never a diagnosis. One corrective regen with the raw evidence bundle
     (`codegen.fidelity.max_attempts`), then accept with a trace rather than
     hard-failing authoring.
  4. **data contract** (`generate_data_contract`): a separate call sharing
     TESTGEN.md + `skill.py` context derives `contract.json` — a single flat
     object of snake_case variable name -> semantic description, for user input
     and SemIf only (no types/validation; that lives in the code + test).
  5. **test** (`generate_skill_tests`): a second shared-context call
     (appending the contract) produces `skill.test.py` — hermetic mechanics,
     loopback fixture injects the endpoint through config, fixtures inline.
  6. **auto-run test** (`run_skill_test`): subprocess in the skill folder,
     `codegen.test_timeout`; on failure a **3-option SemIf decision**
     (`codegen_regen`, trace-only) picks code/contract/test to regenerate,
     the error + existing files are fed back, bounded by `codegen.test_max_attempts`.
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
  (`cli.build_scheduler`); the box `config.json` sets `codegen.timeout: 3600`
  because qwen38-iq3s's card sampler writes routinely run 25–45 min. Raise
  `codegen.timeout` in config for harder prompts.
- **Config tiering + first-fire collection.** After the contract lands, a SemIf
  `choice` per variable auto-populates the skill `config.json` from the global
  config + the category config (`data/skills/<category>/config.json`)
  (`codegen.contract_search`, phase `config:search`, decision-logged). At
  runtime the runner merges global -> category -> skill config (+ per-fire
  answers) into `ctx.config`; contract variables it cannot satisfy pause the run
  **before predict** (`pre_predict`), asking the human one at a time. Each
  answer gets a SemIf `record-as-config vs ask-again-each-fire` choice (phase
  `config:record`); recorded values persist to the skill `config.json`.
- **Requirements elicitation (implementation questions, default on).**
  `codegen.elicitation.enabled` asks the product owner how the new skill should
  connect (which service/account, how to connect, where the credential comes
  from, what success looks like, how failure should behave). Every front end
  defers — the single-slot worker posts
  questions to `scheduler.questions` (`GET/POST /api/questions`) and waits up to
  `codegen.elicitation.wait_timeout` seconds, then proceeds with whatever
  answers arrived (`questions_timeout` trace). Answers ride on the draft into
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
  streaming; see it with `-s` on the box.
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
  budget is a safety net, not the loop cure: qwen38-iq3s's reasoning routinely
  exceeds even the relaxed cap before it settles on a body, which is why the
  sampler (below) is the real fix.
- **Sampler params + escalation.** `codegen.temperature` (default 0.7),
  `top_p` (0.85), `presence_penalty` (1.5), `frequency_penalty` (0.2) follow
  the Qwen3.8 model card's instruct-mode anti-repetition guidance — a high
  `presence_penalty`, not greedy temperature, is the loop cure. Those four are
  the only sampler knobs reachable via ollama's OpenAI-compat API;
  `repeat_penalty`/`min_p`/`top_k` are Modelfile-only, so the per-request
  `presence_penalty` (which overrides the Modelfile) is what actually stops the
  loop and no Modelfile edit is needed. A rejected body retries with the
  escalated sampler (presence 2.0 / temp 0.5) and a fresh short prompt that
  resets the context to SMART; `max_attempts` (default 3) bounds the ladder,
  then a graceful stub. Degeneration does **not** retry: the watchdog raises
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
  it off the box).

### gateway (messenger intake)

- **Run mode.** `python -m semif_agent.cli gateway [--platform simplex]
  [--dashboard]` builds the normal scheduler (engine lazy) and runs the
  configured `gateway.simplex` adapter in the foreground. `--dashboard`
  co-serves the browser UI from a daemon thread. Config lives under
  `gateway.simplex` in `config.json`; `enabled` defaults false. On a
  bootstrap-provisioned box `scripts/bootstrap.sh` renders and enables the
  `semif-simplex.service` bot daemon (pinned `simplex-chat` in **bot mode**,
  profile under `.runtime/simplex/`, port from `simplex_chat.port`) and the
  `semif-gateway.service` agent process, so the gateway runs across
  logout/reboot. The gateway is its own process — the REPL and the gateway are
  independent front ends onto the same on-disk logs/registry (do not run two
  scheduler processes over one skill store concurrently).
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
  (`asyncio.to_thread`) and puts replies on the queue. A second platform means
  a new adapter subclass; the service is unchanged.
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
- **Scheduler glue** (`gateway/service.py`): inbound text →
  `Scheduler.submit_request(text, source=f"simplex:<chat_id>")`; the returned
  request id maps to the chat (`owners`), and `Scheduler.on_request_requeued`
  (a scheduler hook, default `None`) copies that ownership across an updated
  request so its completion still routes home. A `needs_input` pause sets
  `pending_owner` from the pending run's source; the same chat's next message
  goes straight to `Scheduler.answer` (no score/navigation). A *different*
  chat during another chat's pause is told to wait — the single-slot scheduler
  must not silently abandon the first chat's run. A background poll calls
  `Scheduler.run_queue()` (routing each `[run_id] summary` to its owner) and
  surfaces newly posted authoring questions / repair offers (`defer_questions
  = True`; the service routes them by `run_id` → origin, falling back to
  `home_channel`); a chat's plain reply answers the question or picks the
  repair action (`retry`/`repair`/`ask`/`no`).
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
- **Tests.** `tests/test_gateway.py` (stdlib, dev box): adapter allowlist/auth,
  structured send command, batching (real `asyncio`), and `GatewayService`
  routing against a real `Scheduler` (lazy engine, unreachable LLM).
  `tests/test_simplex_ws.py` covers the neutral protocol layer (parse shapes,
  accept ids, corrId round-trips, address show/create). `tests/test_bridges.py`
  exercises `SimplexBridge` over a real loopback server (peek/pop, recipient
  resolution, outbound routing, token/body validation, `/address` success/503/502,
  daemon close on stop, catalog/`describe_bridges()`).
  `tests/test_seed_skills.py` hermetically tests every
  `seeds/<category>/<name>/` package, including `simplex.next_message` and
  `simplex.connect_link`. The live `websockets` transport against a real daemon
  is a jarvis integration concern.
- **Deps.** `websockets` is pinned in `requirements/staging.txt` (staging
  only); the dev box core stays stdlib-only.

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
  single source of bridge specifics — SKILL.md/TESTGEN.md carry only the generic
  pattern. The bridge's optional shared secret is mirrored to the top-level
  `simplex_bridge_token` config var (bootstrap syncs it) so bodies can send the
  `X-Semif-Token` header.
- **Run mode.** `python -m semif_agent.cli bridge [--name simplex]` builds the
  scheduler (engine lazy, used only for trace) and runs the selected + enabled
  bridges until interrupted. Each bridge owns its service daemon: `SimplexBridge`
  connects to a **second** simplex-chat daemon/profile (`bridges.simplex.ws_url`,
  default port 5228, pinning `simplex_chat.forward_port`), separate from the
  command gateway's. On a bootstrap-provisioned box, `scripts/bootstrap.sh`
  renders and enables `semif-simplex-forward.service` (the second daemon) and
  `semif-bridge.service` (`cli bridge`) alongside the gateway units.
- **HTTP surface** (`bridges/base.py` + `bridges/simplex.py`): JSON in/out,
  localhost-bound, optional shared-secret `X-Semif-Token` header. `SimplexBridge`
  buffers **every** inbound DM (no allowlist — the user wants to see who reached
  the bot through its invite link) in a bounded `MessagingInbox` that owns the
  read cursor, and serves `GET /health`, `GET /contacts`, `GET /inbox` (peek),
  `GET /inbox/next?contact=<id>` (pop oldest), `GET /address` (show/create the
  forwarding bot's contact link; 503 when the daemon is not connected, 502 when
  the lookup fails), and `POST /send` `{"recipient","text"}` (resolves a numeric
  id or a known display name and enqueues on the bridge's own daemon).
- **Skill-facing config.** The top-level `simplex_bridge_url` is the address a
  body calls (the data-contract config search auto-populates it); bootstrap keeps
  it in sync with the bridge port. Skills reach a bridge with the ordinary `http`
  transport, so their hermetic tests are loopback HTTP like any other HTTP body.
  `simplex.next_message` and `simplex.connect_link` are the seeds.

## Bridge backlog (one session per item)

The bridge read path (`simplex.next_message`) and contact-link lookup
(`simplex.connect_link`) are covered. Deferred follow-ups:

- [ ] **1. Contact-list refresh from the daemon** (`bridges/simplex.py`,
  `simplex_ws.py`). The bridge learns contacts only from observed inbound
  senders, so `/send` cannot address a contact it has never received from.
  Query `/_contacts <userId>` (active user from `/user` →
  `activeUser.userId`) at startup and on demand, caching `contactId` +
  `profile.displayName`. Reuse the daemon's corrId→Future path
  (`SimplexDaemon._roundtrip`) rather than opening a second WebSocket client.
- [ ] **2. Message history + true unread** (`bridges/simplex.py`). The in-memory
  inbox is a receive buffer, not the daemon's read state. Use `/_get chats
  <userId> count=<n> <json(PaginationByTime)>` with a `ChatListQuery` unread
  filter and `AChat.chatStats{unreadCount, minUnreadItemId}` +
  `chatItem.meta.itemStatus` (`rcvNew`/`rcvRead`) to expose real unread history.
  Note: v7 has **no mark-read command**, so acking is via read receipts, not an
  API call.
- [ ] **3. `simplex.send_message` seed** (`seeds/simplex/send_message/`). The
  outbound counterpart to `simplex.next_message`: resolve the recipient with a
  SemIf sub-decision over `/contacts`, send only on explicit user intent, and
  report the bridge's `contact_id` / errors honestly.
- [ ] **4. More bridges.** Each new third-party API gets its own `bridges/<name>.py`
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
  integration tests on the box.
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
- `tests/integration/` — jarvis only (staging); requires real SemIf + real
  ollama (guppy serves the models over the LAN).
- After touching scheduler/skills/codegen/engine, re-run both; the integration
  tests are the only end-to-end verification.
- **Fixtures are not the runtime.** Generated skill tests (and unit tests of the
  codegen client) use real local endpoints (a loopback `http.server`, never a
  mock); skill bodies must call the user's configured service, never a fixture
  address. Passing a skill test proves mechanics only; a real run proves the
  integration.