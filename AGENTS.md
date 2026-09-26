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

- **SemIf routes, gates, scores, and automates; the LLM only generates and
  self-assesses.** Skills are not chat responses.
- **"No mocking" means the decision engine and LLM are always real.** Skill
  tests are hermetic mechanics checks with fixtures; passing one is not proof
  the real integration works. Only a real run against the user's service is.

When requirements or connection details are unclear, the agent asks the user.
Guessing, inventing data, or shipping a toy is always the wrong answer.

## What this is

A local desktop CLI agent whose entire control flow is a single decision model
(SemIf). Inputs are gated, scored for urgency, queued, and dispatched through a
skill tree. Every SemIf decision is logged as a labeled training row; the
`dream` pass computes the prediction-vs-observation cost (cross-entropy / NLL +
ECE) that later drives fine-tuning. See "Product premise" above for what the
agent is for; this section is the machinery.

The design spec is `IDEA.md`. The key point: **semantic ifs, not text
generation, do the routing.** SemIf returns probabilities conditional on the
supplied options; an LLM is used only for generation and self-assessment.

## Architecture map

```
cli.py          argparse: run (REPL / --script), dream, skills, status, relabel,
                dashboard
scheduler.py    gate (handle/ignore; state carries the user's expectation +
                the available skills, so lookups are not read as small talk)
                -> choice(tau) -> score -> queue; preempt + requeue;
                a skill run paused for input (needs_input) keeps `current`
                busy; `answer` routes straight to the pending run, bypassing
                gate/score/navigation; skill-body authoring is an ASYNC single-slot
                worker (stub authored sync, write queued, gate stays free, the
                original request is re-queued and re-runs the new leaf when the
                body lands); the worker runs the full pipeline: elicitation ->
                codegen body -> fidelity review -> data contract -> test ->
                auto-run test (3-option SemIf regen ladder on failure); config
                search auto-populates the skill config from global/category
                config; elicitation asks implementation questions (REPL inline;
                dashboard deferred via the question queue, `wait_timeout`); a
                failed real run triggers a logged SemIf repair choice
                (retry/repair_skill/ask_user/no_repair) surfaced to the user
queue.py        urgency max-heap (desc weight, FIFO seq), age pulls toward 1.0
skills.py    tree + registry (hardcoded built-in: response.reject, service-free
                behaviors only), navigation = SemIf choices per level (logged),
                create_category
                and create_skill author + register stubs via the decision model
                in generation mode; SkillStore persists one folder per skill
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
skill.py        loop: observe -> predict -> act -> observe -> assess (LLM);
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
                infer_integration / integration_findings; elicitation questions
                + integration hint; TESTGEN.md drives two shared-context calls
                producing contract.json then skill.test.py (hermetic mechanics
                test: inline fixtures, loopback http.server for HTTP bodies, no
                external network, no mock_data.json); run_skill_test executes
                the test as a subprocess
llm.py          OpenAI-compatible client for self-assessment + skill fidelity
                review (real-action vs simulated), stdlib urllib
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
gateway/        messenger intake/reply. base.py: GatewayAdapter contract +
                Inbound/OutboundMessage. service.py: GatewayService maps chats
                onto the single-slot scheduler (submit_request, owner maps,
                pending-run ownership, queue drain, authoring questions/repairs
                routed back to the origin chat). simplex.py: SimplexAdapter —
                local simplex-chat daemon over its JSON WebSocket API
                (lazy `websockets`, allowlist, batching, structured `/_send`).
                Run with `python -m semif_agent.cli gateway [--dashboard]`;
                config under `gateway.simplex` in config.json
```

## Run / verify

Dev machine is a thin client (no GPU, ~1.4G disk): only pure stdlib unit tests
run here (`python3 -m pytest tests/ -q --ignore=tests/integration`).
Integration tests run on the staging machine `jarvis` (see "### jarvis (staging)").

## Git / sync

- Canonical repo lives on Gitea: `git.manyworlds.fit`, **SSH on port 222**
  (`ssh://git@git.manyworlds.fit:222/gabby/semif-agent.git`). Key
  `~/.ssh/id_ed25519` is registered there. The staging machine `jarvis` runs
  from a manual clone at `~/semif-agent` (`scripts/bootstrap.sh` provisions
  everything else and never re-clones the agent repo); the old box working copy
  is at `~/repos/semif-agent` (its editable install still points at the moved
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

Machine split (Sep 2026): **guppy** is the ollama **model server only**;
**jarvis** is the staging/test target (semif-agent + SemIf engine + deps).
Provision jarvis with `scripts/bootstrap.sh` (see "### jarvis (staging)").
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
Core loop, urgency queue, skill tree, skill loop with real SemIf + real LLM
self-assessment, decision logging, `dream` cost pass, REPL + JSONL CLI,
unit tests (24) + box integration tests (2).

### v2
- Real fine-tuning from `decisions.jsonl` at a regular interval ("dreaming"):
  accumulate labeled rows, compute cost, fine-tune the decision model, CI/CD
  validate (accuracy/ECE on a held-out slice, prompt-hash regression), swap the
  pinned model revision. GPU offload: train on a beefier GPU; the running agent
  keeps a frozen inference revision until a swap validates.
- `create_skill` branch: live, mirroring `create_category`. The decision
  model, driven in normal generation mode via `SemIfEngine.generate`, proposes a
  specific skill title + description for the chosen category; the stub is
  persisted to `data/categories.json` (under that category's `skills` list) and
  merged into the running tree as a leaf. Since Sep 2026 the leaf also gets a
  real runnable body via the async authoring pipeline: a larger
  OpenAI-compatible model (`codegen`, default `qwen38-iq3s`) writes
  `predict`/`act` code against `SKILL.md`, then a contract + test
  (against `TESTGEN.md`) are generated and the test is auto-run before the leaf
  is declared ready — all persisted to a folder in `data/skills/` and
  hot-loaded. Authoring is **asynchronous**: the stub is created and the gate
  freed immediately, the body write runs on a single-slot background worker, and
  once the body lands the request that prompted creation is re-queued and re-runs
  navigation onto the new leaf (an empty/in-progress leaf can be restarted via
  `restart <category> <skill>` or the dashboard). A request that prompted a whole
  new category runs the same chain deterministically: `create_category` →
  `create_skill` → async body → re-dispatch. Authoring is still a single pass —
  validating/reusing written bodies across runs is future work.
- **Real integrations, implementation questions, fidelity + repair** (Sep 2026):
  elicitation is on by default and asks implementation questions (which
  service/account, how to connect, where the credential comes from, what success
  looks like); the REPL asks inline, the dashboard defers via the question queue
  (`/api/questions`, worker waits `codegen.elicitation.wait_timeout`). Bodies
  must perform the real action via stdlib transports with values from
  `ctx.config` and declare `INTEGRATION` (service/transport/config_vars); a
  fidelity review (small self-assessment model + static declaration checks)
  regenerates a simulated body once, then accepts it badged `unverified`. A
  failed real run logs a SemIf repair choice (retry / repair_skill / ask_user /
  no_repair) surfaced in REPL/dashboard; repair writes carry the observed
  failure and are bounded by `codegen.repair.max_attempts`. Tests are hermetic
  mechanics checks (loopback `http.server` for HTTP bodies) and never certify
  the live integration — only a real run does.
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
  passwordless sudo. SemIf runs via llama.cpp **CPU** backend (its llamacpp
  backend forces `n_gpu_layers=0`), so the GPU is NOT used by the decision
  engine — it IS used by ollama.
- The SemIf tokenizer is fetched from HF (`Qwen/Qwen3.5-4B` at the pinned
  revision). Set `HF_HOME=/home/abby/hf` or the tokenizer re-downloads.

### SemIf install (box)
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

### ollama (box)
- Installed at `/home/abby/ollama/bin/ollama` (not on PATH), systemd service
  `ollama.service`, ROCm backend with `HSA_OVERRIDE_GFX_VERSION=10.3.0` and KV
  cache q4_0 + flash attention. This is expected, not a bug.
- `semif-hermes` / `semif-hermes-v3` are for ANOTHER project (hermes agent) —
  ignore them; they spew "token repeat limit" errors.
- Use `qwen3.5:4b` for self-assessment (works, ~3s). `qwen38-iq3s` (12G 27B)
  also works but is huge/slow.
- If generation hangs with no log output, restart the service
  (`sudo systemctl restart ollama`) — the ROCm runner can wedge.
- Ollama already binds `OLLAMA_HOST=0.0.0.0`, so jarvis reaches it as a plain
  remote API at `http://192.168.8.181:11434` (no auth — LAN-visible, same
  exposure as the old dashboard POST endpoints).

### jarvis (staging)

- Provision a fresh Ubuntu machine into a running staging box: clone
  `semif-agent` to `~/semif-agent` first (register its SSH key on Gitea), then
  `scripts/bootstrap.sh --peer-ollama http://192.168.8.181:11434`.
  The script must be run from a checkout — it reads pins from that checkout's
  `config.example.json` and never re-clones the agent repo (only the SemIf
  engine). Idempotent and rerunnable; every stage no-ops on existing state, so
  it also boots an unknown-state machine. It installs **no ollama** — llm +
  codegen both point at guppy.
- All pins are read from `config.example.json`'s `engine` block: `semif_repo`
  (public GitHub `TheoLeeCJ/SemIf`), `semif_ref` (pinned commit the box runs),
  `gguf_url`/`gguf_sha256` (verified after download), and the HF tokenizer
  `source`/`revision`. The python dep pins live in `requirements/staging.txt`
  (committed, one versioned artifact — jarvis, guppy, and any future box all
  provision from it). **Maintenance**: bump the pins in `config.example.json` /
  `requirements/staging.txt`, rerun the script, re-run the integration tests.
  The script never guesses.
- pip prints **expected** "dependency conflict" warnings at install time
  (semif-phase1 declares torch/accelerate/protobuf/sentencepiece/numpy 2.2.6
  that we intentionally do not install — the llama.cpp CPU path doesn't need
  them; numpy 2.3.5 is deliberate, 2.2.6 has no cp314 wheel). Same as the box;
  do not "fix" them by installing torch.
- Generates `~/semif-agent/config.json` with `codegen.timeout: 3600` (codegen
  now travels the LAN) and a backup of any prior file. `--threads N` overrides
  engine threads; `--copy-data SRC` rsyncs guppy's `data/` for continuity;
  `--public-dashboard` binds the dashboard to `0.0.0.0`.
- Run / verify on jarvis:
  ```sh
  cd ~/semif-agent && export HF_HOME=~/hf
  ~/semif-venv/bin/python -m semif_agent.cli run              # REPL
  ~/semif-venv/bin/python -m semif_agent.cli dream            # cost report
  ~/semif-venv/bin/python -m pytest tests/integration -q -s   # ~100s, background+poll
  ~/semif-venv/bin/python -m semif_agent.cli dashboard --port 8765
  ```
- First run downloads the 2.8G GGUF and builds `llama-cpp-python` from source
  (~10 min on 6 cores); reruns are fast no-ops.

### codegen (skill bodies, box)
- Skill **bodies** are written by a separate OpenAI-compatible model, configured
  under `codegen` in config.json (default model `qwen38-iq3s`, the 12G 27B
  IQ3_S GGUF — huge/slow). Title + description for new skills still come from
  the **small** decision model (`engine.generate`); only the runnable code body
  uses codegen.
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
  server-side) is the reference seed; `tests/test_seed_skills.py` keeps it
  honest.
- **Trust boundary**: generated skill code is executed locally (it is imported
  as a module and its `predict`/`act` run in-process; `skill.test.py` runs as a
  subprocess in the skill folder). The box is the intended target; treat the
  endpoint as trusted.
- Flow in `scheduler._dispatch_skill`: small model authors title+description →
  stub registered + hot-merged into the tree (navigable immediately) → trace
  `skill_writing` (dashboard shows title/description + a "writing skill body…"
  badge) → the body write is queued to a **single-slot background worker** (the
  12G codegen model can't run twice), the gate stays free for new input. The
  worker runs the full authoring pipeline:
  1. **elicitation** (`generate_elicitation`): implementation questions +
     integration hint. REPL asks inline at dispatch; the dashboard sets
     `defer_questions` and the worker posts to the question queue and waits
     (`codegen.elicitation.wait_timeout`) for answers.
  2. **codegen body** (`generate_skill_body` / `regenerate_skill_body` on
     repair): SKILL.md + request + tree + requirements answers; the body reads
     every operational value from `ctx.config`, performs the real action, and
     declares `INTEGRATION`.
  3. **fidelity review** (`llm.review_skill_body` + `integration_findings`): is
     the action real or simulated/declared-but-unused? One corrective regen
     (`codegen.fidelity.max_attempts`), then accept with a trace + `unverified`
     badge rather than hard-failing authoring.
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
  from, what success looks like, how failure should behave). The REPL asks
  inline at dispatch; the dashboard defers — the single-slot worker posts
  questions to `scheduler.questions` (`GET/POST /api/questions`) and waits up to
  `codegen.elicitation.wait_timeout` seconds, then proceeds with whatever
  answers arrived (`questions_timeout` trace). Answers ride on the draft into
  every body attempt (including retries and regens) and are persisted in the
  registry so a `restart` does not ask again. The prompt keeps an explicit
  anti-pattern block: never config-vs-input cadence questions ("should the
  sender address change?"), never operational data values, never trivia.
- **Fidelity review.** After the body lands, `llm.review_skill_body` (the small
  self-assessment model) plus static `integration_findings` (declared transport
  backed by real calls? declared config_vars actually read?) decide whether the
  body really performs the action. A rejected body is rewritten once with the
  finding (`codegen.fidelity.max_attempts`, reason_kind `fidelity`); a body that
  still fails is accepted but traced (`fidelity_review`,
  `performs_real_action=false`) and badged `unverified` in the dashboard — the
  reviewer never hard-fails authoring, and a missing reviewer degrades to
  accept.
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
  the end of the stream.
- **Sampler params + escalation.** `codegen.temperature` (default 0.7),
  `top_p` (0.85), `presence_penalty` (1.5), `frequency_penalty` (0.2) follow
  the Qwen3.8 model card's instruct-mode anti-repetition guidance — a high
  `presence_penalty`, not greedy temperature, is the loop cure. A rejected
  body retries with the escalated sampler (presence 2.0 / temp 0.5) and a
  fresh short prompt that resets the context to SMART; `max_attempts` (default
  3) bounds the ladder, then a graceful stub.
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
  `gateway.simplex` in `config.json`; `enabled` defaults false. The gateway is
  its own process — the REPL and the gateway are independent front ends onto
  the same on-disk logs/registry (do not run two scheduler processes over one
  skill store concurrently).
- **Transport contract** (`gateway/base.py`): `GatewayAdapter.run(on_inbound,
  outbound_queue)` blocks, delivering `InboundMessage`s and draining a stdlib
  `queue.Queue[OutboundMessage | None]`. Scheduler work is synchronous and can
  block on the decision engine, so the adapter bridges it off its event loop
  (`asyncio.to_thread`) and puts replies on the queue. A second platform means
  a new adapter subclass; the service is unchanged.
- **SimpleX adapter** (`gateway/simplex.py`): connects to `simplex-chat -p
  5225` at `gateway.simplex.ws_url`, XML-JSON WebSocket protocol
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
  attachments, reactions, and typing are out of scope for this cut.
- **Scheduler glue** (`gateway/service.py`): inbound text →
  `Scheduler.submit_request(text, source=f"simplex:<chat_id>")`; the returned
  request id maps to the chat (`owners`), and `Scheduler.on_request_requeued`
  (a scheduler hook, default `None`) copies that ownership across an updated
  request so its completion still routes home. A `needs_input` pause sets
  `pending_owner` from the pending run's source; the same chat's next message
  goes straight to `Scheduler.answer` (no gate/score/navigation). A *different*
  chat during another chat's pause is told to wait — the single-slot scheduler
  must not silently abandon the first chat's run. A background poll calls
  `Scheduler.run_queue()` (routing each `[run_id] summary` to its owner) and
  surfaces newly posted authoring questions / repair offers (`defer_questions
  = True`; the service routes them by `run_id` → origin, falling back to
  `home_channel`); a chat's plain reply answers the question or picks the
  repair action (`retry`/`repair`/`ask`/`no`).
- **Tests.** `tests/test_gateway.py` (stdlib, dev box): allowlist/auth,
  `newChatItems` parsing + echo/group/non-text filtering, structured send
  command, batching (real `asyncio`), and `GatewayService` routing against a
  real `Scheduler` (lazy engine, unreachable LLM). The live `websockets`
  transport against a real daemon is a jarvis integration concern.
- **Deps.** `websockets` is pinned in `requirements/staging.txt` (staging
  only); the dev box core stays stdlib-only.

## Codegen guardrails backlog (one session per item)

Context learned 2026-09-24 on the box: qwen38-iq3s looped ~40 min on one skill
body (1.5 MB streamed) with no guard firing. Root causes: (1) `CodegenClient`
sent `temperature 0.0` (greedy) and no penalties while the Modelfile has
`repeat_penalty 1` (off) + `presence_penalty 0` (now fixed: item 5 ships the
card-endorsed high `presence_penalty` per-request); (2) the total `timeout`
check only runs inside the `if not ready:` branch of `_read_stream`, so a
continuously-streaming runaway never trips it (fixed in item 1). Known model
facts: qwen38-iq3s
runtime window `num_ctx=100000` (native `qwen35.context_length=262144`);
ollama OpenAI-compat `/v1/chat/completions` supports `temperature`, `top_p`,
`presence_penalty`, `frequency_penalty`, `max_tokens`, `reasoning_effort`, and
`stream_options.include_usage` (returns exact token `usage`) — but NOT
`repeat_penalty`/`min_p`/`top_k` (needs a Modelfile edit or native
`/api/chat`). Degradation tracks ABSOLUTE tokens used (SMART<100K,
WARN 100–200K, DUMB>200K), so limits are a total-context budget
(prompt+output), not a fill %.

- [x] **1. Fix the total-timeout stream bug** (`semif_agent/codegen.py`)
  - Move the `now - start >= self.timeout` check out of the `if not ready:`
    branch to the top of the `_read_stream` loop so it fires while streaming.
  - Test: fake SSE server that streams forever → `CodegenError` "total budget"
    must raise while tokens still flow.
  - Verify: `python3 -m pytest tests/ -q --ignore=tests/integration`.

- [x] **2. Auto-detect context window + layered token budget** (`codegen.py`, `cli.py`, `config.example.json`)
  - `CodegenClient` lazily queries `POST /api/show {model}` → `parameters.num_ctx`
    (fallback `model_info.<arch>.context_length`, then `context_window` config,
    then default 100000).
  - Per request: `total_limit = min(smart_limit, window*max_fill_ratio)`;
    `output_limit = min(max_output, total_limit - prompt_est)`;
    `warn_point = min(warn_limit, window*warn_fill_ratio)`.
  - `max_output`: `>=1` = absolute tokens, `0<x<1` = fraction of window,
    default `0.85`. Enforce in `_read_stream` via chars→tokens estimate
    (`chars_per_token`). Abort → `CodegenError` → graceful stub. The transport
    always streams (payload `"stream": true`) so the cap, idle watchdog, and
    item-1 total budget abort a generation in real time; the `stream` config
    now gates only the console echo, and non-stream response reading is gone.
  - Print a start line: context tokens, output cap tokens + %, peak total fill.
  - Config: `codegen.{context_window=0, smart_limit=250000, warn_limit=500000,
    max_fill_ratio=0.9, warn_fill_ratio=0.7, max_output=0.85, chars_per_token=4.0}`.
  - Tests: cap trips at fraction-of-window and absolute forms (fake server +
    explicit window); `/api/show` parse (real fake server, per the no-mocking
    rule — the backlog's "mock the fetch" was implemented as a real endpoint).
  - **Box findings (2026-09-24):** real ollama serves `/api/show`'s
    `parameters` as a **modelfile string** (`"num_ctx 100000\n..."`), not a
    dict — a fake server that returned a dict hid the resulting
    `AttributeError` crash; the parser now handles both shapes (string parses
    the `num_ctx` line, else falls back to `model_info`). The ORIGINAL
    defaults were far too tight for qwen38-iq3s: `max_output 0.25` (25k
    tokens ≈ 100 KB chars) aborted a still-verbosely-reasoning `check_service`
    body at exactly 100,154 chars — the cap fired correctly but too early.
    Defaults were relaxed to the values above so a finite-but-verbose write
    (~100 KB+) can complete while a window-filling degeneration still aborts.
    **The box codegen integration tests remain flaky for a model reason, not
    a code one**: qwen38-iq3s's reasoning often exceeds even the relaxed cap
    before it settles on a body. That is the degeneration items 4–6 fix; the
    budget is the safety net, not the cure. Window detection + cap firing are
    both confirmed live on the box (`context window 100000 tokens` start
    line; abort at the configured cap).

- [x] **3. Exact token accounting via include_usage** (`codegen.py`)
  - Send `stream_options: {"include_usage": true}`; capture the `usage` chunk
    in `_consume_frame`; after the stream log real `prompt/completion/total`
    tokens, % of window, and zone `SMART|WARN|DUMB` (absolute thresholds
    `smart_limit`/`warn_limit`).
  - Test: fake server emits a usage chunk; assert it's captured and logged.
  - Implemented on the dev machine (2026-09-24): `_consume_frame` now also
    tolerates a usage chunk with an empty `choices` list (OpenAI's shape —
    previously an `IndexError`) and returns the `usage` dict; both stream
    readers capture it and `_log_usage` prints real `prompt/completion/total`
    tokens, % of the detected window, and the zone once the stream ends.
    `chat` always sends `stream_options: {"include_usage": true}`. Unit-tested
    (payload carries include_usage; usage captured + zone parametrized
    SMART/WARN/DUMB; a stream without a usage chunk logs nothing extra).

- [x] **4. SemIf degeneration watchdog** (`codegen.py`, `cli.py`, `scheduler.py`)
  - Add optional `degeneration_check: Callable[[str], str|None]` to
    `CodegenClient.chat`/`generate_skill_body`; `_read_stream` calls it every
    `interval` chars with the last `window` chars (content+reasoning);
    non-None → `CodegenError`.
  - Wire in `cli.build_scheduler`: 2-option SemIf decision (continue/stop);
    `P(stop) >= threshold` aborts. Record as a **trace-only** event
    (kind `codegen`, with probs) — never in the decision log.
  - Config: `codegen.degeneration.{enabled, threshold=0.9, interval=8000,
    window=2000, min_chars=4000}`. Disabled when no engine (keeps
    `CodegenClient` standalone pure for unit tests).
  - Tests: a callback returning a reason aborts the stream; not invoked when
    disabled.
  - Implemented on the dev machine (2026-09-24): `CodegenClient` gained
    `degeneration_interval`/`degeneration_window`/`degeneration_min_chars`
    constructor params and a per-`chat` `degeneration_check`; both stream
    readers (`_read_stream` and the blocking fallback) poll it against a
    bounded rolling buffer (content+reasoning), trimming to
    `window + interval` chars so a long degenerating stream never grows
    unbounded memory. `cli.build_scheduler` wires a `degeneration_check_factory`
    (per-run SemIf continue/stop decision; `EngineUnavailable` at call time
    degrades to "keep going", so a machine without the engine never aborts via
    this path) into the `Scheduler`, which builds the per-request check with
    `request.id` and passes it through `generate_skill_body`. Trace event:
    `kind: "codegen"` carrying state/question/options/probs/stop_prob — not in
    the decision log. Config block added to `config.example.json` (`enabled`
    defaults true; the runtime engine check is what disables it off the box).
    Unit-tested: callback reason aborts the stream (deviation from the "not
    invoked when disabled" spec: the min_chars gate is covered too), callback
    returning None continues, no callback completes normally. **Box verified
    (2026-09-24)**: during a real codegen write the watchdog fired 16 trace-only
    `codegen` events, `stop_prob` range 0.056–0.693 (max 0.693 < threshold 0.9);
    the SemIf model occasionally leaned `stop` by argmax but the threshold
    correctly let the healthy-but-verbose generation continue. No
    `DegenerationError` fired, no false abort — the watchdog polls and the
    threshold logic works as designed.

- [x] **5. Sampler params + 3-attempt escalation ladder** (`codegen.py`, `cli.py`)
  - Send `temperature`/`top_p`/`presence_penalty`/`frequency_penalty`.
  - `generate_skill_body` escalates on invalid parse: attempts 2–3 use the
    escalated sampler + a corrective message ("You are looping; emit the final
    Python now."). Max 3 attempts (config `max_attempts`), then graceful stub.
    Retry = fresh short prompt, i.e. context resets to SMART.
  - Tests: payload carries the params; retry uses escalated params.
  - **Revised values after reading the Qwen3.8-27B model card
    (unsloth/Qwen3.8-27B-GGUF, 2026-09-24):** the original spec (`temp 0.15 /
    presence 0.1 / freq 0.2`) kept the model nearly greedy with a near-zero
    presence penalty — the exact loop recipe already documented on the box.
    The card's anti-repetition guidance is a HIGH `presence_penalty` ("adjust
    between 0 and 2 to reduce endless repetition"), recommended
    `temp 0.7 / top_p 0.80 / presence 1.5` in instruct mode. Implemented
    defaults: `temperature 0.7, top_p 0.85, presence_penalty 1.5,
    frequency_penalty 0.2`; escalated (attempts 2+): `temp 0.5, top_p 0.85,
    presence_penalty 2.0, frequency_penalty 0.3`. `top_k`/`min_p`/`repeat_penalty`
    from the card are NOT reachable via ollama OpenAI-compat, so they stay
    unset. **Degeneration does NOT retry**: the item-4 watchdog raises a new
    `DegenerationError(CodegenError)` subclass that propagates straight to the
    scheduler's graceful-stub path — retry-on-degeneration is left as an open
    decision. Unit-tested on the dev machine (payload defaults, escalated retry
    payload + corrective prompt, max_attempts exhaustion, degeneration and
    token-budget errors propagate without retry). **Box verified (2026-09-24)**:
    both codegen writes converged to runnable bodies (`check_service` → 9863
    bytes, materialized, `test_generate_skill_body_codegen` passed;
    `track_drone_delivery` → body written + materialized + ran). No 40-min loop,
    no 1.5 MB stream — the card sampler stopped the degeneration. **But the
    model is now MORE verbose**: ~28–70k tokens of reasoning per body at
    ~25–40 tok/s, so writes take 25–45 min and trip the old 1200s default
    timeout. The box `config.json` now sets `codegen.timeout: 3600`; that
    timeout, not degeneration or the token cap, is the binding constraint now.

- [x] **6. Box Modelfile anti-loop levers** (guppy; infra, not a code change)
  - **Superseded by item 5.** The model card calls for `repetition_penalty 1.0`
    (off) and `min_p 0.0` — the original `repeat_penalty 1.2 / min_p 0.05`
    plan contradicts it. The card-endorsed cure (high `presence_penalty`,
    reachable per-request via OpenAI-compat) is already shipped in item 5, so
    no Modelfile edit is needed. **Box sanity checked (2026-09-24)**:
    `ollama show --modelfile qwen38-iq3s` still shows `repeat_penalty 1` /
    `presence_penalty 0` / `min_p 0` / `num_ctx 100000`; the per-request
    values (presence 1.5 etc.) override them at request time and the writes
    converged — no long loop.

- [x] **7. SKILL.md auditability** (`codegen.py`, `scheduler.py`, `static/app.js`)
  - `read_skill_contract` also yields sha256 of SKILL.md; record
    `contract_sha256` on the `skill_writing` trace event (+ dashboard display).
  - Test: the actually-sent HTTP payload's system message contains a SKILL.md
    phrase (harness already records request bodies).
  - **Reshaped on the dev machine (2026-09-24): commit-ref provenance, not a
    content hash.** A bare sha256 flags stale bodies but can't revive the old
    contract; the ref is the revivable pointer. `skill_contract_ref()`
    (`codegen.py`) returns `{"ref", "dirty"}`: `ref` is the short git commit
    sha the contract was read under — revive with
    `git show <ref>:SKILL.md` — and `dirty` records whether the working-tree
    contract differed from that commit. Both degrade to `None` (never a
    non-revivable hash) outside a git checkout; only a missing contract
    raises, mirroring `read_skill_contract`. Recorded as `contract_ref` /
    `contract_dirty` on the `skill_writing` trace event
    (`scheduler._create_skill`); the dashboard shows `SKILL.md @ <ref>` with a
    `*` when dirty. Unit-tested: ref matches the real `git rev-parse --short
    HEAD` in the repo, degrades off-repo, the sent HTTP payload's system
    message contains the real SKILL.md text, and the ref fields round-trip
    through `/api/trace`.

- [x] **8. Config + AGENTS.md docs**
  - Update `config.example.json` codegen block and the AGENTS.md codegen
    section with every new key from items 2, 4, 5.
  - Implemented on the dev machine (2026-09-24): `config.example.json` was
    already complete — every key from items 2/4/5 landed incrementally with
    those items and matches `cli.build_scheduler` defaults exactly — so the
    only real gap was the AGENTS.md "### codegen (skill bodies, box)" section,
    which gained three bullets (token budget + exact `include_usage` accounting
    with SMART/WARN/DUMB zone; sampler params + escalation ladder; degeneration
    watchdog block), all mirroring the `codegen` block of `config.example.json`.

- [x] **9. Codegen prompt: code-for-reuse + self-generated mock data**
    (`codegen.py`, `SKILL.md`)
  - The generated body is meant to be REUSED across requests, so the prompt
    must tell the model to generate its own mock data (its own source of
    truth) and not assume a human will hand it data at predict/act time. The
    REPL input path during a run (`needs_input`/`answer`) is for CLARIFYING
    questions only — never the primary data source.
  - Root cause (box, 2026-09-24): the generated `track_drone_delivery` body
    asked the human for `tracking_id`/`status`/`latitude`/`longitude` to
    function — reasonable for a one-shot exchange, wrong for a persistent
    skill. `SKILL.md` should encode: skills are reusable modules; prefer an
    internal/mock data model; ask the human only to disambiguate intent.
  - Test: prompt/`SKILL.md` contains the reuse + self-mock-data directives;
    a generated-body parse is unaffected.
  - Implemented on the dev machine (2026-09-24): SKILL.md gained a hard rule
    "Reusable module with its own data model" and the old "Request input when
    data is missing" rule was reworded to "Request input for clarification
    when requirements are unclear from the prompt" (intent only, never
    operational data). Both codegen prompt builders (`build_skill_body_prompt`
    and `_retry_prompt`) now add an explicit user-message line: the body is
    reused across many requests; give the skill its own internal/mock data
    model and ask only to clarify intent. Unit-tested (contract + both prompt
    builders carry the directives; existing body-parse tests unaffected).
  - **Superseded by BRAINSTORM.md item 1 (Sep 2026).** The "own data model"
    doctrine was replaced: skills now own NO data — every operational value is
    requested from the runner via `ctx.config` under a clear snake_case name
    (no fabrication, no embedded mock data, no runtime asks). Mocking/testing
    moved to its own contract `TESTGEN.md`, which drives the data contract
    (`contract.json`) + test (`skill.test.py` with fixtures embedded inline).
    The change-frequency config-vs-input question is answered behaviorally by
    the config step at first fire, never in SKILL.md.

### Code principles
- **No mocking.** The decision engine is always real SemIf; the LLM is always a
  real endpoint. Pure unit tests touch data-structure math only (queue ordering,
  dream cost, contract serialization). Engine-dependent behavior is verified by
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
- CLI subcommands must not crash when the engine is unavailable — `submit`
  catches `EngineUnavailable` and returns `("error", ...)`.
- Keep deps stdlib-only in the core; heavy deps live on the staging venv
  (`~/semif-venv`, provisioned by `scripts/bootstrap.sh`).

## Testing

- `python3 -m pytest tests/ -q --ignore=tests/integration` — anywhere, fast.
  Includes the dashboard API tests (`tests/test_dashboard_api.py`), which spin
  up the stdlib HTTP server on an ephemeral port with the engine never loaded;
  `tests/test_codegen.py` for prompt/parse/validate, integration
  extraction/findings, and body store round-trips; and `tests/test_llm.py` for
  the fidelity review against a throwaway OpenAI-compatible endpoint.
- `tests/integration/` — jarvis only (staging); requires real SemIf + real
  ollama (guppy serves the models over the LAN).
- After touching scheduler/skills/codegen/engine, re-run both; the integration
  tests are the only end-to-end verification.
- **Fixtures are not the runtime.** Generated skill tests (and unit tests of the
  codegen client) use real local endpoints (a loopback `http.server`, never a
  mock); skill bodies must call the user's configured service, never a fixture
  address. Passing a skill test proves mechanics only; a real run proves the
  integration.