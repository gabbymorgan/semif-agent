# AGENTS.md

Guidance for working on the semif agent. Read this before touching code.

## What this is

A local desktop CLI agent whose entire control flow is a single decision model
(SemIf). Inputs are gated, scored for urgency, queued, and dispatched through a
skill tree. Every SemIf decision is logged as a labeled training row; the
`dream` pass computes the prediction-vs-observation cost (cross-entropy / NLL +
ECE) that later drives fine-tuning.

The design spec is `IDEA.md`. The key point: **semantic ifs, not text
generation, do the routing.** SemIf returns probabilities conditional on the
supplied options; an LLM is used only for generation and self-assessment.

## Architecture map

```
cli.py          argparse: run (REPL / --script), dream, skills, status, relabel,
                dashboard
scheduler.py    gate -> choice(tau) -> score -> queue; preempt + requeue;
                a skill run paused for input (needs_input) keeps `current`
                busy; `answer` routes straight to the pending run, bypassing
                gate/score/navigation
queue.py        urgency max-heap (desc weight, FIFO seq), age pulls toward 1.0
skills.py    tree + registry (email.compose, response.reject, tracking.check),
                navigation = SemIf choices per level (logged), create_category
                and create_skill author + register stubs via the decision model
                in generation mode; SkillBodyStore + materialize_skill persist
                and hot-load runnable skill bodies from data/skills/;
                ActionResult.needs_input pauses a run for human input
skill.py        loop: observe -> predict -> act -> observe -> assess (LLM);
                a run paused for input is resumed by re-invoking act with the
                answer on request.user_input (predict is never re-run)
engine.py       SemIfEngine -> semif_phase1.llamacpp_backend (lazy import)
codegen.py      CodegenClient (OpenAI-compatible) writes runnable skill bodies
                against SKILL.md; parse/validate (compile + predict/act)
llm.py          OpenAI-compatible client for self-assessment (stdlib urllib)
log.py          decisions.jsonl rows {state, question, options, predicted_probs,
                selected, observed_outcome, label_source}
trace.py        runs.jsonl lifecycle events keyed by run_id (submit/queued/
                preempted/assessed/...); decisions reference run_id in extra
dream.py        NLL of observed outcome per row; weighted CE, accuracy, ECE
decisions.py    contract dataclasses (Option, DecisionRequest, DecisionResult,
                Request)
dashboard.py    stdlib http.server + JSON API (tree/trace/dream/status +
                POST submit/relabel); static/ frontend served at /
```

## Run / verify

Dev machine is a thin client (no GPU, ~1.4G disk): only pure stdlib unit tests
run here (`python3 -m pytest tests/ -q --ignore=tests/integration`).

## Git / sync

- Canonical repo lives on Gitea: `git.manyworlds.fit`, **SSH on port 222**
  (`ssh://git@git.manyworlds.fit:222/gabby/semif-agent.git`). Key
  `~/.ssh/id_ed25519` is registered there. The box `guppy` keeps a working copy
  at `~/semif-agent`; the dev machine at `~/Repos/semif-agent`. Push/pull from
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

The AMD box `guppy` (`abby@192.168.8.181`) is the real run target. Key facts:

- ssh key `~/.ssh/id_ed25519` is passphrase-protected. Load it into an agent at
  a fixed socket before connecting (the default flatpak `SSH_AUTH_SOCK` refuses):
  ```sh
  SOCK=/tmp/opencode/ssh-agent.sock; rm -f "$SOCK"; eval $(ssh-agent -a "$SOCK")
  printf '#!/bin/sh\necho "<PASSPHRASE>"\n' > /tmp/opencode/askpass.sh; chmod 700 /tmp/opencode/askpass.sh
  SSH_ASKPASS=/tmp/opencode/askpass.sh SSH_ASKPASS_REQUIRE=force setsid -w ssh-add ~/.ssh/id_ed25519
  ```
  The agent dies if this machine restarts; redo it each session.
- `semif_agent` is editable-installed into the box venv
  (`pip install -e ~/semif-agent --no-deps`), so `python -m semif_agent.cli ...`
  works from any directory on the box, not just the repo root.
- Run the agent on the box:
  ```sh
  cd ~/semif-agent && export HF_HOME=/home/abby/hf
  ~/semif-venv/bin/python -m semif_agent.cli run              # REPL
  ~/semif-venv/bin/python -m semif_agent.cli run --script demo.jsonl
  ~/semif-venv/bin/python -m semif_agent.cli dream            # cost report
  ~/semif-venv/bin/python -m semif_agent.cli relabel <id> <outcome>
  ~/semif-venv/bin/python -m semif_agent.cli dashboard --port 8765
  ```
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
  real runnable body: a larger OpenAI-compatible model (`codegen`, default
  `qwen38-iq3s`) writes `predict`/`act` code against `SKILL.md`, persisted to
  `data/skills/` and hot-loaded, then the newly created leaf is executed
  directly so the request that prompted creation is answered. A request that
  prompted a whole new category runs the same chain deterministically:
  `create_category` → `create_skill` → run. Authoring is still a single pass —
  validating/reusing written bodies across runs is future work.
- Queue persistence (durable across restarts).
- Event/timer intake sources beyond typed input.
- Concurrency: SemIf shared-state mode (`score_shared` / `SerialPrefixScorer`)
  for parallel decisions; single execution slot remains for processes.
- Dashboard: run-requeue cross-linking (child run references its parent),
  scheduler sim controls (busy/idle/tau) as a first-class panel.
- **Data-contract tiered data lookup on skill creation**: authoring a new
  skill emits a data contract (what data the skill needs) that kicks off a
  tiered search for a source of truth — a global data object → category-
  specific data objects → a skill-specific data object. The cascade reuses an
  existing source when one exists (no re-querying the user) while keeping the
  data–skill association explicit at each level. If no source of truth is
  found in the cascade, the human is asked for input AFTER code generation to
  complete the skill's function — not during authoring.
- **Post-codegen mock-data test**: after a codegen body write, run the new
  skill against the mock data the model supplied (per backlog item 9) before
  declaring it runnable — a generated body that can't execute its own data
  path fails authoring, not the first real request.
- **Skill-code inspection**: new skills are labeled as new, and the CLI and
  dashboard gain the ability to inspect a skill's generated code body
  (read-only view of `data/skills/<category>/<name>.py` and its trace) so the
  author can audit what was generated.

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
  with `--no-deps` and bring only what's needed:
  `pip install -e ~/semif --no-deps`, then numpy 2.3.5, transformers 5.17.0,
  tokenizers 0.23.2, huggingface-hub, llama-cpp-python 0.3.35.
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
- Bodies are persisted to `data/skills/<category>/<name>.py` (gitignored) and
  loaded back at startup via `importlib`, so skills stay runnable across
  restarts. `SKILL.md` at the repo root is the contract the codegen model is
  prompted with — change it only with intent, it shapes every generated body.
- **Trust boundary**: generated skill code is executed locally (it is imported
  as a module and its `predict`/`act` run in-process). The box is the intended
  target; treat the endpoint as trusted.
- Flow in `scheduler._dispatch_skill`: small model authors title+description →
  trace `skill_writing` (dashboard shows title/description + a "writing skill
  body…" badge) → sync codegen write → `materialize_skill` → hot-merge into the
  tree → the new leaf runs directly so the request is answered. `create_category`
  runs the same chain after authoring the category (`create_category` →
  `create_skill` → run). Codegen failure — including a request timeout — leaves
  a navigable stub and returns a graceful `create_skill` result; a timeout is
  raised as `CodegenError` by the client, never a raw `TimeoutError`. The
  default codegen timeout is 1200s (`cli.build_scheduler`); the box
  `config.json` sets `codegen.timeout: 3600` because qwen38-iq3s's card
  sampler writes routinely run 25–45 min. Raise `codegen.timeout` in config
  for harder prompts.
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

- [ ] **7. SKILL.md auditability** (`codegen.py`, `scheduler.py`, `static/app.js`)
  - `read_skill_contract` also yields sha256 of SKILL.md; record
    `contract_sha256` on the `skill_writing` trace event (+ dashboard display).
  - Test: the actually-sent HTTP payload's system message contains a SKILL.md
    phrase (harness already records request bodies).

- [ ] **8. Config + AGENTS.md docs**
  - Update `config.example.json` codegen block and the AGENTS.md codegen
    section with every new key from items 2, 4, 5.

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
- Keep deps stdlib-only in the core; heavy deps live on the box venv.

## Testing

- `python3 -m pytest tests/ -q --ignore=tests/integration` — anywhere, fast.
  Includes the dashboard API tests (`tests/test_dashboard_api.py`), which spin
  up the stdlib HTTP server on an ephemeral port with the engine never loaded,
  and `tests/test_codegen.py` for prompt/parse/validate + body store round-trips.
- `tests/integration/` — box only; requires real SemIf + real ollama.
- After touching scheduler/skills/codegen/engine, re-run both; the integration
  tests are the only end-to-end verification.