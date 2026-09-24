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
scheduler.py    gate -> choice(tau) -> score -> queue; preempt + requeue
queue.py        urgency max-heap (desc weight, FIFO seq), age pulls toward 1.0
skills.py    tree + registry (email.compose, response.reject, tracking.check),
                navigation = SemIf choices per level (logged), create_category
                and create_skill author + register stubs via the decision model
                in generation mode
skill.py        loop: observe -> predict -> act -> observe -> assess (LLM)
engine.py       SemIfEngine -> semif_phase1.llamacpp_backend (lazy import)
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
- `create_skill` branch: now live, mirroring `create_category`. The decision
  model, driven in normal generation mode via `SemIfEngine.generate`, proposes a
  specific skill title + description for the chosen category; the stub is
  persisted to `data/categories.json` (under that category's `skills` list) and
  merged into the running tree as a leaf. Still deferred: a real skill body —
  opencode authoring at a tree leaf remains future work.
- Queue persistence (durable across restarts).
- Event/timer intake sources beyond typed input.
- Concurrency: SemIf shared-state mode (`score_shared` / `SerialPrefixScorer`)
  for parallel decisions; single execution slot remains for processes.
- Dashboard: run-requeue cross-linking (child run references its parent),
  scheduler sim controls (busy/idle/tau) as a first-class panel.

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
  up the stdlib HTTP server on an ephemeral port with the engine never loaded.
- `tests/integration/` — box only; requires real SemIf + real ollama.
- After touching scheduler/skills/engine, re-run both; the integration tests are
  the only end-to-end verification.