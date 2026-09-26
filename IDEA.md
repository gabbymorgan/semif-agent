# Semif Agent

A local desktop CLI agent whose control flow is a single decision model (SemIf). Inputs are gated, scored, queued, and dispatched through a skill tree; every decision is logged as a labeled training row, and the decision model is fine-tuned at regular intervals on the accumulated prediction-vs-observation cost.

## Engine: SemIf (local decision layer)

- **SemIf** (formerly OpenJev, TheoLeeCJ, MIT) is a local rebuild of Jev's interface pattern. One forward pass reads typed option logits directly from a model; no answer sentence, no decoding loop.
- Contract: input `{state, question, options[]}`, output per-option probabilities conditional on the supplied options. Probabilities are conditional on the option set, not absolute confidence — calibrate per workload (SemIf ships per-workload temperature scaling).
- **Why local**: desktop target, data stays in-network (SemIf's data boundary), runs on a home GPU (or CPU via llama.cpp). Jev (hosted) is not in scope.
- Decision model is pinned to an exact revision; every swap is auditable via prompt hashes in SemIf's output.

## Core loop

```
intake (typed / events / timers / other skills)
  └─ contains_request?          (gate: should the agent handle this? the state
                                 carries the user's expectation + available skills)
       └─ choice                (SemIf: "should this interrupt the current process?")
            ├─ yes → preempt current, requeue it with state preserved
            └─ no  → score      (SemIf: "how urgent?")
                      └─ priority queue, sorted by urgency desc
```

- **Scheduler**: run head of queue when idle. A new interrupt displaces the current process, which itself requeues preserving its state.
- **Concurrency** (TODO resolved): single execution slot for v1. SemIf's shared-state mode (one prefetched state branched across many criteria in parallel) is the lever for parallelizing decisions later, not concurrent process execution.
- **Tie-breaking**: FIFO within equal urgency, recency as secondary key.
- **Priority ageing**: queued items decay upward over time so they can't starve.
- **Bounds**: max queue depth; overflow rejects with a notification.

## Decision templates

All decisions are SemIf calls: `{state, question, options[]}`. State is the current process + input. Options are typed with descriptions.

- **`choice`** — binary: `interrupt` / `defer`. Interrupt iff `P(interrupt) >= τ`.
- **`score`** — ordinal urgency: `critical` / `high` / `medium` / `low`, mapped to numeric weights for sorting.
- **skill navigation** — at each tree level: choose category / descend; the category level offers a `create_category` suggestion and the leaf level a `create_skill` suggestion.
- **`read_next()`** — argument selection within a skill (e.g., which contact is "girlfriend").

**LLM/SemIf boundary**: SemIf for fast, repeated, low-latency decisions (gating, scoring, routing, argument selection). LLM for generation and assessment (email body, self-assessment summary). Never the reverse.

## Skill tree

- Structure: categories → skills → actions. Top level listed at each level.
- Navigation is a chain of SemIf choices, one per level, descending until a leaf skill matches.
- Navigation offers a `create_category` suggestion at the category level and a `create_skill` suggestion at the leaf level. Both are live: the decision model, driven in normal generation mode, authors a title + description (broad bucket for a category, single specific action for a skill), and the stub is persisted to the category registry and merged into the running tree. For a new skill the stub is then promoted to a runnable body: a separate, larger OpenAI-compatible model writes the `predict`/`act` code against the `SKILL.md` contract, persisted under `data/skills/` and hot-loaded. The newly created leaf is then executed directly (no re-dispatch through navigation) so the request that prompted creation is answered: `create_category` → `create_skill` → run, or `create_skill` → run.

### Skill manifest
- name, category, description, allowed inputs, action list, cost budget, decision log reference.

## Skill anatomy

Every skill run follows the same loop:

1. **observe baseline** — capture state relevant to the skill.
2. **make prediction** — predict outcome; log it.
3. **act** — execute actions (possibly calling `read_next()`-style SemIf decisions for arguments).
4. **observe outcome** — capture post-action state.
5. **self-assess** (LLM) — "have I succeeded?" → success/failure + summary + **updated request** back to the priority queue.

- The self-assessment result **labels** every SemIf decision made during that run (`choice`, `score`, navigation, argument selection): the predicted distribution vs. the observed outcome is one labeled training row.
- Per-skill cost budget: a run that exceeds it fails fast and requeues with a degraded goal.

## Learning / fine-tuning pipeline

- **Log**: every decision row → `{state, question, options, predicted_probs, observed_outcome}` — SemIf `decisions.jsonl`-compatible, so logs replay directly into the scorer.
- **Cost function**: loss between the predicted distribution and the observed outcome — cross-entropy / NLL under a proper scoring rule, plus optional calibration loss. This is the training signal.
- **Labels**: self-assessment output by default; human confirmations/corrections override and become high-weight labels.
- **"Dreaming"** — a cronjob at a regular interval: accumulate logged rows → compute cost over the period → fine-tune the decision model on them → CI/CD validates (accuracy / ECE on a held-out slice, prompt-hash regression) → swap in the new pinned model revision.
- **GPU offload**: training runs on a beefier GPU; the running agent keeps a frozen inference revision until a swap validates.
- Cold start: no labels yet — seed from self-assessed runs only; human overrides accelerate early calibration.

## Worked examples

### Example 1
- current_process: none
- input: "send my girlfriend an email that says that I'm going to be late to the party"
- contains_request: yes
- interrupt_current: yes
- skill_selection: email => compose => transcribe(input): [code block that uses SemIf `read_next()` to determine "girlfriend" from contacts, plus a normally functioning LLM for the email body]

### Example 2
- current_process: driving car
- input: "send my girlfriend an email that says that I'm going to be late to the party"
- interrupt_current: no
- priority: .21
- skill_selection: response => rejection(input)

### Example 3
- current_process: driving car
- input: "holy shit stop!"
- contains_request: yes
- interrupt_current: yes
- skill_selection: driving => decelerate(input)

### Example 4 (skill creation)
- current_process: none
- input: "tell me if my package was delivered"
- contains_request: yes
- interrupt_current: yes
- skill_selection: tracking => (no leaf) => create_skill
- the codegen model authors `tracking.check_delivery` (title+description from the decision model; `predict`/`act` body from a larger OpenAI-compatible model per `SKILL.md`) → body persisted + registered → requeue → skill_selection: tracking => check_delivery(input)

### Example 5 (self-assessment + dream)
- current_process: email compose
- input: "send my girlfriend an email…" → email compose succeeds but sends to the wrong contact (`read_next` picked wrong)
- self-assess: fail → labels the `read_next` decision row (predicted_probs vs wrong outcome)
- cost computed, logged
- next dream cycle: row included in fine-tune → model revision swapped → `read_next` recalibrated

## Open questions

- **Safety / authority**: which inputs may interrupt high-stakes processes (driving-grade)? Is interrupt a per-skill permission, not a global default?
- **Queue persistence**: is the priority queue in-memory or durable across restarts?
- **Input taxonomy**: enumerate intake sources and their `contains_request` gating semantics.
- **Privacy boundary**: all data stays local — confirm no telemetry even for dreaming.
- **Override UX**: how do human confirmations/corrections get surfaced and captured cheaply?