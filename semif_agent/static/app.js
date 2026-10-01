"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  runs: [],
  tree: { categories: {} },
  status: { current: null, pending: null, queue: [], tau: 0.6 },
  dream: {},
  selectedRunId: null,
  selectedDecisionId: null,
  phaseFilter: "",
  scrub: 0,
  flowSteps: [],
  answerFeedback: null,
  pendingSubmit: null,
};

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

async function refreshAll() {
  const [trace, tree, status, dream] = await Promise.all([
    getJSON("/api/trace"),
    getJSON("/api/tree"),
    getJSON("/api/status"),
    getJSON("/api/dream"),
  ]);
  state.runs = trace.runs;
  state.tree = tree;
  state.status = status;
  state.dream = dream;
  if (state.selectedRunId && !state.runs.some((r) => r.run_id === state.selectedRunId)) {
    state.selectedRunId = null;
    state.selectedDecisionId = null;
  }
  const pending = state.pendingSubmit;
  if (pending && pending.requestId && state.runs.some((r) => r.run_id === pending.requestId)) {
    state.selectedRunId = pending.requestId;
    state.selectedDecisionId = null;
    state.scrub = 0;
    state.pendingSubmit = null;
  }
  render();
}

function flash(msg) {
  $("#mode-badge").textContent = msg;
}

/* ---------------- phase helpers ---------------- */

function phaseClass(phase) {
  if (phase.startsWith("navigate:")) return "navigate";
  return phase;
}

function shortPhase(phase) {
  return phase.replace("navigate:", "nav:");
}

const ALL_PHASES = ["gate", "choice", "score", "navigate:category", "navigate:leaf", "act"];

function optionProbs(row) {
  return (row.options || []).map((o) => ({
    id: o.id,
    description: o.description,
    p: (row.predicted_probs || {})[o.id] || 0,
  }));
}

/* ---------------- timeline ---------------- */

function renderTimeline() {
  const el = $("#timeline");
  el.innerHTML = "";
  for (const run of state.runs) {
    const decisions = run.decisions.filter(
      (d) => !state.phaseFilter || d.phase === state.phaseFilter
    );
    if (state.phaseFilter && decisions.length === 0) continue;

    const group = document.createElement("div");
    group.className = "run-group";

    const head = document.createElement("div");
    head.className = "run-head";
    head.textContent = run.decisions[0] ? run.decisions[0].state.slice(0, 60) : run.run_id;
    head.title = run.run_id;
    head.addEventListener("click", () => {
      state.selectedRunId = run.run_id;
      state.selectedDecisionId = null;
      state.scrub = 0;
      render();
    });
    group.appendChild(head);

    if (state.selectedRunId === run.run_id) {
      for (const evt of run.events) {
        group.appendChild(eventRow(evt));
      }
      for (const d of decisions) {
        group.appendChild(decisionRow(d));
      }
    } else {
      group.appendChild(dimRow(`${decisions.length} decisions`));
    }
    el.appendChild(group);
  }
}

function dimRow(text) {
  const div = document.createElement("div");
  div.className = "trow event";
  div.textContent = text;
  return div;
}

function eventRow(evt) {
  const div = document.createElement("div");
  div.className = "trow event";
  div.textContent = evt.kind;
  if (evt.kind === "assessed") {
    div.textContent = `assessed → ${evt.success ? "ok" : "fail"}`;
    div.title = evt.summary || "";
  } else if (evt.kind === "queued") {
    div.textContent = `queued (${evt.label})`;
  } else if (evt.kind === "preempted") {
    div.textContent = `preempted ${evt.preempted}`;
  } else if (evt.kind === "skill_writing") {
    div.textContent = `writing skill ${evt.skill} — ${evt.description} (${evt.model || "codegen"})`;
    if (evt.contract_ref) {
      div.title = `SKILL.md @ ${evt.contract_ref}${evt.contract_dirty ? "*" : ""}`;
    }
  } else if (evt.kind === "skill_created") {
    div.textContent = `created ${evt.skill}${evt.written ? " (body written)" : " (stub)"}`;
    div.title = evt.body || evt.description || "";
  } else if (evt.kind === "skill_testing") {
    div.textContent = `test attempt ${evt.attempt}: ${evt.passed ? "passed" : "failed"}`;
    div.title = evt.output || "";
  } else if (evt.kind === "skill_ready") {
    div.textContent = `ready: ${evt.skill}`;
  } else if (evt.kind === "codegen_regen") {
    div.textContent = `regen → ${evt.selected}`;
  } else if (evt.kind === "requirements") {
    div.textContent = "refinement questions asked";
  } else if (evt.kind === "skill_write_failed") {
    div.textContent = `body write failed: ${evt.skill}`;
    div.title = evt.message || "";
  } else if (evt.kind === "skill_restarted") {
    div.textContent = `restarted body write: ${evt.skill}`;
  } else if (evt.kind === "needs_input") {
    div.textContent = "needs input";
    div.title = evt.question || "";
  } else if (evt.kind === "answered") {
    div.textContent = `answered: ${evt.text || ""}`;
  } else if (evt.kind === "pending_abandoned") {
    div.textContent = "pending input abandoned";
  }
  return div;
}

function decisionRow(d) {
  const div = document.createElement("div");
  div.className = "trow" + (d.id === state.selectedDecisionId ? " selected" : "");
  div.appendChild(pill(d.phase));
  if (d.label_source === "human") div.appendChild(star());
  const label = document.createElement("span");
  label.textContent = `${d.selected}  ${(d.predicted_probs[d.selected] || 0).toFixed(2)}`;
  div.appendChild(label);
  if (d.cost && !d.cost.correct) div.appendChild(mark("x"));
  div.title = d.question;
  div.addEventListener("click", () => {
    state.selectedRunId = (d.extra || {}).run_id;
    state.selectedDecisionId = d.id;
    render();
  });
  return div;
}

function pill(text) {
  const span = document.createElement("span");
  span.className = `pill ${phaseClass(text)}`;
  span.textContent = shortPhase(text);
  return span;
}

function star() {
  const span = document.createElement("span");
  span.className = "human-star";
  span.textContent = "★";
  return span;
}

function mark(kind) {
  const span = document.createElement("span");
  span.className = kind === "x" ? "x" : "chk";
  span.textContent = kind === "x" ? "✗" : "✓";
  return span;
}

/* ---------------- flow graph ---------------- */

function buildFlowSteps(run) {
  const items = [];
  for (const evt of run.events) items.push({ kind: "event", ts: evt.ts, data: evt });
  for (const d of run.decisions) items.push({ kind: "decision", ts: d.ts, data: d });
  items.sort((a, b) => a.ts - b.ts);
  return items;
}

function renderFlow() {
  const el = $("#flow");
  const run = state.runs.find((r) => r.run_id === state.selectedRunId);
  $("#run-label").textContent = run ? run.run_id : "";
  $("#scrubber").max = 0;
  $("#scrubber").value = 0;
  state.flowSteps = [];
  el.innerHTML = "";

  if (!run) {
    if (state.pendingSubmit) {
      renderPendingFlow(el);
      return;
    }
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "select a run from the timeline";
    el.appendChild(empty);
    return;
  }

  const steps = buildFlowSteps(run);
  state.flowSteps = steps;
  $("#scrubber").max = Math.max(0, steps.length - 1);
  $("#scrubber").value = 0;

  for (let i = 0; i < steps.length; i++) {
    if (i > 0) el.appendChild(edge(steps[i - 1].data));
    const node = steps[i].kind === "event" ? eventNode(steps[i].data) : decisionNode(steps[i].data);
    node.dataset.step = i;
    if (i > state.scrub) node.classList.add("dim");
    el.appendChild(node);
  }

  if (run.decisions.length === 0 && runInFlight(run)) {
    if (steps.length) el.appendChild(edge(steps[steps.length - 1].data));
    el.appendChild(pendingIndicatorNode("decisions in process…"));
  }
}

const RESTING_EVENTS = ["ran", "rejected", "dropped", "error", "assessed", "needs_input", "pending_abandoned"];

function runInFlight(run) {
  return !run.events.some((evt) => RESTING_EVENTS.includes(evt.kind));
}

function renderPendingFlow(el) {
  const node = document.createElement("div");
  node.className = "node event-node pending-request";
  const kind = document.createElement("div");
  kind.className = "evt-kind";
  kind.textContent = "submitted";
  node.appendChild(kind);
  const body = document.createElement("div");
  body.className = "node-question";
  body.textContent = state.pendingSubmit.text;
  node.appendChild(body);
  el.appendChild(node);
  el.appendChild(edge({ kind: "event" }));
  el.appendChild(pendingIndicatorNode("decisions in process…"));
}

function pendingIndicatorNode(text) {
  const node = document.createElement("div");
  node.className = "node pending-indicator";
  const dot = document.createElement("span");
  dot.className = "spinner";
  node.appendChild(dot);
  const label = document.createElement("span");
  label.textContent = text;
  node.appendChild(label);
  return node;
}

function edge(prev) {
  const div = document.createElement("div");
  div.className = "edge";
  if (prev.kind === "decision") {
    const p = (prev.predicted_probs || {})[prev.selected] || 0;
    if (p >= 0.6) div.classList.add("hot");
  }
  return div;
}

function decisionNode(d) {
  const node = document.createElement("div");
  const ok = d.cost ? d.cost.correct : null;
  node.className = "node" + (d.id === state.selectedDecisionId ? " selected" : "");
  if (ok === true) node.classList.add("ok");
  if (ok === false) node.classList.add("fail");

  const head = document.createElement("div");
  head.className = "node-head";
  head.appendChild(pill(d.phase));
  const q = document.createElement("div");
  q.className = "node-question";
  q.textContent = d.question;
  head.appendChild(q);
  if (d.label_source === "human") head.appendChild(star());
  const id = document.createElement("div");
  id.className = "node-id";
  id.textContent = d.id;
  head.appendChild(id);
  node.appendChild(head);

  for (const opt of optionProbs(d)) {
    node.appendChild(optionRow(opt, d));
  }

  const meta = document.createElement("div");
  meta.className = "node-meta";
  const sel = document.createElement("span");
  sel.textContent = `selected: ${d.selected}`;
  meta.appendChild(sel);
  if (d.label_source === "human") {
    const obs = document.createElement("span");
    obs.textContent = `human override → ${d.observed_outcome}`;
    obs.style.color = "var(--human)";
    meta.appendChild(obs);
  }
  if (d.cost) {
    const nll = document.createElement("span");
    nll.className = "cost-nll";
    nll.textContent = `nll ${d.cost.nll.toFixed(3)} ×${d.cost.weight}`;
    meta.appendChild(nll);
  }
  const ex = d.extra || {};
  const timing = document.createElement("span");
  timing.textContent = ex.total_seconds ? `${ex.total_seconds.toFixed(2)}s` : "";
  meta.appendChild(timing);
  node.appendChild(meta);

  node.addEventListener("click", () => {
    state.selectedDecisionId = d.id;
    state.selectedRunId = (d.extra || {}).run_id;
    render();
  });
  return node;
}

function optionRow(opt, d) {
  const row = document.createElement("div");
  const cls = ["opt-row"];
  if (opt.id === d.selected) cls.push("selected");
  row.className = cls.join(" ");

  const label = document.createElement("span");
  label.className = "opt-label";
  label.textContent = opt.id;
  label.title = opt.description;
  row.appendChild(label);

  const barTrack = document.createElement("span");
  barTrack.className = "opt-bar-track";
  const bar = document.createElement("span");
  bar.className = "opt-bar";
  bar.style.width = `${Math.max(opt.p * 100, 1)}%`;
  barTrack.appendChild(bar);
  row.appendChild(barTrack);

  const marks = document.createElement("span");
  marks.className = "opt-marks";
  if (opt.id === d.observed_outcome && opt.id !== d.selected) marks.appendChild(mark("chk"));
  row.appendChild(marks);

  const pct = document.createElement("span");
  pct.className = "opt-pct";
  pct.textContent = `${(opt.p * 100).toFixed(0)}%`;
  row.appendChild(pct);
  return row;
}

function eventNode(evt) {
  const node = document.createElement("div");
  node.className = "node event-node";
  if (evt.kind === "assessed") node.classList.add(evt.success ? "ok" : "fail");
  const kind = document.createElement("div");
  kind.className = "evt-kind";
  kind.textContent = evt.kind;
  node.appendChild(kind);
  const body = document.createElement("div");
  body.className = "node-question";
  if (evt.kind === "assessed") {
    body.textContent = evt.summary || "";
    if (evt.updated_request) {
      const req = document.createElement("div");
      req.className = "muted";
      req.textContent = `→ requeued: ${evt.updated_request}`;
      node.appendChild(req);
    }
  } else if (evt.kind === "queued") {
    body.textContent = `urgency ${evt.label} (weight ${Number(evt.weight || 0).toFixed(2)})`;
  } else if (evt.kind === "preempted") {
    body.textContent = `interrupted ${evt.preempted}, requeued with state`;
  } else if (evt.kind === "dropped") {
    body.textContent = evt.reason || "";
  } else if (evt.kind === "skill_writing") {
    node.classList.add("writing");
    const title = document.createElement("div");
    title.className = "skill-title";
    title.textContent = evt.skill;
    body.appendChild(title);
    const desc = document.createElement("div");
    desc.className = "muted";
    desc.textContent = evt.description || "";
    body.appendChild(desc);
    if (evt.contract_ref) {
      const ref = document.createElement("div");
      ref.className = "muted";
      ref.textContent = `SKILL.md @ ${evt.contract_ref}${evt.contract_dirty ? "*" : ""}`;
      body.appendChild(ref);
    }
    const badge = document.createElement("span");
    badge.className = "writing-badge";
    badge.textContent = "writing skill body…";
    node.appendChild(badge);
  } else if (evt.kind === "skill_created") {
    const title = document.createElement("div");
    title.className = "skill-title";
    title.textContent = evt.skill;
    body.appendChild(title);
    const desc = document.createElement("div");
    desc.className = "muted";
    desc.textContent = evt.description || "";
    body.appendChild(desc);
    if (evt.body) {
      const p = document.createElement("div");
      p.className = "muted";
      p.textContent = `body: ${evt.body}`;
      node.appendChild(p);
    }
    node.classList.add(evt.written ? "ok" : "stub");
  } else if (evt.kind === "skill_testing") {
    node.classList.add(evt.passed ? "ok" : "fail");
    body.textContent = `test attempt ${evt.attempt}: ${evt.passed ? "passed" : "failed"}`;
    if (evt.output) {
      const out = document.createElement("div");
      out.className = "muted";
      out.textContent = evt.output.slice(0, 200);
      body.appendChild(out);
    }
  } else if (evt.kind === "skill_ready") {
    node.classList.add("ok");
    const title = document.createElement("div");
    title.className = "skill-title";
    title.textContent = `${evt.skill} — ready`;
    body.appendChild(title);
    if (evt.contract_vars) {
      const vars = document.createElement("div");
      vars.className = "muted";
      vars.textContent = `contract: ${evt.contract_vars.join(", ")}`;
      body.appendChild(vars);
    }
  } else if (evt.kind === "codegen_regen") {
    body.textContent = `regen decision → ${evt.selected || ""}`;
    if (evt.reason) {
      const out = document.createElement("div");
      out.className = "muted";
      out.textContent = evt.reason.slice(0, 160);
      body.appendChild(out);
    }
  } else if (evt.kind === "requirements") {
    body.textContent = `refinement: ${(evt.questions || []).join(" | ")}`;
  } else if (evt.kind === "skill_write_failed") {
    node.classList.add("stub");
    const title = document.createElement("div");
    title.className = "skill-title";
    title.textContent = `${evt.skill} — body write failed`;
    body.appendChild(title);
    const msg = document.createElement("div");
    msg.className = "muted";
    msg.textContent = evt.message || "";
    body.appendChild(msg);
    const btn = document.createElement("button");
    btn.className = "restart-btn";
    btn.textContent = "restart";
    btn.addEventListener("click", () => restartSkill(evt.category, evt.skill));
    node.appendChild(btn);
  } else if (evt.kind === "skill_restarted") {
    body.textContent = `restarted body write for ${evt.skill}`;
  } else if (evt.kind === "needs_input") {
    node.classList.add("needs-input");
    body.textContent = `awaiting input: ${evt.question || ""}`;
  } else if (evt.kind === "answered") {
    body.textContent = `answered: ${evt.text || ""}`;
  } else if (evt.kind === "pending_abandoned") {
    body.textContent = "pending input abandoned";
  } else {
    body.textContent = evt.summary || evt.text || "";
  }
  node.appendChild(body);
  return node;
}

/* ---------------- inspector ---------------- */

function selectedDecision() {
  if (!state.selectedDecisionId) return null;
  for (const run of state.runs) {
    for (const d of run.decisions) {
      if (d.id === state.selectedDecisionId) return d;
    }
  }
  return null;
}

function renderInspector() {
  const el = $("#inspector");
  const d = selectedDecision();
  $("#relabel-btn").disabled = !d;
  if (!d) {
    el.innerHTML = '<div class="muted">click a decision node to inspect it</div>';
    return;
  }
  el.innerHTML = "";
  el.appendChild(kv("id", d.id));
  el.appendChild(kv("phase", d.phase || "—"));
  el.appendChild(kv("state", d.state));
  const pre = document.createElement("pre");
  pre.className = "json";
  const copy = { ...d };
  delete copy.cost;
  pre.textContent = JSON.stringify(copy, null, 2);
  el.appendChild(pre);
}

function kv(k, v) {
  const div = document.createElement("div");
  div.className = "kv";
  div.innerHTML = `<span class="k">${k}:</span> `;
  div.appendChild(document.createTextNode(v));
  return div;
}

/* ---------------- skill tree ---------------- */

async function restartSkill(category, name) {
  try {
    const res = await getJSON("/api/restart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category, skill: name }),
    });
    flash(`[${res.status}] ${res.detail}`);
  } catch (err) {
    flash(`restart failed: ${err.message}`);
  }
  await refreshAll();
}

async function answerQuestion(id, text) {
  try {
    const res = await getJSON("/api/questions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, text }),
    });
    flash(`[${res.status}] ${res.detail}`);
  } catch (err) {
    flash(`answer failed: ${err.message}`);
  }
  await refreshAll();
}

async function resolveRepair(id, action) {
  try {
    const res = await getJSON("/api/repair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, action }),
    });
    flash(`[${res.status}] ${res.detail}`);
  } catch (err) {
    flash(`repair failed: ${err.message}`);
  }
  await refreshAll();
}

function skillStatusBadge(status, category, name) {
  const span = document.createElement("span");
  if (status === "writing") {
    span.className = "sbadge writing";
    span.textContent = "writing…";
  } else if (status === "stub") {
    span.className = "sbadge stub";
    span.textContent = "stub";
    const btn = document.createElement("button");
    btn.className = "restart-btn";
    btn.textContent = "restart";
    btn.title = `write the body for ${category}.${name}`;
    btn.addEventListener("click", () => restartSkill(category, name));
    span.appendChild(btn);
  }
  return span;
}

function renderTree() {
  const el = $("#skill-tree");
  el.innerHTML = "";
  const cats = state.tree.categories || {};
  for (const [cat, skills] of Object.entries(cats)) {
    const div = document.createElement("div");
    div.className = "cat";
    const name = document.createElement("div");
    name.className = "cat-name";
    name.textContent = cat;
    div.appendChild(name);
    for (const skill of skills) {
      const row = document.createElement("div");
      row.className = "skill-row";
      const nm = document.createElement("span");
      nm.textContent = skill.name;
      const desc = document.createElement("span");
      desc.className = "sdesc";
      desc.textContent = skill.description;
      row.appendChild(nm);
      row.appendChild(desc);
      if (skill.integration && skill.integration.service && skill.integration.service !== "unknown") {
        const ig = document.createElement("span");
        ig.className = "sdesc";
        ig.textContent = `${skill.integration.service} · ${skill.integration.transport}`;
        row.appendChild(ig);
      }
      if (skill.status === "ready" && skill.integration_source && skill.integration_source !== "declared") {
        const un = document.createElement("span");
        un.className = "sbadge stub";
        un.textContent = "unverified";
        un.title = "no valid INTEGRATION declaration; the body was inferred";
        row.appendChild(un);
      }
      row.appendChild(skillStatusBadge(skill.status, cat, skill.name));
      div.appendChild(row);
    }
    el.appendChild(div);
  }
}

/* ---------------- status / dream ---------------- */

function renderStatus() {
  const el = $("#status");
  el.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "stat-grid";

  const current = state.status.current
    ? `${state.status.current.skill} (${state.status.current.request_id})`
    : "idle";
  grid.appendChild(stat("current", current));
  grid.appendChild(stat("queue", String(state.status.queue.length)));
  grid.appendChild(stat("tau", String(state.status.tau)));
  grid.appendChild(stat("rows", String(state.dream.rows)));

  const acc = state.dream.accuracy;
  const ce = state.dream.cross_entropy;
  const ece = state.dream.ece;
  grid.appendChild(stat("acc", acc == null ? "n/a" : acc.toFixed(3)));
  grid.appendChild(stat("CE", ce == null ? "n/a" : ce.toFixed(4)));
  grid.appendChild(stat("ECE", ece == null ? "n/a" : ece.toFixed(4)));
  grid.appendChild(stat("human", String(state.dream.human_overrides)));

  el.appendChild(grid);
  const q = document.createElement("div");
  q.className = "muted";
  q.style.marginTop = "8px";
  q.textContent = state.status.queue
    .map((item) => `${item.id} w=${item.weight.toFixed(2)} ${item.text}`)
    .join("\n") || "queue empty";
  el.appendChild(q);

  const pending = state.status.pending;
  $("#answer-form").classList.toggle("hidden", !pending);
  if (pending) {
    $("#answer-input").placeholder = `${pending.skill}: ${pending.question}`;
    const p = document.createElement("div");
    p.className = "pending-line";
    p.textContent = `awaiting input (${pending.run_id || pending.skill}): ${pending.question}`;
    el.appendChild(p);
  }
  renderQuestions();
  renderAnswerFeedback();
}

function renderQuestions() {
  const el = $("#status");
  for (const q of state.status.questions || []) {
    const box = document.createElement("div");
    box.className = "pending-line";
    const label = document.createElement("div");
    label.textContent = `? ${q.skill}: ${q.question}`;
    const form = document.createElement("form");
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "answer (empty to skip)";
    const btn = document.createElement("button");
    btn.type = "submit";
    btn.textContent = "answer";
    form.appendChild(input);
    form.appendChild(btn);
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      answerQuestion(q.id, input.value.trim());
      input.value = "";
    });
    box.appendChild(label);
    box.appendChild(form);
    el.appendChild(box);
  }
  for (const r of state.status.repairs || []) {
    const box = document.createElement("div");
    box.className = "pending-line";
    const label = document.createElement("div");
    label.textContent = `repair ${r.skill}: ${r.failure.slice(0, 160)}`;
    box.appendChild(label);
    const actions = document.createElement("div");
    for (const action of ["retry", "repair_skill", "ask_user", "no_repair"]) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = action;
      btn.className = action === r.selected ? "primary" : "";
      btn.addEventListener("click", () => resolveRepair(r.id, action));
      actions.appendChild(btn);
    }
    box.appendChild(actions);
    el.appendChild(box);
  }
}

function renderAnswerFeedback() {
  const el = $("#answer-feedback");
  const fb = state.answerFeedback;
  el.className = "hidden";
  if (!fb) return;
  if (fb.inflight) {
    el.className = "";
    el.textContent = "answering…";
    return;
  }
  if (fb.status === "error") {
    el.className = "feedback-error";
    el.textContent = `✗ ${fb.detail}`;
    return;
  }
  let line = `✓ consumed "${fb.text}"`;
  if (fb.status === "needs_input") {
    el.className = "feedback-awaiting";
    line += ` → awaiting: ${fb.detail}`;
    if (fb.askedQuestion && fb.detail === fb.askedQuestion) {
      line += " (answer received; the skill asked again)";
    }
  } else {
    el.className = "";
    line += ` → ${fb.detail}`;
  }
  el.textContent = line;
}

function stat(k, v) {
  const div = document.createElement("div");
  div.className = "stat";
  div.innerHTML = `<span class="muted">${k}</span><br><span class="v">${v}</span>`;
  return div;
}

/* ---------------- scrubber ---------------- */

function onScrub() {
  state.scrub = Number($("#scrubber").value);
  document.querySelectorAll("#flow .node").forEach((node) => {
    node.classList.toggle("dim", Number(node.dataset.step) > state.scrub);
  });
}

/* ---------------- relabel modal ---------------- */

function openRelabel() {
  const d = selectedDecision();
  if (!d) return;
  $("#relabel-id").textContent = `${d.id} — ${d.question}`;
  const sel = $("#relabel-options");
  sel.innerHTML = "";
  for (const o of d.options) {
    const opt = document.createElement("option");
    opt.value = o.id;
    opt.textContent = `${o.id} — ${o.description}`;
    sel.appendChild(opt);
  }
  sel.value = d.observed_outcome;
  $("#relabel-modal").classList.remove("hidden");
}

async function applyRelabel() {
  const d = selectedDecision();
  if (!d) return;
  const outcome = $("#relabel-options").value;
  const res = await getJSON("/api/relabel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: d.id, outcome }),
  });
  $("#relabel-modal").classList.add("hidden");
  if (res.ok) {
    flash(`relabeled ${d.id} → ${outcome}`);
    await refreshAll();
  } else {
    flash("relabel failed");
  }
}

/* ---------------- wiring ---------------- */

$("#submit-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("#query-input").value.trim();
  if (!text) return;
  $("#query-input").value = "";
  state.pendingSubmit = { text, requestId: null };
  state.selectedRunId = null;
  state.selectedDecisionId = null;
  state.scrub = 0;
  render();
  try {
    const res = await getJSON("/api/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    flash(`[${res.status}] ${res.detail}`);
    if (res.request_id) {
      state.pendingSubmit.requestId = res.request_id;
    } else {
      state.pendingSubmit = null;
    }
  } catch (err) {
    flash(`submit failed: ${err.message}`);
    state.pendingSubmit = null;
  }
  await refreshAll();
});

$("#answer-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("#answer-input").value.trim();
  if (!text) return;
  const btn = $("#answer-btn");
  const pending = state.status.pending || {};
  const runId = pending.run_id || null;
  const askedQuestion = pending.question || null;
  $("#answer-input").value = "";
  btn.disabled = true;
  state.answerFeedback = { inflight: true };
  renderAnswerFeedback();
  try {
    const res = await getJSON("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    flash(`[${res.status}] ${res.detail}`);
    state.answerFeedback = { status: res.status, detail: res.detail, text, askedQuestion };
  } catch (err) {
    flash(`answer failed: ${err.message}`);
    state.answerFeedback = {
      status: "error",
      detail: `answer failed: ${err.message}`,
      text,
      askedQuestion,
    };
  } finally {
    btn.disabled = false;
  }
  await refreshAll();
  if (runId && state.runs.some((r) => r.run_id === runId)) {
    state.selectedRunId = runId;
    state.selectedDecisionId = null;
    state.scrub = 0;
    render();
  }
});

$("#refresh-btn").addEventListener("click", async () => {
  try {
    await refreshAll();
  } catch (err) {
    flash(`refresh failed: ${err.message}`);
  }
});

$("#phase-filter").addEventListener("change", (e) => {
  state.phaseFilter = e.target.value;
  render();
});

$("#scrubber").addEventListener("input", onScrub);
$("#relabel-btn").addEventListener("click", openRelabel);
$("#relabel-cancel").addEventListener("click", () => $("#relabel-modal").classList.add("hidden"));
$("#relabel-apply").addEventListener("click", applyRelabel);

function render() {
  renderTimeline();
  renderFlow();
  renderInspector();
  renderTree();
  renderStatus();
}

function init() {
  const filter = $("#phase-filter");
  for (const p of ALL_PHASES) {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = p;
    filter.appendChild(opt);
  }
  refreshAll().catch((err) => flash(`failed to load: ${err.message}`));
  setInterval(() => refreshAll().catch(() => {}), 4000);
}

init();