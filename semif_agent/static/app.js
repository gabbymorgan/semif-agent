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

const ALL_PHASES = ["gate", "choice", "score", "navigate:category", "navigate:leaf", "predict"];

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
  }
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
  try {
    const res = await getJSON("/api/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    flash(`[${res.status}] ${res.detail}`);
  } catch (err) {
    flash(`submit failed: ${err.message}`);
  }
  $("#query-input").value = "";
  await refreshAll();
});

$("#answer-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("#answer-input").value.trim();
  if (!text) return;
  try {
    const res = await getJSON("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    flash(`[${res.status}] ${res.detail}`);
  } catch (err) {
    flash(`answer failed: ${err.message}`);
  }
  $("#answer-input").value = "";
  await refreshAll();
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