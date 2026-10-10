"""The scheduler: queue, dispatch.

Every SemIf decision (navigation, sub-decision) is logged. Input that arrives
while the single execution slot is busy waits in a strict FIFO queue and is
dispatched in arrival order; there is no urgency scoring and no preemption.
There is no up-front handle/ignore gate: every input is dispatched, and inputs
that are not tasks fall through navigation into the closed `response` tree of
canned replies.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .codegen import (
    CodegenClient,
    CodegenError,
    extract_integration,
    generate_elicitation,
    generate_skill_body,
    generate_skill_tests,
    integration_findings,
    parse_contract,
    regenerate_skill_body,
    run_skill_test,
    skill_contract_ref,
)
from .decisions import DecisionRequest, Option, Request
from .engine import DecisionEngine, EngineUnavailable
from .llm import LLMClient
from .log import DecisionLog
from .provider import ProviderError
from .queue import RequestQueue
from .skill import SkillRunner
from .timers import TimerService
from .skills import (
    CANNED_CATEGORIES,
    DETERMINISTIC_CATEGORIES,
    HARD_LOCKED_CATEGORIES,
    HOUSEKEEPING_CATEGORY,
    ActionContext,
    CategoryRegistry,
    CreateCategory,
    CreateSkill,
    DeletedSkills,
    Skill,
    SkillDraft,
    SkillStore,
    apply_tombstones,
    build_housekeeping_skills,
    build_skills,
    build_tree,
    confirm_skill_fit,
    generate_category,
    generate_skill,
    materialize_skill,
    merge_registry,
    merge_seed_store,
    merge_skill_store,
    navigate,
    parse_meta_command,
)
from .trace import TraceLog

#: The tri-state governing whether a new skill (or category) may be authored.
#: `allow` authors immediately, `deny` blocks creation outright, and `ask`
#: authors only after the human approves the proposal. Set globally with the
#: top-level `new_skill_creation` config value and overridden per category in
#: `data/skills/<category>/config.json` (same key).
NEW_SKILL_MODES = ("allow", "deny", "ask")


def normalize_new_skill_mode(value) -> str | None:
    """Coerce a configured new-skill mode to one of `NEW_SKILL_MODES`.

    Accepts the mode strings and, for backward compatibility with the retired
    `locks.new_skill` boolean, maps `True` -> "deny" and `False` -> "allow".
    Returns None for anything unrecognized so the caller can fall back to the
    next tier (category override, then the global default).
    """
    if isinstance(value, bool):
        return "deny" if value else "allow"
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in NEW_SKILL_MODES:
            return mode
    return None


@dataclass
class Process:
    request: Request
    skill: str


@dataclass
class PendingRun:
    """A skill run paused awaiting human input.

    `question` is what the run asked the human. A `pre_act` pause (contract
    variables collected by the runner before act) re-runs the full resolution
    + act path on resume; an act-driven pause re-invokes `act`.
    """

    request: Request
    skill: Skill
    question: str
    pre_act: bool = False


@dataclass
class SkillWrite:
    """One queued async skill-body write (single-slot codegen worker)."""

    request: Request
    category: str
    draft: SkillDraft
    repair_evidence: dict | None = None
    # Label for a corrective rewrite: "run_failure" (repair loop), "fidelity",
    # "test", or "manual" (a user-requested regen).
    reason_kind: str = "run_failure"


@dataclass
class DraftAuthor:
    """One queued async draft authoring (single-slot `llm` worker).

    `kind` is "category" (author a top-level category) or "skill" (author a
    leaf inside `category`). A category job chains into its skill job so the
    create_category -> create_skill order stays deterministic. `approved` marks
    a chained skill job whose parent category was already approved, so the
    creation-approval prompt is asked once per request, not once per level.
    """

    request: Request
    category: str | None
    kind: str
    approved: bool = False


@dataclass
class PendingQuestion:
    """A question waiting for the human (deferred elicitation or repair).

    Posted by the codegen worker; answered via
    the dashboard (`/api/questions`) or `Scheduler.answer_question`. Elicitation
    posts one at a time and waits `codegen.elicitation.answer_timeout` seconds
    for each (the clock resets on every answer); the worker then continues with
    whatever answers arrived.
    """

    id: str
    run_id: str
    category: str
    skill: str
    question: str
    kind: str = "elicitation"
    offer_id: str | None = None
    answer: str | None = None
    skipped: bool = False
    created_at: float = field(default_factory=time.time)


@dataclass
class RepairOffer:
    """A failed real run plus the SemIf-chosen repair, awaiting the user.

    The app says the task failed and offers to fix/learn; executing a repair is
    user-confirmed because a codegen write takes tens of minutes. `retry` and
    `decline` are cheap; `repair_skill` regenerates the body with the observed
    failure; `ask_user` collects a missing detail (via PendingQuestion) and
    then repairs.
    """

    id: str
    run_id: str
    category: str
    skill: str
    selected: str
    reason: str
    request_text: str
    failure: str
    evidence: dict = field(default_factory=dict)
    status: str = "offered"
    created_at: float = field(default_factory=time.time)


@dataclass
class PendingApproval:
    """A proposed category/skill awaiting the human's yes/no.

    Posted by the single-slot `llm` draft worker after the proposal is authored
    and before anything is registered. Only posted when the effective new-skill
    mode is `ask` (the global default or a per-category override); a denial or
    timeout aborts creation (nothing registered, no body written, the request is
    not re-dispatched). One prompt per creation request: the category prompt
    covers its chained skill, so a new category asks once.
    """

    id: str
    run_id: str
    kind: str  # category | skill
    category: str
    skill: str
    description: str
    status: str = "pending"  # pending | approved | denied | timeout
    created_at: float = field(default_factory=time.time)


@dataclass
class DispatchResult:
    kind: str  # ran | create_category | create_skill | needs_input | error
    summary: str
    skill: str | None = None
    decisions_logged: int = 0
    needs_input: str | None = None
    #: `category.name` when a skill produced the outcome, else None. Carried
    #: structurally so front ends never parse it back out of `summary`.
    skill_ref: str | None = None
    #: The skill's own detail (its `action_log`, or `new_state` when empty) —
    #: the unwrapped result a `result_only` front end speaks.
    result: str | None = None


@dataclass
class SchedulerReply:
    """One scheduler outcome as a structured reply.

    `text` is the human-facing summary (e.g. ``"time.now: ok — 12:00"``). The
    machine-readable fields a front end needs — the run id, the skill ref, and
    the skill's own unwrapped detail — ride alongside it instead of being
    parsed back out of `text`.
    """

    status: str
    text: str
    run_id: str = ""
    skill_ref: str | None = None
    result: str | None = None


class Scheduler:
    def __init__(
        self,
        engine: DecisionEngine,
        llm: LLMClient,
        log: DecisionLog,
        config: dict,
        tau: float = 0.6,
        max_reentries: int = 3,
        trace: TraceLog | None = None,
        codegen: CodegenClient | None = None,
        degeneration_check_factory: Callable[[str], Callable[[str], str | None] | None]
        | None = None,
        regen_decision_factory: Callable[[str], Callable[[str], str] | None]
        | None = None,
        on_request_requeued: Callable[[str, str], None] | None = None,
    ):
        self.engine = engine
        self.llm = llm
        self.log = log
        self.trace = trace if trace is not None else TraceLog()
        self.config = config
        self.tau = tau
        nav_cfg = config.get("navigation", {}) or {}
        self.intent_tau = float(nav_cfg.get("intent_tau", 0.6))
        self.category_tau = float(nav_cfg.get("category_tau", 0.75))
        self.action_tau = float(nav_cfg.get("action_tau", 0.5))
        #: A **category** softmax winner at or above this probability skips the
        #: category scope confirm — the winner is already decisive. The leaf
        #: intent guard is never bypassed (the leaf softmax has no create
        #: option, so its confidence carries no match signal).
        self.bypass_tau = float(nav_cfg.get("softmax_bypass_tau", 0.5))
        self.max_reentries = max_reentries
        self.codegen = codegen
        self.degeneration_check_factory = degeneration_check_factory
        self.regen_decision_factory = regen_decision_factory
        self.on_request_requeued = on_request_requeued
        codegen_cfg = config.get("codegen", {}) or {}
        elicitation_cfg = codegen_cfg.get("elicitation", {}) or {}
        self.elicitation_enabled = bool(elicitation_cfg.get("enabled", True))
        self.elicitation_max = int(elicitation_cfg.get("max_questions", 4))
        self.elicitation_wait = float(elicitation_cfg.get("wait_timeout", 900.0))
        self.elicitation_answer_timeout = float(
            elicitation_cfg.get("answer_timeout", 30.0)
        )
        self.defer_questions = False
        #: Global new-skill creation mode: "allow" (default), "deny", or "ask".
        #: A category may override it in its category config. The retired
        #: boolean `creation_approval` is still read as a fallback (True -> ask,
        #: False -> allow) so an existing config keeps its behaviour.
        self.new_skill_mode = (
            normalize_new_skill_mode(config.get("new_skill_creation"))
            or ("ask" if config.get("creation_approval") else "allow")
        )
        fidelity_cfg = codegen_cfg.get("fidelity", {}) or {}
        self.fidelity_enabled = bool(fidelity_cfg.get("enabled", True))
        self.fidelity_max_attempts = int(fidelity_cfg.get("max_attempts", 1))
        repair_cfg = codegen_cfg.get("repair", {}) or {}
        self.repair_enabled = bool(repair_cfg.get("enabled", True))
        self.repair_max_attempts = int(repair_cfg.get("max_attempts", 1))
        self.test_timeout = float(codegen_cfg.get("test_timeout", 30.0))
        self.test_max_attempts = int(codegen_cfg.get("test_max_attempts", 3))
        self.contract_search = bool(
            (codegen_cfg.get("contract_search", {}) or {}).get("enabled", True)
        )
        self.queue = RequestQueue(
            max_size=int(config.get("queue", {}).get("max_size", 100)),
        )
        self.skills = build_skills(config) + build_housekeeping_skills()
        self.tree = build_tree(self.skills)
        registry_path = config.get("category_registry", "data/categories.json")
        self.registry = CategoryRegistry(registry_path)
        self.body_store = SkillStore(config.get("skill_bodies", "data/skills"))
        self.seed_store = SkillStore(config.get("skill_seeds", "seeds"))
        # Deletions are durable: a tombstone suppresses a seed (or a generated
        # body of the same name) on every future load. Derived from the registry
        # path when not configured, so a test store is self-contained.
        deleted_path = config.get("deleted_skills") or str(
            Path(registry_path).parent / "deleted_skills.json"
        )
        self.deleted = DeletedSkills(deleted_path)
        merge_registry(self.tree, self.registry.read())
        merge_seed_store(self.tree, self.seed_store, self.body_store)
        merge_skill_store(self.tree, self.body_store, self.registry.read())
        apply_tombstones(self.tree, self.deleted)
        self.timers = TimerService(trace=self.trace, timezone=config.get("timezone"))
        self.ctx = ActionContext(
            engine=self.engine, config=config, timers=self.timers, admin=self
        )
        self.runner = SkillRunner(
            self.ctx, self.log, store=self.body_store, tau=self.tau
        )
        self._fatal: str | None = None
        self.current: Process | None = None
        self.pending: PendingRun | None = None
        self._lock = threading.RLock()
        self._drafts: deque[DraftAuthor] = deque()
        self._draft_notify = threading.Condition(self._lock)
        self._draft_thread: threading.Thread | None = None
        self._writes: deque[SkillWrite] = deque()
        # Body writes cancelled while queued or in flight: the worker discards
        # their output instead of materializing the leaf.
        self._cancelled: set[tuple[str, str]] = set()
        self._write_notify = threading.Condition(self._lock)
        self._write_thread: threading.Thread | None = None
        self._question_notify = threading.Condition(self._lock)
        self.questions: list[PendingQuestion] = []
        self.repairs: list[RepairOffer] = []
        self._repair_counts: dict[str, int] = {}
        self._approval_notify = threading.Condition(self._lock)
        self.approvals: list[PendingApproval] = []

    # ---- fatal handling ----

    @property
    def fatal(self) -> str | None:
        """A fatal error (the decision engine went unavailable), or None.

        The app cannot run without real SemIf, so every EngineUnavailable is
        fatal: callers stop dispatching and the CLI exits non-zero.
        """
        return self._fatal

    def _mark_fatal(self, exc: Exception) -> str:
        with self._lock:
            self._fatal = f"decision engine not available: {exc}"
            return self._fatal

    # ---- intake ----

    def submit(self, text: str, source: str = "typed") -> SchedulerReply:
        """Feed one input. Returns the structured reply."""
        return self.submit_request(text, source)

    def submit_request(self, text: str, source: str = "typed") -> SchedulerReply:
        """Feed one input, returning the request id on the reply.

        The id lets out-of-band interfaces (the messenger gateway) map a run
        back to the chat that originated it. The decision engine is real, so an
        EngineUnavailable is fatal: it is recorded and the caller exits.
        """
        try:
            return self._submit(text, source)
        except EngineUnavailable as exc:
            return SchedulerReply("fatal", self._mark_fatal(exc))

    def _submit(self, text: str, source: str = "typed") -> SchedulerReply:
        with self._lock:
            return self._submit_locked(text, source)

    def _submit_locked(self, text: str, source: str = "typed") -> SchedulerReply:
        request = Request(text, source=source)
        self.trace.append("submit", request.id, text=text, source=source)

        if self.current is None:
            self.current = Process(request=request, skill="(scheduling)")
            try:
                outcome = self._dispatch(request)
            finally:
                if self.pending is None:
                    self.current = None
            if outcome.kind == "needs_input":
                return SchedulerReply(
                    "needs_input", outcome.summary, request.id,
                    outcome.skill_ref, outcome.result,
                )
            self.trace.append("ran", request.id, skill=outcome.skill, summary=outcome.summary)
            return SchedulerReply(
                "running", outcome.summary, request.id,
                outcome.skill_ref, outcome.result,
            )

        ok = self.queue.push(request)
        if not ok:
            self.trace.append("rejected", request.id, reason="queue is full")
            return SchedulerReply("rejected", "queue is full", request.id)
        self.trace.append("queued", request.id)
        return SchedulerReply("queued", "queued", request.id)

    def busy(self, text: str, skill: str = "(driving)") -> None:
        """Set a fake in-progress process so the queue path is exercised."""
        with self._lock:
            if self.pending is not None:
                self.trace.append("pending_abandoned", self.pending.request.id)
                self.pending = None
            self.current = Process(request=Request(text, source="busy"), skill=skill)

    def idle(self) -> None:
        with self._lock:
            if self.pending is not None:
                self.trace.append("pending_abandoned", self.pending.request.id)
                self.pending = None
            self.current = None

    def run_queue(self) -> list[SchedulerReply]:
        """Process the queue while idle. Returns the structured outcomes.

        Safe to call from any thread: guarded by the scheduler lock (RLock, so
        the async codegen worker can drain the queue from its completion path).
        """
        with self._lock:
            return self._run_queue_locked()

    def _run_queue_locked(self) -> list[SchedulerReply]:
        results = []
        while self.current is None and len(self.queue) > 0:
            if self._fatal is not None:
                break
            request = self.queue.pop()
            self.trace.append("dequeued", request.id)
            self.current = Process(request=request, skill="(scheduling)")
            try:
                outcome = self._dispatch(request)
            except EngineUnavailable as exc:
                self._mark_fatal(exc)
                results.append(SchedulerReply("fatal", self._fatal))
                break
            except Exception as exc:
                self.trace.append("error", request.id, phase="dispatch", message=str(exc))
                outcome = DispatchResult(kind="error", summary=f"dispatch failed: {exc}")
            finally:
                if self.pending is None:
                    self.current = None
            if outcome.kind == "needs_input":
                results.append(
                    SchedulerReply(
                        "needs_input", outcome.summary, request.id,
                        outcome.skill_ref, outcome.result,
                    )
                )
                continue
            self.trace.append("ran", request.id, skill=outcome.skill, summary=outcome.summary)
            results.append(
                SchedulerReply(
                    "ran", outcome.summary, request.id,
                    outcome.skill_ref, outcome.result,
                )
            )
        return results

    # ---- dispatch ----

    def _dispatch(self, request: Request) -> DispatchResult:
        # Explicit housekeeping commands are deterministic: they route straight
        # to the meta skill, bypassing the generic guards (which misread a
        # task-like skill name as the task). Natural-language phrasings still go
        # through navigation below.
        meta = parse_meta_command(request.text)
        if meta is not None:
            leaf = next(
                (
                    s
                    for s in self.tree.get(HOUSEKEEPING_CATEGORY, [])
                    if s.name == meta
                ),
                None,
            )
            if leaf is not None:
                self.trace.append(
                    "meta_command", request.id, action=meta, text=request.text
                )
                return self._run_skill(leaf, request)
        navigation = navigate(
            self.engine,
            self.log,
            self.trace,
            request,
            self.tree,
            category_tau=self.category_tau,
            action_tau=self.action_tau,
            bypass_tau=self.bypass_tau,
        )
        if isinstance(navigation, CreateCategory):
            if self.new_skill_mode == "deny":
                return self._blocked_create(request, None)
            return self._dispatch_create_category(request)
        if isinstance(navigation, CreateSkill):
            return self._dispatch_skill(request, navigation.category)
        if navigation.category in CANNED_CATEGORIES:
            return self._run_skill(navigation, request)
        if navigation.category in HARD_LOCKED_CATEGORIES:
            # A hard-locked built-in category can never author a leaf, so the
            # reuse-vs-create intent guard has nothing to decide: run the
            # navigation's pick directly (the meta skill asks for any missing
            # target as an input variable).
            return self._run_skill(navigation, request)
        # The leaf softmax offers only existing skills (create_skill is
        # deliberately not an option), so its winner is "the closest existing
        # skill", never a statement that the skill matches. The intent guard is
        # therefore the sole leaf-level create door and is never bypassed: a
        # confident softmax winner among only-wrong options must still be
        # checked, or codegen can never be reached for an unmatched request.
        if not confirm_skill_fit(
            self.engine, self.log, self.trace, request, navigation, self.intent_tau
        ):
            self.trace.append(
                "intent_mismatch",
                request.id,
                selected_skill=f"{navigation.category}.{navigation.name}",
                chosen="create_skill",
            )
            if self._category_locked(navigation.category):
                return self._blocked_create(request, navigation.category)
            return self._dispatch_skill(request, navigation.category)
        return self._run_skill(navigation, request)

    def _dispatch_create_category(self, request: Request) -> DispatchResult:
        """Queue a new top-level category for the `llm` draft worker.

        The gate stays free: the small model authors the category in the
        background and, once it lands, the same worker authors the skill leaf
        inside it and hands off to the codegen body worker. On failure the
        request is not re-dispatched and the user is told.
        """
        self._queue_draft(request, None, "category")
        return DispatchResult(
            kind="create_category",
            summary=(
                "authoring a new category in the background; the request will "
                "re-run when it is ready"
            ),
        )

    def _dispatch_skill(
        self, request: Request, category: str
    ) -> DispatchResult:
        """Queue a new skill leaf for the `llm` draft worker.

        The small model authors the title + description in the background, then
        the codegen worker writes the runnable body; when the body lands the
        original request is re-queued and re-runs navigation onto the new leaf.
        A re-dispatched request whose skill is still unwritten must not author a
        second skill — it reports the pending write instead.
        """
        if self._category_locked(category):
            return self._blocked_create(request, category)
        pending = request.meta.get("awaiting_skill_body")
        if pending is not None:
            category_pending, name = pending
            return DispatchResult(
                kind="create_skill",
                summary=(
                    f"skill {category_pending}.{name} body still being written "
                    f"or failed; restart it with `restart {category_pending} {name}`."
                ),
                skill=name,
            )
        self._queue_draft(request, category, "skill")
        return DispatchResult(
            kind="create_skill",
            summary=(
                f"authoring a new skill for {category} in the background; "
                "the request will re-run when it is ready"
            ),
        )

    # ---- new-skill creation modes (deterministic per-category create door) ----

    def _category_mode(self, category: str) -> str:
        """Effective new-skill mode for a category: allow | deny | ask.

        A deterministic gate, not a SemIf decision. The closed `response` tree
        and the built-in `housekeeping` category are always `deny` (hard-locked
        in code). Every other category reads `new_skill_creation` from its
        category config (`data/skills/<category>/config.json`); when absent it
        falls back to the global `new_skill_creation` mode. The retired boolean
        `locks.new_skill` is still honored (True -> deny, False -> allow) so a
        previously locked category never silently reopens. The mode is
        creation-only: existing skills still run, and repair/regen of an
        existing leaf is allowed.
        """
        if category in CANNED_CATEGORIES or category in HARD_LOCKED_CATEGORIES:
            return "deny"
        config = self.body_store.read_category_config(category) or {}
        mode = normalize_new_skill_mode(config.get("new_skill_creation"))
        if mode is None:
            mode = normalize_new_skill_mode(
                (config.get("locks") or {}).get("new_skill")
            )
        return mode or self.new_skill_mode

    def _category_locked(self, category: str) -> bool:
        """Is this category closed to new-skill creation (mode == deny)?"""
        return self._category_mode(category) == "deny"

    def _blocked_create(
        self, request: Request, category: str | None
    ) -> DispatchResult:
        """Report that a denied category (or a denied category creation) will
        not author."""
        if category is None:
            self.trace.append("category_create_blocked", request.id)
            return DispatchResult(
                kind="error",
                summary=(
                    "cannot create a new category: new-skill creation is denied "
                    '(set new_skill_creation to "allow" or "ask" in config.json)'
                ),
            )
        self.trace.append("skill_create_blocked", request.id, category=category)
        return DispatchResult(
            kind="error",
            summary=(
                f"cannot create a new skill in {category}: new-skill creation "
                f'is denied (set new_skill_creation to "allow" or "ask" in '
                f"data/skills/{category}/config.json, or globally in config.json)"
            ),
        )

    def _run_skill(self, skill: Skill, request: Request) -> DispatchResult:
        if skill.writing:
            return DispatchResult(
                kind="error",
                summary=(
                    f"skill {skill.name} body is still being written; "
                    "it will run when ready."
                ),
                skill=skill.name,
            )
        if skill.is_noop():
            return DispatchResult(
                kind="error",
                summary=(
                    f"skill {skill.name} has no body yet; restart it with "
                    f"`restart {skill.category} {skill.name}`."
                ),
                skill=skill.name,
            )
        outcome = self.runner.run(skill, request)
        return self._finish_run(skill, request, outcome)

    def answer(self, text: str) -> SchedulerReply:
        """Feed the human's answer to a run paused for input.

        Routed directly to the pending run — no gate, score, or navigation —
        and the run resumes by re-invoking `act`. An empty answer is accepted:
        a pre-act contract variable is left unset and marked skipped, and an
        act-driven pause receives the empty string (the body may then fail,
        which is assessed normally).

        Guarded like `submit_request`: a front end calls this synchronously
        (the voice adapter from its worker thread), so an engine failure must
        be fatal-and-reported, and any other error must come back as a reply
        rather than propagate and take the front end down.
        """
        try:
            return self._answer(text)
        except EngineUnavailable as exc:
            return SchedulerReply("fatal", self._mark_fatal(exc))
        except Exception as exc:
            self.trace.append("error", "?", phase="answer", message=str(exc))
            return SchedulerReply("error", f"could not resume the run: {exc}")

    def _answer(self, text: str) -> SchedulerReply:
        with self._lock:
            if self.pending is None:
                return SchedulerReply("error", "no run is waiting for input")
            pending = self.pending
            self.pending = None
            pending.request.user_input = text
            self.trace.append(
                "answered",
                pending.request.id,
                text=text,
                skipped=not (text or "").strip(),
            )
            outcome = self.runner.resume(
                pending.skill, pending.request, pre_act=pending.pre_act
            )
            result = self._finish_run(pending.skill, pending.request, outcome)
            if result.kind == "needs_input":
                return SchedulerReply(
                    "needs_input", result.summary, pending.request.id,
                    result.skill_ref, result.result,
                )
            self.current = None
            self.trace.append("ran", pending.request.id, skill=result.skill, summary=result.summary)
            return SchedulerReply(
                "ran", f"[resumed] {result.summary}", pending.request.id,
                result.skill_ref, result.result,
            )

    def _finish_run(self, skill: Skill, request: Request, outcome) -> DispatchResult:
        if outcome.error:
            self.trace.append(
                "error", request.id, skill=skill.name, message=outcome.error
            )
            self._propose_repair(skill, request, outcome)
            return DispatchResult(kind="error", summary=f"skill error: {outcome.error}")
        if outcome.needs_input:
            self.pending = PendingRun(
                request=request,
                skill=skill,
                question=outcome.needs_input,
                pre_act=outcome.pre_act,
            )
            self.current = Process(request=request, skill=skill.name)
            self.trace.append(
                "needs_input", request.id, skill=skill.name, question=outcome.needs_input
            )
            return DispatchResult(
                kind="needs_input",
                summary=outcome.needs_input,
                skill=skill.name,
                needs_input=outcome.needs_input,
                skill_ref=f"{skill.category}.{skill.name}",
                result=outcome.action_log or outcome.new_state or None,
            )
        self.trace.append(
            "assessed",
            request.id,
            skill=skill.name,
            success=outcome.success,
            summary=outcome.summary,
            assessment_summary=outcome.assessment_summary,
            action_log=outcome.action_log,
            updated_request=outcome.updated_request,
        )
        requeued = False
        if outcome.updated_request and request.reentries < self.max_reentries:
            child = _requeue(request, outcome.updated_request)
            # Remember what this attempt observed so the next assessment can
            # tell a genuine retry (the result changed) from a guaranteed
            # repeat against unchanged state.
            child.meta["prev_result"] = outcome.action_log or outcome.new_state
            self.queue.push(child)
            self.trace.append("requeued", request.id, text=outcome.updated_request)
            if self.on_request_requeued is not None:
                self.on_request_requeued(request.id, child.id)
            requeued = True
        if not outcome.success and not (outcome.updated_request and requeued):
            self._propose_repair(skill, request, outcome)
        return DispatchResult(
            kind="ran",
            summary=outcome.summary,
            skill=skill.name,
            decisions_logged=outcome.decisions_logged,
            skill_ref=f"{skill.category}.{skill.name}",
            result=(outcome.action_log or outcome.new_state or None),
        )

    # ---- repair loop ----

    def _propose_repair(self, skill: Skill, request: Request, outcome) -> None:
        """A real run failed: offer to fix/learn, chosen by a SemIf decision.

        Executing a repair is user-confirmed (a codegen write takes tens of
        minutes), so the offer is recorded and surfaced in REPL/dashboard. The
        choice is logged as a real decision (phase `repair:choice`); the offer
        lifecycle is traced. The raw failure evidence bundle is captured as-is —
        error, action log, new state, and summary — so a repair hands the codegen
        model the observed failure verbatim, never a paraphrase and never
        credentials.
        """
        if (
            not self.repair_enabled
            or skill.is_noop()
            or skill.category in DETERMINISTIC_CATEGORIES
        ):
            return
        evidence = {
            "error": outcome.error,
            "action_log": outcome.action_log,
            "new_state": outcome.new_state,
            "summary": outcome.summary,
            "request": request.text,
            "description": skill.description,
            "requirements": self._skill_requirements(skill),
            "previous_body": self._read_body(skill.category, skill.name),
        }
        failure = (
            outcome.error
            or outcome.action_log
            or outcome.new_state
            or outcome.summary
            or "run failed"
        )
        decision = DecisionRequest(
            state=(
                f"skill {skill.name} failed a real run for {request.text!r}. "
                f"Integration: {skill.integration or 'unknown'}. "
                f"Failure: {failure[:400]}"
            ),
            question="The real run failed. How should the agent recover?",
            options=[
                Option("retry", "Try the same request again."),
                Option("repair_skill", "Repair the skill body from the observed failure."),
                Option("ask_user", "Ask me for the missing detail, then repair."),
                Option("no_repair", "Do nothing; just report the failure."),
            ],
        )
        result = self.engine.call(decision)
        self.log.append(
            decision,
            result,
            extra={"phase": "repair:choice", "run_id": request.id, "skill": skill.name},
        )
        offer = RepairOffer(
            id=uuid.uuid4().hex[:12],
            run_id=request.id,
            category=skill.category,
            skill=skill.name,
            selected=result.selected,
            reason=failure[:400],
            request_text=request.text,
            failure=failure[:2000],
            evidence=evidence,
        )
        with self._lock:
            self.repairs.append(offer)
            self.trace.append(
                "repair_offered",
                request.id,
                category=skill.category,
                skill=skill.name,
                selected=result.selected,
                probs=result.probs,
                failure=failure[:500],
            )
        print(
            f"[repair] {skill.name} failed: {failure[:200]}. "
            f"Suggested: {result.selected}. Use `repairs` / `repair <id> [action]`."
        )

    def pending_questions(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": q.id,
                    "run_id": q.run_id,
                    "category": q.category,
                    "skill": q.skill,
                    "question": q.question,
                    "kind": q.kind,
                    "created_at": q.created_at,
                }
                for q in self.questions
                if q.answer is None and not q.skipped
            ]

    def answer_question(self, question_id: str, text: str) -> tuple[str, str]:
        """Answer a deferred question. An empty answer skips it.

        A repair question carries the detail the user wants the skill repaired
        with; answering it starts the repair directly. Guarded so an engine or
        repair failure is reported instead of propagating into the front end.
        """
        try:
            return self._answer_question(question_id, text)
        except EngineUnavailable as exc:
            return "fatal", self._mark_fatal(exc)
        except Exception as exc:
            return "error", f"could not record the answer: {exc}"

    def _answer_question(self, question_id: str, text: str) -> tuple[str, str]:
        offer_id: str | None = None
        answer = (text or "").strip()
        with self._lock:
            question = next(
                (q for q in self.questions if q.id == question_id), None
            )
            if question is None:
                return "error", f"no pending question {question_id}"
            if answer:
                question.answer = answer
            else:
                question.skipped = True
            self.questions = [q for q in self.questions if q.id != question.id]
            self._question_notify.notify_all()
            self.trace.append(
                "question_answered",
                question.run_id,
                question=question.question,
                answer=answer or None,
                skipped=not answer,
            )
            if question.kind == "repair":
                offer_id = question.offer_id
        if offer_id is not None:
            return self._repair_with_answer(offer_id, answer)
        return "ok", "answer recorded" if answer else "question skipped"

    def pending_approvals(self) -> list[dict]:
        """Creation proposals awaiting the human's yes/no."""
        with self._lock:
            return [
                {
                    "id": a.id,
                    "run_id": a.run_id,
                    "kind": a.kind,
                    "category": a.category,
                    "skill": a.skill,
                    "description": a.description,
                    "created_at": a.created_at,
                }
                for a in self.approvals
                if a.status == "pending"
            ]

    def answer_approval(self, approval_id: str, approved: bool) -> tuple[str, str]:
        """Approve or deny a pending creation proposal.

        The draft worker is blocked in `_request_approval`; setting the status
        and notifying releases it (True registers + writes the body, False
        aborts). The terminal trace event is written by the worker, not here.
        Guarded so an unexpected failure is reported instead of propagating.
        """
        try:
            return self._answer_approval(approval_id, approved)
        except EngineUnavailable as exc:
            return "fatal", self._mark_fatal(exc)
        except Exception as exc:
            return "error", f"could not record the approval: {exc}"

    def _answer_approval(self, approval_id: str, approved: bool) -> tuple[str, str]:
        with self._lock:
            approval = next(
                (a for a in self.approvals if a.id == approval_id), None
            )
            if approval is None:
                return "error", f"no pending approval {approval_id}"
            if approval.status != "pending":
                return "error", f"approval {approval_id} is already {approval.status}"
            approval.status = "approved" if approved else "denied"
            self._approval_notify.notify_all()
            target = (
                approval.category
                if approval.kind == "category"
                else f"{approval.category}.{approval.skill}"
            )
        return (
            "ok",
            f"{'approved' if approved else 'denied'} {approval.kind} {target}",
        )

    def pending_repairs(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": r.id,
                    "run_id": r.run_id,
                    "category": r.category,
                    "skill": r.skill,
                    "selected": r.selected,
                    "reason": r.reason,
                    "failure": r.failure,
                    "status": r.status,
                    "created_at": r.created_at,
                }
                for r in self.repairs
                if r.status == "offered"
            ]

    def resolve_repair(self, offer_id: str, action: str | None = None) -> tuple[str, str]:
        """Execute (or decline) an offered repair. Defaults to the SemIf pick.

        `retry` requeues the original request; `repair_skill` rewrites the body
        from the observed failure; `ask_user` posts a question whose answer
        starts the repair; `no_repair` closes the offer. Repair writes are
        bounded by `codegen.repair.max_attempts` per skill. Guarded so a retry
        run or write failure is reported instead of propagating.
        """
        try:
            return self._resolve_repair(offer_id, action)
        except EngineUnavailable as exc:
            return "fatal", self._mark_fatal(exc)
        except Exception as exc:
            return "error", f"could not execute the repair: {exc}"

    def _resolve_repair(self, offer_id: str, action: str | None = None) -> tuple[str, str]:
        with self._lock:
            offer = next((r for r in self.repairs if r.id == offer_id), None)
            if offer is None:
                return "error", f"no repair offer {offer_id}"
            if offer.status != "offered":
                return "error", f"repair offer {offer_id} is already {offer.status}"
            chosen = action or offer.selected
            if chosen not in ("retry", "repair_skill", "ask_user", "no_repair"):
                return "error", f"unknown repair action {chosen!r}"
            if chosen in ("repair_skill", "ask_user") and self.codegen is None:
                return "error", "no codegen configured"
            if (
                chosen in ("repair_skill", "ask_user")
                and self._repair_counts.get(offer.skill, 0) >= max(self.repair_max_attempts, 0)
            ):
                return "error", f"repair budget for {offer.skill} is exhausted"
            offer.selected = chosen
            if chosen == "no_repair":
                offer.status = "declined"
            elif chosen == "ask_user":
                offer.status = "awaiting_user"
        if chosen == "retry":
            retry = Request(offer.request_text, source="repair_retry")
            self.queue.push(retry)
            with self._lock:
                offer.status = "done"
            self.trace.append("repair_retry", offer.run_id, skill=offer.skill)
            self.run_queue()
            return "running", f"retrying {offer.category}.{offer.skill}"
        if chosen == "no_repair":
            self.trace.append("repair_declined", offer.run_id, skill=offer.skill)
            return "ok", f"declined repair for {offer.category}.{offer.skill}"
        if chosen == "ask_user":
            self._ask_repair_question(offer)
            return "needs_input", f"question posted for {offer.category}.{offer.skill}"
        return self._start_repair(offer, detail=None)

    def _ask_repair_question(self, offer: RepairOffer) -> None:
        question = (
            f"Repairing {offer.category}.{offer.skill} after: {offer.failure[:300]}\n"
            "What detail should I use? (Reply with the fix or leave empty to skip.)"
        )
        pending = PendingQuestion(
            id=uuid.uuid4().hex[:12],
            run_id=offer.run_id,
            category=offer.category,
            skill=offer.skill,
            question=question,
            kind="repair",
            offer_id=offer.id,
        )
        with self._lock:
            self.questions.append(pending)
            self._question_notify.notify_all()
            self.trace.append(
                "repair_question", offer.run_id, skill=offer.skill, question=question
            )

    def _repair_with_answer(self, offer_id: str, answer: str) -> tuple[str, str]:
        with self._lock:
            offer = next((r for r in self.repairs if r.id == offer_id), None)
            if offer is None:
                return "error", f"no repair offer {offer_id}"
        if not answer:
            with self._lock:
                offer.status = "declined"
            self.trace.append("repair_declined", offer.run_id, skill=offer.skill)
            return "ok", f"declined repair for {offer.category}.{offer.skill}"
        return self._start_repair(offer, detail=answer)

    def _skill_requirements(self, skill: Skill) -> dict[str, str]:
        row = next(
            (
                s
                for s in self.registry.read().get(skill.category, {}).get("skills", [])
                if s.get("name") == skill.name
            ),
            {},
        )
        return dict(row.get("requirements") or {})

    def _start_repair(self, offer: RepairOffer, detail: str | None) -> tuple[str, str]:
        draft = self._stub_draft(offer.category, offer.skill)
        if detail:
            draft.requirements[f"how to fix {offer.skill}"] = detail
        with self._lock:
            self._repair_counts[offer.skill] = self._repair_counts.get(offer.skill, 0) + 1
            offer.status = "done"
        request = Request(offer.request_text, source="repair")
        self._start_skill_write(
            request, offer.category, draft, repair_evidence=offer.evidence
        )
        self.trace.append(
            "repair_executed",
            offer.run_id,
            category=offer.category,
            skill=offer.skill,
            detail=detail,
        )
        return "running", f"repairing {offer.category}.{offer.skill}"

    def _stub_draft(self, category: str, name: str) -> SkillDraft:
        entry = self.registry.read().get(category, {})
        row = next(
            (s for s in entry.get("skills", []) if s.get("name") == name), {}
        )
        return SkillDraft(
            name=name,
            description=row.get("description") or name,
            requirements=dict(row.get("requirements") or {}),
        )

    # ---- async draft authoring (single-slot `llm` worker) ----

    def _queue_draft(
        self,
        request: Request,
        category: str | None,
        kind: str,
        approved: bool = False,
    ) -> None:
        """Queue a category or skill draft for the `llm` worker.

        The gate stays free: this returns immediately and one draft at a time is
        authored in the background. A category job chains into its skill job; a
        skill job hands off to the codegen body worker. `approved` marks a
        chained skill job whose category was already approved.
        """
        with self._lock:
            self._drafts.append(
                DraftAuthor(
                    request=request,
                    category=category,
                    kind=kind,
                    approved=approved,
                )
            )
            if self._draft_thread is None or not self._draft_thread.is_alive():
                self._draft_thread = threading.Thread(
                    target=self._draft_worker, name="draft-author", daemon=True
                )
                self._draft_thread.start()
            self._draft_notify.notify()
        target = category or "(new category)"
        print(
            f"[llm] queued {kind} authoring for {target}; "
            "the request will re-run when it is ready."
        )

    def _draft_worker(self) -> None:
        """Drain the draft queue one authoring call at a time."""
        while True:
            with self._draft_notify:
                while not self._drafts and self._fatal is None:
                    self._draft_notify.wait()
                if self._fatal is not None:
                    return
                job = self._drafts.popleft()
            self._author_draft(job)

    def _author_draft(self, job: DraftAuthor) -> None:
        try:
            if job.kind == "category":
                self._author_category(job)
            else:
                self._author_skill(job)
        except (ProviderError, ValueError) as exc:
            self._fail_draft(job, exc)

    def _tree_snapshot(self) -> dict:
        with self._lock:
            return {category: list(skills) for category, skills in self.tree.items()}

    def _author_category(self, job: DraftAuthor) -> None:
        """Author a top-level category, then chain into its skill leaf.

        When the global new-skill mode is `ask`, the human approves the proposed
        category first; the chained skill is covered by that approval. An `llm`
        failure is graceful: only the user is told, the request is not
        re-dispatched — the decision engine stays the single fatal dependency.
        """
        draft = generate_category(self.llm, job.request, self._tree_snapshot())
        if not self._request_approval(job, "category", draft.name, draft.description):
            return
        with self._lock:
            if draft.name in self.tree:
                self.trace.append(
                    "error",
                    job.request.id,
                    phase="create_category",
                    message=f"category {draft.name} already exists",
                )
                return
            self.registry.register(draft.name, draft.description)
            self.tree[draft.name] = []
            # Seed the category config so the new-skill mode is discoverable and
            # editable from the start. It records the mode in effect at creation
            # (the global default), which the operator can then override.
            self.body_store.write_category_config(
                draft.name, {"new_skill_creation": self.new_skill_mode}
            )
            self.trace.append(
                "category_created",
                job.request.id,
                category=draft.name,
                description=draft.description,
            )
        self._queue_draft(job.request, draft.name, "skill", approved=True)

    def _author_skill(self, job: DraftAuthor) -> None:
        """Author a skill leaf stub, then launch the codegen body write.

        When the category's effective new-skill mode is `ask`, the human
        approves the proposed skill (a chained skill inside an already-approved
        category skips this).
        """
        draft = generate_skill(
            self.llm, job.request, job.category, self._tree_snapshot()
        )
        if not self._request_approval(job, "skill", draft.name, draft.description):
            return
        with self._lock:
            if self._category_locked(job.category):
                self.trace.append(
                    "skill_create_blocked",
                    job.request.id,
                    category=job.category,
                    phase="create_skill",
                )
                return
            existing = {s.name for s in self.tree.get(job.category, [])}
            if draft.name in existing:
                self.trace.append(
                    "error",
                    job.request.id,
                    phase="create_skill",
                    category=job.category,
                    message=f"skill {draft.name} already exists",
                )
                return
            self.registry.register_skill(
                job.category,
                draft.name,
                draft.description,
                request_text=job.request.text,
            )
            self.tree.setdefault(job.category, []).append(
                Skill(
                    name=draft.name,
                    category=job.category,
                    description=draft.description,
                )
            )
            if self.codegen is None:
                self.trace.append(
                    "skill_created",
                    job.request.id,
                    category=job.category,
                    skill=draft.name,
                    description=draft.description,
                    body=None,
                    written=False,
                )
        if self.codegen is None:
            return
        self._start_skill_write(job.request, job.category, draft)

    def _fail_draft(self, job: DraftAuthor, exc: Exception) -> None:
        """Surface a failed draft authoring; the request is not re-dispatched."""
        target = job.category or "(new category)"
        with self._lock:
            self.trace.append(
                "draft_failed",
                job.request.id,
                authoring=job.kind,
                category=job.category,
                message=str(exc),
            )
        print(
            f"[llm] {job.kind} authoring for {target} FAILED: {exc}. "
            "The request was not re-dispatched."
        )

    # ---- new-skill creation gate (allow / deny / ask) ----

    def _request_approval(
        self, job: DraftAuthor, kind: str, name: str, description: str
    ) -> bool:
        """Resolve the creation gate for a proposed category/skill. True = proceed.

        The effective mode is the global `new_skill_mode` for a new category and
        the category's own mode for a new skill (global default unless the
        category config overrides it). `allow` returns immediately, `deny`
        aborts, and `ask` blocks the single-slot `llm` worker here — after the
        proposal is authored and before anything is registered — until the human
        answers; the main gate stays free. One prompt per creation request: a
        chained skill job (`job.approved`) is already covered by the category
        approval. Denial or timeout aborts creation: nothing is registered, no
        body is written, and the request is not re-dispatched. Without a
        deferring front end there is nobody to answer, so `ask` denies
        immediately rather than stalling the worker.
        """
        if job.approved:
            return True
        mode = (
            self.new_skill_mode
            if kind == "category"
            else self._category_mode(job.category)
        )
        target = name if kind == "category" else f"{job.category}.{name}"
        if mode == "allow":
            return True
        if mode == "deny":
            # Normally blocked synchronously in dispatch; guard here too so an
            # in-flight draft never registers under a deny.
            if kind == "category":
                self.trace.append("category_create_blocked", job.request.id)
            else:
                self.trace.append(
                    "skill_create_blocked", job.request.id, category=job.category
                )
            print(
                f"[creation] {kind} {target} denied by new-skill mode; "
                "creation aborted."
            )
            return False
        if not self.defer_questions:
            self.trace.append(
                "creation_approval_skipped",
                job.request.id,
                target_kind=kind,
                category=target if kind == "category" else job.category,
                skill="" if kind == "category" else name,
                reason="no front end to answer; creation denied",
            )
            print(
                f"[approval] {kind} {target} needs approval but no front end is "
                "deferring questions; creation denied."
            )
            return False
        approval = PendingApproval(
            id=uuid.uuid4().hex[:12],
            run_id=job.request.id,
            kind=kind,
            category=target if kind == "category" else (job.category or ""),
            skill="" if kind == "category" else name,
            description=description,
        )
        with self._lock:
            self.approvals.append(approval)
            self._approval_notify.notify_all()
        self.trace.append(
            "creation_approval_requested",
            job.request.id,
            target_kind=kind,
            category=approval.category,
            skill=approval.skill,
            description=description,
        )
        print(
            f"[approval] new {kind} {target} — {description}. "
            "Approve in the REPL/dashboard (denies on timeout)."
        )
        deadline = (
            None
            if self.elicitation_wait <= 0
            else time.monotonic() + self.elicitation_wait
        )
        with self._lock:
            while approval.status == "pending":
                if deadline is not None and time.monotonic() >= deadline:
                    approval.status = "timeout"
                    break
                self._approval_notify.wait(timeout=1.0)
            self.approvals = [a for a in self.approvals if a.id != approval.id]
            self._approval_notify.notify_all()
            status = approval.status
        if status == "approved":
            self.trace.append(
                "creation_approval_approved",
                job.request.id,
                target_kind=kind,
                category=approval.category,
                skill=approval.skill,
            )
            return True
        event = (
            "creation_approval_timeout"
            if status == "timeout"
            else "creation_approval_denied"
        )
        self.trace.append(
            event,
            job.request.id,
            target_kind=kind,
            category=approval.category,
            skill=approval.skill,
        )
        print(
            f"[approval] {kind} {target} "
            f"{'timed out' if status == 'timeout' else 'denied'}; creation aborted."
        )
        return False

    # ---- async skill-body writes (single-slot codegen worker) ----

    def _start_skill_write(
        self,
        request: Request,
        category: str,
        draft: SkillDraft,
        repair_evidence: dict | None = None,
        reason_kind: str = "run_failure",
    ) -> None:
        """Mark the leaf in-progress and queue the body write for the worker.

        The gate stays free: this returns immediately and the worker (one write
        at a time, the 12G codegen model can't run twice) writes the body in the
        background. On completion the original request is re-queued and re-runs
        navigation onto the new leaf; on failure the leaf stays a restartable
        stub and only the user is notified. `repair_evidence` makes the write a
        repair: the existing body is rewritten from the raw observed-failure
        bundle.
        """
        with self._lock:
            leaf = next(
                (s for s in self.tree.get(category, []) if s.name == draft.name),
                None,
            )
            if leaf is None:
                return
            # A fresh write clears any stale cancellation for this name.
            self._cancelled.discard((category, draft.name))
            leaf.writing = True
            contract = skill_contract_ref()
            self.trace.append(
                "skill_writing",
                request.id,
                category=category,
                skill=draft.name,
                description=draft.description,
                model=self.codegen.model if self.codegen else None,
                contract_ref=contract["ref"],
                contract_dirty=contract["dirty"],
                repair=repair_evidence is not None,
            )
            self._writes.append(
                SkillWrite(
                    request=request,
                    category=category,
                    draft=draft,
                    repair_evidence=repair_evidence,
                    reason_kind=reason_kind,
                )
            )
            if self._write_thread is None or not self._write_thread.is_alive():
                self._write_thread = threading.Thread(
                    target=self._write_worker, name="skill-writer", daemon=True
                )
                self._write_thread.start()
            self._write_notify.notify()
        print(
            f"[codegen] queued body write for {category}.{draft.name}; "
            "it will run in the background and the request will re-run when ready."
        )

    def _write_worker(self) -> None:
        """Drain the skill-write queue one body at a time.

        A fatal engine failure stops the worker: `_write_skill_body` sets
        `_fatal` and the loop exits so the main loop can observe it and quit.
        """
        while True:
            with self._write_notify:
                while not self._writes and self._fatal is None:
                    self._write_notify.wait()
                if self._fatal is not None:
                    return
                job = self._writes.popleft()
            if self._consume_cancelled(job):
                continue
            self._write_skill_body(job)

    def _is_cancelled(self, category: str, name: str) -> bool:
        with self._lock:
            return (category, name) in self._cancelled

    def _consume_cancelled(self, job: SkillWrite) -> bool:
        """Discard a queued write that was cancelled before it started."""
        if not self._is_cancelled(job.category, job.draft.name):
            return False
        with self._lock:
            self._cancelled.discard((job.category, job.draft.name))
        self.trace.append(
            "skill_write_cancelled",
            job.request.id,
            category=job.category,
            skill=job.draft.name,
        )
        return True

    def _write_skill_body(self, job: SkillWrite) -> None:
        """Run the full authoring pipeline for one queued write (no scheduler
        lock held here): deferred elicitation -> codegen body (or repair
        rewrite) -> fidelity gate -> parse the body's CONTRACT -> test ->
        auto-run test (with a SemIf regen ladder on failure).

        The tree is snapshotted under the lock so the prompt build reads a
        stable view even if the main thread merges another skill meanwhile.
        """
        if self._is_cancelled(job.category, job.draft.name):
            return
        with self._lock:
            tree_snapshot = {category: list(skills) for category, skills in self.tree.items()}
        if job.repair_evidence is None and not job.draft.requirements:
            self._deferred_elicit(job, tree_snapshot)
        try:
            if job.repair_evidence is not None:
                evidence = dict(job.repair_evidence)
                evidence.update({
                    "description": job.draft.description,
                    "requirements": job.draft.requirements,
                    "previous_body": self._read_body(job.category, job.draft.name),
                })
                code = regenerate_skill_body(
                    self.codegen,
                    job.request,
                    job.category,
                    job.draft,
                    self._read_body(job.category, job.draft.name),
                    evidence,
                    reason_kind=job.reason_kind,
                )
            else:
                code = generate_skill_body(
                    self.codegen,
                    job.request,
                    job.category,
                    job.draft,
                    tree_snapshot,
                    requirements=job.draft.requirements,
                    degeneration_check=(
                        self.degeneration_check_factory(job.request.id)
                        if self.degeneration_check_factory is not None
                        else None
                    ),
                )
            code = self._fidelity_gate(job, code)
            contract = parse_contract(code)
        except EngineUnavailable as exc:
            self._mark_fatal(exc)
            self._fail_skill_write(job, exc)
            return
        except (CodegenError, ValueError) as exc:
            self._fail_skill_write(job, exc)
            return
        if self._is_cancelled(job.category, job.draft.name):
            return
        self.body_store.write_body(job.category, job.draft.name, code)
        self.body_store.write_contract(job.category, job.draft.name, contract)
        try:
            if self.contract_search:
                self._config_search(job, contract)
            self._test_and_fix(job, code, contract)
        except EngineUnavailable as exc:
            self._mark_fatal(exc)
            self._fail_skill_write(job, exc)
            return
        except (CodegenError, ValueError) as exc:
            self._fail_skill_write(job, exc)
            return
        self._complete_skill_write(job, code)

    def _read_body(self, category: str, name: str) -> str:
        try:
            return (self.body_store.dir(category, name) / "skill.py").read_text()
        except OSError:
            return ""

    def _deferred_elicit(self, job: SkillWrite, tree: dict) -> None:
        """Generate questions in the background, post them, wait for answers.

        The single-slot worker itself asks: the REPL, dashboard, and gateway all
        defer questions to `self.questions` now that authoring is async. It asks
        them one at a time, waiting up to `elicitation.answer_timeout` seconds
        for each (the clock resets on every answer); the gate stays free while
        it waits and other requests proceed. A timeout stops the sequence and
        the worker proceeds with whatever answers arrived (possibly none).
        """
        if not (self.defer_questions and self.elicitation_enabled and self.codegen):
            if self.elicitation_enabled and self.codegen is not None:
                self.trace.append(
                    "requirements_skipped",
                    job.request.id,
                    category=job.category,
                    skill=job.draft.name,
                    reason="questions not deferred by this front end",
                )
            return
        try:
            elicited = generate_elicitation(
                self.codegen, job.request, job.category, job.draft, tree,
                max_questions=self.elicitation_max,
            )
        except CodegenError:
            return
        if elicited.integration:
            job.draft.integration = elicited.integration
        if not elicited.questions:
            return
        answers = self._post_questions(job, elicited.questions)
        job.draft.requirements.update(answers)
        if job.draft.requirements or job.draft.integration:
            self._registry_requirements(job)
            self.trace.append(
                "requirements",
                job.request.id,
                category=job.category,
                skill=job.draft.name,
                questions=list(job.draft.requirements.keys()),
                integration=job.draft.integration or None,
            )

    def _post_questions(
        self, job: SkillWrite, questions: list[str]
    ) -> dict[str, str]:
        """Ask the questions one at a time, waiting for each answer.

        Only one question is live at a time: the next is posted only after the
        current is answered (an empty answer is accepted) or skipped. Each
        question gets a fresh `elicitation.answer_timeout` seconds — the clock
        resets on every answer submit. A timeout stops the sequence so the
        worker proceeds with whatever answers arrived.
        """
        self.trace.append(
            "questions_asked",
            job.request.id,
            category=job.category,
            skill=job.draft.name,
            questions=questions,
        )
        print(
            f"[codegen] {len(questions)} question(s) about "
            f"{job.category}.{job.draft.name} — answer in the dashboard."
        )
        answers: dict[str, str] = {}
        asked = 0
        timed_out = False
        for question in questions:
            pending = PendingQuestion(
                id=uuid.uuid4().hex[:12],
                run_id=job.request.id,
                category=job.category,
                skill=job.draft.name,
                question=question,
            )
            with self._lock:
                self.questions.append(pending)
                self._question_notify.notify_all()
            asked += 1
            deadline = (
                None
                if self.elicitation_answer_timeout <= 0
                else time.monotonic() + self.elicitation_answer_timeout
            )
            with self._lock:
                while pending.answer is None and not pending.skipped:
                    if deadline is not None and time.monotonic() >= deadline:
                        timed_out = True
                        break
                    self._question_notify.wait(timeout=1.0)
                if pending.answer is not None:
                    answers[question] = pending.answer
                self.questions = [q for q in self.questions if q is not pending]
                self._question_notify.notify_all()
            if pending.answer is None and not pending.skipped:
                # The current question timed out; don't post the rest.
                break
        if timed_out:
            self.trace.append(
                "questions_timeout",
                job.request.id,
                category=job.category,
                skill=job.draft.name,
                answered=len(answers),
                asked=asked,
            )
        return answers

    def _registry_requirements(self, job: SkillWrite) -> None:
        entry = self.registry.read().get(job.category, {})
        row = next(
            (s for s in entry.get("skills", []) if s.get("name") == job.draft.name), {}
        )
        self.registry.register_skill(
            job.category,
            job.draft.name,
            job.draft.description,
            request_text=row.get("request_text", job.request.text),
            requirements=job.draft.requirements,
        )

    def _fidelity_gate(self, job: SkillWrite, code: str) -> str:
        """Rapid SemIf sanity gate on the body; regen once only on reconsider.

        `authoring:fidelity` (accept/reconsider) runs beside static findings
        (declaration vs. code). A non-empty findings list forces reconsider.
        The gate never diagnoses: it only triggers a rewrite, which is handed
        the raw evidence bundle as-is. Bounded by
        `codegen.fidelity.max_attempts`; a body that still fails is accepted
        (never hard-fail authoring) but traced and badged `unverified`. The
        engine is real: an EngineUnavailable propagates and the worker marks
        the app fatal.
        """
        if not self.fidelity_enabled or self.codegen is None:
            return code
        integration, source = extract_integration(code)
        findings = integration_findings(integration, source, code)
        attempts = max(self.fidelity_max_attempts, 0)
        reviewed = code
        for attempt in range(attempts + 1):
            decision = DecisionRequest(
                state=(
                    f"[authoring fidelity] skill {job.category}.{job.draft.name}\n"
                    f"request: {job.request.text}\n"
                    f"description: {job.draft.description}\n"
                    f"INTEGRATION: {json.dumps(integration, sort_keys=True)}\n"
                    f"findings: {json.dumps(findings)}\n"
                    f"body:\n{reviewed[:4000]}"
                ),
                question=(
                    "Does this skill body really perform the requested action "
                    "against the user's service, or does it simulate it?"
                ),
                options=[
                    Option("accept", "It really performs the action."),
                    Option("reconsider", "It does not; rewrite it."),
                ],
            )
            result = self.engine.call(decision)
            reconsider = result.prob("reconsider") >= self.tau or bool(findings)
            self.log.append(
                decision,
                result,
                extra={
                    "phase": "authoring:fidelity",
                    "run_id": job.request.id,
                    "skill": f"{job.category}.{job.draft.name}",
                    "attempt": attempt,
                    "reconsider": reconsider,
                    "findings": findings,
                },
            )
            self.trace.append(
                "fidelity_review",
                job.request.id,
                category=job.category,
                skill=job.draft.name,
                attempt=attempt,
                performs_real_action=not reconsider,
                reconsider=reconsider,
                findings=findings,
                integration=integration,
                integration_source=source,
            )
            if not reconsider:
                return reviewed
            if attempt >= attempts:
                return reviewed
            evidence = {
                "request": job.request.text,
                "description": job.draft.description,
                "requirements": job.draft.requirements,
                "INTEGRATION": integration,
                "findings": findings,
                "verdict": result.selected,
                "probabilities": result.probs,
                "previous_body": reviewed,
            }
            try:
                reviewed = regenerate_skill_body(
                    self.codegen,
                    job.request,
                    job.category,
                    job.draft,
                    reviewed,
                    evidence,
                    reason_kind="fidelity",
                )
            except (CodegenError, ValueError) as exc:
                self.trace.append(
                    "fidelity_regen_failed",
                    job.request.id,
                    category=job.category,
                    skill=job.draft.name,
                    message=str(exc),
                )
                return reviewed
            integration, source = extract_integration(reviewed)
            findings = integration_findings(integration, source, reviewed)
        return reviewed

    def _test_and_fix(self, job: SkillWrite, code: str, contract: dict) -> None:
        """Generate the test artifact and auto-run it; regen the failing piece.

        On failure a SemIf decision picks whether to regenerate the body or the
        test; whichever it is, the error and the existing files are fed back
        into the corrective call. The contract rides in the body, so a code
        regen re-parses it (and rewrites the `contract.json` mirror). Fixture
        data lives inside the test, so a fixture fix is a test regen. Bounded by
        `test_max_attempts`.
        """
        attempts = max(self.test_max_attempts, 1)
        reason: str | None = None
        target: str | None = None
        for attempt in range(1, attempts + 1):
            if target == "regen_code":
                code = regenerate_skill_body(
                    self.codegen,
                    job.request,
                    job.category,
                    job.draft,
                    code,
                    {
                        "test_output": reason,
                        "description": job.draft.description,
                        "requirements": job.draft.requirements,
                        "previous_body": code,
                    },
                    reason_kind="test",
                )
                self.body_store.write_body(job.category, job.draft.name, code)
                contract = parse_contract(code)
                self.body_store.write_contract(job.category, job.draft.name, contract)
            test = generate_skill_tests(
                self.codegen, job.request, job.category, job.draft, code, contract,
                reason=reason, target=target,
            )
            self.body_store.write_test(job.category, job.draft.name, test)
            passed, output = run_skill_test(
                self.body_store.dir(job.category, job.draft.name), timeout=self.test_timeout
            )
            self.trace.append(
                "skill_testing",
                job.request.id,
                category=job.category,
                skill=job.draft.name,
                attempt=attempt,
                passed=passed,
                output=output[-400:],
            )
            if passed:
                return
            if attempt >= attempts:
                raise ValueError(
                    f"skill test failed after {attempts} attempts: {output[-2000:]}"
                )
            reason = output
            target = self._decide_regen(job, output)

    def _decide_regen(self, job: SkillWrite, reason: str) -> str:
        """Which artifact to regenerate: a SemIf decision, or regen_test by default."""
        if self.regen_decision_factory is None:
            return "regen_test"
        decide = self.regen_decision_factory(job.request.id)
        if decide is None:
            return "regen_test"
        try:
            return decide(reason)
        except Exception:
            return "regen_test"

    def _config_search(self, job: SkillWrite, contract: dict) -> None:
        """Auto-populate the skill config: a SemIf choice per contract variable
        maps it against candidate values from the whole config cascade (global
        -> category -> skill). Unmatched variables are left to the first-fire
        ask. The engine is always real: an EngineUnavailable propagates to the
        worker, which marks the app fatal."""
        merged: dict = {}
        merged.update(self.config)
        merged.update(self.body_store.read_category_config(job.category))
        merged.update(self.body_store.read_config(job.category, job.draft.name))
        try:
            for key in contract:
                candidates = _config_candidates(key, merged)
                if not candidates:
                    continue
                decision = DecisionRequest(
                    state=(
                        f"[config search] skill {job.category}.{job.draft.name} "
                        f"needs {key!r}: {contract[key]}"
                    ),
                    question=f"Which config value satisfies the skill variable {key!r}?",
                    options=[Option(c, str(merged[c])[:80]) for c in candidates]
                    + [Option("ask", "None of these; ask the user.")],
                )
                result = self.engine.call(decision)
                self.log.append(
                    decision,
                    result,
                    extra={
                        "phase": "config:search",
                        "run_id": job.request.id,
                        "skill": f"{job.category}.{job.draft.name}",
                        "variable": key,
                    },
                )
                if result.selected != "ask":
                    current = self.body_store.read_config(job.category, job.draft.name)
                    current[key] = merged[result.selected]
                    self.body_store.write_config(job.category, job.draft.name, current)
        except (ValueError, TypeError):
            return

    def _complete_skill_write(self, job: SkillWrite, code: str) -> None:
        """Materialize the body, hot-merge it into the tree, and re-dispatch the
        original request so it is answered by the new leaf."""
        if self._is_cancelled(job.category, job.draft.name):
            self._discard_cancelled_write(job)
            return
        job.draft.code = code
        try:
            skill = materialize_skill(job.draft, job.category, self.body_store)
        except ValueError as exc:
            self._fail_skill_write(job, exc)
            return
        with self._lock:
            skills = self.tree.setdefault(job.category, [])
            for index, existing in enumerate(skills):
                if existing.name == job.draft.name:
                    skills[index] = skill
                    break
            else:
                skills.append(skill)
            skills.sort(key=lambda s: s.name)
            body_path = self.body_store.dir(job.category, job.draft.name).as_posix()
            self.trace.append(
                "skill_created",
                job.request.id,
                category=job.category,
                skill=job.draft.name,
                description=job.draft.description,
                body=body_path,
                written=True,
            )
            self.trace.append(
                "skill_ready",
                job.request.id,
                category=job.category,
                skill=job.draft.name,
                contract_vars=list(skill.contract or {}),
            )
            requeued = job.request.copy_for_requeue()
            requeued.meta["awaiting_skill_body"] = [job.category, job.draft.name]
            requeued.meta["parent_run"] = job.request.id
            self.queue.push(requeued)
            self.trace.append(
                "skill_requeued",
                job.request.id,
                text=requeued.text,
            )
        print(
            f"[codegen] body ready for {job.category}.{job.draft.name}; "
            "original request re-queued."
        )
        self.run_queue()

    def _discard_cancelled_write(self, job: SkillWrite) -> None:
        """Clean up a write that was cancelled mid-flight.

        The cancel path already removed the leaf and folder; the generation may
        have rewritten files after that, so delete the folder again and clear
        the in-progress flag. The cancellation marker is consumed.
        """
        with self._lock:
            self._cancelled.discard((job.category, job.draft.name))
            leaf = next(
                (s for s in self.tree.get(job.category, []) if s.name == job.draft.name),
                None,
            )
            if leaf is not None:
                leaf.writing = False
        try:
            self.body_store.delete(job.category, job.draft.name)
        except (OSError, ValueError):
            pass

    def _fail_skill_write(self, job: SkillWrite, exc: Exception) -> None:
        """Clear the in-progress flag and notify; the leaf stays a restartable stub."""
        if self._is_cancelled(job.category, job.draft.name):
            self._discard_cancelled_write(job)
            return
        with self._lock:
            leaf = next(
                (s for s in self.tree.get(job.category, []) if s.name == job.draft.name),
                None,
            )
            if leaf is not None:
                leaf.writing = False
            self.trace.append(
                "skill_write_failed",
                job.request.id,
                category=job.category,
                skill=job.draft.name,
                message=str(exc),
            )
        print(
            f"[codegen] body write for {job.category}.{job.draft.name} FAILED: {exc}. "
            f"The leaf is a stub — restart it with `restart {job.category} {job.draft.name}`."
        )

    def restart_skill(self, category: str, name: str) -> tuple[str, str]:
        """Kick off (or re-kick) the body write for an empty not-in-progress leaf."""
        with self._lock:
            leaf = next(
                (s for s in self.tree.get(category, []) if s.name == name),
                None,
            )
            if leaf is None:
                return "error", f"no skill {category}.{name}"
            if category in CANNED_CATEGORIES:
                return "error", f"{category} is a closed canned tree; there is no body to write"
            if leaf.origin == "builtin":
                return "error", f"{category}.{name} is a built-in skill; there is no body to write"
            if self._category_locked(category):
                return "error", f"{category} is locked to new skills"
            if leaf.writing:
                return "error", f"skill {category}.{name} is already being written"
            if self.codegen is None:
                return "error", "no codegen configured"
            draft = self._stub_draft(category, name)
            if leaf.description:
                draft.description = leaf.description
            row = next(
                (
                    s
                    for s in self.registry.read().get(category, {}).get("skills", [])
                    if s.get("name") == name
                ),
                {},
            )
            request_text = row.get("request_text")
            origin = (
                Request(request_text, source="restart")
                if request_text
                else Request(f"write the body for {category}.{name}: {draft.description}", source="restart")
            )
        self._start_skill_write(origin, category, draft)
        self.trace.append("skill_restarted", origin.id, category=category, skill=name)
        return "running", f"restarting body write for {category}.{name}"

    # ---- housekeeping (the meta skills' operations) ----

    def _cancel_pending_write(self, category: str, name: str) -> bool:
        """Cancel any queued/in-flight body write for a leaf.

        Removes a still-queued write and marks the name cancelled so an
        in-flight generation discards its output. Returns whether a write was
        in progress.
        """
        with self._lock:
            leaf = next(
                (s for s in self.tree.get(category, []) if s.name == name), None
            )
            was_writing = bool(leaf and leaf.writing)
            self._cancelled.add((category, name))
            self._writes = deque(
                w
                for w in self._writes
                if not (w.category == category and w.draft.name == name)
            )
            if leaf is not None:
                leaf.writing = False
        return was_writing

    def delete_skill(self, category: str, name: str) -> tuple[str, str]:
        """Delete a skill: tree leaf, registry entry, folder, and tombstone.

        Durable: the tombstone suppresses a seed (or a same-named generated
        body) on every future load. Built-in skills cannot be deleted.
        """
        with self._lock:
            leaf = next(
                (s for s in self.tree.get(category, []) if s.name == name), None
            )
            if leaf is None:
                return "error", f"no skill {category}.{name}"
            if leaf.origin == "builtin" or category in CANNED_CATEGORIES:
                return "error", (
                    f"{category}.{name} is a built-in skill and cannot be deleted"
                )
            self._cancel_pending_write(category, name)
            self.tree[category] = [
                s for s in self.tree.get(category, []) if s.name != name
            ]
            self.registry.unregister_skill(category, name)
            self.deleted.add(category, name)
            self.trace.append("skill_deleted", "?", category=category, skill=name)
        try:
            self.body_store.delete(category, name)
        except (OSError, ValueError) as exc:
            self.trace.append(
                "skill_delete_failed",
                "?",
                category=category,
                skill=name,
                message=str(exc),
            )
        return "ok", f"deleted {category}.{name}"

    def clear_skill_config(self, category: str, name: str) -> tuple[str, str]:
        """Clear a skill's recorded config so it is asked for again next fire."""
        with self._lock:
            leaf = next(
                (s for s in self.tree.get(category, []) if s.name == name), None
            )
            if leaf is None:
                return "error", f"no skill {category}.{name}"
            current = self.body_store.read_config(category, name) or {}
            cleared = sorted(set(current) | set(leaf.config or {}))
            if not cleared:
                return "ok", f"{category}.{name} has no recorded config"
            self.body_store.write_config(category, name, {})
            leaf.config = {}
            self.trace.append(
                "config_cleared",
                "?",
                category=category,
                skill=name,
                variables=cleared,
            )
        return "ok", (
            f"cleared {len(cleared)} config variable(s) for {category}.{name}: "
            + ", ".join(cleared)
        )

    def regen_skill(
        self, category: str, name: str, guidance: str = ""
    ) -> tuple[str, str]:
        """Regenerate a skill body from the user's guidance.

        A stub with no body is simply (re)written; an existing body is rewritten
        through the corrective path with the guidance as evidence. Built-in
        skills cannot be regenerated; a locked category may still regenerate an
        existing leaf (the lock is creation-only).
        """
        with self._lock:
            leaf = next(
                (s for s in self.tree.get(category, []) if s.name == name), None
            )
            if leaf is None:
                return "error", f"no skill {category}.{name}"
            if category in CANNED_CATEGORIES or leaf.origin == "builtin":
                return "error", (
                    f"{category}.{name} is a built-in skill and cannot be regenerated"
                )
            if self.codegen is None:
                return "error", "no codegen configured"
            if leaf.writing:
                return "error", f"skill {category}.{name} is already being written"
            draft = self._stub_draft(category, name)
            if leaf.description:
                draft.description = leaf.description
            row = next(
                (
                    s
                    for s in self.registry.read().get(category, {}).get("skills", [])
                    if s.get("name") == name
                ),
                {},
            )
            request_text = row.get("request_text")
            origin = (
                Request(request_text, source="regen")
                if request_text
                else Request(
                    f"regenerate the body for {category}.{name}: {draft.description}",
                    source="regen",
                )
            )
        previous = self._read_body(category, name)
        evidence = {"guidance": guidance or "(none given)"} if previous else None
        self._start_skill_write(
            origin,
            category,
            draft,
            repair_evidence=evidence,
            reason_kind="manual",
        )
        self.trace.append(
            "skill_regen_started",
            origin.id,
            category=category,
            skill=name,
            guidance=guidance or None,
        )
        return "running", f"regenerating body for {category}.{name}"

    def cancel_skill_build(self, category: str, name: str) -> tuple[str, str]:
        """Cancel a skill body build and delete its half-built leaf."""
        with self._lock:
            leaf = next(
                (s for s in self.tree.get(category, []) if s.name == name), None
            )
            if leaf is None:
                return "error", f"no skill {category}.{name}"
            if leaf.origin == "builtin" or category in CANNED_CATEGORIES:
                return "error", (
                    f"{category}.{name} is a built-in skill and cannot be cancelled"
                )
            was_writing = self._cancel_pending_write(category, name)
            self.tree[category] = [
                s for s in self.tree.get(category, []) if s.name != name
            ]
            self.registry.unregister_skill(category, name)
            self.deleted.add(category, name)
            self.trace.append(
                "skill_build_cancelled",
                "?",
                category=category,
                skill=name,
                was_writing=was_writing,
            )
        try:
            self.body_store.delete(category, name)
        except (OSError, ValueError) as exc:
            self.trace.append(
                "skill_delete_failed",
                "?",
                category=category,
                skill=name,
                message=str(exc),
            )
        detail = (
            f"cancelled the build for {category}.{name} and deleted the leaf"
            if was_writing
            else f"deleted {category}.{name} (no build was in progress)"
        )
        return "ok", detail

    def set_category_mode(self, category: str, mode: str) -> tuple[str, str]:
        """Set a category's new-skill mode (`allow` | `deny` | `ask`).

        Written to the category config (`new_skill_creation`); passing an empty
        value clears the override so the category inherits the global mode.
        """
        with self._lock:
            if category in CANNED_CATEGORIES or category in HARD_LOCKED_CATEGORIES:
                return "error", (
                    f"{category} is a built-in category and is always denied"
                )
            if category not in self.tree:
                return "error", f"no category {category}"
            config = self.body_store.read_category_config(category) or {}
            # Drop the retired boolean lock so it cannot shadow the new mode.
            config.pop("locks", None)
            if mode:
                normalized = normalize_new_skill_mode(mode)
                if normalized is None:
                    return "error", (
                        f"invalid mode {mode!r}: expected one of "
                        f"{', '.join(NEW_SKILL_MODES)}"
                    )
                config["new_skill_creation"] = normalized
            else:
                config.pop("new_skill_creation", None)
            self.body_store.write_category_config(category, config)
            effective = self._category_mode(category)
            self.trace.append(
                "category_mode_set", "?", category=category, mode=effective
            )
        return "ok", f"{category} new-skill creation is now {effective}"

    def category_modes(self) -> dict[str, str]:
        with self._lock:
            return {category: self._category_mode(category) for category in self.tree}

    def status(self) -> str:
        with self._lock:
            lines = []
            current = f"{self.current.skill} ({self.current.request.id})" if self.current else "idle"
            lines.append(f"current: {current}")
            if self.pending is not None:
                lines.append(f"awaiting input: {self.pending.question}")
            waiting = [q for q in self.questions if q.answer is None and not q.skipped]
            if waiting:
                lines.append(f"questions waiting: {len(waiting)} (answer in the dashboard)")
            offered = [r for r in self.repairs if r.status == "offered"]
            if offered:
                lines.append(f"repair offers: {len(offered)} (use `repairs`)")
            lines.append(f"queue: {len(self.queue)} pending")
            for request in self.queue.items():
                lines.append(f"  {request.id}  {request.text[:60]}")
            pending_timers = self.timers.pending()
            if pending_timers:
                lines.append(f"timers: {len(pending_timers)} scheduled")
                for timer in pending_timers:
                    lines.append(
                        f"  {timer['kind']} {timer['id']} due {timer['due_at_local'][:16]} "
                        f"({timer['label']})"
                    )
            return "\n".join(lines)


def _requeue(request: Request, updated_text: str) -> Request:
    updated = Request(updated_text, source="requeue")
    updated.reentries = request.reentries + 1
    updated.meta["parent_run"] = request.id
    return updated


def _config_candidates(key: str, merged: dict) -> list[str]:
    """Candidate config keys that plausibly satisfy a contract variable.

    Cheap keyword overlap on snake_case tokens — enough to offer the SemIf
    config search a small, sane option set. Only scalar values are candidates.
    """
    parts = set(key.split("_"))
    candidates = []
    for cfg_key, value in merged.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cfg_parts = set(str(cfg_key).split("_"))
            if parts & cfg_parts:
                candidates.append(str(cfg_key))
    return sorted(candidates)