"""The scheduler: gate, choice, score, queue, dispatch.

Every SemIf decision (gate, choice, score, navigation, prediction) is logged.
A high-priority input can preempt the current process, which requeues with its
state preserved; a deferred input is scored and queued by urgency.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .codegen import (
    CodegenClient,
    CodegenError,
    extract_integration,
    generate_data_contract,
    generate_elicitation,
    generate_skill_body,
    generate_skill_tests,
    integration_findings,
    regenerate_skill_body,
    run_skill_test,
    skill_contract_ref,
)
from .decisions import DecisionRequest, Option, Request
from .engine import EngineUnavailable, SemIfEngine
from .llm import LLMClient
from .log import DecisionLog
from .queue import UrgencyQueue
from .skill import SkillRunner
from .skills import (
    ActionContext,
    CategoryRegistry,
    CreateCategory,
    CreateSkill,
    Prediction,
    Skill,
    SkillDraft,
    SkillStore,
    build_skills,
    build_tree,
    compose_state,
    confirm_skill_fit,
    generate_category,
    generate_skill,
    materialize_skill,
    merge_registry,
    merge_seed_store,
    merge_skill_store,
    navigate,
    tree_summary,
)
from .trace import TraceLog

GATE_YES = "yes"
GATE_CAPABILITY_CHARS = 1200
CHOICE_INTERRUPT = "interrupt"
URGENCY_OPTIONS = [
    ("critical", "Immediate danger or critical failure."),
    ("high", "Important but not dangerous."),
    ("medium", "Should be handled reasonably soon."),
    ("low", "Can wait."),
]
URGENCY_WEIGHTS = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}


def build_gate_decision(request: Request, tree: dict[str, list[Skill]]) -> DecisionRequest:
    """The top-level handle/ignore filter, with end-user expectation and the
    agent's real capabilities in the state.

    Without them the decision model reads an information request ("tell me the
    next event in my nextcloud calendar") as small talk and drops it: "actionable"
    implies a side effect, and the model cannot see that looking a value up in a
    configured service is exactly what this agent does. The available skills are
    listed so a request naming one of them is visibly within the agent's remit.
    """
    capabilities = tree_summary(tree)[:GATE_CAPABILITY_CHARS] or "  (none yet)"
    state = (
        f"User input: {request.text}\n"
        "[Context: the user is talking to a personal task agent. Any imperative "
        "request to perform a task — send, message, tell, check, create, remind, "
        "or look something up — must be handled, even if no skill matches yet. "
        "Only chatter, thanks, and bare greetings should be ignored.]\n"
        f"[Available skills:\n{capabilities}]"
    )
    return DecisionRequest(
        state=state,
        question="Should the agent handle this input?",
        options=[
            Option(
                GATE_YES,
                "Yes — do what it asks, including looking something up, reporting "
                "information, or performing an action using the agent's skills.",
            ),
            Option(
                "no",
                "No — it is not a request for the agent to do or answer anything "
                "(background text, noise, or a bare greeting).",
            ),
        ],
    )


@dataclass
class Process:
    request: Request
    skill: str
    weight: float


@dataclass
class PendingRun:
    """A skill run paused awaiting human input.

    `prediction` is kept so resume re-invokes only `act` (predict is not
    re-run, avoiding duplicate SemIf sub-decisions); `question` is what the
    run asked the human. A `pre_predict` pause (contract variables collected
    by the runner before predict) has no prediction yet — resume runs the full
    predict+act path.
    """

    request: Request
    skill: Skill
    prediction: Prediction | None
    question: str
    pre_predict: bool = False


@dataclass
class SkillWrite:
    """One queued async skill-body write (single-slot codegen worker)."""

    request: Request
    category: str
    draft: SkillDraft
    weight: float
    repair_evidence: dict | None = None


@dataclass
class PendingQuestion:
    """A question waiting for the human (deferred elicitation or repair).

    Posted by the codegen worker when no inline asker is wired; answered via
    the dashboard (`/api/questions`) or `Scheduler.answer_question`. The worker
    waits (bounded by `codegen.elicitation.wait_timeout`) and continues with
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
class DispatchResult:
    kind: str  # ran | create_category | create_skill | needs_input | error
    summary: str
    skill: str | None = None
    decisions_logged: int = 0
    body_written: bool = False
    needs_input: str | None = None
    draft: SkillDraft | None = None


class Scheduler:
    def __init__(
        self,
        engine: SemIfEngine,
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
        asker: Callable[[str], str | None] | None = None,
        on_request_requeued: Callable[[str, str], None] | None = None,
    ):
        self.engine = engine
        self.llm = llm
        self.log = log
        self.trace = trace if trace is not None else TraceLog()
        self.config = config
        self.tau = tau
        nav_cfg = config.get("navigation", {}) or {}
        self.create_tau = float(nav_cfg.get("create_tau", 0.5))
        self.create_margin = float(nav_cfg.get("create_margin", 0.35))
        self.max_reentries = max_reentries
        self.codegen = codegen
        self.degeneration_check_factory = degeneration_check_factory
        self.regen_decision_factory = regen_decision_factory
        self.asker = asker
        self.on_request_requeued = on_request_requeued
        codegen_cfg = config.get("codegen", {}) or {}
        elicitation_cfg = codegen_cfg.get("elicitation", {}) or {}
        self.elicitation_enabled = bool(elicitation_cfg.get("enabled", True))
        self.elicitation_max = int(elicitation_cfg.get("max_questions", 4))
        self.elicitation_wait = float(elicitation_cfg.get("wait_timeout", 900.0))
        self.defer_questions = False
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
        self.queue = UrgencyQueue(
            max_size=int(config.get("queue", {}).get("max_size", 100)),
            age_rate=float(config.get("queue", {}).get("age_rate", 0.0)),
        )
        self.skills = build_skills(config)
        self.tree = build_tree(self.skills)
        self.registry = CategoryRegistry(config.get("category_registry", "data/categories.json"))
        self.body_store = SkillStore(config.get("skill_bodies", "data/skills"))
        self.seed_store = SkillStore(config.get("skill_seeds", "seeds"))
        merge_registry(self.tree, self.registry.read())
        merge_seed_store(self.tree, self.seed_store, self.body_store)
        merge_skill_store(self.tree, self.body_store, self.registry.read())
        self.ctx = ActionContext(engine=self.engine, config=config)
        self.runner = SkillRunner(
            self.ctx, self.log, store=self.body_store, tau=self.tau
        )
        self._fatal: str | None = None
        self.current: Process | None = None
        self.pending: PendingRun | None = None
        self._lock = threading.RLock()
        self._writes: deque[SkillWrite] = deque()
        self._write_notify = threading.Condition(self._lock)
        self._write_thread: threading.Thread | None = None
        self._question_notify = threading.Condition(self._lock)
        self.questions: list[PendingQuestion] = []
        self.repairs: list[RepairOffer] = []
        self._repair_counts: dict[str, int] = {}

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

    # ---- decision templates (all real SemIf, all logged) ----

    def _contains_request(self, request: Request) -> bool:
        decision = build_gate_decision(request, self.tree)
        result = self.engine.call(decision)
        self.log.append(
            decision, result, extra={"phase": "gate", "run_id": request.id}
        )
        return result.prob(GATE_YES) >= self.tau

    def _choice(self, request: Request, current: Process) -> bool:
        decision = DecisionRequest(
            state=compose_state(request, current=current.skill),
            question="Should this be allowed to interrupt the current process?",
            options=[
                Option(CHOICE_INTERRUPT, "Yes, interrupt the current process."),
                Option("defer", "No, wait until the current process finishes."),
            ],
        )
        result = self.engine.call(decision)
        self.log.append(
            decision,
            result,
            extra={"phase": "choice", "current": current.skill, "run_id": request.id},
        )
        return result.prob(CHOICE_INTERRUPT) >= self.tau

    def _score(self, request: Request, current: str | None = None) -> tuple[float, str]:
        decision = DecisionRequest(
            state=compose_state(request, current=current),
            question="How urgent is this request?",
            options=[Option(option_id, description) for option_id, description in URGENCY_OPTIONS],
        )
        result = self.engine.call(decision)
        self.log.append(
            decision, result, extra={"phase": "score", "run_id": request.id}
        )
        label = result.selected
        return URGENCY_WEIGHTS[label], label

    # ---- intake ----

    def submit(self, text: str, source: str = "typed") -> tuple[str, str]:
        """Feed one input. Returns (status, detail)."""
        status, detail, _ = self.submit_request(text, source)
        return status, detail

    def submit_request(
        self, text: str, source: str = "typed"
    ) -> tuple[str, str, str]:
        """Feed one input, also returning the request id.

        The id lets out-of-band interfaces (the messenger gateway) map a run
        back to the chat that originated it. The decision engine is real, so an
        EngineUnavailable is fatal: it is recorded and the caller exits.
        """
        try:
            return self._submit(text, source)
        except EngineUnavailable as exc:
            return "fatal", self._mark_fatal(exc), ""

    def _submit(self, text: str, source: str = "typed") -> tuple[str, str, str]:
        with self._lock:
            return self._submit_locked(text, source)

    def _submit_locked(self, text: str, source: str = "typed") -> tuple[str, str, str]:
        request = Request(text, source=source)
        self.trace.append("submit", request.id, text=text, source=source)
        if not self._contains_request(request):
            self.trace.append("dropped", request.id, reason="no actionable request")
            return "dropped", "no actionable request", request.id

        if self.current is None:
            weight, label = self._score(request)
            self.current = Process(request=request, skill="(scheduling)", weight=weight)
            try:
                outcome = self._dispatch(request, weight=weight)
            finally:
                if self.pending is None:
                    self.current = None
            if outcome.kind == "needs_input":
                return "needs_input", outcome.summary, request.id
            self.trace.append("ran", request.id, skill=outcome.skill, summary=outcome.summary)
            return "running", f"[{label}] {outcome.summary}", request.id

        interrupt = self._choice(request, self.current)
        if interrupt:
            if self.pending is not None:
                self.trace.append("pending_abandoned", self.pending.request.id)
                self.pending = None
            previous = self.current
            previous.request.resume["from_skill"] = previous.skill
            self.queue.push(previous.request, previous.weight)
            self.current = Process(request=request, skill="(scheduling)", weight=1.0)
            self.trace.append("preempted", request.id, preempted=previous.skill)
            outcome = self._dispatch(request, weight=1.0)
            if outcome.kind == "needs_input":
                return "preempted", f"interrupted {previous.skill}; {outcome.summary}", request.id
            self.current = None
            self.trace.append("ran", request.id, skill=outcome.skill, summary=outcome.summary)
            return "preempted", f"interrupted {previous.skill}; {outcome.summary}", request.id

        weight, label = self._score(request, current=self.current.skill)
        ok = self.queue.push(request, weight)
        if not ok:
            self.trace.append("rejected", request.id, reason="queue is full")
            return "rejected", "queue is full", request.id
        self.trace.append("queued", request.id, weight=weight, label=label)
        return "queued", f"urgency {label} (weight {weight:.2f})", request.id

    def busy(self, text: str, skill: str = "(driving)") -> None:
        """Set a fake in-progress process so the choice/score path is exercised."""
        with self._lock:
            if self.pending is not None:
                self.trace.append("pending_abandoned", self.pending.request.id)
                self.pending = None
            self.current = Process(request=Request(text, source="busy"), skill=skill, weight=1.0)

    def idle(self) -> None:
        with self._lock:
            if self.pending is not None:
                self.trace.append("pending_abandoned", self.pending.request.id)
                self.pending = None
            self.current = None

    def run_queue(self) -> list[tuple[str, str]]:
        """Process the queue while idle. Returns the outcomes.

        Safe to call from any thread: guarded by the scheduler lock (RLock, so
        the async codegen worker can drain the queue from its completion path).
        """
        with self._lock:
            return self._run_queue_locked()

    def _run_queue_locked(self) -> list[tuple[str, str]]:
        results = []
        while self.current is None and len(self.queue) > 0:
            if self._fatal is not None:
                break
            request = self.queue.pop()
            self.trace.append("dequeued", request.id)
            self.current = Process(request=request, skill="(scheduling)", weight=0.0)
            try:
                outcome = self._dispatch(request, weight=0.0)
            except EngineUnavailable as exc:
                self._mark_fatal(exc)
                results.append(("fatal", self._fatal))
                break
            except Exception as exc:
                self.trace.append("error", request.id, phase="dispatch", message=str(exc))
                outcome = DispatchResult(kind="error", summary=f"dispatch failed: {exc}")
            finally:
                if self.pending is None:
                    self.current = None
            if outcome.kind == "needs_input":
                results.append(("needs_input", f"[{request.id}] {outcome.summary}"))
                continue
            self.trace.append("ran", request.id, skill=outcome.skill, summary=outcome.summary)
            results.append(("ran", f"[{request.id}] {outcome.summary}"))
        return results

    # ---- dispatch ----

    def _dispatch(self, request: Request, weight: float = 0.0) -> DispatchResult:
        navigation = navigate(
            self.engine,
            self.log,
            self.trace,
            request,
            self.tree,
            create_tau=self.create_tau,
            create_margin=self.create_margin,
        )
        if isinstance(navigation, CreateCategory):
            created = self._create_category(request)
            if created.kind != "create_category":
                return created
            return self._dispatch_skill(request, created.skill, weight)
        if isinstance(navigation, CreateSkill):
            return self._dispatch_skill(request, navigation.category, weight)
        if not confirm_skill_fit(
            self.engine, self.log, self.trace, request, navigation, self.tau
        ):
            self.trace.append(
                "intent_mismatch",
                request.id,
                selected_skill=f"{navigation.category}.{navigation.name}",
                chosen="create_skill",
            )
            return self._dispatch_skill(request, navigation.category, weight)
        return self._run_skill(navigation, request)

    def _dispatch_skill(
        self, request: Request, category: str, weight: float = 0.0
    ) -> DispatchResult:
        """create_skill in `category`, then launch an async body write.

        The stub is authored and registered synchronously so the leaf is
        navigable immediately; the runnable body is written in the background
        by the single-slot codegen worker. When the body lands, the original
        request is re-queued and re-runs navigation onto the new leaf. A
        re-dispatched request whose skill is still unwritten must not author a
        second skill — it reports the pending write instead.
        """
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
        created = self._create_skill(request, category)
        if created.kind != "create_skill" or created.draft is None:
            return created
        if self.asker is not None:
            self._elicit_requirements(request, category, created.draft)
        elif self.elicitation_enabled:
            if self.defer_questions:
                self.trace.append(
                    "requirements_deferred",
                    request.id,
                    category=category,
                    skill=created.draft.name,
                )
            else:
                self.trace.append(
                    "requirements_skipped",
                    request.id,
                    category=category,
                    skill=created.draft.name,
                    reason="no asker wired",
                )
        self._start_skill_write(request, category, created.draft, weight)
        return created

    def _elicit_requirements(
        self, request: Request, category: str, draft: SkillDraft
    ) -> None:
        """Ask the product owner implementation questions before the body is written.

        Inline path (the REPL wires an asker): questions are generated and asked
        synchronously at dispatch. The dashboard uses the deferred path instead —
        the worker posts questions to `self.questions` and waits for answers.
        Answers seed the draft's requirements and the intended integration. Any
        failure degrades to no requirements — elicitation must never block
        authoring.
        """
        if not self.elicitation_enabled or self.codegen is None:
            return
        try:
            elicited = generate_elicitation(
                self.codegen, request, category, draft, self.tree,
                max_questions=self.elicitation_max,
            )
        except CodegenError:
            return
        if elicited.integration:
            draft.integration = elicited.integration
        for question in elicited.questions:
            try:
                answer = self.asker(question)
            except Exception:
                return
            if answer is None or not answer.strip():
                return
            draft.requirements[question] = answer.strip()
        if draft.requirements or draft.integration:
            self.trace.append(
                "requirements",
                request.id,
                category=category,
                skill=draft.name,
                questions=list(draft.requirements.keys()),
                integration=draft.integration or None,
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

    def answer(self, text: str) -> tuple[str, str]:
        """Feed the human's answer to a run paused for input.

        Routed directly to the pending run — no gate, score, or navigation —
        and the run resumes by re-invoking only `act` with the same prediction.
        """
        with self._lock:
            if self.pending is None:
                return "error", "no run is waiting for input"
            pending = self.pending
            self.pending = None
            pending.request.user_input = text
            self.trace.append("answered", pending.request.id, text=text)
            outcome = self.runner.resume(
                pending.skill, pending.request, pending.prediction,
                pre_predict=pending.pre_predict,
            )
            result = self._finish_run(pending.skill, pending.request, outcome)
            if result.kind == "needs_input":
                return "needs_input", result.summary
            self.current = None
            self.trace.append("ran", pending.request.id, skill=result.skill, summary=result.summary)
            return "ran", f"[resumed] {result.summary}"

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
                prediction=outcome.prediction,
                question=outcome.needs_input,
                pre_predict=outcome.pre_predict,
            )
            self.current = Process(request=request, skill=skill.name, weight=0.5)
            self.trace.append(
                "needs_input", request.id, skill=skill.name, question=outcome.needs_input
            )
            return DispatchResult(
                kind="needs_input",
                summary=outcome.needs_input,
                skill=skill.name,
                needs_input=outcome.needs_input,
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
            self.queue.push(child, 0.5)
            self.trace.append("requeued", request.id, text=outcome.updated_request)
            if self.on_request_requeued is not None:
                self.on_request_requeued(request.id, child.id)
            requeued = True
        if not outcome.success and not (outcome.updated_request and requeued):
            self._propose_repair(skill, request, outcome)
        return DispatchResult(
            kind="ran",
            summary=(
                f"{skill.name}: {'ok' if outcome.success else 'failed'} — "
                f"{outcome.action_log or outcome.summary}"
            ),
            skill=skill.name,
            decisions_logged=outcome.decisions_logged,
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
        if not self.repair_enabled or skill.is_noop():
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
        with; answering it starts the repair directly.
        """
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
        bounded by `codegen.repair.max_attempts` per skill.
        """
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
            self.queue.push(retry, 0.5)
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
            request, offer.category, draft, 0.5, repair_evidence=offer.evidence
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

    def _create_category(self, request: Request) -> DispatchResult:
        """Author a new category stub with the decision model in generation mode.

        The engine is always real, so an EngineUnavailable is fatal: it is
        recorded and re-raised for the caller to stop on.
        """
        try:
            draft = generate_category(self.engine, request, self.tree)
        except EngineUnavailable as exc:
            self._mark_fatal(exc)
            raise
        except ValueError as exc:
            self.trace.append("error", request.id, phase="create_category", message=str(exc))
            return DispatchResult(kind="error", summary=f"create_category failed: {exc}")
        if draft.name in self.tree:
            self.trace.append(
                "error",
                request.id,
                phase="create_category",
                message=f"category {draft.name} already exists",
            )
            return DispatchResult(
                kind="error",
                summary=f"create_category failed: {draft.name} already exists",
            )
        self.registry.register(draft.name, draft.description)
        self.tree[draft.name] = []
        self.trace.append(
            "category_created",
            request.id,
            category=draft.name,
            description=draft.description,
        )
        return DispatchResult(
            kind="create_category",
            summary=f"created category {draft.name}: {draft.description}",
            skill=draft.name,
        )

    def _create_skill(self, request: Request, category: str) -> DispatchResult:
        """Author a new skill leaf with the decision model in generation mode.

        The small model writes the title + description; the stub is registered
        and merged into the tree so the leaf is navigable immediately. If
        codegen is configured, the runnable body is written asynchronously (see
        _start_skill_write) and the request is re-dispatched once the body
        lands; the returned result carries the draft so _dispatch_skill can
        launch that write. Without codegen the stub is final. The engine is
        always real, so an EngineUnavailable is fatal: it is recorded and
        re-raised for the caller to stop on.
        """
        try:
            draft = generate_skill(self.engine, request, category, self.tree)
        except EngineUnavailable as exc:
            self._mark_fatal(exc)
            raise
        except ValueError as exc:
            self.trace.append("error", request.id, phase="create_skill", message=str(exc))
            return DispatchResult(kind="error", summary=f"create_skill failed: {exc}")
        existing = {s.name for s in self.tree.get(category, [])}
        if draft.name in existing:
            self.trace.append(
                "error",
                request.id,
                phase="create_skill",
                category=category,
                message=f"skill {draft.name} already exists",
            )
            return DispatchResult(
                kind="error",
                summary=f"create_skill failed: {draft.name} already exists",
            )

        self.registry.register_skill(
            category, draft.name, draft.description, request_text=request.text
        )
        self.tree.setdefault(category, []).append(
            Skill(name=draft.name, category=category, description=draft.description)
        )
        if self.codegen is None:
            self.trace.append(
                "skill_created",
                request.id,
                category=category,
                skill=draft.name,
                description=draft.description,
                body=None,
                written=False,
            )
            return DispatchResult(
                kind="create_skill",
                summary=f"created stub {category}.{draft.name}: {draft.description} (no codegen configured)",
                skill=draft.name,
            )
        return DispatchResult(
            kind="create_skill",
            summary=(
                f"created stub {category}.{draft.name}: {draft.description} — "
                "body writing in background; request will re-run when ready"
            ),
            skill=draft.name,
            draft=draft,
        )

    # ---- async skill-body writes (single-slot codegen worker) ----

    def _start_skill_write(
        self,
        request: Request,
        category: str,
        draft: SkillDraft,
        weight: float,
        repair_evidence: dict | None = None,
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
                    weight=weight,
                    repair_evidence=repair_evidence,
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
            self._write_skill_body(job)

    def _write_skill_body(self, job: SkillWrite) -> None:
        """Run the full authoring pipeline for one queued write (no scheduler
        lock held here): deferred elicitation -> codegen body (or repair
        rewrite) -> fidelity gate -> data contract -> test -> auto-run test
        (with a SemIf regen ladder on failure).

        The tree is snapshotted under the lock so the prompt build reads a
        stable view even if the main thread merges another skill meanwhile.
        """
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
                    reason_kind="run_failure",
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
            contract = generate_data_contract(
                self.codegen, job.request, job.category, job.draft, code
            )
        except EngineUnavailable as exc:
            self._mark_fatal(exc)
            self._fail_skill_write(job, exc)
            return
        except (CodegenError, ValueError) as exc:
            self._fail_skill_write(job, exc)
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

        The dashboard path: no inline asker exists, so the single-slot worker
        itself asks. It pauses here for up to `elicitation.wait_timeout` while
        the gate stays free and other requests proceed; whatever answers arrive
        (possibly none) feed the body writer.
        """
        if not (self.defer_questions and self.elicitation_enabled and self.codegen):
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
        """Post a question group and block the worker until answered or timed out."""
        group = [
            PendingQuestion(
                id=uuid.uuid4().hex[:12],
                run_id=job.request.id,
                category=job.category,
                skill=job.draft.name,
                question=question,
            )
            for question in questions
        ]
        with self._lock:
            self.questions.extend(group)
            self._question_notify.notify_all()
        self.trace.append(
            "questions_asked",
            job.request.id,
            category=job.category,
            skill=job.draft.name,
            questions=questions,
        )
        print(
            f"[codegen] {len(group)} question(s) waiting about "
            f"{job.category}.{job.draft.name} — answer in the dashboard."
        )
        deadline = (
            None if self.elicitation_wait <= 0
            else time.monotonic() + self.elicitation_wait
        )
        with self._lock:
            while not all(q.answer is not None or q.skipped for q in group):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                self._question_notify.wait(timeout=1.0)
            answers = {q.question: q.answer for q in group if q.answer}
            self.questions = [q for q in self.questions if q not in group]
            self._question_notify.notify_all()
        if len(answers) < len(group):
            self.trace.append(
                "questions_timeout",
                job.request.id,
                category=job.category,
                skill=job.draft.name,
                answered=len(answers),
                asked=len(group),
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

        On failure a SemIf decision picks which of code/contract/test to
        regenerate; whichever it is, the error and the existing files are fed
        back into the corrective call. Fixture data lives inside the test, so a
        fixture fix is a test regen. Bounded by `test_max_attempts`.
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
            if target == "regen_contract":
                contract = generate_data_contract(
                    self.codegen, job.request, job.category, job.draft, code, reason=reason
                )
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
        maps it against candidate values from the global config and the
        category config. Unmatched variables are left to the first-fire ask.
        The engine is always real: an EngineUnavailable propagates to the
        worker, which marks the app fatal."""
        merged: dict = {}
        merged.update(self.config)
        merged.update(self.body_store.read_category_config(job.category))
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
            self.queue.push(requeued, job.weight)
            self.trace.append(
                "skill_requeued",
                job.request.id,
                text=requeued.text,
                weight=job.weight,
            )
        print(
            f"[codegen] body ready for {job.category}.{job.draft.name}; "
            "original request re-queued."
        )
        self.run_queue()

    def _fail_skill_write(self, job: SkillWrite, exc: Exception) -> None:
        """Clear the in-progress flag and notify; the leaf stays a restartable stub."""
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
        self._start_skill_write(origin, category, draft, 0.5)
        self.trace.append("skill_restarted", origin.id, category=category, skill=name)
        return "running", f"restarting body write for {category}.{name}"

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
            for weight, request in self.queue.items():
                lines.append(f"  {request.id}  w={weight:.2f}  {request.text[:60]}")
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