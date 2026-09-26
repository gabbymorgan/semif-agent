"""The scheduler: gate, choice, score, queue, dispatch.

Every SemIf decision (gate, choice, score, navigation, prediction) is logged.
A high-priority input can preempt the current process, which requeues with its
state preserved; a deferred input is scored and queued by urgency.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .codegen import (
    CodegenClient,
    CodegenError,
    generate_data_contract,
    generate_requirements,
    generate_skill_body,
    generate_skill_tests,
    regenerate_skill_body,
    run_skill_test,
    skill_contract_ref,
)
from .decisions import DecisionRequest, Option, Request
from .engine import SemIfEngine
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
    generate_category,
    generate_skill,
    materialize_skill,
    merge_registry,
    merge_skill_store,
    navigate,
)
from .trace import TraceLog

GATE_YES = "yes"
CHOICE_INTERRUPT = "interrupt"
URGENCY_OPTIONS = [
    ("critical", "Immediate danger or critical failure."),
    ("high", "Important but not dangerous."),
    ("medium", "Should be handled reasonably soon."),
    ("low", "Can wait."),
]
URGENCY_WEIGHTS = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}


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
    ):
        self.engine = engine
        self.llm = llm
        self.log = log
        self.trace = trace if trace is not None else TraceLog()
        self.config = config
        self.tau = tau
        self.max_reentries = max_reentries
        self.codegen = codegen
        self.degeneration_check_factory = degeneration_check_factory
        self.regen_decision_factory = regen_decision_factory
        self.asker = asker
        codegen_cfg = config.get("codegen", {}) or {}
        elicitation_cfg = codegen_cfg.get("elicitation", {}) or {}
        self.elicitation_enabled = bool(elicitation_cfg.get("enabled", False))
        self.elicitation_max = int(elicitation_cfg.get("max_questions", 3))
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
        merge_registry(self.tree, self.registry.read())
        merge_skill_store(self.tree, self.body_store, self.registry.read())
        self.ctx = ActionContext(engine=self.engine, config=config)
        self.runner = SkillRunner(self.ctx, self.llm, self.log, store=self.body_store)
        self.current: Process | None = None
        self.pending: PendingRun | None = None
        self._lock = threading.RLock()
        self._writes: deque[SkillWrite] = deque()
        self._write_notify = threading.Condition(self._lock)
        self._write_thread: threading.Thread | None = None

    # ---- decision templates (all real SemIf, all logged) ----

    def _contains_request(self, request: Request) -> bool:
        decision = DecisionRequest(
            state=compose_state(request),
            question="Does this input contain an actionable request?",
            options=[Option(GATE_YES, "Yes, it is actionable."), Option("no", "No, it is not.")],
        )
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
        from .engine import EngineUnavailable

        try:
            return self._submit(text, source)
        except EngineUnavailable as exc:
            return "error", f"decision engine unavailable: {exc}"

    def _submit(self, text: str, source: str = "typed") -> tuple[str, str]:
        with self._lock:
            return self._submit_locked(text, source)

    def _submit_locked(self, text: str, source: str = "typed") -> tuple[str, str]:
        request = Request(text, source=source)
        self.trace.append("submit", request.id, text=text, source=source)
        if not self._contains_request(request):
            self.trace.append("dropped", request.id, reason="no actionable request")
            return "dropped", "no actionable request"

        if self.current is None:
            weight, label = self._score(request)
            self.current = Process(request=request, skill="(scheduling)", weight=weight)
            try:
                outcome = self._dispatch(request, weight=weight)
            finally:
                if self.pending is None:
                    self.current = None
            if outcome.kind == "needs_input":
                return "needs_input", outcome.summary
            self.trace.append("ran", request.id, skill=outcome.skill, summary=outcome.summary)
            return "running", f"[{label}] {outcome.summary}"

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
                return "preempted", f"interrupted {previous.skill}; {outcome.summary}"
            self.current = None
            self.trace.append("ran", request.id, skill=outcome.skill, summary=outcome.summary)
            return "preempted", f"interrupted {previous.skill}; {outcome.summary}"

        weight, label = self._score(request, current=self.current.skill)
        ok = self.queue.push(request, weight)
        if not ok:
            self.trace.append("rejected", request.id, reason="queue is full")
            return "rejected", "queue is full"
        self.trace.append("queued", request.id, weight=weight, label=label)
        return "queued", f"urgency {label} (weight {weight:.2f})"

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
        from .engine import EngineUnavailable

        results = []
        while self.current is None and len(self.queue) > 0:
            request = self.queue.pop()
            self.trace.append("dequeued", request.id)
            self.current = Process(request=request, skill="(scheduling)", weight=0.0)
            try:
                outcome = self._dispatch(request, weight=0.0)
            except EngineUnavailable as exc:
                outcome = DispatchResult(kind="error", summary=f"engine unavailable: {exc}")
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
        navigation = navigate(self.engine, self.log, self.trace, request, self.tree)
        if isinstance(navigation, CreateCategory):
            created = self._create_category(request)
            if created.kind != "create_category":
                return created
            return self._dispatch_skill(request, created.skill, weight)
        if isinstance(navigation, CreateSkill):
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
        self._elicit_requirements(request, category, created.draft)
        self._start_skill_write(request, category, created.draft, weight)
        return created

    def _elicit_requirements(
        self, request: Request, category: str, draft: SkillDraft
    ) -> None:
        """Ask the product owner refinement questions before the body is written.

        Opt-in (`codegen.elicitation.enabled`) and requires an asker (the REPL
        wires one; the dashboard degrades to skip). Requirements answers ride on
        the draft into the body prompt. Any failure degrades to no requirements —
        elicitation must never block authoring.
        """
        if not (self.elicitation_enabled and self.asker is not None):
            return
        if self.codegen is None:
            return
        try:
            questions = generate_requirements(
                self.codegen, request, category, draft, self.tree,
                max_questions=self.elicitation_max,
            )
        except CodegenError:
            return
        for question in questions:
            try:
                answer = self.asker(question)
            except Exception:
                return
            if answer is None or not answer.strip():
                return
            draft.requirements[question] = answer.strip()
        if draft.requirements:
            self.trace.append(
                "requirements",
                request.id,
                category=category,
                skill=draft.name,
                questions=list(draft.requirements.keys()),
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
            updated_request=outcome.updated_request,
        )
        if outcome.updated_request and request.reentries < self.max_reentries:
            self.queue.push(_requeue(request, outcome.updated_request), 0.5)
            self.trace.append("requeued", request.id, text=outcome.updated_request)
        return DispatchResult(
            kind="ran",
            summary=f"{skill.name}: {'ok' if outcome.success else 'failed'} — {outcome.summary}",
            skill=skill.name,
            decisions_logged=outcome.decisions_logged,
        )

    def _create_category(self, request: Request) -> DispatchResult:
        """Author a new category stub with the decision model in generation mode."""
        from .engine import EngineUnavailable

        try:
            draft = generate_category(self.engine, request, self.tree)
        except (EngineUnavailable, ValueError) as exc:
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
        launch that write. Without codegen the stub is final.
        """
        from .engine import EngineUnavailable

        try:
            draft = generate_skill(self.engine, request, category, self.tree)
        except (EngineUnavailable, ValueError) as exc:
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
        self, request: Request, category: str, draft: SkillDraft, weight: float
    ) -> None:
        """Mark the leaf in-progress and queue the body write for the worker.

        The gate stays free: this returns immediately and the worker (one write
        at a time, the 12G codegen model can't run twice) writes the body in the
        background. On completion the original request is re-queued and re-runs
        navigation onto the new leaf; on failure the leaf stays a restartable
        stub and only the user is notified.
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
            )
            self._writes.append(
                SkillWrite(request=request, category=category, draft=draft, weight=weight)
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
        """Drain the skill-write queue one body at a time."""
        while True:
            with self._write_notify:
                while not self._writes:
                    self._write_notify.wait()
                job = self._writes.popleft()
            self._write_skill_body(job)

    def _write_skill_body(self, job: SkillWrite) -> None:
        """Run the full authoring pipeline for one queued write (no scheduler
        lock held here): codegen body -> data contract -> test -> auto-run test
        (with a SemIf regen ladder on failure).

        The tree is snapshotted under the lock so the prompt build reads a
        stable view even if the main thread merges another skill meanwhile.
        """
        with self._lock:
            tree_snapshot = {category: list(skills) for category, skills in self.tree.items()}
        try:
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
            contract = generate_data_contract(
                self.codegen, job.request, job.category, job.draft, code
            )
        except (CodegenError, ValueError) as exc:
            self._fail_skill_write(job, exc)
            return
        self.body_store.write_body(job.category, job.draft.name, code)
        self.body_store.write_contract(job.category, job.draft.name, contract)
        if self.contract_search:
            self._config_search(job, contract)
        try:
            self._test_and_fix(job, code, contract)
        except (CodegenError, ValueError) as exc:
            self._fail_skill_write(job, exc)
            return
        self._complete_skill_write(job, code)

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
                    self.codegen, job.request, job.category, job.draft, code, reason
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
        category config. Unmatched variables are left to the first-fire ask."""
        from .engine import EngineUnavailable

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
        except (EngineUnavailable, ValueError, TypeError):
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
            entry = self.registry.read().get(category, {})
            row = next(
                (s for s in entry.get("skills", []) if s.get("name") == name),
                {},
            )
            request_text = row.get("request_text")
            draft = SkillDraft(
                name=name,
                description=leaf.description or row.get("description") or name,
            )
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