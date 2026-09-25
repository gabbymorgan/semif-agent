"""The scheduler: gate, choice, score, queue, dispatch.

Every SemIf decision (gate, choice, score, navigation, prediction) is logged.
A high-priority input can preempt the current process, which requeues with its
state preserved; a deferred input is scored and queued by urgency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .codegen import CodegenClient, CodegenError, generate_skill_body
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
    SkillBodyStore,
    build_skills,
    build_tree,
    compose_state,
    generate_category,
    generate_skill,
    materialize_skill,
    merge_registry,
    merge_skill_bodies,
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
    run asked the human.
    """

    request: Request
    skill: Skill
    prediction: Prediction
    question: str


@dataclass
class DispatchResult:
    kind: str  # ran | create_category | create_skill | needs_input | error
    summary: str
    skill: str | None = None
    decisions_logged: int = 0
    body_written: bool = False
    needs_input: str | None = None


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
        self.queue = UrgencyQueue(
            max_size=int(config.get("queue", {}).get("max_size", 100)),
            age_rate=float(config.get("queue", {}).get("age_rate", 0.0)),
        )
        self.skills = build_skills(config)
        self.tree = build_tree(self.skills)
        self.registry = CategoryRegistry(config.get("category_registry", "data/categories.json"))
        self.body_store = SkillBodyStore(config.get("skill_bodies", "data/skills"))
        merge_registry(self.tree, self.registry.read())
        merge_skill_bodies(self.tree, self.body_store, self.registry.read())
        self.ctx = ActionContext(engine=self.engine, config=config)
        self.runner = SkillRunner(self.ctx, self.llm, self.log)
        self.current: Process | None = None
        self.pending: PendingRun | None = None

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
        request = Request(text, source=source)
        self.trace.append("submit", request.id, text=text, source=source)
        if not self._contains_request(request):
            self.trace.append("dropped", request.id, reason="no actionable request")
            return "dropped", "no actionable request"

        if self.current is None:
            weight, label = self._score(request)
            self.current = Process(request=request, skill="(scheduling)", weight=weight)
            try:
                outcome = self._dispatch(request)
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
            outcome = self._dispatch(request)
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
        if self.pending is not None:
            self.trace.append("pending_abandoned", self.pending.request.id)
            self.pending = None
        self.current = Process(request=Request(text, source="busy"), skill=skill, weight=1.0)

    def idle(self) -> None:
        if self.pending is not None:
            self.trace.append("pending_abandoned", self.pending.request.id)
            self.pending = None
        self.current = None

    def run_queue(self) -> list[tuple[str, str]]:
        """Process the queue while idle. Returns the outcomes."""
        from .engine import EngineUnavailable

        results = []
        while self.current is None and len(self.queue) > 0:
            request = self.queue.pop()
            self.trace.append("dequeued", request.id)
            self.current = Process(request=request, skill="(scheduling)", weight=0.0)
            try:
                outcome = self._dispatch(request)
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

    def _dispatch(self, request: Request) -> DispatchResult:
        navigation = navigate(self.engine, self.log, self.trace, request, self.tree)
        if isinstance(navigation, CreateCategory):
            created = self._create_category(request)
            if created.kind != "create_category":
                return created
            return self._dispatch_skill(request, created.skill)
        if isinstance(navigation, CreateSkill):
            return self._dispatch_skill(request, navigation.category)
        return self._run_skill(navigation, request)

    def _dispatch_skill(self, request: Request, category: str) -> DispatchResult:
        """create_skill in `category`, then run the new skill so the request is answered.

        The created leaf is executed directly, not via a re-dispatch that would
        re-run navigation on a tree that just changed.
        """
        created = self._create_skill(request, category)
        if created.kind != "create_skill" or not created.body_written:
            return created
        skill = next(
            (s for s in self.tree.get(category, []) if s.name == created.skill),
            None,
        )
        if skill is None:
            return created
        return self._run_skill(skill, request)

    def _run_skill(self, skill: Skill, request: Request) -> DispatchResult:
        outcome = self.runner.run(skill, request)
        return self._finish_run(skill, request, outcome)

    def answer(self, text: str) -> tuple[str, str]:
        """Feed the human's answer to a run paused for input.

        Routed directly to the pending run — no gate, score, or navigation —
        and the run resumes by re-invoking only `act` with the same prediction.
        """
        if self.pending is None:
            return "error", "no run is waiting for input"
        pending = self.pending
        self.pending = None
        pending.request.user_input = text
        self.trace.append("answered", pending.request.id, text=text)
        outcome = self.runner.resume(pending.skill, pending.request, pending.prediction)
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

        The small model writes the title + description; a larger OpenAI-
        compatible model then writes the runnable body against SKILL.md. The
        stub is registered first so the leaf is navigable even if the body
        write fails; a successful write is merged into the tree as a runnable
        skill and executed directly by _dispatch_skill.
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

        self.registry.register_skill(category, draft.name, draft.description)
        self.tree.setdefault(category, []).append(
            Skill(name=draft.name, category=category, description=draft.description)
        )
        self.trace.append(
            "skill_writing",
            request.id,
            category=category,
            skill=draft.name,
            description=draft.description,
            model=self.codegen.model if self.codegen else None,
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

        try:
            draft.code = generate_skill_body(
                self.codegen,
                request,
                category,
                draft,
                self.tree,
                degeneration_check=(
                    self.degeneration_check_factory(request.id)
                    if self.degeneration_check_factory is not None
                    else None
                ),
            )
            skill = materialize_skill(draft, category, self.body_store)
        except (CodegenError, ValueError) as exc:
            self.trace.append(
                "error",
                request.id,
                phase="create_skill",
                category=category,
                message=f"skill body write failed: {exc}",
            )
            return DispatchResult(
                kind="create_skill",
                summary=f"created stub {category}.{draft.name}: {draft.description} (body write failed: {exc})",
                skill=draft.name,
            )

        skills = self.tree.setdefault(category, [])
        for index, existing in enumerate(skills):
            if existing.name == draft.name:
                skills[index] = skill
                break
        else:
            skills.append(skill)
        skills.sort(key=lambda s: s.name)
        body_path = self.body_store.body_path(category, draft.name).as_posix()
        self.trace.append(
            "skill_created",
            request.id,
            category=category,
            skill=draft.name,
            description=draft.description,
            body=body_path,
            written=True,
        )
        return DispatchResult(
            kind="create_skill",
            summary=f"created skill {category}.{draft.name}: {draft.description}",
            skill=draft.name,
            body_written=True,
        )

    def status(self) -> str:
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