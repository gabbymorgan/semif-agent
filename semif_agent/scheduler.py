"""The scheduler: gate, choice, score, queue, dispatch.

Every SemIf decision (gate, choice, score, navigation, prediction) is logged.
A high-priority input can preempt the current process, which requeues with its
state preserved; a deferred input is scored and queued by urgency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .decisions import DecisionRequest, Option, Request
from .engine import SemIfEngine
from .llm import LLMClient
from .log import DecisionLog
from .queue import UrgencyQueue
from .skill import SkillRunner
from .skills import (
    ActionContext,
    CreateSkill,
    Skill,
    build_skills,
    build_tree,
    compose_state,
    navigate,
)

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
class DispatchResult:
    kind: str  # ran | create_skill | error
    summary: str
    skill: str | None = None
    decisions_logged: int = 0


class Scheduler:
    def __init__(
        self,
        engine: SemIfEngine,
        llm: LLMClient,
        log: DecisionLog,
        config: dict,
        tau: float = 0.6,
        max_reentries: int = 3,
    ):
        self.engine = engine
        self.llm = llm
        self.log = log
        self.config = config
        self.tau = tau
        self.max_reentries = max_reentries
        self.queue = UrgencyQueue(
            max_size=int(config.get("queue", {}).get("max_size", 100)),
            age_rate=float(config.get("queue", {}).get("age_rate", 0.0)),
        )
        self.skills = build_skills(config)
        self.tree = build_tree(self.skills)
        self.ctx = ActionContext(engine=self.engine, config=config)
        self.runner = SkillRunner(self.ctx, self.llm, self.log)
        self.current: Process | None = None

    # ---- decision templates (all real SemIf, all logged) ----

    def _contains_request(self, request: Request) -> bool:
        decision = DecisionRequest(
            state=compose_state(request),
            question="Does this input contain an actionable request?",
            options=[Option(GATE_YES, "Yes, it is actionable."), Option("no", "No, it is not.")],
        )
        result = self.engine.call(decision)
        self.log.append(decision, result, extra={"phase": "gate"})
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
        self.log.append(decision, result, extra={"phase": "choice", "current": current.skill})
        return result.prob(CHOICE_INTERRUPT) >= self.tau

    def _score(self, request: Request, current: str | None = None) -> tuple[float, str]:
        decision = DecisionRequest(
            state=compose_state(request, current=current),
            question="How urgent is this request?",
            options=[Option(option_id, description) for option_id, description in URGENCY_OPTIONS],
        )
        result = self.engine.call(decision)
        self.log.append(decision, result, extra={"phase": "score"})
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
        if not self._contains_request(request):
            return "dropped", "no actionable request"

        if self.current is None:
            weight, label = self._score(request)
            self.current = Process(request=request, skill="(scheduling)", weight=weight)
            outcome = self._dispatch(request)
            self.current = None
            return "running", f"[{label}] {outcome.summary}"

        interrupt = self._choice(request, self.current)
        if interrupt:
            previous = self.current
            previous.request.resume["from_skill"] = previous.skill
            self.queue.push(previous.request, previous.weight)
            self.current = Process(request=request, skill="(scheduling)", weight=1.0)
            outcome = self._dispatch(request)
            self.current = None
            return "preempted", f"interrupted {previous.skill}; {outcome.summary}"

        weight, label = self._score(request, current=self.current.skill)
        ok = self.queue.push(request, weight)
        if not ok:
            return "rejected", "queue is full"
        return "queued", f"urgency {label} (weight {weight:.2f})"

    def busy(self, text: str, skill: str = "(driving)") -> None:
        """Set a fake in-progress process so the choice/score path is exercised."""
        self.current = Process(request=Request(text, source="busy"), skill=skill, weight=1.0)

    def idle(self) -> None:
        self.current = None

    def run_queue(self) -> list[tuple[str, str]]:
        """Process the queue while idle. Returns the outcomes."""
        from .engine import EngineUnavailable

        results = []
        while self.current is None and len(self.queue) > 0:
            request = self.queue.pop()
            self.current = Process(request=request, skill="(scheduling)", weight=0.0)
            try:
                outcome = self._dispatch(request)
            except EngineUnavailable as exc:
                outcome = DispatchResult(kind="error", summary=f"engine unavailable: {exc}")
            self.current = None
            results.append(("ran", f"[{request.id}] {outcome.summary}"))
        return results

    # ---- dispatch ----

    def _dispatch(self, request: Request) -> DispatchResult:
        navigation = navigate(self.engine, request, self.tree)
        if isinstance(navigation, CreateSkill):
            return DispatchResult(
                kind="create_skill",
                summary="skill authoring via opencode is deferred to v2; request logged.",
            )
        outcome = self.runner.run(navigation, request)
        if outcome.error:
            return DispatchResult(kind="error", summary=f"skill error: {outcome.error}")
        if outcome.updated_request and request.reentries < self.max_reentries:
            self.queue.push(_requeue(request, outcome.updated_request), 0.5)
        return DispatchResult(
            kind="ran",
            summary=f"{navigation.name}: {'ok' if outcome.success else 'failed'} — {outcome.summary}",
            skill=navigation.name,
            decisions_logged=outcome.decisions_logged,
        )

    def status(self) -> str:
        lines = []
        current = f"{self.current.skill} ({self.current.request.id})" if self.current else "idle"
        lines.append(f"current: {current}")
        lines.append(f"queue: {len(self.queue)} pending")
        for weight, request in self.queue.items():
            lines.append(f"  {request.id}  w={weight:.2f}  {request.text[:60]}")
        return "\n".join(lines)


def _requeue(request: Request, updated_text: str) -> Request:
    updated = Request(updated_text, source="requeue")
    updated.reentries = request.reentries + 1
    return updated