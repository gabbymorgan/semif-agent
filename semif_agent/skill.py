"""The skill execution loop: observe -> predict -> act -> observe -> assess.

Assessment is a SemIf decision, not generation: `assess:outcome` decides
success/failure (P(success) >= tau) and, on failure, `assess:requeue` decides
complete/retry. The run summary is deterministic — same inputs, same string —
built from the category, skill, outcome flag, and action log. Every SemIf
decision made during a run is logged as a training row; the assessment outcome
is kept on the row so the dream pass can weigh failed runs.

A skill with a data contract (contract.json) may need variables collected from
the human before it runs. Those are asked pre-predict, runner-driven: `run`
resolves the tiered config (global -> category -> skill) plus any per-fire
answers, and if contract variables are still unresolved it pauses with a
`needs_input` instead of invoking the skill. The answer is then either recorded
as config (SemIf "record or ask again" choice) or kept as a per-fire input, and
the resolution continues.
"""

from __future__ import annotations

from dataclasses import dataclass

from .decisions import DecisionRequest, Option, Request
from .log import DecisionLog
from .skills import (
    ActionContext,
    Prediction,
    Skill,
    SkillStore,
    resolve_skill_config,
    unresolved_variables,
)


def deterministic_summary(category: str, name: str, success: bool, action_log: str, new_state: str) -> str:
    """The run summary, computed without generation.

    Same inputs always produce the same string. Feeds both `RunResult.summary`
    and `assessment_summary`; consumers keep reading whichever field they read
    before.
    """
    outcome = "ok" if success else "failed"
    detail = (action_log or new_state or "").strip()
    return f"{category}.{name}: {outcome} — {detail}"


@dataclass
class RunResult:
    skill: str
    success: bool
    summary: str
    action_log: str
    new_state: str
    updated_request: str | None = None
    decisions_logged: int = 0
    assessment_summary: str = ""
    error: str | None = None
    needs_input: str | None = None
    prediction: Prediction | None = None
    pre_predict: bool = False


class SkillRunner:
    def __init__(
        self,
        ctx: ActionContext,
        log: DecisionLog,
        store: SkillStore | None = None,
        tau: float = 0.6,
    ):
        self.ctx = ctx
        self.log = log
        self.store = store
        self.tau = tau

    # ---- contract resolution ----

    def _skill_context(self, skill: Skill, request: Request) -> ActionContext:
        """Tiered config merged into an ActionContext for one run."""
        answered = request.meta.get("config_answers", {})
        if self.store is not None:
            merged = resolve_skill_config(self.store, skill, self.ctx.config, answered)
        else:
            merged = {**self.ctx.config}
            merged.update(skill.config or {})
            merged.update(answered)
        return ActionContext(engine=self.ctx.engine, config=merged)

    def _unresolved(self, skill: Skill, request: Request) -> list[str]:
        if not skill.contract:
            return []
        answered = request.meta.get("config_answers", {})
        if self.store is not None:
            return unresolved_variables(self.store, skill, self.ctx.config, answered)
        merged = {**self.ctx.config}
        merged.update(skill.config or {})
        merged.update(answered)
        return [name for name in skill.contract if name not in merged]

    def _variable_question(self, skill: Skill, variable: str) -> str:
        desc = (skill.contract or {}).get(variable, "")
        return (
            f"To run {skill.category}.{skill.name} I need `{variable}`. {desc}\n"
            "Reply with the value, or 'skip' to leave it empty."
        )

    def _decide_record(self, skill: Skill, request: Request, variable: str, value: str) -> bool:
        """SemIf choice: record this answer as skill config, or ask again each
        fire. The engine is always real; an unavailable engine propagates and
        the scheduler marks the app fatal."""
        decision = DecisionRequest(
            state=(
                f"[config] skill {skill.category}.{skill.name} got "
                f"{variable} = {value}"
            ),
            question=(
                f"Record `{variable}` as this skill's config, or ask again "
                "each time it fires?"
            ),
            options=[
                Option("record", "Record it in the skill config."),
                Option("ask_again", "Ask again each time the skill fires."),
            ],
        )
        result = self.ctx.engine.call(decision)
        self.log.append(
            decision,
            result,
            extra={
                "phase": "config:record",
                "run_id": request.id,
                "skill": f"{skill.category}.{skill.name}",
                "variable": variable,
            },
        )
        return result.selected == "record"

    def _record_answer(self, skill: Skill, request: Request) -> None:
        """Persist (or stash per-fire) the human's answer to the last question.

        `request.user_input` is cleared after consumption so a contract answer
        does not leak into a later act-driven pause or the skill's own
        user_input checks.
        """
        variable = request.meta.pop("awaiting_var", None)
        answer = request.user_input
        request.user_input = None
        if variable is None or answer is None:
            return
        value = answer.strip()
        if not value or value.lower() == "skip":
            return
        if self._decide_record(skill, request, variable, value):
            skill.config = {**(skill.config or {}), variable: value}
            if self.store is not None:
                self.store.write_config(skill.category, skill.name, skill.config)
        else:
            request.meta.setdefault("config_answers", {})[variable] = value

    # ---- run / resume ----

    def run(self, skill: Skill, request: Request) -> RunResult:
        baseline = request.text
        missing = self._unresolved(skill, request)
        if missing:
            variable = missing[0]
            request.meta["awaiting_var"] = variable
            return RunResult(
                skill=skill.name,
                success=False,
                summary="",
                action_log="",
                new_state=baseline,
                needs_input=self._variable_question(skill, variable),
                prediction=None,
                pre_predict=True,
            )
        ctx = self._skill_context(skill, request)
        try:
            prediction = skill.predict(ctx, request) if skill.predict else Prediction(text="")
            action = skill.act(ctx, request, prediction)
        except Exception as exc:
            return RunResult(
                skill=skill.name,
                success=False,
                summary="",
                action_log="",
                new_state=baseline,
                error=str(exc),
            )
        if action.needs_input:
            return RunResult(
                skill=skill.name,
                success=False,
                summary="",
                action_log=action.action_log,
                new_state=baseline,
                needs_input=action.needs_input,
                prediction=prediction,
            )
        return self._finish(skill, request, prediction, action)

    def resume(
        self,
        skill: Skill,
        request: Request,
        prediction: Prediction | None,
        pre_predict: bool = False,
    ) -> RunResult:
        """Re-invoke the run with the human's answer (on request.user_input).

        For an act-driven pause (predict already ran) only `act` is re-invoked
        with the same prediction. For a pre-predict pause (contract collection)
        the answer is recorded and the whole resolution + predict + act path is
        re-run — more variables may be missing, pausing again until satisfied.
        """
        if pre_predict:
            self._record_answer(skill, request)
            return self.run(skill, request)
        baseline = request.text
        ctx = self._skill_context(skill, request)
        try:
            action = skill.act(ctx, request, prediction)
        except Exception as exc:
            return RunResult(
                skill=skill.name,
                success=False,
                summary="",
                action_log="",
                new_state=baseline,
                error=str(exc),
            )
        if action.needs_input:
            return RunResult(
                skill=skill.name,
                success=False,
                summary="",
                action_log=action.action_log,
                new_state=baseline,
                needs_input=action.needs_input,
                prediction=prediction,
            )
        return self._finish(skill, request, prediction, action)

    def _assess(self, skill: Skill, request: Request, action) -> tuple[bool, str | None]:
        """SemIf assessment: did the run succeed, and should it run again?

        `assess:outcome` decides success (`P(success) >= tau`); a failure is
        then offered to `assess:requeue`, which picks complete/retry. On retry
        the original request text is re-dispatched (the scheduler bounds it by
        `max_reentries`). Both are real logged decision rows. The engine is
        always real: EngineUnavailable propagates to the scheduler, which marks
        the app fatal.
        """
        label = f"{skill.category}.{skill.name}"
        outcome = DecisionRequest(
            state=(
                f"skill: {label}\n"
                f"goal: {request.text}\n"
                f"action log:\n{action.action_log}"
            ),
            question="Did the skill achieve the user's goal?",
            options=[
                Option("success", "Yes — the goal was met."),
                Option("failure", "No — the goal was not met."),
            ],
        )
        result = self.ctx.engine.call(outcome)
        self.log.append(
            outcome,
            result,
            extra={"phase": "assess:outcome", "run_id": request.id, "skill": label},
        )
        success = result.prob("success") >= self.tau
        if success:
            return True, None

        requeue = DecisionRequest(
            state=(
                f"skill: {label}\n"
                f"goal: {request.text}\n"
                f"action log:\n{action.action_log}\n"
                "outcome: failed"
            ),
            question="Is the request complete, or should it run again?",
            options=[
                Option("complete", "It is complete; do nothing further."),
                Option("retry", "Run the original request again."),
            ],
        )
        requeue_result = self.ctx.engine.call(requeue)
        self.log.append(
            requeue,
            requeue_result,
            extra={"phase": "assess:requeue", "run_id": request.id, "skill": label},
        )
        updated = request.text if requeue_result.prob("retry") >= self.tau else None
        return False, updated

    def _finish(
        self, skill: Skill, request: Request, prediction: Prediction | None, action
    ) -> RunResult:
        observed = action.new_state
        success, updated_request = self._assess(skill, request, action)

        decisions = getattr(prediction, "decisions", [])
        for decision, result in decisions:
            self.log.append(
                decision,
                result,
                extra={
                    "phase": "predict",
                    "skill": skill.name,
                    "run_ok": success,
                    "run_id": request.id,
                },
            )

        summary = deterministic_summary(
            skill.category, skill.name, success, action.action_log, observed
        )
        return RunResult(
            skill=skill.name,
            success=success,
            summary=summary,
            action_log=action.action_log,
            new_state=observed,
            updated_request=updated_request,
            decisions_logged=len(decisions),
            assessment_summary=summary,
            prediction=prediction,
        )