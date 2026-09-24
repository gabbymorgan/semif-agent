"""The skill execution loop: observe -> predict -> act -> observe -> assess.

Every SemIf decision made during a run is logged as a training row; the
assessment outcome is kept on the row so the dream pass can weigh failed runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .decisions import Request
from .llm import Assessment, LLMClient
from .log import DecisionLog
from .skills import ActionContext, Prediction, Skill


@dataclass
class RunResult:
    skill: str
    success: bool
    summary: str
    action_log: str
    new_state: str
    updated_request: str | None = None
    decisions_logged: int = 0
    error: str | None = None
    needs_input: str | None = None
    prediction: Prediction | None = None


class SkillRunner:
    def __init__(self, ctx: ActionContext, llm: LLMClient, log: DecisionLog):
        self.ctx = ctx
        self.llm = llm
        self.log = log

    def run(self, skill: Skill, request: Request) -> RunResult:
        baseline = request.text
        try:
            prediction = skill.predict(self.ctx, request) if skill.predict else Prediction(text="")
            action = skill.act(self.ctx, request, prediction)
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

    def resume(self, skill: Skill, request: Request, prediction: Prediction) -> RunResult:
        """Re-invoke act with the human's answer (on request.user_input) and finish.

        predict is not re-run: its SemIf sub-decisions were already made and are
        logged here, at completion, so their run_ok label reflects the outcome.
        """
        baseline = request.text
        try:
            action = skill.act(self.ctx, request, prediction)
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

    def _finish(
        self, skill: Skill, request: Request, prediction: Prediction, action
    ) -> RunResult:
        baseline = request.text
        observed = action.new_state
        try:
            assessment: Assessment = self.llm.assess(skill.name, baseline, action.action_log)
        except Exception as exc:
            return RunResult(
                skill=skill.name,
                success=False,
                summary="",
                action_log="",
                new_state=observed,
                error=str(exc),
            )

        run_ok = assessment.success
        decisions = getattr(prediction, "decisions", [])
        for decision, result in decisions:
            self.log.append(
                decision,
                result,
                extra={
                    "phase": "predict",
                    "skill": skill.name,
                    "run_ok": run_ok,
                    "run_id": request.id,
                },
            )

        return RunResult(
            skill=skill.name,
            success=assessment.success,
            summary=assessment.summary,
            action_log=action.action_log,
            new_state=observed,
            updated_request=assessment.updated_request,
            decisions_logged=len(decisions),
            prediction=prediction,
        )