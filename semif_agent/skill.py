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
            observed = action.new_state
            assessment: Assessment = self.llm.assess(skill.name, baseline, action.action_log)
        except Exception as exc:
            return RunResult(
                skill=skill.name,
                success=False,
                summary="",
                action_log="",
                new_state=baseline,
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
        )