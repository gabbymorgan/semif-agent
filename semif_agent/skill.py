"""The skill execution loop: observe -> act -> observe -> assess.

The body is a single `act(ctx, request)` phase. It reads resolved values from
`ctx.config` (the runner resolves the contract from the tiered config and asks
the human for anything unresolved), performs the real action, and returns an
`ActionResult`. Any SemIf sub-decisions it made ride back on
`ActionResult.decisions` and are logged by the runner with the run outcome.

Assessment is a SemIf decision, not generation: `assess:outcome` decides
success/failure (P(success) >= tau) and, on failure, `assess:requeue` decides
complete/retry. The run summary is deterministic — same inputs, same string —
built from the category, skill, outcome flag, and action log. Every SemIf
decision made during a run is logged as a training row; the assessment outcome
is kept on the row so the dream pass can weigh failed runs.

A skill with a data contract may need variables collected from the human before
it runs. Those are asked pre-act, runner-driven: `run` resolves the tiered
config (global -> category -> skill) plus any per-fire answers, and if contract
variables are still unresolved it pauses with a `needs_input` instead of
invoking the skill. The answer is then either recorded as config (SemIf "record
or ask again" choice) or kept as a per-fire input, and the resolution
continues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .decisions import DecisionRequest, Option, Request
from .log import DecisionLog
from .skills import (
    CANNED_CATEGORIES,
    ActionContext,
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


# Variable names whose resolved value must never be written to the decision log
# (which is a training artifact): credentials, tokens, secrets.
_SECRET_NAME_RE = re.compile(r"pass|token|secret|key|credential|auth", re.IGNORECASE)


def _resolved_inputs(skill: Skill, config: dict) -> str:
    """Render the resolved contract values for the assessment state.

    Gives the outcome decision the full query -> resolution -> action path so it
    can weigh a wrong resolution, not just a bad action. Secret-named variables
    are redacted so credentials never reach the decision log.
    """
    contract = skill.contract or {}
    if not contract:
        return "(none)"
    lines = []
    for name in contract:
        value = config.get(name)
        if value is None:
            rendered = "(unset)"
        elif _SECRET_NAME_RE.search(name):
            rendered = "***"
        else:
            rendered = str(value)
        lines.append(f"- {name}: {rendered}")
    return "\n".join(lines)


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
    pre_act: bool = False


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
        return ActionContext(engine=self.ctx.engine, config=merged, timers=self.ctx.timers)

    def _unresolved(self, skill: Skill, request: Request) -> list[str]:
        if not skill.contract:
            return []
        answered = request.meta.get("config_answers", {})
        if self.store is not None:
            missing = unresolved_variables(self.store, skill, self.ctx.config, answered)
        else:
            merged = {**self.ctx.config}
            merged.update(skill.config or {})
            merged.update(answered)
            missing = [name for name in skill.contract if name not in merged]
        # A variable the human chose to skip (empty answer) stays unresolved but
        # must never be re-asked: the run proceeds with it left unset.
        skipped = request.meta.get("config_skipped", ())
        return [name for name in missing if name not in skipped]

    def _variable_question(self, skill: Skill, variable: str) -> str:
        desc = (skill.contract or {}).get(variable, "")
        return (
            f"To run {skill.category}.{skill.name} I need `{variable}`. {desc}\n"
            "Reply with the value, or leave it empty to skip."
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

        An empty answer (or the literal "skip") leaves the variable unset and
        marks it skipped for this run, so the run continues instead of re-asking.
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
            request.meta.setdefault("config_skipped", set()).add(variable)
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
                pre_act=True,
            )
        ctx = self._skill_context(skill, request)
        try:
            action = skill.act(ctx, request)
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
            )
        return self._finish(skill, request, ctx, action)

    def resume(
        self,
        skill: Skill,
        request: Request,
        pre_act: bool = False,
    ) -> RunResult:
        """Re-invoke the run with the human's answer (on request.user_input).

        For a pre-act pause (contract collection) the answer is recorded and the
        whole resolution + act path is re-run — more variables may be missing,
        pausing again until satisfied. For an act-driven pause, `act` is
        re-invoked; because the body is single-phase there is no frozen
        prediction to replay, so `act` runs again from the top.
        """
        if pre_act:
            self._record_answer(skill, request)
            return self.run(skill, request)
        baseline = request.text
        ctx = self._skill_context(skill, request)
        try:
            action = skill.act(ctx, request)
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
            )
        return self._finish(skill, request, ctx, action)

    def _assess(
        self, skill: Skill, request: Request, ctx: ActionContext, action
    ) -> tuple[bool, str | None]:
        """SemIf assessment: did the run succeed, and should it run again?

        `assess:outcome` decides success (`P(success) >= tau`); a failure is
        then offered to `assess:requeue`, which picks complete/retry. On retry
        the original request text is re-dispatched (the scheduler bounds it by
        `max_reentries`). Both are real logged decision rows. The state carries
        the resolved inputs so a wrong resolution is visible to the decision.
        The engine is always real: EngineUnavailable propagates to the
        scheduler, which marks the app fatal.
        """
        label = f"{skill.category}.{skill.name}"
        resolved = _resolved_inputs(skill, ctx.config)
        outcome = DecisionRequest(
            state=(
                f"skill: {label}\n"
                f"goal: {request.text}\n"
                f"resolved inputs:\n{resolved}\n"
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
                f"resolved inputs:\n{resolved}\n"
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
        self, skill: Skill, request: Request, ctx: ActionContext, action
    ) -> RunResult:
        observed = action.new_state
        if skill.category in CANNED_CATEGORIES:
            summary = deterministic_summary(
                skill.category, skill.name, True, action.action_log, observed
            )
            return RunResult(
                skill=skill.name,
                success=True,
                summary=summary,
                action_log=action.action_log,
                new_state=observed,
                assessment_summary=summary,
            )
        success, updated_request = self._assess(skill, request, ctx, action)

        decisions = getattr(action, "decisions", [])
        for decision, result in decisions:
            self.log.append(
                decision,
                result,
                extra={
                    "phase": "act",
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
        )