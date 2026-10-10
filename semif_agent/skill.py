"""The skill execution loop: observe -> act -> observe -> assess.

The body is a single `act(ctx, request)` phase. It reads resolved values from
`ctx.config` (the runner resolves the contract from the tiered config and asks
the human for anything unresolved), performs the real action, and returns an
`ActionResult`. Any SemIf sub-decisions it made ride back on
`ActionResult.decisions` and are logged by the runner with the run outcome.

Assessment is a SemIf decision, not generation: `assess:outcome` decides
whether there is any additional work to do — `done` (request satisfied),
`continue` (the step worked but more work is needed; the scheduler routes the
next step), or `failed` (the step errored → the repair path). The run summary is
deterministic — same inputs, same string — built from the category, skill,
outcome flag, and action log. Every SemIf decision made during a run is logged as
a training row; the assessment outcome is kept on the row so the dream pass can
weigh failed runs.

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
    DETERMINISTIC_CATEGORIES,
    ActionContext,
    Skill,
    SkillStore,
    bound_output,
    render_ledger,
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
    #: The three-way outcome for a chained run: "done" (request satisfied),
    #: "continue" (the step worked but more work is needed → the scheduler routes
    #: the next step), or "failed" (the step errored → the repair path). None
    #: only for a pause for input (needs_input/error), which never reaches the
    #: outcome branch.
    outcome: str | None = None
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
        done_tau: float = 0.5,
        fail_tau: float = 0.5,
        result_chars: int = 300,
    ):
        self.ctx = ctx
        self.log = log
        self.store = store
        self.tau = tau
        #: Thresholds for the three-way `assess:outcome` decision. Precedence is
        #: failed > done > continue: a step is only declared "continue" when the
        #: model is genuinely unsure, so a confident success/failure is honored
        #: and the chain is not entered needlessly.
        self.done_tau = done_tau
        self.fail_tau = fail_tau
        #: The shared speakable budget: every skill's action_log/new_state is
        #: truncated to this many characters (the same bound the voice front end
        #: and the humanizer use), so a result fits one message / ~20s of speech
        #: and the run ledger stays bounded.
        self.result_chars = result_chars

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
        return ActionContext(
            engine=self.ctx.engine,
            config=merged,
            timers=self.ctx.timers,
            admin=self.ctx.admin,
        )

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
    ) -> str:
        """SemIf assessment: is there any *additional work* to be done?

        A single three-way `assess:outcome` decision over the step's own action
        plus the run ledger so far:

        - ``done``     — the request is satisfied (a definitive "nothing found"
          counts as done);
        - ``continue`` — the step worked, but the request is not yet satisfied;
          the scheduler routes the next step (bounded by ``max_reentries`` and
          the monotonic-progress guard);
        - ``failed``   — the step errored / did not work → the repair path.

        The verdict uses thresholds, not argmax, with precedence
        ``failed > done > continue`` (``fail_tau`` then ``done_tau``), so a
        confident success/failure is honored and ``continue`` — the over-trigger
        — only wins when the model is genuinely unsure. Real logged decision row
        (phase `assess:outcome`). The state carries the skill's declared action,
        the goal, the resolved inputs, the (bounded) action log, and the steps
        already taken. The engine is always real: EngineUnavailable propagates to
        the scheduler, which marks the app fatal.
        """
        label = f"{skill.category}.{skill.name}"
        resolved = _resolved_inputs(skill, ctx.config)
        observed = bound_output(action.action_log, self.result_chars)
        if not observed:
            observed = bound_output(action.new_state, self.result_chars)
        state = (
            f"skill: {label}\n"
            f"skill action: {skill.description}\n"
            f"goal: {request.text}\n"
            f"resolved inputs:\n{resolved}\n"
            f"action log:\n{observed}"
        )
        ledger = render_ledger(request.run_ledger)
        if ledger:
            state += f"\nsteps already taken:\n{ledger}"
        decision = DecisionRequest(
            state=state,
            question="Is there any additional work to be done to satisfy the request?",
            options=[
                Option(
                    "done",
                    "No — the request is satisfied. A definitive 'nothing found' "
                    "or 'nothing to do' result (empty inbox, no unread messages, "
                    "no matching event) counts as done.",
                ),
                Option(
                    "continue",
                    "Yes — this step worked, but the request is not yet satisfied; "
                    "another skill should run next toward the same request.",
                ),
                Option(
                    "failed",
                    "The step did not work: the result is an error, a refusal, or "
                    "an unresolved action.",
                ),
            ],
        )
        result = self.ctx.engine.call(decision)
        self.log.append(
            decision,
            result,
            extra={"phase": "assess:outcome", "run_id": request.id, "skill": label},
        )
        probs = result.probs
        if probs.get("failed", 0.0) >= self.fail_tau:
            return "failed"
        if probs.get("done", 0.0) >= self.done_tau:
            return "done"
        return "continue"

    def _finish(
        self, skill: Skill, request: Request, ctx: ActionContext, action
    ) -> RunResult:
        action_log = bound_output(action.action_log, self.result_chars)
        observed = bound_output(action.new_state, self.result_chars)
        if skill.category in CANNED_CATEGORIES or skill.category in DETERMINISTIC_CATEGORIES:
            summary = deterministic_summary(
                skill.category, skill.name, True, action_log, observed
            )
            return RunResult(
                skill=skill.name,
                success=True,
                summary=summary,
                action_log=action_log,
                new_state=observed,
                assessment_summary=summary,
                outcome="done",
            )
        verdict = self._assess(skill, request, ctx, action)
        success = verdict != "failed"

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
            skill.category, skill.name, success, action_log, observed
        )
        return RunResult(
            skill=skill.name,
            success=success,
            summary=summary,
            action_log=action_log,
            new_state=observed,
            outcome=verdict,
            decisions_logged=len(decisions),
            assessment_summary=summary,
        )