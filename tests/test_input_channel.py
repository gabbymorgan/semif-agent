"""Pure-stdlib tests for the runtime user-input channel.

A skill can pause its run by returning ActionResult(..., needs_input="<q>");
the scheduler keeps the run pending, and `answer` resumes it by re-invoking
only `act` with the human's answer on request.user_input. Assessment is now a
SemIf decision, so these mechanics tests drive the loop with a deterministic
scripted engine (see conftest) rather than loading the real GGUF.
"""

from semif_agent.decisions import DecisionRequest, DecisionResult, Option, Request
from semif_agent.llm import LLMClient
from semif_agent.log import DecisionLog
from semif_agent.scheduler import Scheduler
from semif_agent.skills import ActionResult, Skill
from semif_agent.trace import TraceLog

from tests.conftest import ScriptedEngine


def build_scheduler(tmp_path, choices=None):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    engine = ScriptedEngine(choices=choices, default="success")
    llm = LLMClient(base_url="http://localhost:1/v1", model="test")
    return Scheduler(
        engine=engine,
        llm=llm,
        log=log,
        config={"skills": {}, "skill_seeds": str(tmp_path / "seeds")},
        trace=trace,
    )


def need_input_skill(seen):
    def act(ctx, request):
        if request.user_input:
            seen.append(request.user_input)
            return ActionResult(
                action_log=f"got {request.user_input}", new_state=f"done {request.user_input}"
            )
        return ActionResult(
            action_log="need a tracking number",
            new_state=request.text,
            needs_input="What's the tracking number?",
        )

    return Skill(
        name="track.manual",
        category="tracking",
        description="Resolve a tracking number with the human.",
        act=act,
    )


def test_skill_pause_and_answer(tmp_path):
    scheduler = build_scheduler(tmp_path)
    seen = []
    request = Request("track my package manually")

    result = scheduler._run_skill(need_input_skill(seen), request)

    assert result.kind == "needs_input"
    assert result.needs_input == "What's the tracking number?"
    assert scheduler.pending is not None
    assert scheduler.pending.question == "What's the tracking number?"
    assert scheduler.current is not None
    assert scheduler.current.skill == "track.manual"
    assert scheduler.log.read() == [], "sub-decisions must be deferred until completion"

    status, detail = scheduler.answer("AB123")
    assert status == "ran"
    assert seen == ["AB123"]
    assert scheduler.pending is None
    assert scheduler.current is None

    kinds = [e["kind"] for e in scheduler.trace.read()]
    assert "needs_input" in kinds
    assert "answered" in kinds
    assert "assessed" in kinds
    assert "ran" in kinds


def test_answer_without_pending_is_error(tmp_path):
    scheduler = build_scheduler(tmp_path)
    status, detail = scheduler.answer("hello")
    assert status == "error"
    assert "waiting for input" in detail


def test_act_decisions_logged_on_completion(tmp_path):
    decision = DecisionRequest(
        state="s", question="which?", options=[Option("a", "A."), Option("b", "B.")]
    )
    result = DecisionResult(
        request=decision, option_ids=["a", "b"], probabilities=[0.3, 0.7]
    )

    def act(ctx, request):
        if request.user_input:
            return ActionResult(
                action_log="ok", new_state="done", decisions=[(decision, result)]
            )
        return ActionResult(
            action_log="ask", new_state=request.text, needs_input="confirm?"
        )

    skill = Skill(name="t.x", category="t", description="", act=act)
    scheduler = build_scheduler(
        tmp_path,
        choices={"achieve the user's goal": "failure", "complete, or should it run again": "complete"},
    )

    scheduler._run_skill(skill, Request("x"))
    assert scheduler.log.read() == []

    scheduler.answer("yes")
    rows = scheduler.log.read()
    act_rows = [r for r in rows if r.get("extra", {}).get("phase") == "act"]
    assert len(act_rows) == 1
    assert act_rows[0]["extra"]["run_ok"] is False
    phases = [r.get("extra", {}).get("phase") for r in rows]
    assert "assess:outcome" in phases and "assess:requeue" in phases


def test_busy_abandons_pending(tmp_path):
    scheduler = build_scheduler(tmp_path)
    request = Request("track")
    scheduler._run_skill(need_input_skill([]), request)
    assert scheduler.pending is not None

    scheduler.busy("driving on the freeway", skill="driving")
    assert scheduler.pending is None
    assert any(e["kind"] == "pending_abandoned" for e in scheduler.trace.read())


def test_idle_abandons_pending(tmp_path):
    scheduler = build_scheduler(tmp_path)
    scheduler._run_skill(need_input_skill([]), Request("track"))
    assert scheduler.pending is not None

    scheduler.idle()
    assert scheduler.pending is None
    assert scheduler.current is None


def test_resume_can_ask_again(tmp_path):
    seen = []

    def act(ctx, request):
        if request.user_input == "AB123":
            return ActionResult(action_log="done", new_state="resolved")
        if request.user_input:
            seen.append(request.user_input)
            return ActionResult(
                action_log="wrong format",
                new_state=request.text,
                needs_input="That wasn't a valid tracking number. Try again?",
            )
        return ActionResult(
            action_log="ask", new_state=request.text, needs_input="Tracking number?"
        )

    skill = Skill(name="t.x", category="t", description="", act=act)
    scheduler = build_scheduler(tmp_path)

    scheduler._run_skill(skill, Request("track"))
    status, detail = scheduler.answer("XYZ")
    assert status == "needs_input"
    assert seen == ["XYZ"]
    assert scheduler.pending is not None

    status, detail = scheduler.answer("AB123")
    assert status == "ran"
    assert scheduler.pending is None
    assert scheduler.current is None