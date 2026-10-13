"""Hermetic mechanics test for calculator.calculate.

Run from this folder: `python skill.test.py`. No external network and no
configuration — this body is pure local compute. A scripted `FakeEngine` stands
in for SemIf so the operator choice and the operand disambiguation are exercised
deterministically. This proves the body parses numbers (digits and words),
resolves one operator and up to two operands, spells the result out for the ear,
asks for missing numbers and resumes, and fails honestly on an undefined
calculation. It does NOT prove the live decision engine — only a real run does.
"""

import sys

from semif_agent.decisions import DecisionResult, Request
from semif_agent.skills import ActionContext

import skill


class FakeEngine:
    """Picks the queued option id per call (default: the first option)."""

    def __init__(self, picks=None):
        self.picks = list(picks or [])
        self.calls = []

    def call(self, decision):
        self.calls.append(decision)
        option_ids = [option.id for option in decision.options]
        chosen = self.picks.pop(0) if self.picks else option_ids[0]
        if chosen not in option_ids:
            chosen = option_ids[0]
        return DecisionResult(
            request=decision,
            option_ids=option_ids,
            probabilities=[1.0 if i == chosen else 0.0 for i in option_ids],
        )


def run(text, picks=None):
    engine = FakeEngine(picks)
    ctx = ActionContext(engine=engine, config={})
    request = Request(text)
    action = skill.act(ctx, request)
    return action, engine, request


def check(text, picks, expected):
    action, engine, _ = run(text, picks)
    assert action.action_log == expected, (text, action.action_log, expected)
    assert action.needs_input is None, (text, action.needs_input)
    return action, engine


def main():
    # The operator is a SemIf choice; the answer is spoken in words.
    check("what is 3 plus 4?", ["add"], "Three plus four is seven.")
    check("9 minus 4", ["subtract"], "Nine minus four is five.")
    check("6 times 7", ["multiply"], "Six times seven is forty-two.")
    check("10 divided by 4", ["divide"], "Ten divided by four is two point five.")
    check("2 to the power of 10", ["exponent"],
          "Two to the power of ten is one thousand twenty-four.")

    # Squares, cubes, roots, fractions, percentages.
    check("7 squared", ["square"], "Seven squared is forty-nine.")
    check("3 cubed", ["cube"], "Three cubed is twenty-seven.")
    check("square root of 144", ["square_root"],
          "The square root of one hundred forty-four is twelve.")
    check("cube root of 27", ["cube_root"],
          "The cube root of twenty-seven is three.")
    check("3/4", ["fraction"],
          "The fraction three over four is zero point seven five.")
    check("20% of 50", ["percent"], "Twenty percent of fifty is ten.")

    # An inexact result is announced as approximate.
    check("square root of 2", ["square_root"],
          "The square root of two is approximately one point four one four two.")

    # Number words are read as operands too.
    check("five plus five", ["add"], "Five plus five is ten.")
    check("twenty one times two", ["multiply"],
          "Twenty-one times two is forty-two.")

    # A sign at the start of the request is part of the number.
    check("-5 plus 3", ["add"], "Negative five plus three is negative two.")

    # More numbers than the operation consumes: SemIf picks the operands.
    _action, engine = check("add 1 2 3", ["add", "0", "0"], "One plus two is three.")
    assert len(engine.calls) == 3, "operator + two operand choices expected"

    # Undefined calculations are reported honestly, not fabricated.
    action, _ = check("5 divided by 0", ["divide"], "I can't divide by zero.")
    assert action.needs_input is None

    # Too few numbers: ask once, then finish on the resume.
    engine = FakeEngine(["add"])
    ctx = ActionContext(engine=engine, config={})
    request = Request("what is 3 plus?")
    first = skill.act(ctx, request)
    assert first.needs_input, "a missing operand should be asked for"
    assert first.action_log == "waiting for the numbers to calculate"
    request.user_input = "4"
    second = skill.act(ctx, request)
    assert second.action_log == "Three plus four is seven.", second.action_log
    assert second.needs_input is None

    print(second.action_log)
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
