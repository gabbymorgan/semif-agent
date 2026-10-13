"""Hermetic mechanics test for time.date.

Run from this folder: `python skill.test.py`. No external network, no config.
"""

import sys

from semif_agent.decisions import Request
from semif_agent.skills import ActionContext

import skill


def main():
    ctx = ActionContext(engine=None, config={})
    action = skill.act(ctx, Request("what is today's date?"))
    assert action.new_state, "act must report the date"
    assert "Today is" in action.action_log, action.action_log
    assert action.needs_input is None
    print(action.action_log)
    print("ok")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
