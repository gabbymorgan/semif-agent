#!/usr/bin/env python3
"""Tune the two-stage navigation guards.

Navigation is two-stage at both levels, and the guards are the create doors.
Before either stage, the **actionability guard** (`confirm_non_action`) decides
whether the input is a request at all: a non-request goes straight to the closed
`response` tree and never reaches the category softmax, so a category-level
measurement of a non-request is reported as `response` with no guard
probability.

- **Category**: the softmax offers every existing (real) category WITH its
  description plus `create_category`; `confirm_category_fit` confirms the
  winner's scope. `navigation.category_tau` is that guard's P(covers) floor.
- **Leaf**: the softmax offers only existing skills; `confirm_skill_fit`
  compares the winner's action to the request. `navigation.intent_tau` is that
  guard's P(same) floor.

This replays a hand-labeled eval set through the **real** decision engine and
the **real** skill tree, measuring each guard's probability per request so the
floors can be placed between the true matches and the true mismatches. Run it on
a host with the pinned engine:

    .runtime/venv/bin/python scripts/tune_navigation.py
    .runtime/venv/bin/python scripts/tune_navigation.py --level leaf --tau 0.6

The tree is whatever the host currently has loaded (committed seeds + the runtime
`data/skills/`), so an expectation naming a skill that is not present is
reported as N/A rather than counted wrong. Labels are hand-written ground truth;
self-labeled decision rows are not evidence. Nothing is written to the runtime
logs.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from semif_agent.cli import build_scheduler, load_config
from semif_agent.decisions import Request
from semif_agent.log import DecisionLog
from semif_agent.skills import (
    CreateCategory,
    category_descriptions,
    confirm_category_fit,
    confirm_skill_fit,
    navigate,
)
from semif_agent.trace import TraceLog

# (request, expected) at the LEAF level: "category.skill" or "create".
LEAF_EVAL: list[tuple[str, str]] = [
    ("what is my next message on simplex?", "simplex.next_message"),
    ("check my next message on simplex", "simplex.next_message"),
    ("read the next simplex message", "simplex.next_message"),
    ("show me my simplex messages", "simplex.next_message"),
    ('send a message on simplex to pepper: "hey!"', "simplex.send_message"),
    ("send a simplex message to pepper saying hi", "simplex.send_message"),
    ("what is my simplex connection link?", "simplex.connect_link"),
    ("give me my simplex contact link", "simplex.connect_link"),
    ("tell me the next event in my nextcloud calendar", "nextcloud.next_event"),
    ("what is 15% of 80?", "calculator.calculate"),
    ("square root of 144", "calculator.calculate"),
    ("translate my last simplex message to French", "create"),
    ("delete my next simplex message", "create"),
    ("book a flight to japan", "create"),
]

# (request, expected) at the CATEGORY level: a category name or "create".
CATEGORY_EVAL: list[tuple[str, str]] = [
    ("what is my next message on simplex?", "simplex"),
    ("send a simplex message to pepper saying hi", "simplex"),
    ("what is my simplex connection link?", "simplex"),
    ("tell me the next event in my nextcloud calendar", "nextcloud"),
    ("what is 15% of 80?", "calculator"),
    ("square root of 144", "calculator"),
    ("hello there", "response"),
    ("thanks!", "response"),
    ("what", "response"),
    ("book a flight to japan", "create"),
    ("what is the weather in paris", "create"),
    ("order me a pizza", "create"),
    ("tell me if my package was delivered", "create"),
]

SWEEP = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def _present(tree: dict, qualified: str) -> bool:
    if qualified == "create":
        return True
    if "." not in qualified:
        return qualified in tree
    category, _, name = qualified.partition(".")
    return any(s.name == name for s in tree.get(category, []))


def measure(
    scheduler, log, trace, request: Request, level: str, tau: float
) -> dict:
    """Run one navigation pass and return the guard probability for `level`."""
    before = len(trace.read())
    nav = navigate(
        scheduler.engine, log, trace, request, scheduler.tree, category_tau=tau
    )
    events = trace.read()[before:]

    if level == "category":
        if isinstance(nav, CreateCategory):
            return {"picked": "create", "p": None}
        # navigate already ran the guard internally; read its event.
        guard = next((e for e in events if e["kind"] == "category_scope"), None)
        return {"picked": nav.category, "p": guard["probs"]["covers"] if guard else None}

    # leaf: navigate returns a Skill, CreateSkill, or CreateCategory.
    if not hasattr(nav, "name"):
        return {"picked": "create", "p": None}
    confirm_skill_fit(scheduler.engine, log, trace, request, nav, tau=tau)
    events = trace.read()[before:]
    guard = next((e for e in events if e["kind"] == "intent_guard"), None)
    return {"picked": f"{nav.category}.{nav.name}", "p": guard["probs"]["same"] if guard else None}


def verdict(measured: dict, tau: float) -> str:
    if measured["p"] is None:
        return measured["picked"]
    if measured["p"] >= tau:
        return measured["picked"]
    return "create"


def run_level(scheduler, log, trace, level: str, tau: float, eval_set, sweep) -> None:
    print(f"### {level.upper()} level (real engine; ~1s/decision)\n")
    results = []
    for text, expected in eval_set:
        present = _present(scheduler.tree, expected)
        measured = measure(scheduler, log, trace, Request(text), level, tau)
        results.append((text, expected, present, measured))

    print(f"{'request':52s} {'picked':28s} {'p':>8s}  {'expected':24s}")
    print("-" * 116)
    for text, expected, present, m in results:
        picked = m["picked"] or "(create)"
        p = f"{m['p']:.3f}" if m["p"] is not None else "   -"
        exp = expected + ("" if present else "  [N/A]")
        print(f"{text[:52]:52s} {picked:28s} {p:>8s}  {exp:24s}")

    print(f"\n{'tau':>6s} {'correct':>8s} {'false_create':>13s} {'false_reuse':>12s} {'na':>4s}")
    print("-" * 46)
    for t in sweep:
        correct = fc = fr = na = 0
        for _text, expected, present, m in results:
            if not present:
                na += 1
                continue
            got = verdict(m, t)
            if got == expected:
                correct += 1
            elif expected == "create":
                fr += 1
            else:
                fc += 1
        print(f"{t:6.2f} {correct:8d} {fc:13d} {fr:12d} {na:4d}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "config.json"))
    parser.add_argument("--tau", type=float, default=None, help="report a single tau only")
    parser.add_argument(
        "--level", choices=["leaf", "category", "both"], default="both"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    scheduler, _ = build_scheduler(config)
    scratch = Path(tempfile.mkdtemp(prefix="tune_navigation."))
    log = DecisionLog(str(scratch / "decisions.jsonl"))
    trace = TraceLog(str(scratch / "runs.jsonl"))
    sweep = [args.tau] if args.tau is not None else SWEEP

    if args.level in ("category", "both"):
        run_level(
            scheduler, log, trace, "category", args.tau or 0.5, CATEGORY_EVAL, sweep
        )
    if args.level in ("leaf", "both"):
        run_level(scheduler, log, trace, "leaf", args.tau or 0.6, LEAF_EVAL, sweep)

    print(
        "false_create = should have reused something existing but authored a new one\n"
        "false_reuse  = should have authored something new but used an existing one"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
