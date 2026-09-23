"""The skill tree, registry, and SemIf-driven navigation.

A skill is a leaf reached by a chain of SemIf choices (category -> skill).
At every level a "create_skill" branch exists; opencode is the authoring tool
there (deferred to v2, stubbed as CreateSkill).

Only the real skills live here; navigation uses the real decision engine.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .decisions import DecisionRequest, Option, Request
from .engine import SemIfEngine
from .log import DecisionLog


@dataclass
class ActionResult:
    action_log: str
    new_state: str


@dataclass
class Prediction:
    """The predict phase: a forecast plus any SemIf decisions it made."""

    text: str
    decisions: list[tuple[DecisionRequest, object]] = field(default_factory=list)


@dataclass
class ActionContext:
    engine: SemIfEngine
    config: dict


@dataclass
class Skill:
    name: str
    category: str
    description: str
    cost_budget: float = 1.0
    predict: Callable[[ActionContext, Request], Prediction] = field(
        default=lambda ctx, req: Prediction(text="")
    )
    act: Callable[[ActionContext, Request, Prediction], ActionResult] = field(
        default=lambda ctx, req, pred: ActionResult("", "")
    )


@dataclass
class CreateSkill:
    """Sentinel for the 'create a missing skill' branch at a tree level."""

    category: str | None = None


def compose_state(request: Request, current: str | None = None) -> str:
    parts = [request.text]
    if current:
        parts.append(f"[current process: {current}]")
    return " ".join(parts)


def _contacts(ctx: ActionContext) -> list[dict]:
    path = Path(ctx.config.get("contacts", "data/contacts.json"))
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def _email_predict(ctx: ActionContext, request: Request) -> Prediction:
    contacts = _contacts(ctx)
    if not contacts:
        return Prediction(text="no contacts available", decisions=[])
    decision = DecisionRequest(
        state=compose_state(request),
        question="Which contact is the intended recipient?",
        options=[Option(c["name"], c.get("description", "")) for c in contacts]
        + [Option("none", "None of the listed contacts.")],
    )
    result = ctx.engine.call(decision)
    return Prediction(text=f"recipient is {result.selected}", decisions=[(decision, result)])


def _email_compose(ctx: ActionContext, request: Request, prediction: Prediction) -> ActionResult:
    recipient = prediction.text.removeprefix("recipient is ")
    if recipient == "no contacts available" or recipient == "none":
        return ActionResult(
            action_log="email.compose aborted: recipient not resolved.",
            new_state=request.text,
        )
    drafts = Path(ctx.config.get("drafts", "data/drafts"))
    drafts.mkdir(parents=True, exist_ok=True)
    target = drafts / f"{request.id}.txt"
    target.write_text(f"To: {recipient}\nBody: {request.text}\n")
    return ActionResult(
        action_log=f"email.compose: wrote draft {target} for {recipient!r}.",
        new_state=f"Draft written to {target.name} for {recipient}.",
    )


def _response_reject(ctx: ActionContext, request: Request, prediction: Prediction) -> ActionResult:
    message = f"Rejected: I cannot act on this while busy ({request.text})."
    return ActionResult(action_log=f"response.reject: {message}", new_state=message)


def _tracking_check(ctx: ActionContext, request: Request, prediction: Prediction) -> ActionResult:
    path = Path(ctx.config.get("packages", "data/packages.json"))
    if not path.is_file():
        return ActionResult(
            action_log="tracking.check aborted: no packages file.",
            new_state=request.text,
        )
    packages = json.loads(path.read_text())
    lines = [f"{p.get('id')}: {p.get('status')}" for p in packages]
    report = "Tracking statuses:\n" + "\n".join(lines)
    return ActionResult(action_log="tracking.check: " + report, new_state=report)


def build_skills(config: dict) -> list[Skill]:
    skills = config.get("skills", {})
    return [
        Skill(
            name="email.compose",
            category="email",
            description="Compose and dispatch an email.",
            predict=_email_predict,
            act=_email_compose,
            cost_budget=float(skills.get("email", {}).get("cost_budget", 1.0)),
        ),
        Skill(
            name="response.reject",
            category="response",
            description="Politely reject a request because the agent is busy.",
            act=_response_reject,
        ),
        Skill(
            name="tracking.check",
            category="tracking",
            description="Check the delivery status of a package.",
            act=_tracking_check,
        ),
    ]


def build_tree(skills: list[Skill]) -> dict[str, list[Skill]]:
    tree: dict[str, list[Skill]] = {}
    for skill in skills:
        tree.setdefault(skill.category, []).append(skill)
    for category in tree:
        tree[category].sort(key=lambda s: s.name)
    return tree


def navigate(
    engine: SemIfEngine,
    log: DecisionLog,
    request: Request,
    tree: dict[str, list[Skill]],
) -> Skill | CreateSkill:
    """Descend the tree one SemIf choice per level. Every choice is logged."""
    categories = sorted(tree.keys())
    create = Option("create_skill", "Create a new skill for this.")
    top = DecisionRequest(
        state=compose_state(request),
        question="Which top-level category handles this request?",
        options=[Option(c, c) for c in categories] + [create],
    )
    top_result = engine.call(top)
    log.append(top, top_result, extra={"phase": "navigate:category", "run_id": request.id})
    category = top_result.selected
    if category == "create_skill":
        return CreateSkill(category=None)
    skills = tree[category]
    leaf = DecisionRequest(
        state=compose_state(request, current=category),
        question=f"Within {category}, which skill?",
        options=[Option(s.name, s.description) for s in skills] + [create],
    )
    leaf_result = engine.call(leaf)
    log.append(leaf, leaf_result, extra={"phase": "navigate:leaf", "run_id": request.id})
    pick = leaf_result.selected
    if pick == "create_skill":
        return CreateSkill(category=category)
    return next(s for s in skills if s.name == pick)


def tree_summary(tree: dict[str, list[Skill]]) -> str:
    lines = []
    for category in sorted(tree):
        names = ", ".join(s.name for s in tree[category])
        lines.append(f"  {category}: {names}")
    return "\n".join(lines)
