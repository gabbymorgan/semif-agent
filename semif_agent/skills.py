"""The skill tree, registry, and SemIf-driven navigation.

A skill is a leaf reached by a chain of SemIf choices (category -> skill).
The category level carries a "create_category" branch and the leaf level a
"create_skill" branch. Both are live: the decision model is driven in normal
generation mode to propose a title + description — a broad new category or a
specific new skill leaf — which is persisted to a category registry and merged
into the running tree as a stub.

Only the real skills live here; navigation uses the real decision engine.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .decisions import DecisionRequest, Option, Request
from .engine import SemIfEngine
from .llm import LLMClient
from .log import DecisionLog
from .trace import TraceLog


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
    """Suggestion that the current category needs a new skill.

    Handled live, like CreateCategory: the decision model authors the new skill
    stub, which is persisted and merged into the tree. `category` names the
    category that needs the new skill.
    """

    category: str


@dataclass
class CreateCategory:
    """Suggestion that the request needs a brand-new top-level category.

    Unlike CreateSkill this is handled live: the decision model is used in
    normal generation mode to author the category stub.
    """


@dataclass
class CategoryDraft:
    """An authored category stub: a broad bucket for future skills."""

    name: str
    description: str


@dataclass
class SkillDraft:
    """An authored skill leaf stub: one specific action within a category."""

    name: str
    description: str


class CategoryRegistry:
    """Persisted category stubs, one file on disk.

    Format: {name: {"description": str, "skills": [{"name": str, "description":
    str}, ...]}}. The skills list is filled by create_skill; each entry becomes
    a stub leaf merged into the running tree.
    """

    def __init__(self, path: str = "data/categories.json"):
        self.path = Path(path)

    def read(self) -> dict[str, dict]:
        if not self.path.is_file():
            return {}
        return json.loads(self.path.read_text())

    def register(self, name: str, description: str) -> None:
        categories = self.read()
        categories[name] = {"description": description, "skills": []}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(categories, indent=2) + "\n")

    def register_skill(self, category: str, name: str, description: str) -> None:
        """Add a skill leaf to a category, creating the category entry if needed."""
        categories = self.read()
        entry = categories.setdefault(category, {"description": "", "skills": []})
        skills = entry.setdefault("skills", [])
        if not any(s.get("name") == name for s in skills):
            skills.append({"name": name, "description": description})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(categories, indent=2) + "\n")


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


def merge_registry(tree: dict[str, list[Skill]], categories: dict[str, dict]) -> None:
    """Fold persisted categories and their skills into a running tree.

    Category stubs become empty buckets; registered skills become stub leaves
    (no-op bodies) so they are navigable and rerunnable immediately.
    """
    for category, data in categories.items():
        tree.setdefault(category, [])
        existing = {s.name for s in tree[category]}
        for skill in data.get("skills", []):
            name = skill.get("name")
            if not name or name in existing:
                continue
            tree[category].append(
                Skill(
                    name=name,
                    category=category,
                    description=skill.get("description", ""),
                )
            )
            existing.add(name)


def navigate(
    engine: SemIfEngine,
    log: DecisionLog,
    trace: TraceLog,
    request: Request,
    tree: dict[str, list[Skill]],
) -> Skill | CreateCategory | CreateSkill:
    """Descend the tree one SemIf choice per level. Every choice is logged.

    The category level offers a "create_category" branch and the leaf level a
    "create_skill" branch; both are handled live by dispatch and log a
    suggestion event to the trace.
    """
    categories = sorted(tree.keys())
    create_category = Option("create_category", "Suggest a new category for this.")
    top = DecisionRequest(
        state=compose_state(request),
        question="Which top-level category handles this request?",
        options=[Option(c, c) for c in categories] + [create_category],
    )
    top_result = engine.call(top)
    log.append(top, top_result, extra={"phase": "navigate:category", "run_id": request.id})
    category = top_result.selected
    if category == "create_category":
        trace.append(
            "create_category",
            request.id,
            state=top.state,
            question=top.question,
            options=[o.id for o in top.options],
            selected=top_result.selected,
            probs=top_result.probs,
        )
        return CreateCategory()
    skills = tree[category]
    create_skill = Option("create_skill", "Suggest creating a new skill.")
    leaf = DecisionRequest(
        state=compose_state(request, current=category),
        question=f"Within {category}, which skill?",
        options=[Option(s.name, s.description) for s in skills] + [create_skill],
    )
    leaf_result = engine.call(leaf)
    log.append(leaf, leaf_result, extra={"phase": "navigate:leaf", "run_id": request.id})
    pick = leaf_result.selected
    if pick == "create_skill":
        trace.append(
            "skill_needed",
            request.id,
            category=category,
            state=leaf.state,
            question=leaf.question,
            options=[o.id for o in leaf.options],
            selected=leaf_result.selected,
            probs=leaf_result.probs,
        )
        return CreateSkill(category=category)
    return next(s for s in skills if s.name == pick)


def tree_summary(tree: dict[str, list[Skill]]) -> str:
    lines = []
    for category in sorted(tree):
        names = ", ".join(s.name for s in tree[category])
        lines.append(f"  {category}: {names}")
    return "\n".join(lines)


def build_category_prompt(request: Request, tree: dict[str, list[Skill]]) -> list[dict]:
    """Chat messages for the decision model used as the category author.

    The category must be a general bucket that many tools could fit under, not
    a single skill. The existing tree is included so the model avoids duplicating
    categories and stays broad enough to be useful.
    """
    system = (
        "You are the skill-tree authoring step of a local agent. A request did "
        "not fit any existing category. Propose one new top-level category of "
        "tools/skills that would encompass this request. It must be broad enough "
        "that many tools could fit under it — a general-purpose bucket, not a "
        "single skill. Reply with JSON only: "
        '{"title": "<short lowercase snake_case id, no spaces>", '
        '"description": "<one to two sentence purpose>"}'
    )
    user = (
        f"Request: {request.text}\n"
        f"Existing categories and their skills:\n{tree_summary(tree)}\n"
        "Proposed new category (JSON only):"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_category_draft(raw: str) -> CategoryDraft:
    """Parse the model's JSON reply into a CategoryDraft."""
    parsed = LLMClient._parse_json(raw)
    title = str(parsed.get("title", "")).strip()
    description = str(parsed.get("description", "")).strip()
    if not title or not description:
        raise ValueError(f"category draft missing title/description: {raw!r}")
    name = re.sub(r"\s+", "_", title.lower())
    if not name.replace("_", "").isalnum():
        raise ValueError(f"category title must be snake_case alnum: {title!r}")
    return CategoryDraft(name=name, description=description)


def generate_category(
    engine: SemIfEngine, request: Request, tree: dict[str, list[Skill]]
) -> CategoryDraft:
    """Author a new category stub with the decision model in generation mode."""
    raw = engine.generate(build_category_prompt(request, tree))
    return parse_category_draft(raw)


def build_skill_prompt(
    request: Request, category: str, tree: dict[str, list[Skill]]
) -> list[dict]:
    """Chat messages for the decision model used as the skill author.

    The skill must be one specific, single-purpose action that fits inside the
    given category — not a broad bucket. Existing skills in the category are
    included so the model avoids duplicating them.
    """
    system = (
        "You are the skill-tree authoring step of a local agent. A request inside "
        f"the '{category}' category did not fit any existing skill. Propose ONE "
        "new skill for this category: a specific, single-purpose action the agent "
        "can take. Reply with JSON only: "
        '{"title": "<short lowercase snake_case id, no spaces>", '
        '"description": "<one to two sentence purpose>"}'
    )
    existing = ", ".join(s.name for s in tree.get(category, [])) or "(none)"
    user = (
        f"Request: {request.text}\n"
        f"Category: {category}\n"
        f"Existing skills in this category: {existing}\n"
        "Proposed new skill (JSON only):"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_skill_draft(raw: str) -> SkillDraft:
    """Parse the model's JSON reply into a SkillDraft."""
    parsed = LLMClient._parse_json(raw)
    title = str(parsed.get("title", "")).strip()
    description = str(parsed.get("description", "")).strip()
    if not title or not description:
        raise ValueError(f"skill draft missing title/description: {raw!r}")
    name = re.sub(r"\s+", "_", title.lower())
    if not name.replace("_", "").isalnum():
        raise ValueError(f"skill title must be snake_case alnum: {title!r}")
    return SkillDraft(name=name, description=description)


def generate_skill(
    engine: SemIfEngine, request: Request, category: str, tree: dict[str, list[Skill]]
) -> SkillDraft:
    """Author a new skill leaf stub with the decision model in generation mode."""
    raw = engine.generate(build_skill_prompt(request, category, tree))
    return parse_skill_draft(raw)
