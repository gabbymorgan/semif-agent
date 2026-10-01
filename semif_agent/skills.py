"""The skill tree, registry, and SemIf-driven navigation.

A skill is a leaf reached by a chain of SemIf choices (category -> skill).
The category level carries a "create_category" branch and the leaf level a
"create_skill" branch. Both are live: a small OpenAI-compatible model (`llm`,
separate from `codegen`) proposes a title + description — a broad new category
or a specific new skill leaf — which is persisted to a category registry and
merged into the running tree as a stub.

Only the real skills live here; navigation uses the real decision engine.
"""

from __future__ import annotations

import importlib.util
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
    needs_input: str | None = None
    # Any (DecisionRequest, DecisionResult) pairs the body made during `act`;
    # the runner logs them with the run outcome (phase `act`).
    decisions: list[tuple[DecisionRequest, object]] = field(default_factory=list)


@dataclass
class ActionContext:
    engine: SemIfEngine
    config: dict


def _noop_act(ctx: ActionContext, request: Request) -> ActionResult:
    return ActionResult("", "")


@dataclass
class Skill:
    name: str
    category: str
    description: str
    cost_budget: float = 1.0
    act: Callable[[ActionContext, Request], ActionResult] = field(default=_noop_act)
    writing: bool = False
    config: dict = field(default_factory=dict)
    contract: dict = field(default_factory=dict)
    integration: dict = field(default_factory=dict)
    integration_source: str = "unknown"

    def is_noop(self) -> bool:
        """A stub leaf: authored (title + description) but no runnable body yet."""
        return self.act is _noop_act

    @property
    def status(self) -> str:
        """Leaf readiness: `writing` (codegen in flight), `stub` (no body), `ready`."""
        if self.writing:
            return "writing"
        if self.is_noop():
            return "stub"
        return "ready"


@dataclass
class CreateSkill:
    """Suggestion that the current category needs a new skill.

    Handled live, like CreateCategory: the small `llm` provider authors the new
    skill stub, which is persisted and merged into the tree. `category` names the
    category that needs the new skill.
    """

    category: str


@dataclass
class CreateCategory:
    """Suggestion that the request needs a brand-new top-level category.

    Like CreateSkill this is handled live: the small `llm` provider authors the
    category stub.
    """


@dataclass
class CategoryDraft:
    """An authored category stub: a broad bucket for future skills."""

    name: str
    description: str


@dataclass
class SkillDraft:
    """An authored skill leaf stub: one specific action within a category.

    `requirements` maps elicitation questions (asked of the human during
    authoring) to their answers; they are fed to the body-writer so the body
    reflects the refined product goal. `integration` carries the service and
    transport the answers point at; it is fed to the body writer as an advisory
    hint only — the SKILL.md contract, request, and requirements still win.
    """

    name: str
    description: str
    code: str = ""
    requirements: dict[str, str] = field(default_factory=dict)
    integration: dict = field(default_factory=dict)


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

    def register_skill(
        self,
        category: str,
        name: str,
        description: str,
        request_text: str = "",
        requirements: dict[str, str] | None = None,
    ) -> None:
        """Add a skill leaf to a category, creating the category entry if needed.

        `request_text` is the originating request, kept so a later restart of the
        stub can re-drive codegen with the same context. `requirements` are the
        elicitation answers, kept so a restart does not have to ask again.
        """
        categories = self.read()
        entry = categories.setdefault(category, {"description": "", "skills": []})
        skills = entry.setdefault("skills", [])
        entry_row = next((s for s in skills if s.get("name") == name), None)
        if entry_row is None:
            entry_row = {"name": name, "description": description}
            skills.append(entry_row)
        entry_row["request_text"] = request_text
        if requirements:
            entry_row["requirements"] = requirements
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(categories, indent=2) + "\n")


class SkillStore:
    """Persists one skill leaf as a folder of deliverables.

    Layout: <base>/<category>/<name>/{skill.py, skill.test.py, contract.json,
    config.json}. skill.py is the runnable body; the rest are produced by the
    contract/test generation steps and read back so skills stay runnable and
    configurable across restarts. The old single-file layout
    (<base>/<category>/<name>.py) is NOT read — this is a clean switch.
    """

    def __init__(self, path: str = "data/skills"):
        self.path = Path(path)

    def dir(self, category: str, name: str) -> Path:
        return self.path / category / name

    def write_body(self, category: str, name: str, code: str) -> Path:
        directory = self.dir(category, name)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "skill.py"
        target.write_text(code.rstrip() + "\n")
        return target

    def write_contract(self, category: str, name: str, contract: dict) -> Path:
        return self._write_json(category, name, "contract.json", contract)

    def write_config(self, category: str, name: str, config: dict) -> Path:
        return self._write_json(category, name, "config.json", config)

    def write_test(self, category: str, name: str, code: str) -> Path:
        directory = self.dir(category, name)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "skill.test.py"
        target.write_text(code.rstrip() + "\n")
        return target

    def _write_json(self, category: str, name: str, filename: str, value) -> Path:
        directory = self.dir(category, name)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / filename
        target.write_text(json.dumps(value, indent=2) + "\n")
        return target

    def read_contract(self, category: str, name: str) -> dict:
        return self._read_json(category, name, "contract.json")

    def read_config(self, category: str, name: str) -> dict:
        return self._read_json(category, name, "config.json")

    def read_manifest(self, category: str, name: str) -> dict:
        return self._read_json(category, name, "manifest.json")

    def _read_json(self, category: str, name: str, filename: str):
        path = self.dir(category, name) / filename
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text())
        except ValueError:
            return {}

    def read_category_config(self, category: str) -> dict:
        """Category-scoped config: <base>/<category>/config.json."""
        path = self.path / category / "config.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text())
        except ValueError:
            return {}

    def write_category_config(self, category: str, config: dict) -> Path:
        directory = self.path / category
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "config.json"
        target.write_text(json.dumps(config, indent=2) + "\n")
        return target

    def list_skills(self) -> list[tuple[str, str]]:
        if not self.path.is_dir():
            return []
        skills = []
        for category_dir in sorted(self.path.iterdir()):
            if not category_dir.is_dir():
                continue
            for entry in sorted(category_dir.iterdir()):
                if entry.is_dir() and (entry / "skill.py").is_file():
                    skills.append((category_dir.name, entry.name))
        return skills


def load_skill_module(category: str, name: str, base: str = "data/skills"):
    """Import a persisted skill body and return its module."""
    path = Path(base) / category / name / "skill.py"
    module_name = f"_skill_{category}_{name}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load skill module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_integration(code: str) -> tuple[dict, str]:
    """Integration declaration (or inference) for a body, lazily.

    Lives in codegen.py; imported here so skills.py stays free of codegen at
    module import time (codegen imports skills).
    """
    from .codegen import extract_integration

    return extract_integration(code)


def _extract_contract(code: str) -> dict:
    """The body's flat `CONTRACT` declaration, or {} when absent/unparseable.

    The module constant is the source of truth; `contract.json` is a derived
    mirror. Lives in codegen.py; imported lazily so skills.py stays free of
    codegen at module import time.
    """
    from .codegen import parse_contract

    try:
        return parse_contract(code)
    except ValueError:
        return {}


def materialize_skill(
    draft: SkillDraft, category: str, store: SkillStore
) -> Skill:
    """Persist the draft's code body and build a runnable Skill from it.

    The skill's config and contract are read back from the store so the runner
    can resolve its data needs across restarts; its integration declaration is
    extracted from the body so the tree can show what it talks to.
    """
    if not draft.code:
        raise ValueError(f"skill {draft.name} has no code body to materialize")
    store.write_body(category, draft.name, draft.code)
    try:
        module = load_skill_module(category, draft.name, store.path)
    except Exception as exc:
        raise ValueError(f"skill {category}.{draft.name} body failed to import: {exc}") from exc
    if not callable(getattr(module, "act", None)):
        raise ValueError(f"skill {category}.{draft.name} body must define act")
    integration, integration_source = _extract_integration(draft.code)
    contract = _extract_contract(draft.code) or store.read_contract(category, draft.name)
    store.write_contract(category, draft.name, contract)
    return Skill(
        name=draft.name,
        category=category,
        description=draft.description,
        act=module.act,
        config=store.read_config(category, draft.name),
        contract=contract,
        integration=integration,
        integration_source=integration_source,
    )


def merge_skill_store(
    tree: dict[str, list[Skill]], store: SkillStore, registry: dict[str, dict]
) -> int:
    """Upgrade persisted skill folders in the tree to runnable skills.

    A skill folder makes a stub leaf executable; where the registry entry was
    lost (or never written), the category is created and the description falls
    back to the skill name. Returns the number of skills made runnable.
    """
    upgraded = 0
    for category, name in store.list_skills():
        description = ""
        entry = registry.get(category, {})
        for skill in entry.get("skills", []):
            if skill.get("name") == name:
                description = skill.get("description", "")
        try:
            module = load_skill_module(category, name, store.path)
        except Exception:
            continue
        if not callable(getattr(module, "act", None)):
            continue
        code_path = store.dir(category, name) / "skill.py"
        try:
            code = code_path.read_text()
            integration, integration_source = _extract_integration(code)
        except OSError:
            code, integration, integration_source = "", {}, "unknown"
        contract = _extract_contract(code) or store.read_contract(category, name)
        skill = Skill(
            name=name,
            category=category,
            description=description or name,
            act=module.act,
            config=store.read_config(category, name),
            contract=contract,
            integration=integration,
            integration_source=integration_source,
        )
        skills = tree.setdefault(category, [])
        for index, existing in enumerate(skills):
            if existing.name == name:
                skills[index] = skill
                break
        else:
            skills.append(skill)
            skills.sort(key=lambda s: s.name)
        upgraded += 1
    return upgraded


def merge_seed_store(
    tree: dict[str, list[Skill]],
    seed_store: SkillStore,
    config_store: SkillStore | None = None,
) -> int:
    """Load committed starter skills from a read-only seed store into the tree.

    A seed is a real skill in the same folder layout as a generated one
    (`skill.py`, `contract.json`, `skill.test.py`, plus a `manifest.json` with
    its description), so it is inspectable and testable exactly like a codegen
    body. `config_store` is the runtime store (`data/skills`): recorded answers
    — credentials collected at first fire — live there, never in the committed
    seed folder. Returns the number of skills loaded.
    """
    loaded = 0
    for category, name in seed_store.list_skills():
        directory = seed_store.dir(category, name)
        try:
            module = load_skill_module(category, name, seed_store.path)
        except Exception:
            continue
        act = getattr(module, "act", None)
        if not callable(act):
            continue
        try:
            code = (directory / "skill.py").read_text()
        except OSError:
            continue
        integration, integration_source = _extract_integration(code)
        contract = _extract_contract(code) or seed_store.read_contract(category, name)
        manifest = seed_store.read_manifest(category, name)
        description = str(manifest.get("description") or name)
        skill = Skill(
            name=name,
            category=category,
            description=description,
            act=act,
            config=(config_store or seed_store).read_config(category, name),
            contract=contract,
            integration=integration,
            integration_source=integration_source,
        )
        skills = tree.setdefault(category, [])
        for index, existing in enumerate(skills):
            if existing.name == name:
                skills[index] = skill
                break
        else:
            skills.append(skill)
            skills.sort(key=lambda s: s.name)
        loaded += 1
    return loaded


def resolve_skill_config(
    store: SkillStore, skill: Skill, global_config: dict, answered: dict | None = None
) -> dict:
    """Tiered config lookup: global config.json -> category config -> skill
    config, plus any per-fire answers collected this run.

    Higher tiers win. This is the merged view a skill reads from `ctx.config`.
    """
    merged: dict = {}
    merged.update(global_config)
    merged.update(store.read_category_config(skill.category))
    merged.update(skill.config or {})
    for key, value in (answered or {}).items():
        merged[key] = value
    return merged


def unresolved_variables(
    store: SkillStore, skill: Skill, global_config: dict, answered: dict | None = None
) -> list[str]:
    """Contract variables the runner has not satisfied yet (not in the merged
    config and not answered this run). A missing input var is asked of the
    human before act runs."""
    merged = resolve_skill_config(store, skill, global_config, answered)
    return [name for name in (skill.contract or {}) if name not in merged]


def compose_state(request: Request, additional_context: str | None = None) -> str:
    parts = [request.text]
    if current:
        parts.append(f"({additional_context})")
    return " ".join(parts)


def _canned_response(name: str, message: str):
    """An act that returns a fixed line. No transport, no generation, no config."""

    def act(ctx: ActionContext, request: Request) -> ActionResult:
        return ActionResult(action_log=f"{name}: {message}", new_state=message)

    return act


# The `response` category is a closed tree of canned replies: it never authors a
# skill (no create route) and its runs are not assessed (there is no side effect
# to succeed or fail). Every non-task input lands here instead of being gated.
CANNED_CATEGORIES = frozenset({"response"})
RESPONSE_FALLBACK = "response.clarify"

_CANNED_RESPONSES = [
    (
        "greeting",
        "The user is greeting the agent (hello, hi, good morning).",
        "Hello! What can I help you with?",
    ),
    (
        "thanks",
        "The user is thanking the agent.",
        "You're welcome! Anything else?",
    ),
    (
        "acknowledge",
        "The user is acknowledging or making filler the agent should nod to.",
        "Got it.",
    ),
    (
        "farewell",
        "The user is signing off (bye, goodbye, see you).",
        "Goodbye! Talk to you later.",
    ),
    (
        "affirm",
        "The user is agreeing or confirming (yes, ok, sure, sounds good).",
        "Okay!",
    ),
    (
        "unable",
        "A request the agent understands but has no way to perform.",
        "I'm not able to do that yet.",
    ),
    (
        "clarify",
        "The input is too vague to act on; ask the user to be more specific.",
        "Could you try being more specific?",
    ),
]

_RESPONSE_MESSAGES = {f"response.{name}": message for name, _, message in _CANNED_RESPONSES}


def build_skills(config: dict) -> list[Skill]:
    """The hardcoded built-ins: internal behaviors only.

    Real integrations do not live here — they are authored by the codegen
    pipeline into `data/skills/` or shipped as seed skill packages under
    `seeds/`. A fabricated built-in here shadows the authoring path for a real
    one (navigation routes to it), so keep this list to behaviors that need no
    external service.

    The `response` category is a closed tree of canned replies (see
    CANNED_CATEGORIES): it is the catchall for inputs that are not tasks, and it
    never goes through codegen.
    """
    return [
        Skill(
            name=f"response.{name}",
            category="response",
            description=description,
            act=_canned_response(f"response.{name}", message),
        )
        for name, description, message in _CANNED_RESPONSES
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


def _clears_create_gate(
    result,
    create_id: str,
    existing_ids: list[str],
    tau: float,
    margin: float,
) -> bool:
    """Should the create branch actually fire, or is it a weak plurality win?

    The create option shares a softmax with the real categories/skills, so its
    probability is diluted by every existing option — it is not a measure of
    "confidence that nothing matches". Require both an absolute floor (`tau`)
    and a clear lead over the best existing option (`margin`); a create that
    only squeaks past on a crowded tree is suppressed and navigation falls back
    to the best existing branch (the intent guard remains a second door to
    authoring). Returns True when the gate is effectively disabled.
    """
    probs = result.probs
    if create_id not in probs:
        return True
    create_p = probs[create_id]
    best = max((probs[i] for i in existing_ids if i in probs), default=0.0)
    return create_p >= tau and (create_p - best) >= margin


def _navigate_canned(
    engine: SemIfEngine,
    log: DecisionLog,
    trace: TraceLog,
    request: Request,
    category: str,
    skills: list[Skill],
) -> Skill:
    """Pick one canned reply from a closed category. No create branch, ever.

    The catchall (`RESPONSE_FALLBACK`) is offered last so the decision question
    can point at it by name; it is the answer when nothing specific fits.
    """
    ordered = [s for s in skills if not s.is_noop()] or list(skills)
    fallback = next((s for s in ordered if s.name == RESPONSE_FALLBACK), None)
    ordered = [s for s in ordered if s is not fallback] + ([fallback] if fallback else [])
    if len(ordered) == 1:
        return ordered[0]
    leaf = DecisionRequest(
        state=compose_state(request, additional_context: f"all choices are within the {category} category"),
        question=(
            "Which canned response best fits this input? "
            f"Choose {ordered[-1].name} if none of the others does."
        ),
        options=[Option(s.name, s.description) for s in ordered],
    )
    result = engine.call(leaf)
    log.append(leaf, result, extra={"phase": "navigate:response", "run_id": request.id})
    chosen = next((s for s in ordered if s.name == result.selected), ordered[-1])
    trace.append(
        "response_selected",
        request.id,
        category=category,
        skill=chosen.name,
        selected=result.selected,
        probs=result.probs,
    )
    return chosen


def navigate(
    engine: SemIfEngine,
    log: DecisionLog,
    trace: TraceLog,
    request: Request,
    tree: dict[str, list[Skill]],
    create_tau: float = 0.5,
    create_margin: float = 0.35,
) -> Skill | CreateCategory | CreateSkill:
    """Descend the tree one SemIf choice per level. Every choice is logged.

    The category level offers a "create_category" branch and the leaf level a
    "create_skill" branch; both are handled live by dispatch and log a
    suggestion event to the trace. A level with nothing to choose from
    (an empty tree, or a category with no skills yet) short-circuits straight to
    the create branch: SemIf decisions need at least two options, and asking
    "which of one?" is meaningless.

    A create branch that wins the softmax but fails the confidence gate
    (`create_tau` floor and `create_margin` lead over the best existing option)
    is suppressed: navigation falls back to the best existing option so a
    genuinely unmatched action still reaches authoring through the intent guard,
    while a crowded tree no longer drifts into create on a weak plurality.
    """
    categories = sorted(tree.keys())
    create_category = Option("create_category", "Suggest a new category for this.")
    top = DecisionRequest(
        state=compose_state(request),
        question="Which top-level category handles this request?",
        options=[Option(c, c) for c in categories] + [create_category],
    )
    if not categories:
        trace.append(
            "create_category",
            request.id,
            state=top.state,
            question=top.question,
            options=[o.id for o in top.options],
            selected="create_category",
            probs={},
        )
        return CreateCategory()
    top_result = engine.call(top)
    log.append(top, top_result, extra={"phase": "navigate:category", "run_id": request.id})
    category = top_result.selected
    if category == "create_category" and not _clears_create_gate(
        top_result, "create_category", categories, create_tau, create_margin
    ):
        fallback = max(
            (c for c in categories if c in top_result.probs),
            key=lambda c: top_result.probs[c],
        )
        trace.append(
            "create_suppressed",
            request.id,
            level="category",
            fallback=fallback,
            probs=top_result.probs,
            tau=create_tau,
            margin=create_margin,
        )
        category = fallback
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
    if category in CANNED_CATEGORIES:
        return _navigate_canned(engine, log, trace, request, category, skills)
    create_skill = Option(
        "create_skill",
        "No existing skill performs this action; create a new skill for it.",
    )
    leaf = DecisionRequest(
        state=compose_state(request),
        question=f"Which {category} skill performs the action this request asks for? "
        "Choose create_skill if none does.",
        options=[Option(s.name, s.description) for s in skills] + [create_skill],
    )
    if not skills:
        trace.append(
            "skill_needed",
            request.id,
            category=category,
            state=leaf.state,
            question=leaf.question,
            options=[o.id for o in leaf.options],
            selected="create_skill",
            probs={},
        )
        return CreateSkill(category=category)
    leaf_result = engine.call(leaf)
    log.append(leaf, leaf_result, extra={"phase": "navigate:leaf", "run_id": request.id})
    pick = leaf_result.selected
    if pick == "create_skill" and not _clears_create_gate(
        leaf_result, "create_skill", [s.name for s in skills], create_tau, create_margin
    ):
        fallback = max(
            (s.name for s in skills if s.name in leaf_result.probs),
            key=lambda n: leaf_result.probs[n],
        )
        trace.append(
            "create_suppressed",
            request.id,
            level="leaf",
            category=category,
            fallback=fallback,
            probs=leaf_result.probs,
            tau=create_tau,
            margin=create_margin,
        )
        pick = fallback
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


def confirm_skill_fit(
    engine: SemIfEngine,
    log: DecisionLog,
    trace: TraceLog,
    request: Request,
    skill: Skill,
    tau: float = 0.6,
) -> bool:
    """Does this skill's action actually match what the request asks for?

    Navigation picks the closest leaf; a seeded read skill can outscore the
    generic "create a new skill" branch on a request whose verb (send) no
    existing action performs. This one SemIf decision guards against silently
    running the wrong skill: a mismatch sends dispatch to author a new leaf
    instead. Wording tuned against the real model: comparing the two *actions*
    ("different action") is unambiguous where "can this serve the request" was
    not — e.g. "message Sam" (an implicit send in instant-messaging syntax)
    reads as a different action from "read the next message". Do not add an
    instant-messaging context hint: it over-fires and pulls read phrasings to
    the send side.
    """
    decision = DecisionRequest(
        state=(
            f"Requested action: {request.text}\n"
            f"Action of the existing skill: {skill.description}"
        ),
        question="Compare the requested action and the skill's action. Are they the same action?",
        options=[
            Option("same", "Same action."),
            Option(
                "different",
                "Different action: the skill does not do what the user asks.",
            ),
        ],
    )
    result = engine.call(decision)
    log.append(
        decision,
        result,
        extra={
            "phase": "navigate:intent",
            "run_id": request.id,
            "skill": f"{skill.category}.{skill.name}",
        },
    )
    fits = result.prob("same") >= tau
    trace.append(
        "intent_guard",
        request.id,
        category=skill.category,
        skill=skill.name,
        selected=result.selected,
        probs=result.probs,
        fits=fits,
    )
    return fits


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
    client: LLMClient, request: Request, tree: dict[str, list[Skill]]
) -> CategoryDraft:
    """Author a new category stub with the `llm` provider."""
    raw = client.chat(build_category_prompt(request, tree), max_tokens=128)
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
        '{"title": "<short lowercase snake_case id, no spaces, generic and reusable>", '
        '"description": "<one to two sentence description of how it benefits user>"}'
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
    if not re.fullmatch(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", name):
        raise ValueError(f"skill title must be snake_case alnum (dots allowed): {title!r}")
    return SkillDraft(name=name, description=description)


def generate_skill(
    client: LLMClient, request: Request, category: str, tree: dict[str, list[Skill]]
) -> SkillDraft:
    """Author a new skill leaf stub with the `llm` provider."""
    raw = client.chat(build_skill_prompt(request, category, tree), max_tokens=128)
    return parse_skill_draft(raw)
