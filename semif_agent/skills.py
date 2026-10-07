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
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .decisions import DecisionRequest, Option, Request
from .engine import DecisionEngine
from .llm import LLMClient
from .log import DecisionLog
from .timers import TimerService
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
    engine: DecisionEngine
    config: dict
    timers: TimerService | None = None
    # The scheduler, exposed to built-in meta skills (housekeeping) so they can
    # act on the agent's own tree/registry/store. None for ordinary bodies, which
    # must never reach into the agent.
    admin: object | None = None


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
    category_description: str = ""
    # Where the leaf came from: "builtin" (hardcoded), "seed" (committed starter
    # package), or "generated" (codegen-authored). Built-ins are protected from
    # delete/restart/regen; seeds and generated leaves are deletable (durably,
    # via a tombstone).
    origin: str = "generated"

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
    hint only — the CODEGEN.md contract, request, and requirements still win.
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

    def unregister_skill(self, category: str, name: str) -> None:
        """Remove a skill entry from its category (the category itself stays).

        The category bucket is kept even when it becomes empty: removing it would
        make the category vanish from the navigation softmax until something
        re-registers it.
        """
        categories = self.read()
        entry = categories.get(category)
        if entry is None:
            return
        entry["skills"] = [
            s for s in entry.get("skills", []) if s.get("name") != name
        ]
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

    def delete(self, category: str, name: str) -> bool:
        """Remove a skill folder. Returns True when something was removed.

        Path-guarded: the resolved folder must sit directly under this store's
        root, so a crafted category/name can never escape it.
        """
        target = (self.path / category / name).resolve()
        root = self.path.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"refusing to delete outside the skill store: {target}")
        if not target.is_dir():
            return False
        shutil.rmtree(target)
        return True


class DeletedSkills:
    """A persisted set of deleted skill refs (`category.name`).

    Seeds reload from the committed `seeds/` tree on every startup, so deleting
    one from the running tree would be undone on the next boot. Recording the
    ref here suppresses it (and any generated body of the same name) at load
    time, making deletion durable. Restoring a skill means removing its ref.
    """

    def __init__(self, path: str = "data/deleted_skills.json"):
        self.path = Path(path)

    def read(self) -> set[str]:
        if not self.path.is_file():
            return set()
        try:
            data = json.loads(self.path.read_text())
        except ValueError:
            return set()
        if isinstance(data, dict):
            data = data.get("skills", [])
        return {str(ref) for ref in data or []}

    def add(self, category: str, name: str) -> None:
        refs = self.read()
        refs.add(f"{category}.{name}")
        self._write(refs)

    def remove(self, category: str, name: str) -> None:
        refs = self.read()
        refs.discard(f"{category}.{name}")
        self._write(refs)

    def contains(self, category: str, name: str) -> bool:
        return f"{category}.{name}" in self.read()

    def _write(self, refs: set[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(refs), indent=2) + "\n")


def apply_tombstones(tree: dict[str, list[Skill]], deleted: DeletedSkills) -> int:
    """Drop tombstoned leaves from the tree. Returns the number removed."""
    removed = 0
    for ref in deleted.read():
        category, _, name = ref.partition(".")
        if not name:
            continue
        skills = tree.get(category)
        if not skills:
            continue
        kept = [s for s in skills if s.name != name]
        removed += len(skills) - len(kept)
        tree[category] = kept
    return removed


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
        origin="generated",
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
        category_description = str(
            entry.get("description") or CATEGORY_DESCRIPTIONS.get(category, "")
        )
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
            category_description=category_description,
            origin="generated",
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
            category_description=CATEGORY_DESCRIPTIONS.get(category, ""),
            origin="seed",
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
    if additional_context:
        parts.append(f"({additional_context})")
    return " ".join(parts)


def _canned_response(name: str, message: str):
    """An act that returns a fixed line. No transport, no generation, no config."""

    def act(ctx: ActionContext, request: Request) -> ActionResult:
        return ActionResult(action_log=message, new_state=message)

    return act


# The `response` category is a closed tree of canned replies: it never authors a
# skill (no create route) and its runs are not assessed (there is no side effect
# to succeed or fail). Every non-task input lands here instead of being gated.
CANNED_CATEGORIES = frozenset({"response"})
RESPONSE_FALLBACK = "response.clarify"

# The housekeeping category holds the agent's own maintenance skills (delete a
# skill, clear its config, regenerate it, cancel a build). It is a built-in
# category that is **deterministically locked** from new-skill creation — the
# lock is a code constant, not a config value, so it can never be unlocked by
# editing a file and navigation can never author into it.
HOUSEKEEPING_CATEGORY = "housekeeping"
HARD_LOCKED_CATEGORIES = frozenset({HOUSEKEEPING_CATEGORY})

# Categories whose runs are deterministic internal actions (a known found/not-
# found result, no external service): no SemIf assessment and no repair loop, so
# a "skill not found" can never offer to codegen-repair the meta skill.
DETERMINISTIC_CATEGORIES = frozenset({HOUSEKEEPING_CATEGORY})

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
        "clarify",
        "The input is too vague to act on; ask the user to be more specific.",
        "Could you try being more specific?",
    ),
]

_RESPONSE_MESSAGES = {f"response.{name}": message for name, _, message in _CANNED_RESPONSES}

# Built-in category descriptions. A category level that offers only bare names
# carries no signal (real runs sent "book a flight to japan" to the `simplex`
# bucket 0.49 vs create_category 0.12); giving the model the category's scope
# flips that to create_category 0.87 vs simplex 0.02. Authored categories carry
# their own description from the registry; this is the seed for the built-ins and
# the fallback for a legacy entry with no description.
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "response": (
        "Anything that is not a request to perform an action or produce a result: "
        "greetings, thanks, acknowledgements, chit-chat, and off-topic statements."
    ),
    "calendar": (
        "Calendar and scheduling: reading or changing the user's calendar "
        "events and reminders."
    ),
    "simplex": (
        "SimpleX messaging: reading incoming messages, sending messages to "
        "contacts, and showing the user's contact link."
    ),
    "time": (
        "Time and date utilities: telling the current time or date, setting "
        "countdown timers, and setting alarms."
    ),
    "housekeeping": (
        "Agent self-maintenance: deleting, reconfiguring, regenerating, and "
        "cancelling the agent's own skills."
    ),
}


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
            category_description=CATEGORY_DESCRIPTIONS["response"],
            origin="builtin",
        )
        for name, description, message in _CANNED_RESPONSES
    ]


def _confirmed(request: Request) -> bool:
    """Whether the human's answer to a confirmation prompt was affirmative."""
    return str(request.user_input or "").strip().lower() in (
        "y",
        "yes",
        "confirm",
        "confirmed",
        "ok",
        "okay",
    )


_TARGET_QUESTION = (
    "Which skill? Reply with its name as category.skill (or a unique skill name)."
)

# Explicit housekeeping command forms. They are routed deterministically to the
# housekeeping skill (see parse_meta_command) because the generic SemIf guards
# read a task-like skill name in the request as the task itself (e.g.
# "regen skill calendar.create_event" scores as a calendar task). The skill name
# is just a variable the skill resolves — or asks for.
_META_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    (
        "delete_skill",
        re.compile(
            r"^\s*(?:please\s+)?(?:delete|remove|drop)\s+(?:the\s+|a\s+)?skill\b",
            re.IGNORECASE,
        ),
    ),
    (
        "clear_config",
        re.compile(
            r"^\s*(?:please\s+)?(?:clear|reset|erase|wipe)\s+(?:the\s+|a\s+)?"
            r"(?:skill'?s?\s+)?(?:config(?:uration)?|settings?|variables?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "regen_skill",
        re.compile(
            r"^\s*(?:please\s+)?(?:regen|regenerate|rewrite|rebuild|fix|repair)\b"
            r".*\b(?:skill|code)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cancel_build",
        re.compile(
            r"^\s*(?:please\s+)?(?:"
            r"(?:cancel|abort|stop)\s+(?:the\s+)?skill\s+build(?:ing)?\b"
            r"|(?:cancel|abort)\s+(?:the\s+)?build\b"
            r"|stop\s+building\b.*\bskill\b"
            r")",
            re.IGNORECASE,
        ),
    ),
]


def parse_meta_command(text: str) -> str | None:
    """Recognize an explicit housekeeping command, returning its skill name.

    The command forms ("delete skill [name]", "clear config variables for
    [name]", "regen skill [name]", "cancel skill build") are routed
    deterministically, because the generic SemIf navigation misreads a
    task-like skill name in the request as the task itself. Anything else
    returns None and goes through normal navigation. The target skill name, when
    present, is left in the request for the skill to resolve as an input
    variable.
    """
    stripped = (text or "").strip()
    for name, pattern in _META_PATTERNS:
        if pattern.match(stripped):
            return name
    return None


def _resolve_target(admin, request: Request):
    """Resolve the skill a housekeeping request targets.

    The target may be named in the request ("delete skill calendar.foo") or
    supplied as an input variable when the request is name-free ("delete
    skill") — in which case the skill asks for it. Keeping the name out of the
    routed request is deliberate: a skill name like `calendar.create_event`
    reads as a task and pulls navigation to the wrong category. Returns a
    `(category, name)` tuple, or an `ActionResult` (needs_input / not-found /
    ambiguous) to return as-is.
    """
    target = request.meta.get("housekeeping_target")
    if target:
        return tuple(target)
    resolved = resolve_skill_ref(admin.tree, request.text)
    if isinstance(resolved, tuple):
        return resolved
    if isinstance(resolved, list):
        names = ", ".join(f"{c}.{n}" for c, n in resolved)
        return ActionResult(
            action_log=(
                f"several skills match ({names}); "
                "name one as category.skill"
            ),
            new_state="ambiguous",
        )
    answer = (request.user_input or "").strip()
    if not answer:
        return ActionResult(
            action_log="awaiting skill name",
            new_state="awaiting target",
            needs_input=_TARGET_QUESTION,
        )
    request.user_input = None
    ref = resolve_skill_ref(admin.tree, answer)
    if isinstance(ref, tuple):
        request.meta["housekeeping_target"] = ref
        return ref
    if isinstance(ref, list):
        names = ", ".join(f"{c}.{n}" for c, n in ref)
        return ActionResult(
            action_log=f"several skills match ({names}); be specific",
            new_state="ambiguous",
        )
    return ActionResult(
        action_log=f"no skill matches {answer!r}",
        new_state="not found",
    )


def _delete_skill_act(ctx: ActionContext, request: Request) -> ActionResult:
    admin = ctx.admin
    if admin is None:
        return ActionResult("no admin available", "error")
    target = _resolve_target(admin, request)
    if isinstance(target, ActionResult):
        return target
    category, name = target
    ref = f"{category}.{name}"
    if request.meta.get("housekeeping_confirm") != ref:
        request.meta["housekeeping_confirm"] = ref
        return ActionResult(
            action_log=f"awaiting confirmation for {ref}",
            new_state="awaiting confirmation",
            needs_input=f"Delete {ref} and all its files? Reply yes to confirm.",
        )
    if not _confirmed(request):
        return ActionResult(
            action_log=f"cancelled by user for {ref}",
            new_state="cancelled",
        )
    _status, detail = admin.delete_skill(category, name)
    return ActionResult(detail, detail)


def _clear_config_act(ctx: ActionContext, request: Request) -> ActionResult:
    admin = ctx.admin
    if admin is None:
        return ActionResult("no admin available", "error")
    target = _resolve_target(admin, request)
    if isinstance(target, ActionResult):
        return target
    category, name = target
    _status, detail = admin.clear_skill_config(category, name)
    return ActionResult(detail, detail)


def _regen_skill_act(ctx: ActionContext, request: Request) -> ActionResult:
    admin = ctx.admin
    if admin is None:
        return ActionResult("no admin available", "error")
    target = _resolve_target(admin, request)
    if isinstance(target, ActionResult):
        return target
    category, name = target
    ref = f"{category}.{name}"
    if request.meta.get("housekeeping_regen") != ref:
        request.meta["housekeeping_regen"] = ref
        return ActionResult(
            action_log=f"awaiting guidance for {ref}",
            new_state="awaiting guidance",
            needs_input="What needs to be fixed?",
        )
    guidance = (request.user_input or "").strip()
    _status, detail = admin.regen_skill(category, name, guidance)
    return ActionResult(detail, detail)


def _cancel_build_act(ctx: ActionContext, request: Request) -> ActionResult:
    admin = ctx.admin
    if admin is None:
        return ActionResult("no admin available", "error")
    writing = [
        (category, skill.name)
        for category, skills in admin.tree.items()
        for skill in skills
        if skill.writing
    ]
    target = request.meta.get("housekeeping_target")
    if target:
        category, name = tuple(target)
    else:
        resolved = resolve_skill_ref(admin.tree, request.text)
        if isinstance(resolved, list):
            names = ", ".join(f"{c}.{n}" for c, n in resolved)
            return ActionResult(
                action_log=(
                    f"several skills match ({names}); "
                    "name one as category.skill"
                ),
                new_state="ambiguous",
            )
        if isinstance(resolved, tuple):
            if resolved not in writing:
                category, name = resolved
                return ActionResult(
                    action_log=(
                        f"no build is in progress for "
                        f"{category}.{name}"
                    ),
                    new_state="no build",
                )
            category, name = resolved
        elif len(writing) == 1:
            category, name = writing[0]
        elif not writing:
            return ActionResult(
                action_log="no skill build is in progress",
                new_state="no build",
            )
        else:
            answer = (request.user_input or "").strip()
            if not answer:
                return ActionResult(
                    action_log="awaiting skill name",
                    new_state="awaiting target",
                    needs_input="Which skill build should I cancel? Reply with its name.",
                )
            request.user_input = None
            ref = resolve_skill_ref(admin.tree, answer)
            if isinstance(ref, tuple) and ref in writing:
                category, name = ref
            else:
                return ActionResult(
                    action_log=(
                        f"no build in progress matching "
                        f"{answer!r}"
                    ),
                    new_state="no build",
                )
        request.meta["housekeeping_target"] = (category, name)
    ref = f"{category}.{name}"
    if request.meta.get("housekeeping_confirm") != ref:
        request.meta["housekeeping_confirm"] = ref
        return ActionResult(
            action_log=f"awaiting confirmation for {ref}",
            new_state="awaiting confirmation",
            needs_input=(
                f"Cancel the build for {ref} and delete its half-built leaf? "
                "Reply yes to confirm."
            ),
        )
    if not _confirmed(request):
        return ActionResult(
            action_log=f"cancelled by user for {ref}",
            new_state="cancelled",
        )
    _status, detail = admin.cancel_skill_build(category, name)
    return ActionResult(detail, detail)


def build_housekeeping_skills() -> list[Skill]:
    """The built-in meta skills that maintain the agent's own skill tree.

    They act on the scheduler (reached through `ctx.admin`) rather than an
    external service, so they live here beside the canned `response` tree, not
    in the codegen/seed pipeline. Their runs are deterministic internal actions
    (see DETERMINISTIC_CATEGORIES) and the category is deterministically locked
    from new-skill creation (HARD_LOCKED_CATEGORIES).
    """
    description = CATEGORY_DESCRIPTIONS[HOUSEKEEPING_CATEGORY]
    return [
        Skill(
            name="delete_skill",
            category=HOUSEKEEPING_CATEGORY,
            description=(
                "Delete an existing skill: remove it from the tree, delete its "
                "code body and files, and forget its recorded config."
            ),
            act=_delete_skill_act,
            category_description=description,
            origin="builtin",
        ),
        Skill(
            name="clear_config",
            category=HOUSEKEEPING_CATEGORY,
            description=(
                "Clear the saved configuration variables recorded for an "
                "existing skill, so they are asked for again next time it runs."
            ),
            act=_clear_config_act,
            category_description=description,
            origin="builtin",
        ),
        Skill(
            name="regen_skill",
            category=HOUSEKEEPING_CATEGORY,
            description=(
                "Regenerate (rewrite) an existing skill's code body from "
                "guidance about what needs to be fixed."
            ),
            act=_regen_skill_act,
            category_description=description,
            origin="builtin",
        ),
        Skill(
            name="cancel_build",
            category=HOUSEKEEPING_CATEGORY,
            description=(
                "Cancel a skill body that is still being generated and delete "
                "its half-built skill leaf."
            ),
            act=_cancel_build_act,
            category_description=description,
            origin="builtin",
        ),
    ]


def resolve_skill_ref(
    tree: dict[str, list[Skill]], text: str
) -> tuple[str, str] | list[tuple[str, str]] | None:
    """Find the skill(s) named in a free-form request.

    Matches `category.name` or a bare unique `name`, case-insensitively, on a
    token boundary (so `time` does not match inside `sometimes`). The longest
    match wins; a tie across different skills returns the list of candidates so
    the caller can ask the human to be specific. Returns None when nothing
    matches.
    """
    lowered = text.lower()
    matches: list[tuple[str, str, int]] = []
    for category, skills in tree.items():
        for skill in skills:
            for candidate in (f"{category}.{skill.name}", skill.name):
                if _mentions(lowered, candidate.lower()):
                    matches.append((category, skill.name, len(candidate)))
    if not matches:
        return None
    longest = max(match[2] for match in matches)
    top = {(category, name) for category, name, length in matches if length == longest}
    if len(top) == 1:
        return next(iter(top))
    return sorted(top)


def _mentions(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    pattern = r"(?<![a-z0-9_.])" + re.escape(needle) + r"(?![a-z0-9_.])"
    return re.search(pattern, haystack) is not None


def category_descriptions(tree: dict[str, list[Skill]]) -> dict[str, str]:
    """The description to offer for each category, best-effort.

    An authored skill carries the registry description on its `Skill`
    (merge_registry/merge_skill_store); a seed or a legacy entry with no
    description falls back to `CATEGORY_DESCRIPTIONS`. An empty category bucket
    (no skills to carry it) falls back too, so a category is never offered bare.
    """
    merged: dict[str, str] = {}
    for category, skills in tree.items():
        for skill in skills:
            if skill.category_description:
                merged[category] = skill.category_description
                break
    for category in tree:
        if not merged.get(category):
            merged[category] = CATEGORY_DESCRIPTIONS.get(category, "")
    return merged



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
        description = str(data.get("description") or CATEGORY_DESCRIPTIONS.get(category, ""))
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
                    category_description=description,
                )
            )
            existing.add(name)


def _navigate_canned(
    engine: DecisionEngine,
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
        state=compose_state(
            request,
            additional_context=f"all choices are within the {category} category",
        ),
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
    engine: DecisionEngine,
    log: DecisionLog,
    trace: TraceLog,
    request: Request,
    tree: dict[str, list[Skill]],
    category_tau: float = 0.75,
    action_tau: float = 0.5,
    bypass_tau: float = 0.5,
) -> Skill | CreateCategory | CreateSkill:
    """Descend the tree one SemIf choice per level. Every choice is logged.

    **Confident softmax winners skip their confirm guard.** When the category
    softmax winner's probability is >= `bypass_tau` (default 0.5) the scope
    confirm (`confirm_category_fit`) is skipped — the winner is taken as-is
    (traced `category_scope_bypassed`); below it the guard runs and can reject
    the winner into create. The leaf softmax winner's probability is handed to
    dispatch (`request.meta["leaf_softmax_prob"]`) so `confirm_skill_fit` is
    likewise skipped above the threshold (traced `intent_bypass`). The guards
    exist to catch an unsure softmax; a winner at 0.5+ is already decisive, and
    each skipped guard is one fewer SemIf call on the hot path. Below the
    threshold the two-stage behavior is unchanged. The recorded data agrees:
    every logged winner >= 0.5 also passed its guard.

    The **actionability guard** (`confirm_non_action`, phase
    `navigate:actionability`, keyed by `navigation.action_tau`) runs at the **top**
    of navigation, before any category is considered: the closed `response` tree
    is reached **only** when the input is not a request (a greeting, thanks,
    acknowledgement, small talk, or a stray statement). A request — even one no
    skill covers, e.g. "what is 255 * 12?" — authors a skill instead, never a
    canned reply. Deciding this first (rather than only when the category softmax
    happens to propose `response`/`create_category`) is what keeps a non-request
    out of a real category: a description alone cannot draw the line — "1+1=2" (a
    statement) and "what is 255 * 12?" (a request) both read as math to the
    category softmax — and `confirm_category_fit` is deliberately permissive.

    Only once the guard says the input is a request does the category softmax
    run, over the **real** categories (the canned `response` tree is never an
    option) plus `create_category`. The winner is then confirmed by
    `confirm_category_fit`, a name-anchored scope check: a rejected winner (or a
    `create_category` win) authors a new category. Descriptions are load-bearing —
    with bare names the model sent "book a flight to japan" to the `simplex`
    bucket (0.49 vs create_category 0.12); with descriptions it picks
    `create_category` 0.87. An empty tree short-circuits straight to create: SemIf
    decisions need at least two options.

    The leaf level is two-stage too: navigation only ever picks among the
    existing skills (`create_skill` is deliberately NOT in the softmax), and the
    reuse-vs-create decision is the intent guard in dispatch
    (`confirm_skill_fit`), keyed by `navigation.intent_tau`.
    """
    if not tree:
        trace.append(
            "create_category",
            request.id,
            state=compose_state(request),
            question="Which top-level category handles this request?",
            options=[],
            selected="create_category",
            probs={},
        )
        return CreateCategory()

    # Actionability first: is this input a request at all? A non-request goes to
    # the closed `response` tree (a canned reply) and never reaches the category
    # softmax; a request always proceeds to category selection, so it can never
    # be canned.
    if confirm_non_action(engine, log, trace, request, action_tau):
        response_skills = tree.get("response")
        if response_skills:
            return _navigate_canned(
                engine, log, trace, request, "response", response_skills
            )
        trace.append(
            "create_category",
            request.id,
            state=compose_state(request),
            question="Which top-level category handles this request?",
            options=[],
            selected="create_category",
            probs={},
        )
        return CreateCategory()

    categories = [c for c in sorted(tree.keys()) if c not in CANNED_CATEGORIES]
    descriptions = category_descriptions(tree)
    create_category = Option(
        "create_category",
        "No existing category covers this request; a new top-level category is "
        "needed.",
    )
    top = DecisionRequest(
        state=compose_state(request),
        question="Which top-level category handles this request?",
        options=[Option(c, descriptions.get(c, c)) for c in categories]
        + [create_category],
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
    if category != "create_category":
        winner_prob = float(top_result.probs.get(category, 0.0))
        if winner_prob >= bypass_tau:
            # Confident winner: skip the scope confirm (one fewer SemIf call).
            trace.append(
                "category_scope_bypassed",
                request.id,
                category=category,
                prob=winner_prob,
            )
        elif not confirm_category_fit(
            engine, log, trace, request, category, descriptions.get(category, ""),
            category_tau,
        ):
            trace.append(
                "category_scope_rejected",
                request.id,
                category=category,
                probs=top_result.probs,
            )
            category = "create_category"
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
    if not skills:
        trace.append(
            "skill_needed",
            request.id,
            category=category,
            state=compose_state(
                request,
                additional_context=f"all choices are within the {category} category",
            ),
            question="(this category has no skills yet)",
            options=[],
            selected="create_skill",
            probs={},
        )
        return CreateSkill(category=category)
    if len(skills) == 1:
        # A one-option softmax is meaningless; hand the sole skill to the intent
        # guard, which owns the reuse-vs-create decision.
        return skills[0]
    leaf = DecisionRequest(
        state=compose_state(
            request,
            additional_context=f"all choices are within the {category} category",
        ),
        question=f"Which {category} skill performs the action this request asks for?",
        options=[Option(s.name, s.description) for s in skills],
    )
    leaf_result = engine.call(leaf)
    log.append(leaf, leaf_result, extra={"phase": "navigate:leaf", "run_id": request.id})
    pick = leaf_result.selected
    # Hand the winner's confidence to dispatch so it can skip the intent guard
    # when the softmax is already decisive (see `navigate` docstring).
    request.meta["leaf_softmax_prob"] = float(leaf_result.probs.get(pick, 0.0))
    return next((s for s in skills if s.name == pick), skills[0])


def confirm_category_fit(
    engine: DecisionEngine,
    log: DecisionLog,
    trace: TraceLog,
    request: Request,
    category: str,
    description: str,
    tau: float = 0.75,
) -> bool:
    """Does this category's scope actually cover what the request asks for?

    The category-level counterpart of `confirm_skill_fit`, and the create door
    at that level: the described-category softmax proposes the best category,
    and this name-anchored scope check confirms it. A rejected winner (or a
    `create_category` win) authors a new category instead. The threshold is
    `navigation.category_tau` (default 0.75 — measured so true non-tasks, which
    score 0.87–0.999 on `response`, sit above it and a task misrouted into
    `response` at 0.61 sits below).

    Name-anchored on purpose: asking whether *the winner we picked* covers the
    request is far cleaner than a global "does any category cover this?" — the
    global wording collapsed in-scope and out-of-scope requests into one band
    (0.47-0.98) and over-covered a non-task ("thank pepper ..."). This compares
    one category's stated scope to the request. The check is deliberately
    permissive: with real descriptions the in-scope cases score 0.9-1.0, so a
    borderline over-cover routes into a plausible category and the leaf intent
    guard catches it (a skill is only authored if no existing leaf matches).
    """
    decision = DecisionRequest(
        state=(
            f"Request: {request.text}\n"
            f"Category: {category}\n"
            f"Scope: {description}"
        ),
        question=f"Does the scope of the {category} category cover this request?",
        options=[
            Option("covers", "Yes, this category covers the request."),
            Option(
                "none",
                "No, this category does not cover the request; a new category is needed.",
            ),
        ],
    )
    result = engine.call(decision)
    log.append(
        decision,
        result,
        extra={
            "phase": "navigate:category_scope",
            "run_id": request.id,
            "category": category,
        },
    )
    fits = result.prob("covers") >= tau
    trace.append(
        "category_scope",
        request.id,
        category=category,
        selected=result.selected,
        probs=result.probs,
        fits=fits,
    )
    return fits


def confirm_non_action(
    engine: DecisionEngine,
    log: DecisionLog,
    trace: TraceLog,
    request: Request,
    tau: float = 0.5,
) -> bool:
    """Is this input non-action (not a request) rather than an actionable request?

    The actionability guard, run at the top of `navigate` before any category is
    considered. A request — even one no skill covers — must author a skill; only
    non-action input gets a canned reply. Non-action covers chit-chat and
    statements ("1+1=2") **and** fragments too vague/incomplete to act on
    ("what"), which reach `response.clarify` ("Could you try being more
    specific?"). This is what keeps "what is 255 * 12?" (a specific request) out
    of the closed `response` tree while "what" (nothing to act on) stays in it —
    a line no single category description can draw, because both read as
    questions to the softmax. Returns True when the input is non-action (route to
    `response`); the threshold is `navigation.action_tau` (default 0.5,
    deliberately separate from the other taus so tuning actionability moves
    nothing else).
    """
    decision = DecisionRequest(
        state=request.text,
        question=(
            "Is this input a request to the agent (it asks for a specific answer, "
            "result, or task), or is it non-request input (a greeting, thanks, "
            "acknowledgement, small talk, a statement, or a fragment too "
            "vague/incomplete to act on)?"
        ),
        options=[
            Option(
                "action",
                "It is a request: the user asks the agent for a specific answer, a "
                "result, or an action (including one the agent has no skill for "
                "yet), or gives a command to manage a skill (delete, clear config, "
                "regenerate, cancel a build).",
            ),
            Option(
                "non_action",
                "It is not a request: a greeting, thanks, acknowledgement, small "
                "talk, a statement, or an incomplete fragment that names nothing to "
                "act on (e.g. 'what', 'hmm', 'and?').",
            ),
        ],
    )
    result = engine.call(decision)
    log.append(
        decision,
        result,
        extra={"phase": "navigate:actionability", "run_id": request.id},
    )
    non_action = result.prob("non_action") >= tau
    trace.append(
        "actionability",
        request.id,
        selected=result.selected,
        probs=result.probs,
        non_action=non_action,
    )
    return non_action


def confirm_skill_fit(
    engine: DecisionEngine,
    log: DecisionLog,
    trace: TraceLog,
    request: Request,
    skill: Skill,
    tau: float = 0.6,
) -> bool:
    """Does this skill's action actually match what the request asks for?

    Navigation picks the closest existing leaf; this guard is now the **sole**
    leaf-level reuse-vs-create decision (create_skill is no longer an option in
    the leaf softmax). If the skill's action is not the same action the request
    asks for, dispatch authors a new leaf instead. The threshold is
    `navigation.intent_tau` (deliberately separate from the run-assessment
    `tau`, so tuning reuse-vs-create does not move assessment/fidelity).

    Wording tuned against the real model: comparing the two *actions*
    ("different action") is unambiguous where "can this serve the request" was
    not — e.g. "message Sam" (an implicit send in instant-messaging syntax)
    reads as a different action from "read the next message". Do not add an
    instant-messaging context hint: it over-fires and pulls read phrasings to
    the send side. The comparison is over-permissive on near-synonyms (read vs
    send score "same" ~0.74-0.78), so it is paired with the softmax that
    disambiguates them — the guard only ever sees the softmax winner.
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
