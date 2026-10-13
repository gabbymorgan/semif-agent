"""Skill packs: bundles of skills associated with a concern, product, or bridge.

A pack is a folder with a ``pack.json`` manifest and a ``skills/`` store::

    packs/nextcloud/
      pack.json
      skills/nextcloud/{create_event,create_task,next_event}/

The manifest binds the pieces that used to be scattered by convention — the
concern, its category descriptions, the bridge(s) it talks to, its runtime
requirements, and its config schema/defaults — so "the Nextcloud product, its
bridge, its skills, and its settings" are one thing. Bridge *code* stays in
``semif_agent/bridges`` (isolation is non-negotiable); a pack references a
bridge by name.

A pack can be enabled/disabled as a unit: disabling it drops its categories from
the tree and stops its bridge (see ``pack_enabled``). This is the same switch
for skills and the bridge, so a host never offers a skill whose bridge is off.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .skills import CATEGORY_DESCRIPTIONS, SkillStore, merge_seed_store

MANIFEST = "pack.json"


@dataclass
class SkillPack:
    """One concern/product bundle: manifest metadata + its skill store."""

    name: str
    title: str
    concern: str
    categories: dict[str, str] = field(default_factory=dict)
    bridges: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    enabled_by_default: bool = True
    config: dict = field(default_factory=dict)
    path: Path = field(default_factory=Path)
    store: SkillStore = field(default_factory=SkillStore)

    def category_names(self) -> list[str]:
        return list(self.categories) or sorted(
            {category for category, _ in self.store.list_skills()}
        )


def _parse_pack(manifest_path: Path) -> SkillPack:
    data = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    name = str(data.get("name") or root.name).strip()
    categories = {
        str(key): str(value)
        for key, value in (data.get("categories") or {}).items()
    }
    return SkillPack(
        name=name,
        title=str(data.get("title") or name),
        concern=str(data.get("concern") or data.get("description") or "").strip(),
        categories=categories,
        bridges=tuple(str(b) for b in (data.get("bridges") or [])),
        requires=tuple(str(r) for r in (data.get("requires") or [])),
        enabled_by_default=bool(data.get("enabled_by_default", True)),
        config=dict(data.get("config") or {}),
        path=root,
        store=SkillStore(str(root / "skills")),
    )


class PackRegistry:
    """Discovers packs from one or more roots.

    A root is either a single pack directory (contains ``pack.json``), a
    directory of packs (each subdirectory with a ``pack.json``), or a plain
    skills directory (no manifest — treated as one manifest-less pack whose
    store is the root itself).
    """

    def __init__(self, roots):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.roots = [Path(root) for root in roots]
        self._packs: list[SkillPack] | None = None

    def discover(self) -> list[SkillPack]:
        if self._packs is not None:
            return self._packs
        packs: list[SkillPack] = []
        seen: set[str] = set()
        for root in self.roots:
            for pack in self._packs_from_root(root):
                if pack.name in seen:
                    continue
                seen.add(pack.name)
                packs.append(pack)
        self._packs = packs
        return packs

    @staticmethod
    def _packs_from_root(root: Path) -> list[SkillPack]:
        if (root / MANIFEST).is_file():
            return [_parse_pack(root / MANIFEST)]
        if root.is_dir():
            child_manifests = sorted(
                child / MANIFEST
                for child in root.iterdir()
                if child.is_dir() and (child / MANIFEST).is_file()
            )
            if child_manifests:
                return [_parse_pack(manifest) for manifest in child_manifests]
            # A plain skills directory: one implicit, manifest-less pack.
            if any(child.is_dir() for child in root.iterdir()):
                return [
                    SkillPack(
                        name=root.name or "pack",
                        title=root.name or "pack",
                        concern="",
                        path=root,
                        store=SkillStore(str(root)),
                    )
                ]
        return []

    def get(self, name: str) -> SkillPack | None:
        return next((p for p in self.discover() if p.name == name), None)


def load_packs(config: dict) -> list[SkillPack]:
    """Discover every pack from the ``skill_packs`` config root(s)."""
    roots = config.get("skill_packs") or "packs"
    return PackRegistry(roots).discover()


def pack_enabled(config: dict, pack: SkillPack) -> bool:
    """Is this pack enabled? One switch governs skills *and* its bridge.

    An explicit ``packs.<name>.enabled`` wins. Otherwise a bridge pack follows
    its bridge's ``bridges.<bridge>.enabled`` (the historical switch), and a
    bridge-less pack falls back to ``enabled_by_default``.
    """
    override = (config.get("packs", {}) or {}).get(pack.name)
    if isinstance(override, dict) and "enabled" in override:
        return bool(override["enabled"])
    if pack.bridges:
        block = (config.get("bridges", {}) or {}).get(pack.bridges[0], {}) or {}
        return bool(block.get("enabled", pack.enabled_by_default))
    return pack.enabled_by_default


def disabled_categories(config: dict, packs: list[SkillPack]) -> set[str]:
    """Categories owned by a disabled pack (dropped from the tree)."""
    cats: set[str] = set()
    for pack in packs:
        if not pack_enabled(config, pack):
            cats.update(pack.category_names())
    return cats


def merge_packs(
    tree: dict,
    packs: list[SkillPack],
    config: dict,
    config_store: SkillStore | None = None,
) -> int:
    """Load every enabled pack's skills into the tree. Returns the count.

    A disabled pack contributes nothing; a later pass drops any of its
    categories that a generated body may have added (see the scheduler).
    """
    loaded = 0
    for pack in packs:
        if not pack_enabled(config, pack):
            continue
        descriptions = {**CATEGORY_DESCRIPTIONS, **pack.categories}
        loaded += merge_seed_store(
            tree,
            pack.store,
            config_store,
            category_descriptions=descriptions,
            origin="seed",
        )
    return loaded
