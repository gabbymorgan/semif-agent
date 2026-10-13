"""The bridge catalog: discovery, prompt description, and standalone runner.

A bridge implementation ships **with its pack** (``packs/<name>/bridge/``); the
core owns only the shared contract (``bridges/base.py``) and this registry. A
pack's ``pack.json`` names its bridge(s) in ``bridges`` and ships one module per
name under ``bridge/`` (``bridge/<name>.py``, plus optional sibling modules such
as ``bridge/<name>_client.py``). The registry imports each enabled pack's bridge
by convention, so nothing in the core names a concrete bridge — a new bridge is a
new pack, not a core edit.

The shared, pack-less **LLM bridge** stays a core built-in (``CORE_BRIDGES``):
many packs call it, so it belongs to no single concern. ``known_infos()`` exposes
every bridge's static metadata, ``describe_bridges()`` renders the catalog block
injected into the code-generation prompts, and ``run_bridges()`` starts the
enabled ones as a standalone process (``python -m semif_agent.cli bridge``).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import threading
import types
from pathlib import Path

from .base import BridgeInfo, BridgeService
from .llm import LLMBridge

#: The agent checkout root; a relative pack root resolves against it, and the
#: default pack root is ``<checkout>/packs``.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK_ROOT = REPO_ROOT / "packs"

#: Bridges that belong to no pack — shared services any pack may call. They
#: still live under ``bridges.<name>`` so ``derived_config_vars`` and
#: ``run_bridges`` find them. Order is presentation order.
CORE_BRIDGES: tuple[type[BridgeService], ...] = (LLMBridge,)

#: Pack-discovered bridge classes, keyed by ``pack:bridge:path`` (loaded once
#: per process).
_CLASS_CACHE: dict[str, type[BridgeService]] = {}

#: Packs the running scheduler discovered; ``describe_bridges()`` (called from
#: codegen with no config) prefers these over a fresh default-root scan.
_ACTIVE_PACKS: list | None = None


def set_bridge_packs(packs) -> None:
    """Record the packs the running agent loaded, for prompt-time discovery.

    ``describe_bridges()`` is called from codegen, which has no config in hand;
    ``build_scheduler`` calls this after loading packs so the catalog reflects
    the host's real ``skill_packs`` roots. ``None`` restores default discovery.
    """
    global _ACTIVE_PACKS
    _ACTIVE_PACKS = list(packs) if packs is not None else None


def _resolve_roots(roots) -> list[Path]:
    if isinstance(roots, (str, Path)):
        roots = [roots]
    if not roots:
        roots = [DEFAULT_PACK_ROOT]
    resolved = []
    for root in roots:
        path = Path(str(root))
        resolved.append(path if path.is_absolute() else REPO_ROOT / path)
    return resolved


def _packs_from_roots(roots=None) -> list:
    from ..packs import PackRegistry

    return PackRegistry(_resolve_roots(roots)).discover()


def _packs_from_config(config: dict) -> list:
    return _packs_from_roots((config or {}).get("skill_packs"))


def _packs(packs=None) -> list:
    if packs is not None:
        return list(packs)
    if _ACTIVE_PACKS is not None:
        return list(_ACTIVE_PACKS)
    return _packs_from_roots(None)


def _load_package(module_name: str, directory: Path) -> types.ModuleType:
    """Import ``directory`` as a package under ``module_name`` (idempotent).

    The pack's ``bridge/`` folder is loaded from its real path (never relying on
    it being importable by name), so sibling modules and relative imports work
    regardless of where the pack was cloned. An absent ``__init__.py`` is fine —
    a synthetic package namespace is registered instead.
    """
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    init = directory / "__init__.py"
    if not init.is_file():
        package = types.ModuleType(module_name)
        package.__path__ = [str(directory)]  # type: ignore[attr-defined]
        package.__package__ = module_name
        sys.modules[module_name] = package
        return package
    spec = importlib.util.spec_from_file_location(
        module_name, init, submodule_search_locations=[str(directory)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load bridge package at {directory}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = package
    spec.loader.exec_module(package)
    return package


def _find_bridge_class(module: types.ModuleType, name: str) -> type[BridgeService] | None:
    """The ``BridgeService`` subclass a pack bridge module defines.

    An explicit ``BRIDGE`` attribute wins; otherwise scan for subclasses defined
    in the module, preferring one whose ``INFO.name`` matches the pack's declared
    bridge name.
    """
    explicit = getattr(module, "BRIDGE", None)
    if isinstance(explicit, type) and issubclass(explicit, BridgeService):
        return explicit
    candidates = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, BridgeService)
        and value is not BridgeService
        and getattr(value, "__module__", None) == module.__name__
    ]
    for candidate in candidates:
        info = getattr(candidate, "INFO", None)
        if info is not None and getattr(info, "name", None) == name:
            return candidate
    return candidates[0] if candidates else None


def load_pack_bridge_class(pack, bridge_name: str) -> type[BridgeService] | None:
    """Import one pack's bridge module and return its bridge class (or None)."""
    bridge_dir = Path(pack.path) / "bridge"
    module_file = bridge_dir / f"{bridge_name}.py"
    if not module_file.is_file():
        return None
    package_name = f"semif_pack_bridge_{pack.name}"
    _load_package(package_name, bridge_dir)
    module = importlib.import_module(f"{package_name}.{bridge_name}")
    return _find_bridge_class(module, bridge_name)


def load_pack_bridge_package(pack_name: str, packs=None) -> types.ModuleType:
    """Import and return a pack's ``bridge/`` package.

    The package is registered under the stable name
    ``semif_pack_bridge_<pack>``, so a caller (a pack's own test, or a tool) can
    import a sibling module with ``importlib.import_module(f"{pkg.__name__}.<mod>")``.
    """
    pack = next((p for p in _packs(packs) if p.name == pack_name), None)
    if pack is None:
        raise KeyError(f"unknown pack: {pack_name!r}")
    bridge_dir = Path(pack.path) / "bridge"
    if not bridge_dir.is_dir():
        raise KeyError(f"pack {pack_name!r} ships no bridge/ directory")
    return _load_package(f"semif_pack_bridge_{pack.name}", bridge_dir)


def load_pack_bridge_module(
    pack_name: str, bridge_name: str, packs=None
) -> types.ModuleType:
    """Import and return one pack's bridge module (``bridge/<bridge_name>.py``)."""
    package = load_pack_bridge_package(pack_name, packs)
    return importlib.import_module(f"{package.__name__}.{bridge_name}")


def discover_bridge_classes(packs=None) -> list[type[BridgeService]]:
    """Every bridge class: the packs' bridges (in pack order) then core built-ins.

    With no ``packs``, uses the running scheduler's packs, else a scan of the
    default pack root. A pack that declares a bridge but ships no ``bridge/``
    module is skipped (a half-authored pack never breaks the catalog).
    """
    classes: list[type[BridgeService]] = []
    for pack in _packs(packs):
        for bridge_name in getattr(pack, "bridges", ()) or ():
            key = f"{pack.name}:{bridge_name}:{getattr(pack, 'path', '')}"
            if key not in _CLASS_CACHE:
                found = load_pack_bridge_class(pack, bridge_name)
                if found is None:
                    continue
                _CLASS_CACHE[key] = found
            bridge_class = _CLASS_CACHE[key]
            if bridge_class not in classes:
                classes.append(bridge_class)
    for bridge_class in CORE_BRIDGES:
        if bridge_class not in classes:
            classes.append(bridge_class)
    return classes


def _class_for(name: str, packs=None) -> type[BridgeService] | None:
    for bridge_class in discover_bridge_classes(packs):
        if bridge_class.INFO.name == name:
            return bridge_class
    return None


def known_infos(packs=None) -> list[BridgeInfo]:
    return [bridge_class.INFO for bridge_class in discover_bridge_classes(packs)]


def derived_config_vars(config: dict) -> dict[str, str]:
    """Skill-facing config vars for every bridge, derived from its own block.

    The `bridges.<name>` block is the single source of truth: its `host`/`port`
    are the listener a skill body calls and its `token` is the shared secret.
    There is deliberately no duplicate top-level `*_bridge_url`/`*_bridge_token`
    to keep in sync — the values are derived here and injected into the config a
    skill reads (`ctx.config`). A bridge with no block (or no port) contributes
    nothing, so an unconfigured bridge still surfaces as an unresolved contract
    variable the operator is asked for.
    """
    blocks = config.get("bridges", {}) or {}
    derived: dict[str, str] = {}
    for info in known_infos(_packs_from_config(config)):
        block = blocks.get(info.name)
        if not isinstance(block, dict):
            continue
        host = str(block.get("host") or "127.0.0.1").strip()
        # A bind-all address is not a callable one; skills always call loopback.
        if host in ("", "0.0.0.0", "::", "[::]"):
            host = "127.0.0.1"
        port = block.get("port")
        if info.url_config_var and port:
            derived[info.url_config_var] = f"http://{host}:{int(port)}"
        if info.auth_config_var:
            derived[info.auth_config_var] = str(block.get("token") or "")
    return derived


def describe_bridges(packs=None) -> str:
    """A prompt block describing every available bridge service.

    Injected into the skill-body, retry, regen, and testgen
    prompts so codegen knows which real services it may build a skill against,
    how to reach and authenticate to each one, and which `ctx.config` variables
    carry its URL, token, and other values. This is the single source of bridge
    specifics — CODEGEN.md/TESTGEN.md carry only the generic pattern. Static: no
    bridge needs to be running or enabled.
    """
    lines = [
        "Available bridge services — call a service's own HTTP API directly "
        "when it is a plain HTTP API; use one of these bridges when the service "
        "needs state/auth, a native or heavy protocol, or a sanitized surface. "
        "For a service a bridge fronts, call the bridge over HTTP with "
        "urllib.request rather than reaching around it; read every value from "
        "ctx.config:",
    ]
    for info in known_infos(packs):
        lines.append(f"- {info.name} (service: {info.service}): {info.description}")
        lines.append(f"    base URL config var: {info.url_config_var}")
        if info.auth_header and info.auth_config_var:
            lines.append(
                f"    auth: send the value of {info.auth_config_var} in the "
                f"{info.auth_header} header when it is set"
            )
        docs = dict(info.config_var_docs)
        if info.config_vars:
            lines.append("    config vars:")
            for name in info.config_vars:
                description = docs.get(name)
                suffix = f" — {description}" if description else ""
                lines.append(f"      - {name}{suffix}")
        if info.endpoints:
            lines.append("    endpoints:")
            for endpoint in info.endpoints:
                lines.append(f"      - {endpoint}")
    return "\n".join(lines) + "\n"


def build_bridge(
    config: dict, name: str, trace=None, llm=None
) -> BridgeService:
    """Instantiate one bridge from the top-level `bridges.<name>` config block.

    The class is discovered from the host's packs (plus the core built-ins); it
    is constructed through ``BridgeService.from_config``, so a bridge that needs
    the scheduler's ``llm`` client (the shared LLM bridge) or top-level fallback
    values (Nextcloud's ``nextcloud_*``) declares that itself rather than the
    core special-casing it by name.
    """
    bridge_class = _class_for(name, _packs_from_config(config))
    if bridge_class is None:
        raise KeyError(f"unknown bridge: {name!r}")
    block = (config.get("bridges", {}) or {}).get(name, {}) or {}
    return bridge_class.from_config(block, config=config, trace=trace, llm=llm)


def run_bridges(
    config: dict,
    names: list[str] | None = None,
    trace=None,
    llm=None,
) -> int:
    """Run the selected (+ enabled) bridges until interrupted.

    With explicit `names`, a bridge runs even if disabled is not set; with no
    names, only `enabled` bridges run. Returns a process exit code. `llm` is the
    scheduler's configured language-model client (see `build_bridge`).
    """
    blocks = config.get("bridges", {}) or {}
    selected: list[BridgeService] = []
    for bridge_class in discover_bridge_classes(_packs_from_config(config)):
        name = bridge_class.INFO.name
        if names and name not in names:
            continue
        block = blocks.get(name, {}) or {}
        if not names and not block.get("enabled", False):
            continue
        bridge = build_bridge(config, name, trace=trace, llm=llm)
        ok, hint = bridge.check_requirements()
        if not ok:
            print(f"bridge {name} unavailable: {hint}")
            for running in selected:
                running.stop()
            return 1
        port = bridge.start()
        selected.append(bridge)
        print(
            f"bridge {name} listening on http://{bridge.host}:{port} "
            f"(skills use the {bridge_class.INFO.url_config_var} config var)"
        )
    if not selected:
        print("no bridge services enabled (set bridges.<name>.enabled)")
        return 1

    stop = threading.Event()
    try:
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        for bridge in selected:
            bridge.stop()
    return 0
