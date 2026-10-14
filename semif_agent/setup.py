"""First-run setup: the bootstrap interview, config generation, and the plan.

``scripts/bootstrap.sh`` clones the packs, then calls this module (via
``python -m semif_agent.cli setup``) to ask the few questions that make a fresh
machine work — which decision engine, whether the agent may author new skills,
which capability packs, and which gateways — and to write the split ``config/``
plus ``secrets.json`` from the answers. It also emits a machine-readable *plan*
so bootstrap installs **only what the answers require** (no simplex-chat binary
for a REPL-only host, no Playwright for a host that did not pick the browser
pack, no voice stack without the voice gateway).

The two ends of the spectrum are both reachable here:

* **Minimum** — the local SemIf 4B decision engine + the ``essentials`` pack
  (canned replies, time/timers, calculator). No account, no network, no gateway.
* **Maximum** — every pack (SimpleX, browser, Nextcloud, Outlook), a remote
  Winnow decision engine, remote ``llm``/``codegen`` endpoints, and all
  gateways (SimpleX, LXMF, voice).

The module is pure stdlib and takes injectable ``input_fn``/``output`` so the
interview is unit-testable without a TTY.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .config import (
    DEFAULT_SECRETS_FILE,
    default_config,
    load_secrets,
    write_split_config,
)
from .packs import SkillPack, pack_enabled

#: ``llm``/``codegen`` providers the wizard can emit.
PROVIDERS = ("semif", "ollama", "opencode", "openai")
#: ``engine`` providers.
ENGINE_PROVIDERS = ("semif", "winnow")
#: New-skill creation modes.
AUTHORING_MODES = ("deny", "ask", "allow")

#: Capability packs the wizard offers (essentials is always included).
PACK_CHOICES: tuple[tuple[str, str], ...] = (
    ("simplex", "SimpleX messaging (read/send messages, show your contact link)"),
    ("browser", "Browser automation (Playwright: open/read/click/fill a site)"),
    ("nextcloud", "Nextcloud (calendar, tasks, files, contacts, notes)"),
    ("outlook", "Outlook / Microsoft 365 (mail, calendar, To Do, OneDrive)"),
)
#: Gateway platforms the wizard offers.
GATEWAY_CHOICES: tuple[tuple[str, str], ...] = (
    ("simplex", "SimpleX command gateway (DM the bot to run tasks)"),
    ("lxmf", "LXMF / Reticulum command gateway (in-process, no daemon)"),
    ("voice", "Voice gateway (wake word + speech, needs a microphone)"),
)

#: The always-on pack: the built-in local utilities.
BASE_PACK = "essentials"


@dataclass
class Answers:
    """The wizard's answers, and the source of truth for the generated config."""

    # Decision engine.
    engine_provider: str = "semif"  # semif | winnow
    winnow_base_url: str = ""
    winnow_model: str = ""
    winnow_api_key: str = ""

    # New-skill authoring: deny | ask | allow.
    new_skill_creation: str = "deny"

    # The small `llm` (title/description) provider. `semif` reuses the loaded
    # decision model in-process, so the minimum path needs no endpoint at all.
    llm_provider: str = "semif"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""

    # The `codegen` (skill-body) provider. Empty model == not configured; the
    # agent then has no authoring path (which is the minimum path's intent).
    codegen_provider: str = "ollama"
    codegen_base_url: str = ""
    codegen_model: str = ""
    codegen_api_key: str = ""

    # Capability packs to enable (essentials is always included).
    packs: list[str] = field(default_factory=lambda: [BASE_PACK])

    # Gateway platforms to enable.
    gateways: list[str] = field(default_factory=list)

    # IANA timezone; empty uses the host zone.
    timezone: str = ""

    # Nextcloud connection (optional; collected only when the pack is picked).
    nextcloud_url: str = ""
    nextcloud_username: str = ""
    nextcloud_app_password: str = ""

    # Outlook public-client id (not a secret).
    outlook_client_id: str = ""

    def __post_init__(self) -> None:
        self.engine_provider = _norm(self.engine_provider, ENGINE_PROVIDERS, "semif")
        self.new_skill_creation = _norm(
            self.new_skill_creation, AUTHORING_MODES, "deny"
        )
        self.llm_provider = _norm(self.llm_provider, PROVIDERS, "semif")
        self.codegen_provider = _norm(self.codegen_provider, PROVIDERS, "ollama")
        self.packs = _clean_list(self.packs)
        self.gateways = _clean_list(self.gateways)
        self.normalize()

    def normalize(self) -> "Answers":
        """Ensure the always-on base pack is present (idempotent)."""
        if BASE_PACK not in self.packs:
            self.packs.insert(0, BASE_PACK)
        return self

    @classmethod
    def minimal(cls) -> "Answers":
        """The minimum working path: local engine, no authoring, essentials only."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict) -> "Answers":
        """Build answers from a JSON object (an ``--answers FILE`` payload)."""
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in fields})

    def to_dict(self) -> dict:
        return {
            f: getattr(self, f) for f in self.__dataclass_fields__  # type: ignore[attr-defined]
        }


def _norm(value: str, allowed: Sequence[str], default: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in allowed else default


def _clean_list(values: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        token = str(value).strip().lower()
        if token and token not in out:
            out.append(token)
    return out


# --------------------------------------------------------------------------- #
# Interview
# --------------------------------------------------------------------------- #


def _ask_text(
    input_fn: Callable[[str], str],
    output: Callable[[str], None],
    prompt: str,
    default: str = "",
    *,
    secret: bool = False,
) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        raw = input_fn(f"{prompt}{suffix}: ").strip()
    except EOFError:
        raw = ""
    return raw or default


def _ask_choice(
    input_fn: Callable[[str], str],
    output: Callable[[str], None],
    prompt: str,
    options: Sequence[tuple[str, str]],
    default_index: int = 1,
) -> str:
    for index, (_, label) in enumerate(options, 1):
        marker = " (default)" if index == default_index else ""
        output(f"  {index}) {label}{marker}")
    while True:
        raw = _ask_text(input_fn, output, prompt, str(default_index))
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        output(f"  please enter a number between 1 and {len(options)}")


def _ask_multi(
    input_fn: Callable[[str], str],
    output: Callable[[str], None],
    prompt: str,
    options: Sequence[tuple[str, str]],
    default: Iterable[str] = (),
) -> list[str]:
    letters = {chr(ord("a") + i): key for i, (key, _) in enumerate(options)}
    for index, (_, label) in enumerate(options):
        output(f"  {chr(ord('a') + index)}) {label}")
    default = list(default)
    default_str = ",".join(
        chr(ord("a") + i) for i, (key, _) in enumerate(options) if key in default
    )
    raw = _ask_text(input_fn, output, prompt, default_str)
    if not raw:
        return []
    picked: list[str] = []
    for token in raw.replace(" ", "").split(","):
        key = letters.get(token.lower())
        if key and key not in picked:
            picked.append(key)
    return picked


def _ask_provider(
    input_fn: Callable[[str], str],
    output: Callable[[str], None],
    label: str,
    default: str,
    *,
    allow_semif: bool = True,
) -> str:
    options: list[tuple[str, str]] = []
    if allow_semif:
        options.append(("semif", "the loaded decision engine (in-process, no endpoint)"))
    options += [
        ("ollama", "a local ollama endpoint"),
        ("opencode", "the hosted OpenCode Console"),
        ("openai", "another OpenAI-compatible endpoint"),
    ]
    default_index = next(
        (i for i, (key, _) in enumerate(options, 1) if key == default), 1
    )
    output(f"  Which endpoint should {label} use?")
    return _ask_choice(input_fn, output, f"{label} provider", options, default_index)


def _collect_llm(
    input_fn: Callable[[str], str],
    output: Callable[[str], None],
    answers: Answers,
    *,
    label: str,
    allow_semif: bool,
) -> None:
    """Ask for an OpenAI-compatible endpoint and record it on `answers`."""
    provider = _ask_provider(
        input_fn, output, label, answers.llm_provider, allow_semif=allow_semif
    )
    setattr(answers, f"{label}_provider", provider)
    if provider == "semif":
        setattr(answers, f"{label}_model", "")
        setattr(answers, f"{label}_base_url", "")
        setattr(answers, f"{label}_api_key", "")
        return
    if provider == "opencode":
        base_default = "https://opencode.ai/inference/openai/v1"
    elif provider == "ollama":
        base_default = "http://localhost:11434/v1"
    else:
        base_default = ""
    setattr(
        answers,
        f"{label}_base_url",
        _ask_text(input_fn, output, f"{label} base URL", base_default),
    )
    model = ""
    while not model:
        model = _ask_text(input_fn, output, f"{label} model (required)")
        if not model:
            output(f"  {label}.model is required; the agent fails fast without it.")
    setattr(answers, f"{label}_model", model)
    key = _ask_text(
        input_fn, output, f"{label} API key (blank for none)", "", secret=True
    )
    setattr(answers, f"{label}_api_key", key)


def run_interview(
    *,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
    interactive: bool = True,
    prefill: Answers | None = None,
) -> Answers:
    """Run the setup interview and return the answers.

    ``interactive=False`` returns the prefill (or the minimum path) without
    prompting — the ``--yes`` behaviour.
    """
    answers = prefill or Answers.minimal()
    if not interactive:
        return answers

    output("")
    output("== SemIf agent setup ==")
    output(
        "This writes this machine's config. Press Enter to accept the default\n"
        "shown in [brackets]. Nothing is installed that your answers don't need."
    )

    # 1. Decision engine.
    output("")
    output("Decision engine")
    output(
        "  The decision engine routes every request. It runs locally on the\n"
        "  pinned SemIf 4B GGUF by default (no network), or against a remote\n"
        "  Winnow /v1/systemone server."
    )
    answers.engine_provider = _ask_choice(
        input_fn,
        output,
        "engine",
        [
            ("semif", "local SemIf 4B (llama.cpp, pinned GGUF)"),
            ("winnow", "remote Winnow decision server"),
        ],
        1,
    )
    if answers.engine_provider == "winnow":
        answers.winnow_base_url = _ask_text(
            input_fn, output, "  winnow base URL", "http://127.0.0.1:8091"
        )
        model = ""
        while not model:
            model = _ask_text(input_fn, output, "  winnow model (required)")
            if not model:
                output("  engine.winnow.model is required.")
        answers.winnow_model = model
        answers.winnow_api_key = _ask_text(
            input_fn, output, "  winnow API key (blank for none)", "", secret=True
        )

    # 2. Skill authoring.
    output("")
    output("Skill authoring")
    output(
        "  May the agent write brand-new skills when no shipped skill fits?\n"
        "  This needs two endpoints: a small `llm` to name the skill and a\n"
        "  code-capable `codegen` to write its body."
    )
    mode = _ask_choice(
        input_fn,
        output,
        "authoring",
        [
            ("deny", "no — use the shipped skills only (recommended)"),
            ("ask", "ask me before each new skill"),
            ("allow", "yes — write new skills automatically"),
        ],
        1,
    )
    answers.new_skill_creation = mode
    if mode != "deny":
        output("")
        _collect_llm(input_fn, output, answers, label="llm", allow_semif=True)
        output("")
        _collect_llm(input_fn, output, answers, label="codegen", allow_semif=False)

    # 3. Capability packs.
    output("")
    output("Capabilities (skill packs)")
    output(
        "  Essentials (time, timers, calculator) is always included. Enable more\n"
        "  packs by their letter; separate several with commas (e.g. a,c)."
    )
    answers.packs = _ask_multi(
        input_fn,
        output,
        "packs (blank for essentials only)",
        PACK_CHOICES,
        default=[p for p in answers.packs if p != BASE_PACK],
    )

    # 4. Pack-specific connection details.
    if "nextcloud" in answers.packs:
        output("")
        output("Nextcloud connection (blank to fill in later)")
        answers.nextcloud_url = _ask_text(
            input_fn, output, "  Nextcloud URL", answers.nextcloud_url
        )
        answers.nextcloud_username = _ask_text(
            input_fn, output, "  Nextcloud username", answers.nextcloud_username
        )
        answers.nextcloud_app_password = _ask_text(
            input_fn,
            output,
            "  Nextcloud app password",
            answers.nextcloud_app_password,
            secret=True,
        )
    if "outlook" in answers.packs:
        output("")
        output("Outlook / Microsoft 365")
        output(
            "  Register an Entra ID public-client app and paste its application\n"
            "  (client) id; run packs/outlook/scripts/outlook-auth.py once after."
        )
        answers.outlook_client_id = _ask_text(
            input_fn, output, "  Outlook client id", answers.outlook_client_id
        )

    # 5. Gateways.
    output("")
    output("Gateways (how you talk to the agent)")
    output(
        "  You can always use the terminal REPL and dashboard. Optionally reach\n"
        "  the agent over a messenger or the local microphone too."
    )
    answers.gateways = _ask_multi(
        input_fn,
        output,
        "gateways (blank for none)",
        GATEWAY_CHOICES,
        default=answers.gateways,
    )

    # 6. Timezone.
    output("")
    answers.timezone = _ask_text(
        input_fn,
        output,
        "Timezone (IANA name, e.g. America/Chicago; blank for the host zone)",
        answers.timezone,
    )
    return answers.normalize()


# --------------------------------------------------------------------------- #
# Config generation
# --------------------------------------------------------------------------- #


def _set_secret_or_literal(block: dict, key: str, value: str, secret_key: str) -> None:
    """Write ``!secret <secret_key>`` when a value is present, else an empty literal."""
    block[key] = f"!secret {secret_key}" if value else ""


def build_config(answers: Answers, packs: Sequence[SkillPack]) -> dict:
    """The merged config dict for these answers (defaults + overrides)."""
    answers.normalize()
    config = default_config(packs)

    # Decision engine.
    engine = config.setdefault("engine", {})
    engine["provider"] = answers.engine_provider
    if answers.engine_provider == "winnow":
        winnow = engine.setdefault("winnow", {})
        if answers.winnow_base_url:
            winnow["base_url"] = answers.winnow_base_url
        winnow["model"] = answers.winnow_model
        _set_secret_or_literal(
            winnow, "api_key", answers.winnow_api_key, "engine_winnow_api_key"
        )

    # Authoring mode.
    config["new_skill_creation"] = answers.new_skill_creation

    # `llm`: semif reuses the loaded engine; anything else needs an endpoint.
    llm = config.setdefault("llm", {})
    llm["provider"] = answers.llm_provider
    if answers.llm_provider == "semif":
        llm["model"] = ""
        llm["api_key"] = ""
    else:
        if answers.llm_base_url:
            llm["base_url"] = answers.llm_base_url
        llm["model"] = answers.llm_model
        _set_secret_or_literal(llm, "api_key", answers.llm_api_key, "llm_api_key")

    # `codegen`: an empty model means "not configured" (no authoring path).
    codegen = config.setdefault("codegen", {})
    if answers.codegen_model:
        codegen["provider"] = answers.codegen_provider
        if answers.codegen_base_url:
            codegen["base_url"] = answers.codegen_base_url
        codegen["model"] = answers.codegen_model
        _set_secret_or_literal(
            codegen, "api_key", answers.codegen_api_key, "codegen_api_key"
        )
    else:
        codegen["model"] = ""
        codegen["api_key"] = ""

    if answers.timezone:
        config["timezone"] = answers.timezone

    # Packs: enable exactly the chosen ones (essentials always on).
    for pack in packs:
        enabled = pack.name in answers.packs
        if pack.bridges:
            config.setdefault("bridges", {}).setdefault(pack.bridges[0], {})[
                "enabled"
            ] = enabled
        else:
            config.setdefault("packs", {}).setdefault(pack.name, {})["enabled"] = enabled

    # The shared LLM bridge is only useful when a bridge-owning pack is on.
    bridge_packs = [p for p in packs if p.bridges and p.name in answers.packs]
    config.setdefault("bridges", {}).setdefault("llm", {})["enabled"] = bool(
        bridge_packs
    )

    # Nextcloud / Outlook connection details.
    if "nextcloud" in answers.packs:
        nc = config["bridges"].setdefault("nextcloud", {})
        if answers.nextcloud_url:
            nc["url"] = answers.nextcloud_url
        if answers.nextcloud_username:
            nc["username"] = answers.nextcloud_username
        _set_secret_or_literal(
            nc, "app_password", answers.nextcloud_app_password, "nextcloud_app_password"
        )
    if "outlook" in answers.packs:
        config["bridges"].setdefault("outlook", {})["client_id"] = (
            answers.outlook_client_id
        )

    # Gateways.
    gateway = config.setdefault("gateway", {})
    for platform in ("simplex", "lxmf", "voice"):
        gateway.setdefault(platform, {})["enabled"] = platform in answers.gateways
    gateway.setdefault("humanize", {})["enabled"] = bool(answers.gateways)

    return config


def build_secrets(answers: Answers) -> dict:
    """The secrets to store for these answers (only keys that were provided)."""
    secrets: dict[str, str] = {}
    if answers.engine_provider == "winnow" and answers.winnow_api_key:
        secrets["engine_winnow_api_key"] = answers.winnow_api_key
    if answers.llm_provider != "semif" and answers.llm_api_key:
        secrets["llm_api_key"] = answers.llm_api_key
    if answers.codegen_api_key:
        secrets["codegen_api_key"] = answers.codegen_api_key
    if "nextcloud" in answers.packs and answers.nextcloud_app_password:
        secrets["nextcloud_app_password"] = answers.nextcloud_app_password
    return secrets


def write_secrets(root: str | Path, secrets: dict, *, merge: bool = True) -> Path:
    """Write ``secrets.json`` under `root`, merging with any existing entries."""
    path = Path(root) / DEFAULT_SECRETS_FILE
    current = load_secrets(path) if merge else {}
    current.update({k: v for k, v in secrets.items() if v})
    path.write_text(json.dumps(current, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Install plan
# --------------------------------------------------------------------------- #


def build_plan(answers: Answers, packs: Sequence[SkillPack]) -> dict:
    """What bootstrap should install for these answers.

    Drives the conditional stages so a REPL-only host never downloads the
    simplex-chat binary, Playwright, or the voice stack.
    """
    answers.normalize()
    bridge_packs = [
        p.name for p in packs if p.bridges and p.name in answers.packs
    ]
    daemon_gateways = [g for g in answers.gateways if g != "voice"]
    return {
        "engine_provider": answers.engine_provider,
        "authoring": answers.new_skill_creation,
        "packs": list(answers.packs),
        "bridge_packs": bridge_packs,
        "gateways": list(answers.gateways),
        # Platforms the systemd gateway unit runs (voice is foreground-only).
        "gateway_platforms": daemon_gateways,
        "install_engine": answers.engine_provider == "semif",
        "install_bridge_service": bool(bridge_packs),
        "install_gateway_service": bool(daemon_gateways),
        "install_simplex_binary": ("simplex" in answers.packs)
        or ("simplex" in answers.gateways),
        "install_simplex_forward": "simplex" in answers.packs,
        "install_browser": "browser" in answers.packs,
        "install_voice": "voice" in answers.gateways,
    }


def plan_from_config(config: dict, packs: Sequence[SkillPack]) -> dict:
    """Derive the same plan from an already-merged config (a rerun)."""
    gateways = [
        platform
        for platform in ("simplex", "lxmf", "voice")
        if ((config.get("gateway", {}) or {}).get(platform, {}) or {}).get("enabled")
    ]
    enabled = [p.name for p in packs if pack_enabled(config, p)]
    bridge_packs = [p.name for p in packs if p.bridges and p.name in enabled]
    daemon_gateways = [g for g in gateways if g != "voice"]
    engine_provider = str(
        (config.get("engine", {}) or {}).get("provider", "semif") or "semif"
    ).lower()
    return {
        "engine_provider": engine_provider,
        "authoring": str(config.get("new_skill_creation", "deny") or "deny"),
        "packs": enabled,
        "bridge_packs": bridge_packs,
        "gateways": gateways,
        "gateway_platforms": daemon_gateways,
        "install_engine": engine_provider == "semif",
        "install_bridge_service": bool(bridge_packs),
        "install_gateway_service": bool(daemon_gateways),
        "install_simplex_binary": ("simplex" in enabled) or ("simplex" in gateways),
        "install_simplex_forward": "simplex" in enabled,
        "install_browser": "browser" in enabled,
        "install_voice": "voice" in gateways,
    }


# --------------------------------------------------------------------------- #
# Top-level entry used by the CLI
# --------------------------------------------------------------------------- #


def run_setup(
    *,
    root: str | Path,
    path: str | Path,
    packs: Sequence[SkillPack],
    interview: bool = False,
    yes: bool = False,
    answers_file: str | None = None,
    plan_only: bool = False,
    plan_out: str | None = None,
    force: bool = False,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> int:
    """Seed (or re-seed) the split config from an interview, or emit a plan.

    Returns a process exit code. In ``plan_only`` mode it reads the existing
    config and writes the derived plan without touching anything.
    """
    config_path = Path(path)

    if plan_only:
        from .config import load_config_file

        try:
            raw = load_config_file(config_path)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean error
            output(f"could not read {config_path}: {exc}")
            return 1
        plan = plan_from_config(raw, packs)
        _emit_plan(plan, plan_out, output)
        return 0

    if config_path.exists() and not force:
        output(f"{config_path} already exists; leaving it alone (use --force)")
        return 1

    prefill = None
    if answers_file:
        try:
            prefill = Answers.from_dict(json.loads(Path(answers_file).read_text()))
        except (OSError, ValueError) as exc:
            output(f"could not read answers file {answers_file}: {exc}")
            return 1

    if interview and not yes:
        answers = run_interview(
            input_fn=input_fn, output=output, interactive=True, prefill=prefill
        )
    else:
        answers = prefill or Answers.minimal()

    config = build_config(answers, packs)
    written = write_split_config(Path(root), config, packs)
    secrets_path = write_secrets(Path(root), build_secrets(answers), merge=True)
    plan = build_plan(answers, packs)

    output("")
    output("== config written")
    output(f"   {len(written)} files under {Path(root) / 'config'}")
    output(f"   {secrets_path.name} (credentials)")
    _print_summary(answers, output)
    _emit_plan(plan, plan_out, output)
    return 0


def _print_summary(answers: Answers, output: Callable[[str], None]) -> None:
    output(
        "   engine: "
        + (
            "local SemIf 4B"
            if answers.engine_provider == "semif"
            else f"remote Winnow ({answers.winnow_model or '?'})"
        )
    )
    output(f"   authoring: {answers.new_skill_creation}")
    output(f"   packs: {', '.join(answers.packs)}")
    output(f"   gateways: {', '.join(answers.gateways) or 'none (REPL + dashboard)'}")
    if answers.timezone:
        output(f"   timezone: {answers.timezone}")


def _emit_plan(plan: dict, plan_out: str | None, output: Callable[[str], None]) -> None:
    if plan_out:
        Path(plan_out).parent.mkdir(parents=True, exist_ok=True)
        Path(plan_out).write_text(json.dumps(plan, indent=2) + "\n")
    else:
        output("PLAN " + json.dumps(plan))
