"""CLI entrypoint: interactive REPL, scripted JSONL mode, and subcommands.

Run on a host with SemIf + a GGUF + a local OpenAI-compatible server:

    python -m semif_agent.cli run            # REPL
    python -m semif_agent.cli run --script inputs.jsonl
    python -m semif_agent.cli dream          # prediction-observation cost report
    python -m semif_agent.cli skills
    python -m semif_agent.cli status
    python -m semif_agent.cli relabel <id> <outcome>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from .dream import dream as run_dream
from .codegen import CodegenClient
from .console import (
    OPENCODE_BASE_URL,
    ConsoleCodegenClient,
    ConsoleLLMClient,
)
from .decisions import DecisionRequest, Option
from .engine import EngineConfig, EngineUnavailable, SemIfEngine
from .llm import LLMClient
from .log import DecisionLog
from .scheduler import Scheduler
from .skills import tree_summary
from .trace import TraceLog


REPO_ROOT = Path(__file__).resolve().parent.parent

_PATH_KEYS = ("skill_seeds", "skill_bodies", "category_registry", "log", "trace")
_PATH_DEFAULTS = {
    "skill_seeds": "seeds",
    "skill_bodies": "data/skills",
    "category_registry": "data/categories.json",
    "log": "data/decisions.jsonl",
    "trace": "data/runs.jsonl",
}


def _anchor_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _resolve_api_key(cfg: dict) -> str:
    """Service-account key for an authenticated provider.

    A literal `api_key` wins; otherwise `api_key_env` names an environment
    variable (preferred, so the secret never lands in config.json).
    """
    key = cfg.get("api_key") or ""
    if key:
        return str(key)
    env = cfg.get("api_key_env")
    if env:
        return os.environ.get(str(env), "")
    return ""


def _opt_float(value) -> float | None:
    """Config number for an opt-in provider parameter.

    A missing/`null` key yields `None`, which the provider omits from the
    request so the server's model default applies. A present value is coerced
    to `float`.
    """
    return None if value is None else float(value)


def _required_model(cfg: dict, provider: str) -> str:
    """The configured model name, or a clear fatal error.

    There is no default: which model an endpoint serves is deployment-specific,
    so the per-machine config must name it rather than the code guessing a model
    the endpoint may not have. Failing fast turns a missing model into an
    actionable config error instead of a request to a model that isn't there.
    """
    model = str(cfg.get("model") or "").strip()
    if not model:
        raise SystemExit(
            f"config.json: {provider}.model is required "
            f"(set it to a model your {provider} endpoint serves)"
        )
    return model


def _provider_endpoint(cfg: dict, default_base: str) -> dict:
    """Endpoint kwargs shared by the llm/codegen clients.

    `provider` selects the class (`opencode` → the Console client) and the base
    URL default: an explicit `base_url` always wins, else the Console URL for
    `opencode` and `default_base` (ollama) otherwise. Auth comes from
    `api_key`/`api_key_env`; `extra_headers` passes through for custom schemes.
    """
    provider = str(cfg.get("provider", "ollama") or "ollama").lower()
    base_url = cfg.get("base_url") or (
        OPENCODE_BASE_URL if provider == "opencode" else default_base
    )
    return {
        "base_url": base_url,
        "api_key": _resolve_api_key(cfg),
        "extra_headers": cfg.get("extra_headers") or None,
        "user_agent": cfg.get("user_agent"),
    }


def load_config(path: str = "config.json") -> dict:
    """Load config.json and anchor its runtime paths to the checkout root.

    config.json is the per-machine user config and is required: bootstrap.sh
    seeds it from config.example.json once, and only a human edits it. There is
    no in-code fallback — a missing config.json is a hard error.

    Path-valued keys (`skill_seeds`, `skill_bodies`, `category_registry`, `log`,
    `trace`) resolve against REPO_ROOT, not the process cwd, so a fresh clone
    loads its committed seeds and writes runtime artifacts into the checkout no
    matter where the CLI is invoked from. Absolute values pass through untouched.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise SystemExit(
            f"config.json not found at {config_path} — run scripts/bootstrap.sh "
            f"to seed it from config.example.json, then configure it"
        )
    config = json.loads(config_path.read_text())
    for key in _PATH_KEYS:
        config[key] = _anchor_path(config.get(key) or _PATH_DEFAULTS[key])
    return config


def load_pins(path: str | None = None) -> dict:
    """Load the committed pin manifest (pins.json) that owns every external ref.

    Pins are code, not per-machine config: the SemIf engine commit, GGUF
    url+sha256, HF tokenizer revision, and simplex-chat version/url+sha256 live
    in pins.json so a `git pull` bump propagates on the next bootstrap run
    instead of being frozen in each machine's config.json.
    """
    pins_path = Path(path) if path else REPO_ROOT / "pins.json"
    if not pins_path.is_file():
        raise SystemExit(f"pins.json not found at {pins_path}")
    return json.loads(pins_path.read_text())


def build_engine_config(config: dict, pins: dict | None = None) -> EngineConfig:
    """Engine settings for a run: per-machine tuning from config.json, pinned
    refs from pins.json.

    `engine.backend`/`context_tokens`/`threads` are per-machine config. The
    tokenizer source/revision and the GGUF are pins; the GGUF path is derived
    from the pin's filename under `.runtime/models/`, unless config.json sets an
    explicit `engine.gguf` override.
    """
    pins = load_pins() if pins is None else pins
    eng = config.get("engine", {}) or {}
    pin_eng = pins.get("engine", {}) or {}
    override = eng.get("gguf")
    gguf = _anchor_path(override) if override else str(
        REPO_ROOT / ".runtime" / "models" / os.path.basename(pin_eng.get("gguf_url", ""))
    )
    return EngineConfig(
        backend=eng.get("backend", "llamacpp"),
        source=pin_eng.get("source", ""),
        revision=pin_eng.get("revision", ""),
        gguf=gguf,
        context_tokens=int(eng.get("context_tokens", 4096)),
        threads=eng.get("threads"),
    )


def build_codegen_client(config: dict) -> CodegenClient:
    """Build the codegen client from the `codegen` config block.

    `provider` selects the class (`opencode` -> the Console client, which adds
    bearer auth and a User-Agent) and the endpoint/auth come from
    `_provider_endpoint`. Shared by `build_scheduler` and the integration tests
    so both exercise the configured provider rather than assuming ollama.
    """
    codegen_cfg = config.get("codegen", {}) or {}
    deg_cfg = codegen_cfg.get("degeneration", {}) or {}
    provider = str(codegen_cfg.get("provider", "ollama") or "ollama").lower()
    client_cls = ConsoleCodegenClient if provider == "opencode" else CodegenClient
    endpoint = _provider_endpoint(
        codegen_cfg,
        config.get("llm", {}).get("base_url", "http://localhost:11434/v1"),
    )
    return client_cls(
        base_url=endpoint["base_url"],
        model=_required_model(codegen_cfg, "codegen"),
        timeout=float(codegen_cfg.get("timeout", 1200.0)),
        stream=bool(codegen_cfg.get("stream", False)),
        idle_warn=float(codegen_cfg.get("idle_warn", 60.0)),
        idle_timeout=float(codegen_cfg.get("idle_timeout", 180.0)),
        context_window=float(codegen_cfg.get("context_window", 0.0)),
        smart_limit=int(codegen_cfg.get("smart_limit", 250000)),
        warn_limit=int(codegen_cfg.get("warn_limit", 500000)),
        max_fill_ratio=float(codegen_cfg.get("max_fill_ratio", 0.9)),
        warn_fill_ratio=float(codegen_cfg.get("warn_fill_ratio", 0.7)),
        max_output=float(codegen_cfg.get("max_output", 0.85)),
        chars_per_token=float(codegen_cfg.get("chars_per_token", 4.0)),
        degeneration_interval=int(deg_cfg.get("interval", 8000)),
        degeneration_window=int(deg_cfg.get("window", 2000)),
        degeneration_min_chars=int(deg_cfg.get("min_chars", 4000)),
        temperature=_opt_float(codegen_cfg.get("temperature")),
        top_p=_opt_float(codegen_cfg.get("top_p")),
        presence_penalty=_opt_float(codegen_cfg.get("presence_penalty")),
        frequency_penalty=_opt_float(codegen_cfg.get("frequency_penalty")),
        max_attempts=int(codegen_cfg.get("max_attempts", 3)),
        api_key=endpoint["api_key"],
        extra_headers=endpoint["extra_headers"],
        user_agent=endpoint["user_agent"],
    )


def build_scheduler(config: dict) -> tuple[Scheduler, dict]:
    engine = SemIfEngine(build_engine_config(config))
    llm_cfg = config.get("llm", {}) or {}
    llm_provider = str(llm_cfg.get("provider", "ollama") or "ollama").lower()
    llm_cls = ConsoleLLMClient if llm_provider == "opencode" else LLMClient
    llm_endpoint = _provider_endpoint(llm_cfg, "http://localhost:11434/v1")
    llm = llm_cls(
        base_url=llm_endpoint["base_url"],
        model=_required_model(llm_cfg, "llm"),
        timeout=float(llm_cfg.get("timeout", 600.0)),
        stream=bool(llm_cfg.get("stream", False)),
        idle_warn=float(llm_cfg.get("idle_warn", 30.0)),
        idle_timeout=float(llm_cfg.get("idle_timeout", 120.0)),
        context_window=float(llm_cfg.get("context_window", 0.0)),
        temperature=_opt_float(llm_cfg.get("temperature")),
        top_p=_opt_float(llm_cfg.get("top_p")),
        presence_penalty=_opt_float(llm_cfg.get("presence_penalty")),
        frequency_penalty=_opt_float(llm_cfg.get("frequency_penalty")),
        disable_thinking=bool(llm_cfg.get("disable_thinking", True)),
        api_key=llm_endpoint["api_key"],
        extra_headers=llm_endpoint["extra_headers"],
        user_agent=llm_endpoint["user_agent"],
    )
    log = DecisionLog(config.get("log", "data/decisions.jsonl"))
    trace = TraceLog(config.get("trace", "data/runs.jsonl"))
    codegen_cfg = config.get("codegen", {})
    deg_cfg = codegen_cfg.get("degeneration", {}) or {}
    codegen = build_codegen_client(config)

    def make_degeneration_check(run_id: str):
        """2-option SemIf decision (continue/stop) that aborts a degenerating
        codegen stream: P(stop) >= threshold returns an abort reason. Recorded
        as a trace-only event (kind `codegen`, with probs) — never in the
        decision log. Disabled when no engine (EngineUnavailable at call time
        degrades to "keep going"), which keeps CodegenClient standalone pure."""
        if not bool(deg_cfg.get("enabled", True)):
            return None
        threshold = float(deg_cfg.get("threshold", 0.9))

        def check(recent: str) -> str | None:
            decision = DecisionRequest(
                state=f"[codegen {codegen.model}] {recent[-2000:]}",
                question="Is this skill-body generation degenerating (looping or repeating instead of converging)?",
                options=[
                    Option("continue", "Continue; it is still making progress."),
                    Option("stop", "Stop; it is degenerating."),
                ],
            )
            try:
                result = engine.call(decision)
            except EngineUnavailable:
                return None
            trace.append(
                "codegen",
                run_id,
                state=decision.state,
                question=decision.question,
                options=[o.id for o in decision.options],
                selected=result.selected,
                probs=result.probs,
                stop_prob=result.prob("stop"),
            )
            if result.prob("stop") >= threshold:
                return (
                    f"P(stop)={result.prob('stop'):.2f} >= threshold {threshold}"
                )
            return None

        return check

    def make_regen_decision(run_id: str):
        """2-option SemIf decision on a failing skill test: which artifact to
        regenerate (the body — which carries the contract — or the test; fixture
        data lives inside the test, so a fixture fix is a test regen). Recorded
        as a trace-only event (kind `codegen_regen`) — never in the decision
        log. Engine unavailable at call time degrades to `regen_test`."""
        def decide(reason: str) -> str:
            decision = DecisionRequest(
                state=f"[testgen {codegen.model}] skill test failed. {reason[-1200:]}",
                question="The auto-run test failed. What should be regenerated to fix it?",
                options=[
                    Option("regen_code", "Regenerate the skill body code."),
                    Option("regen_test", "Regenerate the test only."),
                ],
            )
            try:
                result = engine.call(decision)
            except EngineUnavailable:
                return "regen_test"
            trace.append(
                "codegen_regen",
                run_id,
                state=decision.state,
                question=decision.question,
                options=[o.id for o in decision.options],
                selected=result.selected,
                probs=result.probs,
                reason=reason[-2000:],
            )
            return result.selected

        return decide

    scheduler = Scheduler(
        engine=engine,
        llm=llm,
        log=log,
        config=config,
        tau=float(config.get("tau", 0.6)),
        max_reentries=int(config.get("max_reentries", 3)),
        trace=trace,
        codegen=codegen,
        degeneration_check_factory=make_degeneration_check,
        regen_decision_factory=make_regen_decision,
    )
    return scheduler, config


def try_warm(scheduler: Scheduler) -> str:
    try:
        scheduler.engine._ensure_loaded()
        return "decision engine loaded."
    except EngineUnavailable as exc:
        return f"decision engine unavailable: {exc}"


def _answer_questions(scheduler: Scheduler) -> None:
    """Ask any deferred authoring/repair questions at the REPL."""
    while True:
        pending = scheduler.pending_questions()
        if not pending:
            return
        question = pending[0]
        answer = input(
            f"[{question['skill']}] {question['question']}\n"
            "answer (empty to skip): "
        )
        status, detail = scheduler.answer_question(question["id"], answer)
        print(f"[{status}] {detail}")


def _print_repairs(scheduler: Scheduler) -> None:
    for offer in scheduler.pending_repairs():
        print(
            f"[repair] {offer['id']}  {offer['category']}.{offer['skill']}  "
            f"suggested={offer['selected']}  {offer['failure'][:120]}"
        )


def _answer_approvals(scheduler: Scheduler) -> None:
    """Ask any pending creation approvals at the REPL.

    Only prompts when `creation_approval` is on (the scheduler posts nothing
    otherwise). An empty answer denies.
    """
    while True:
        pending = scheduler.pending_approvals()
        if not pending:
            return
        item = pending[0]
        target = (
            item["category"]
            if item["kind"] == "category"
            else f"{item['category']}.{item['skill']}"
        )
        answer = input(
            f"[approval] create new {item['kind']} {target}: {item['description']}\n"
            "approve? (y/N): "
        ).strip().lower()
        status, detail = scheduler.answer_approval(
            item["id"], answer in ("y", "yes")
        )
        print(f"[{status}] {detail}")


def _print_approvals(scheduler: Scheduler) -> None:
    for item in scheduler.pending_approvals():
        target = (
            item["category"]
            if item["kind"] == "category"
            else f"{item['category']}.{item['skill']}"
        )
        print(f"[approval] {item['id']}  new {item['kind']} {target}: {item['description']}")


def _fatal_exit(scheduler: Scheduler) -> int | None:
    """If the scheduler hit a fatal engine failure, print it and return non-zero.

    The decision engine is always real; if it goes unavailable mid-run the app
    cannot make decisions and must exit rather than degrade.
    """
    if scheduler.fatal is None:
        return None
    print(scheduler.fatal, file=sys.stderr)
    return 1


def _start_timer_printer(scheduler: Scheduler) -> None:
    """Print fired timers/alarms while the REPL is blocked on `input()`.

    A background daemon thread polls the scheduler's timer service and prints
    each fired notification, so a timer set during a session is delivered even
    when the prompt is idle. Timers are in-process: they die with this process.
    """

    def loop() -> None:
        while True:
            time.sleep(0.5)
            for fired in scheduler.timers.drain():
                print(f"\n[timer] {fired.message}")

    threading.Thread(target=loop, name="timer-printer", daemon=True).start()


def repl(scheduler: Scheduler, config: dict) -> int:
    scheduler.defer_questions = True
    _start_timer_printer(scheduler)
    print(try_warm(scheduler))
    print(
        "type a request, or one of: busy <text> | idle | status | skills | dream | "
        "relabel <id> <outcome> | restart <category> <skill> | "
        "repairs | repair <offer-id> [action] | approvals | quit"
    )
    while True:
        _answer_questions(scheduler)
        _answer_approvals(scheduler)
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        lower = line.lower()
        if lower in ("quit", "exit"):
            break
        if lower == "status":
            print(scheduler.status())
            continue
        if lower == "skills":
            print(tree_summary(scheduler.tree))
            continue
        if lower == "dream":
            print(run_dream(scheduler.log).render())
            continue
        if lower == "repairs":
            _print_repairs(scheduler)
            continue
        if lower == "approvals":
            _print_approvals(scheduler)
            continue
        if lower.startswith(("repair ", "/repair ")):
            parts = line.lstrip("/").split()
            if len(parts) not in (2, 3):
                print("usage: repair <offer-id> [retry|repair_skill|ask_user|no_repair]")
                continue
            status, detail = scheduler.resolve_repair(
                parts[1], parts[2] if len(parts) == 3 else None
            )
            print(f"[{status}] {detail}")
            _answer_questions(scheduler)
            _answer_approvals(scheduler)
            for result_status, result_detail in scheduler.run_queue():
                print(f"[{result_status}] {result_detail}")
            continue
        if lower.startswith("relabel "):
            parts = line.split()
            if len(parts) != 3:
                print("usage: relabel <id> <outcome>")
                continue
            ok = scheduler.log.relabel(parts[1], parts[2])
            print("relabeled." if ok else f"no row with id {parts[1]}")
            continue
        if lower.startswith(("restart ", "/restart ")):
            parts = line.lstrip("/").split()
            if len(parts) != 3:
                print("usage: restart <category> <skill>")
                continue
            status, detail = scheduler.restart_skill(parts[1], parts[2])
            print(f"[{status}] {detail}")
            continue
        if lower == "idle":
            scheduler.idle()
            print("current process cleared.")
            continue
        if lower.startswith("busy "):
            scheduler.busy(line[5:].strip())
            print("current process set (busy).")
            continue
        status, detail = scheduler.submit(line)
        print(f"[{status}] {detail}")
        if _fatal_exit(scheduler) is not None:
            return 1
        while scheduler.pending is not None:
            hint = "" if scheduler.pending.pre_act else " [leave empty to skip]"
            answer = input(f"{scheduler.pending.question}{hint} ")
            status, detail = scheduler.answer(answer)
            print(f"[{status}] {detail}")
            if _fatal_exit(scheduler) is not None:
                return 1
        for result_status, result_detail in scheduler.run_queue():
            print(f"[{result_status}] {result_detail}")
        if _fatal_exit(scheduler) is not None:
            return 1
        _answer_questions(scheduler)
        _answer_approvals(scheduler)
        _print_repairs(scheduler)
    return _fatal_exit(scheduler) or 0


def scripted(scheduler: Scheduler, path: str) -> int:
    print(try_warm(scheduler))
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        status, detail = scheduler.submit(str(row["text"]), source=row.get("source", "scripted"))
        print(f"[{status}] {detail}")
        if _fatal_exit(scheduler) is not None:
            return 1
    for result_status, result_detail in scheduler.run_queue():
        print(f"[{result_status}] {result_detail}")
    return _fatal_exit(scheduler) or 0


def _gateway_address_callback(adapter):
    """Return an `on_connected` coroutine that prints the bot's contact link once.

    A human cannot reach the gateway without the bot's SimpleX contact address,
    and the only thing that surfaced it before was `scripts/simplex-address.py`
    run once at provision time. Print it on connect, from the operator front end
    (the CLI), through the daemon the gateway already owns — show-or-create, no
    second connection, no new transport surface. The lookup is best-effort: a
    failure is reported once (to stderr) and retried on the next connect, and it
    never takes the socket down.
    """
    printed = False
    warned = False

    async def _announce() -> None:
        nonlocal printed, warned
        if printed:
            return
        try:
            link = await adapter.daemon.address()
        except Exception as exc:
            if not warned:
                warned = True
                print(
                    f"gateway simplex: could not read contact link: {exc}",
                    file=sys.stderr,
                )
            return
        short = (link or {}).get("short_link") or ""
        full = (link or {}).get("full_link") or ""
        if not (short or full):
            if not warned:
                warned = True
                print("gateway simplex: no contact link available", file=sys.stderr)
            return
        printed = True
        print(f"gateway simplex contact link: {short}")
        if full:
            print(full)

    return _announce


def _lxmf_address_callback(adapter):
    """Return a sync `on_connected` hook that prints the bot's LXMF address once.

    Unlike SimpleX there is no daemon round-trip: the router knows its delivery
    address as soon as it is registered, so the hook just prints it once.
    """
    printed = False

    def _announce() -> None:
        nonlocal printed
        if printed:
            return
        address = getattr(adapter.daemon, "address", "") or ""
        if not address:
            print("gateway lxmf: no address available", file=sys.stderr)
            return
        printed = True
        print(f"gateway lxmf address: {address}")

    return _announce


def _attach_address_announcer(adapter) -> None:
    """Wire the platform's startup address announcement onto its daemon.

    A human cannot reach a gateway without its address. Print it once on
    connect, from the operator front end (the CLI), through the transport the
    adapter already owns — no second connection, no new transport surface.
    """
    if adapter.name == "simplex":
        adapter.daemon.on_connected = _gateway_address_callback(adapter)
    elif adapter.name == "lxmf":
        adapter.daemon.on_connected = _lxmf_address_callback(adapter)


def _build_gateway_adapter(platform: str, cfg: dict, trace):
    """Construct one adapter from its config, or None for an unknown platform."""
    if platform == "simplex":
        from .gateway.simplex import SimplexAdapter

        return SimplexAdapter(cfg, trace=trace)
    if platform == "lxmf":
        from .gateway.lxmf import LxmfAdapter

        cfg = dict(cfg or {})
        # Contain Reticulum's config + identity under the checkout's .runtime/.
        cfg.setdefault("config_dir", str(REPO_ROOT / ".runtime" / "lxmf" / "reticulum"))
        cfg.setdefault("storage_path", str(REPO_ROOT / ".runtime" / "lxmf" / "router"))
        return LxmfAdapter(cfg, trace=trace)
    return None


def _resolve_platforms(requested: str, gateway_cfg: dict) -> list[str]:
    """Turn `--platform` into a list of platform names.

    `all`/`*` means every **enabled** gateway block in config; a
    comma-separated list is taken as-is (unknown names are kept so the caller
    can report them).
    """
    requested = (requested or "simplex").strip()
    if requested in ("all", "*"):
        return [
            name
            for name, block in (gateway_cfg or {}).items()
            if isinstance(block, dict)
            and not name.startswith("_")
            and block.get("enabled", False)
        ]
    return [name.strip() for name in requested.split(",") if name.strip()]


def run_gateway(
    scheduler: Scheduler,
    config: dict,
    platform: str = "simplex",
    serve_dashboard: bool = False,
) -> int:
    """Run one or more messenger gateways in the foreground.

    All enabled adapters share a single scheduler/process (see AGENTS.md: never
    run two scheduler processes over one skill store), so each gets its own
    transport thread but they route through one `GatewayService`.
    """
    gateway_cfg = config.get("gateway", {}) or {}
    platforms = _resolve_platforms(platform, gateway_cfg)

    adapters: dict = {}
    for name in platforms:
        cfg = gateway_cfg.get(name) or {}
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            print(f"gateway.{name} is not enabled in config")
            continue
        adapter = _build_gateway_adapter(name, cfg, scheduler.trace)
        if adapter is None:
            print(f"unknown gateway platform {name!r}")
            continue
        ok, hint = adapter.check_requirements()
        if not ok:
            print(f"gateway {name} unavailable: {hint}")
            continue
        _attach_address_announcer(adapter)
        adapters[name] = adapter

    if not adapters:
        print("no gateway platforms enabled/available")
        return 1

    # Reticulum installs process signal handlers, which only works on the main
    # thread. Run the LXMF transport's main-thread setup (Reticulum + router)
    # here before the per-adapter transport threads start; `run()` reuses it.
    for name in list(adapters):
        prepare = getattr(adapters[name].daemon, "prepare", None)
        if callable(prepare):
            try:
                prepare()
            except Exception as exc:
                print(f"gateway {name} unavailable: {exc}", file=sys.stderr)
                del adapters[name]

    if not adapters:
        print("no gateway platforms enabled/available")
        return 1

    from .gateway.service import GatewayService

    scheduler.defer_questions = True
    service = GatewayService(scheduler, adapters, config=gateway_cfg)
    scheduler.on_request_requeued = service.on_request_requeued
    service.start()

    if serve_dashboard:
        from .dashboard import serve

        dash_cfg = config.get("dashboard", {}) or {}
        threading.Thread(
            target=serve,
            args=(scheduler,),
            kwargs={
                "port": int(dash_cfg.get("port", 8765)),
                "host": str(dash_cfg.get("host", "127.0.0.1")),
            },
            daemon=True,
        ).start()

    threads: list[threading.Thread] = []
    for name, adapter in adapters.items():

        def _run(adapter=adapter, name=name) -> None:
            try:
                adapter.run(
                    lambda msg, _name=name: service.handle_inbound(
                        msg, platform=_name
                    ),
                    service.outbound_for(name),
                )
            except Exception as exc:  # a dead transport must not kill the rest
                print(f"gateway {name} stopped: {exc}", file=sys.stderr)

        thread = threading.Thread(target=_run, name=f"gateway-{name}", daemon=True)
        thread.start()
        threads.append(thread)
        endpoint = getattr(adapter, "ws_url", "") or getattr(adapter, "config_dir", "")
        print(f"gateway {name} listening ({endpoint}) (Ctrl-C to stop)")

    try:
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        for adapter in adapters.values():
            try:
                adapter.close()
            except Exception:
                pass
        for thread in threads:
            thread.join(timeout=2.0)
    return 0


def run_bridge(scheduler: Scheduler, config: dict, name: str | None = None) -> int:
    """Run standalone third-party API bridges (SimpleX first).

    Bridges are their own processes so generated skills never touch a system's
    native protocol and the command gateway never grows read/send surface. See
    `semif_agent.bridges`.
    """
    from .bridges.registry import run_bridges

    names = [name] if name else None
    return run_bridges(
        config, names=names, trace=scheduler.trace, llm=scheduler.llm
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else list(sys.argv[1:])
    config = load_config(_extract_config_path(argv))
    parser = argparse.ArgumentParser(prog="semif-agent")
    parser.add_argument("--config", default="config.json")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="interactive REPL or scripted input")
    run_p.add_argument("--script", default=None, help="JSONL file of {\"text\": ...} rows")

    sub.add_parser("dream", help="compute the prediction-observation cost report")
    sub.add_parser("skills", help="list the skill tree")
    sub.add_parser("status", help="show current process and queue")

    relabel_p = sub.add_parser("relabel", help="human override of a decision label")
    relabel_p.add_argument("id")
    relabel_p.add_argument("outcome")

    dash_p = sub.add_parser("dashboard", help="run the local browser dashboard")
    dash_p.add_argument("--port", type=int, default=int(config.get("dashboard", {}).get("port", 8765)))
    dash_p.add_argument(
        "--host",
        default=str(config.get("dashboard", {}).get("host", "127.0.0.1")),
        help="bind address (0.0.0.0 to expose on the LAN)",
    )
    dash_p.add_argument(
        "--replay",
        action="store_true",
        help="replay mode: do not warm the decision engine",
    )

    gw_p = sub.add_parser("gateway", help="run the messenger gateway (SimpleX, LXMF)")
    gw_p.add_argument(
        "--platform",
        default="simplex",
        help="gateway platform(s): simplex, lxmf, a comma-separated list, or all (default simplex)",
    )
    gw_p.add_argument(
        "--dashboard",
        action="store_true",
        help="also serve the browser dashboard from the same process",
    )

    bridge_p = sub.add_parser(
        "bridge", help="run standalone third-party API bridges (SimpleX first)"
    )
    bridge_p.add_argument(
        "--name",
        default=None,
        help="bridge service to run (default: all enabled)",
    )

    args = parser.parse_args(argv)
    scheduler, config = build_scheduler(config)

    if args.command == "run":
        if args.script:
            return scripted(scheduler, args.script)
        return repl(scheduler, config)
    elif args.command == "dream":
        print(run_dream(scheduler.log).render())
    elif args.command == "skills":
        print(tree_summary(scheduler.tree))
    elif args.command == "status":
        print(scheduler.status())
    elif args.command == "relabel":
        ok = scheduler.log.relabel(args.id, args.outcome)
        print("relabeled." if ok else f"no row with id {args.id}")
    elif args.command == "dashboard":
        from .dashboard import serve

        scheduler.defer_questions = True
        if args.replay:
            print("replay mode: reading decision log + trace; engine not warmed.")
        serve(scheduler, port=args.port, host=args.host)
    elif args.command == "gateway":
        return run_gateway(
            scheduler,
            config,
            platform=args.platform,
            serve_dashboard=args.dashboard,
        )
    elif args.command == "bridge":
        return run_bridge(scheduler, config, name=args.name)
    else:
        parser.print_help()
    return 0


def _extract_config_path(argv: list[str]) -> str:
    """Pull --config out of argv before the full parser runs, so the dashboard
    subcommand can read dashboard.port from config for its default."""
    for index, arg in enumerate(argv):
        if arg == "--config" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return "config.json"


if __name__ == "__main__":
    sys.exit(main())