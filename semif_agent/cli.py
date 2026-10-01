"""CLI entrypoint: interactive REPL, scripted JSONL mode, and subcommands.

Run on the box with SemIf + a GGUF + a local OpenAI-compatible server:

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
import sys
from pathlib import Path

from .dream import dream as run_dream
from .codegen import CodegenClient
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


def load_config(path: str = "config.json") -> dict:
    """Load config.json and anchor its runtime paths to the checkout root.

    Path-valued keys (`skill_seeds`, `skill_bodies`, `category_registry`, `log`,
    `trace`) resolve against REPO_ROOT, not the process cwd, so a fresh clone
    loads its committed seeds and writes runtime artifacts into the checkout no
    matter where the CLI is invoked from. Absolute values pass through untouched.
    """
    config = json.loads(Path(path).read_text())
    for key in _PATH_KEYS:
        config[key] = _anchor_path(config.get(key) or _PATH_DEFAULTS[key])
    return config


def build_scheduler(config: dict) -> tuple[Scheduler, dict]:
    engine = SemIfEngine(
        EngineConfig(
            backend=config.get("engine", {}).get("backend", "llamacpp"),
            source=config.get("engine", {}).get("source", ""),
            revision=config.get("engine", {}).get("revision", ""),
            gguf=config.get("engine", {}).get("gguf", ""),
            context_tokens=int(config.get("engine", {}).get("context_tokens", 4096)),
            threads=config.get("engine", {}).get("threads"),
        )
    )
    llm_cfg = config.get("llm", {}) or {}
    llm = LLMClient(
        base_url=llm_cfg.get("base_url", "http://localhost:11434/v1"),
        model=llm_cfg.get("model", "qwen2.5:3b"),
        timeout=float(llm_cfg.get("timeout", 600.0)),
        stream=bool(llm_cfg.get("stream", False)),
        idle_warn=float(llm_cfg.get("idle_warn", 30.0)),
        idle_timeout=float(llm_cfg.get("idle_timeout", 120.0)),
        context_window=float(llm_cfg.get("context_window", 0.0)),
        temperature=float(llm_cfg.get("temperature", 0.2)),
        top_p=float(llm_cfg.get("top_p", 0.9)),
        presence_penalty=float(llm_cfg.get("presence_penalty", 0.0)),
        frequency_penalty=float(llm_cfg.get("frequency_penalty", 0.0)),
        disable_thinking=bool(llm_cfg.get("disable_thinking", True)),
    )
    log = DecisionLog(config.get("log", "data/decisions.jsonl"))
    trace = TraceLog(config.get("trace", "data/runs.jsonl"))
    codegen_cfg = config.get("codegen", {})
    deg_cfg = codegen_cfg.get("degeneration", {}) or {}
    codegen = CodegenClient(
        base_url=codegen_cfg.get(
            "base_url", config.get("llm", {}).get("base_url", "http://localhost:11434/v1")
        ),
        model=codegen_cfg.get("model", "qwen38-iq3s"),
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
        temperature=float(codegen_cfg.get("temperature", 0.7)),
        top_p=float(codegen_cfg.get("top_p", 0.85)),
        presence_penalty=float(codegen_cfg.get("presence_penalty", 1.5)),
        frequency_penalty=float(codegen_cfg.get("frequency_penalty", 0.2)),
        max_attempts=int(codegen_cfg.get("max_attempts", 3)),
    )

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
        """3-option SemIf decision on a failing skill test: which artifact to
        regenerate (code, contract, or test — fixture data lives inside the
        test, so a fixture fix is a test regen). Recorded as a trace-only event
        (kind `codegen_regen`) — never in the decision log. Engine unavailable
        at call time degrades to `regen_test`."""
        def decide(reason: str) -> str:
            decision = DecisionRequest(
                state=f"[testgen {codegen.model}] skill test failed. {reason[-1200:]}",
                question="The auto-run test failed. What should be regenerated to fix it?",
                options=[
                    Option("regen_code", "Regenerate the skill body code."),
                    Option("regen_test", "Regenerate the test only."),
                    Option("regen_contract", "Regenerate the data contract."),
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


def _fatal_exit(scheduler: Scheduler) -> int | None:
    """If the scheduler hit a fatal engine failure, print it and return non-zero.

    The decision engine is always real; if it goes unavailable mid-run the app
    cannot make decisions and must exit rather than degrade.
    """
    if scheduler.fatal is None:
        return None
    print(scheduler.fatal, file=sys.stderr)
    return 1


def repl(scheduler: Scheduler, config: dict) -> int:
    scheduler.defer_questions = True
    print(try_warm(scheduler))
    print(
        "type a request, or one of: busy <text> | idle | status | skills | dream | "
        "relabel <id> <outcome> | restart <category> <skill> | "
        "repairs | repair <offer-id> [action] | quit"
    )
    while True:
        _answer_questions(scheduler)
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
            answer = input(f"{scheduler.pending.question} ")
            status, detail = scheduler.answer(answer)
            print(f"[{status}] {detail}")
            if _fatal_exit(scheduler) is not None:
                return 1
        for result_status, result_detail in scheduler.run_queue():
            print(f"[{result_status}] {result_detail}")
        if _fatal_exit(scheduler) is not None:
            return 1
        _answer_questions(scheduler)
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


def run_gateway(
    scheduler: Scheduler,
    config: dict,
    platform: str = "simplex",
    serve_dashboard: bool = False,
) -> int:
    """Run the messenger gateway in the foreground (SimpleX first)."""
    cfg = (config.get("gateway", {}) or {}).get(platform, {}) or {}
    if not cfg.get("enabled", False):
        print(f"gateway.{platform} is not enabled in config")
        return 1

    if platform == "simplex":
        from .gateway.simplex import SimplexAdapter

        adapter = SimplexAdapter(cfg, trace=scheduler.trace)
    else:
        print(f"unknown gateway platform {platform!r}")
        return 1

    ok, hint = adapter.check_requirements()
    if not ok:
        print(f"gateway {platform} unavailable: {hint}")
        return 1

    from .gateway.service import GatewayService

    scheduler.defer_questions = True
    service = GatewayService(scheduler, adapter, config=cfg)
    scheduler.on_request_requeued = service.on_request_requeued
    service.start()

    if serve_dashboard:
        import threading

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

    print(f"gateway {platform} listening on {adapter.ws_url} (Ctrl-C to stop)")
    try:
        adapter.run(service.handle_inbound, service.outbound)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        adapter.close()
    return 0


def run_bridge(scheduler: Scheduler, config: dict, name: str | None = None) -> int:
    """Run standalone third-party API bridges (SimpleX first).

    Bridges are their own processes so generated skills never touch a system's
    native protocol and the command gateway never grows read/send surface. See
    `semif_agent.bridges`.
    """
    from .bridges.registry import run_bridges

    names = [name] if name else None
    return run_bridges(config, names=names, trace=scheduler.trace)


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

    gw_p = sub.add_parser("gateway", help="run the messenger gateway (SimpleX)")
    gw_p.add_argument("--platform", default="simplex", help="gateway platform (default simplex)")
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