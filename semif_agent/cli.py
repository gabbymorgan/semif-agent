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
from .engine import EngineConfig, EngineUnavailable, SemIfEngine
from .llm import LLMClient
from .log import DecisionLog
from .scheduler import Scheduler
from .skills import build_skills, build_tree, tree_summary


def load_config(path: str = "config.json") -> dict:
    return json.loads(Path(path).read_text())


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
    llm = LLMClient(
        base_url=config.get("llm", {}).get("base_url", "http://localhost:11434/v1"),
        model=config.get("llm", {}).get("model", "qwen2.5:3b"),
    )
    log = DecisionLog(config.get("log", "data/decisions.jsonl"))
    scheduler = Scheduler(
        engine=engine,
        llm=llm,
        log=log,
        config=config,
        tau=float(config.get("tau", 0.6)),
        max_reentries=int(config.get("max_reentries", 3)),
    )
    return scheduler, config


def try_warm(scheduler: Scheduler) -> str:
    try:
        scheduler.engine._ensure_loaded()
        return "decision engine loaded."
    except EngineUnavailable as exc:
        return f"decision engine unavailable: {exc}"


def repl(scheduler: Scheduler, config: dict) -> None:
    print(try_warm(scheduler))
    print("type a request, or one of: busy <text> | idle | status | skills | dream | relabel <id> <outcome> | quit")
    while True:
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
            print(tree_summary(build_tree(build_skills(config))))
            continue
        if lower == "dream":
            print(run_dream(scheduler.log).render())
            continue
        if lower.startswith("relabel "):
            parts = line.split()
            if len(parts) != 3:
                print("usage: relabel <id> <outcome>")
                continue
            ok = scheduler.log.relabel(parts[1], parts[2])
            print("relabeled." if ok else f"no row with id {parts[1]}")
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


def scripted(scheduler: Scheduler, path: str) -> None:
    print(try_warm(scheduler))
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        status, detail = scheduler.submit(str(row["text"]), source=row.get("source", "scripted"))
        print(f"[{status}] {detail}")


def main(argv: list[str] | None = None) -> int:
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

    args = parser.parse_args(argv)
    config = load_config(args.config)
    scheduler, config = build_scheduler(config)

    if args.command == "run":
        if args.script:
            scripted(scheduler, args.script)
        else:
            repl(scheduler, config)
    elif args.command == "dream":
        print(run_dream(scheduler.log).render())
    elif args.command == "skills":
        print(tree_summary(build_tree(build_skills(config))))
    elif args.command == "status":
        print(scheduler.status())
    elif args.command == "relabel":
        ok = scheduler.log.relabel(args.id, args.outcome)
        print("relabeled." if ok else f"no row with id {args.id}")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())