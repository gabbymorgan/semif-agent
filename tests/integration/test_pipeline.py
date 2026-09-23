"""End-to-end pipeline test. Run ONLY on the box with real SemIf + real LLM.

    python -m pytest tests/integration -q

The decision engine is real (llamacpp GGUF) and self-assessment is a real
local LLM endpoint. If either is unavailable this fails loudly — no mocking.
"""

import json
from pathlib import Path

import pytest

from semif_agent.cli import build_scheduler, load_config
from semif_agent.dream import dream
from semif_agent.engine import EngineUnavailable


def require_real(config: dict):
    from semif_agent.engine import SemIfEngine, EngineConfig

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
    try:
        engine._ensure_loaded()
    except EngineUnavailable as exc:
        pytest.fail(f"real engine unavailable: {exc}")


def test_pipeline_end_to_end(tmp_path):
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    scheduler, config = build_scheduler(config)

    inputs = [
        "send my girlfriend an email that says I'm going to be late to the party",
        "tell me if my package was delivered",
    ]
    for text in inputs:
        status, detail = scheduler.submit(text)
        print(f"[{status}] {detail}")
        assert status in ("running", "preempted", "queued", "rejected")

    scheduler.run_queue()

    rows = scheduler.log.read()
    assert len(rows) > 0, "expected SemIf decisions to be logged"

    report = dream(scheduler.log)
    assert report.cross_entropy is not None
    print(report.render())

    skills = scheduler.status()
    assert "queue:" in skills

    contacts_path = Path(config.get("skills", {}).get("contacts", "data/contacts.json"))
    assert contacts_path.is_file()


def test_busy_choice_path(tmp_path):
    config = load_config()
    require_real(config)
    config["log"] = str(tmp_path / "decisions.jsonl")
    scheduler, config = build_scheduler(config)

    scheduler.busy("driving on the freeway", skill="driving")
    status, detail = scheduler.submit("send my girlfriend an email that says I'm going to be late")
    print(f"[{status}] {detail}")
    assert status in ("preempted", "queued", "dropped")
    scheduler.idle()