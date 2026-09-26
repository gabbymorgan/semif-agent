"""The committed seed skills are real, inspectable, and hermetically tested.

A seed is a starter integration shipped in the repo in the SKILL.md folder
format. These tests keep it honest: the declared integration must match the
code, the contract must cover what the body reads, and the hermetic mechanics
test must pass. They are not proof of the live integration — only a real run
against the user's service is (see AGENTS.md).
"""

import json
import os
from pathlib import Path

from semif_agent.codegen import integration_findings, parse_integration, run_skill_test

SEED = Path(__file__).resolve().parent.parent / "seeds" / "calendar" / "next_event"


def test_seed_declares_a_real_integration():
    code = (SEED / "skill.py").read_text()
    integration = parse_integration(code)
    assert integration["service"] == "nextcloud_calendar"
    assert integration["transport"] == "caldav"
    assert integration_findings(integration, "declared", code) == []


def test_seed_contract_is_flat_and_covered_by_declaration():
    code = (SEED / "skill.py").read_text()
    contract = json.loads((SEED / "contract.json").read_text())
    assert contract, "the seed must carry a data contract"
    for name, description in contract.items():
        assert name.islower() and " " not in name, name
        assert isinstance(description, str) and description.strip()
    integration = parse_integration(code)
    assert set(integration["config_vars"]) == set(contract)


def test_seed_has_manifest_description():
    manifest = json.loads((SEED / "manifest.json").read_text())
    assert manifest["description"].strip()


def test_seed_hermetic_test_passes(monkeypatch):
    root = str(SEED.parents[2])
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", root + os.pathsep + existing)
    passed, output = run_skill_test(SEED, timeout=30)
    assert passed, output
