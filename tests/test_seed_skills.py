"""The committed seed skills are real, inspectable, and hermetically tested.

A seed is a starter integration shipped in the repo in the SKILL.md folder
format. These tests keep every seed honest: the declared integration must match
the code, the contract must cover what the body reads, and the hermetic
mechanics test must pass. They are not proof of the live integration — only a
real run against the user's service is (see AGENTS.md).
"""

import json
import os
from pathlib import Path

import pytest

from semif_agent.codegen import (
    integration_findings,
    parse_contract,
    parse_integration,
    run_skill_test,
)

SEEDS = Path(__file__).resolve().parent.parent / "seeds"
SEED_DIRS = sorted(path.parent for path in SEEDS.glob("*/*/skill.py"))


def seed_id(seed: Path) -> str:
    return f"{seed.parent.name}.{seed.name}"


def test_expected_seeds_exist():
    ids = {seed_id(seed) for seed in SEED_DIRS}
    assert "calendar.next_event" in ids
    assert "simplex.next_message" in ids
    assert "simplex.connect_link" in ids


@pytest.mark.parametrize("seed", SEED_DIRS, ids=[seed_id(s) for s in SEED_DIRS])
def test_seed_declares_a_real_integration(seed):
    code = (seed / "skill.py").read_text()
    integration = parse_integration(code)
    assert integration["service"]
    assert integration["transport"]
    assert integration_findings(integration, "declared", code) == []


@pytest.mark.parametrize("seed", SEED_DIRS, ids=[seed_id(s) for s in SEED_DIRS])
def test_seed_contract_is_flat_and_covered_by_declaration(seed):
    code = (seed / "skill.py").read_text()
    contract = parse_contract(code)
    assert contract, "the seed must declare a CONTRACT constant"
    for name, description in contract.items():
        assert name.islower() and " " not in name, name
        assert isinstance(description, str) and description.strip()
    integration = parse_integration(code)
    assert set(integration["config_vars"]) == set(contract)
    mirror = json.loads((seed / "contract.json").read_text())
    assert mirror == contract, "contract.json must mirror the CONTRACT constant"


@pytest.mark.parametrize("seed", SEED_DIRS, ids=[seed_id(s) for s in SEED_DIRS])
def test_seed_has_manifest_description(seed):
    manifest = json.loads((seed / "manifest.json").read_text())
    assert manifest["description"].strip()


@pytest.mark.parametrize("seed", SEED_DIRS, ids=[seed_id(s) for s in SEED_DIRS])
def test_seed_hermetic_test_passes(seed, monkeypatch):
    root = str(seed.parents[2])
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", root + os.pathsep + existing)
    passed, output = run_skill_test(seed, timeout=30)
    assert passed, output
