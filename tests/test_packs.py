"""The committed pack skills are real, inspectable, and hermetically tested.

A pack ships starter integrations in the repo in the CODEGEN.md folder format,
grouped by concern under ``packs/<name>/skills/<category>/<leaf>/``. These tests
keep every pack skill honest: the declared integration must match the code, the
contract must cover what the body reads, and the hermetic mechanics test must
pass. They are not proof of the live integration — only a real run against the
user's service is (see AGENTS.md).
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
from semif_agent.packs import (
    disabled_categories,
    load_packs,
    merge_packs,
    pack_enabled,
)

PACKS = Path(__file__).resolve().parent.parent / "packs"
SKILL_DIRS = sorted(path.parent for path in PACKS.glob("*/skills/*/*/skill.py"))


def seed_id(seed: Path) -> str:
    return f"{seed.parent.name}.{seed.name}"


def test_pack_manifests_are_valid():
    packs = load_packs({"skill_packs": str(PACKS)})
    assert {p.name for p in packs} >= {
        "nextcloud",
        "simplex",
        "outlook",
        "browser",
        "essentials",
    }
    for pack in packs:
        assert pack.concern, f"{pack.name} must declare a concern"
        assert pack.categories, f"{pack.name} must declare its categories"


def test_pack_enabled_resolution():
    packs = {p.name: p for p in load_packs({"skill_packs": str(PACKS)})}
    # A bridge pack follows its bridge's enabled flag.
    assert pack_enabled({"bridges": {"nextcloud": {"enabled": False}}}, packs["nextcloud"]) is False
    assert pack_enabled({"bridges": {"nextcloud": {"enabled": True}}}, packs["nextcloud"]) is True
    # An explicit packs.<name>.enabled wins over the bridge flag.
    assert (
        pack_enabled(
            {"bridges": {"nextcloud": {"enabled": False}}, "packs": {"nextcloud": {"enabled": True}}},
            packs["nextcloud"],
        )
        is True
    )
    # A bridge-less pack uses enabled_by_default.
    assert pack_enabled({}, packs["essentials"]) is True
    assert pack_enabled({"packs": {"essentials": {"enabled": False}}}, packs["essentials"]) is False
    # browser defaults off (needs the optional deps).
    assert pack_enabled({}, packs["browser"]) is False


def test_merge_packs_loads_enabled_and_skips_disabled():
    packs = load_packs({"skill_packs": str(PACKS)})
    tree: dict = {}
    config = {"bridges": {"nextcloud": {"enabled": True}, "outlook": {"enabled": False}}}
    merge_packs(tree, packs, config)
    assert "nextcloud" in tree
    assert "time" in tree and "calculator" in tree  # the essentials pack
    assert "outlook" not in tree  # disabled pack contributes nothing
    assert "browser" not in tree
    assert disabled_categories(config, packs) >= {"outlook", "browser"}
    # A pack skill is labeled `seed` (a committed starter), not `generated`.
    assert all(s.origin == "seed" for s in tree["nextcloud"])
    # The pack's category description rides on the skill (feeds navigation).
    assert tree["nextcloud"][0].category_description


def test_manifestless_pack_root_is_one_implicit_pack(tmp_path):
    directory = tmp_path / "my_skills" / "widgets" / "frobnicate"
    directory.mkdir(parents=True)
    (directory / "skill.py").write_text("def act(ctx, request):\n    return None\n")
    packs = load_packs({"skill_packs": str(tmp_path / "my_skills")})
    assert len(packs) == 1
    assert ("widgets", "frobnicate") in packs[0].store.list_skills()


def test_expected_seeds_exist():
    ids = {seed_id(seed) for seed in SKILL_DIRS}
    assert "nextcloud.next_event" in ids
    assert "nextcloud.create_event" in ids
    assert "simplex.next_message" in ids
    assert "simplex.send_message" in ids
    assert "simplex.connect_link" in ids
    assert "time.now" in ids
    assert "time.date" in ids
    assert "time.set_timer" in ids
    assert "time.set_alarm" in ids
    assert "nextcloud.create_task" in ids
    assert "calculator.calculate" in ids
    assert "browser.open_page" in ids
    assert "browser.read_page" in ids
    assert "browser.screenshot" in ids


@pytest.mark.parametrize("seed", SKILL_DIRS, ids=[seed_id(s) for s in SKILL_DIRS])
def test_seed_declares_a_real_integration(seed):
    code = (seed / "skill.py").read_text()
    integration = parse_integration(code)
    assert integration["service"]
    assert integration["transport"]
    assert integration_findings(integration, "declared", code) == []


@pytest.mark.parametrize("seed", SKILL_DIRS, ids=[seed_id(s) for s in SKILL_DIRS])
def test_seed_contract_is_flat_and_covered_by_declaration(seed):
    code = (seed / "skill.py").read_text()
    contract = parse_contract(code)
    integration = parse_integration(code)
    assert set(integration["config_vars"]) == set(contract)
    mirror = json.loads((seed / "contract.json").read_text())
    assert mirror == contract, "contract.json must mirror the CONTRACT constant"
    if integration["transport"] == "compute":
        # A local compute skill acts on the machine itself and needs no
        # configuration; an empty contract is legitimate.
        return
    assert contract, "an external-integration seed must declare a CONTRACT constant"
    for name, description in contract.items():
        assert name.islower() and " " not in name, name
        assert isinstance(description, str) and description.strip()


@pytest.mark.parametrize("seed", SKILL_DIRS, ids=[seed_id(s) for s in SKILL_DIRS])
def test_seed_has_manifest_description(seed):
    manifest = json.loads((seed / "manifest.json").read_text())
    assert manifest["description"].strip()


@pytest.mark.parametrize("seed", SKILL_DIRS, ids=[seed_id(s) for s in SKILL_DIRS])
def test_seed_hermetic_test_passes(seed, monkeypatch):
    root = str(seed.parents[4])
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", root + os.pathsep + existing)
    passed, output = run_skill_test(seed, timeout=30)
    assert passed, output
