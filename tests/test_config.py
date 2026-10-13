"""The layered config: index includes, !secret resolution, and schema validation.

These exercise the Home-Assistant-style split directly: config.json is a thin
index over fragments, secrets live in a separate file referenced as
``"!secret <key>"``, and the core + pack schemas own defaults and validate a
live config.
"""

import json
from pathlib import Path

from semif_agent.config import (
    ConfigError,
    generate_example_config,
    iter_secret_refs,
    load_config_file,
    load_secrets,
    missing_secret_refs,
    resolve_secrets,
    split_config,
    validate_config,
    write_split_config,
)
from semif_agent.packs import load_packs


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_index_includes_deep_merge_later_wins(tmp_path):
    _write(tmp_path / "config" / "core.json", {"tau": 0.6, "queue": {"max_size": 10}})
    _write(tmp_path / "config" / "gateway.json", {"gateway": {"simplex": {"enabled": True}}})
    _write(
        tmp_path / "config.json",
        {"include": ["config/core.json", "config/gateway.json"]},
    )
    config = load_config_file(tmp_path / "config.json")
    assert config["tau"] == 0.6
    assert config["queue"]["max_size"] == 10
    assert config["gateway"]["simplex"]["enabled"] is True


def test_include_cycle_is_an_error(tmp_path):
    _write(tmp_path / "a.json", {"include": ["b.json"]})
    _write(tmp_path / "b.json", {"include": ["a.json"]})
    try:
        load_config_file(tmp_path / "a.json")
    except ConfigError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("expected a cycle error")


def test_secret_resolution_and_missing_refs(tmp_path):
    _write(tmp_path / "secrets.json", {"nextcloud_app_password": "s3cret"})
    secrets = load_secrets(tmp_path / "secrets.json")
    config = {
        "bridges": {"nextcloud": {"app_password": "!secret nextcloud_app_password"}},
        "llm": {"api_key": "!secret absent_key"},
    }
    assert list(iter_secret_refs(config)) == ["nextcloud_app_password", "absent_key"]
    resolved = resolve_secrets(config, secrets)
    assert resolved["bridges"]["nextcloud"]["app_password"] == "s3cret"
    # A missing secret resolves to "" at load (startup never hard-fails) ...
    assert resolved["llm"]["api_key"] == ""
    # ... and is reported by the validator.
    assert missing_secret_refs(config, secrets) == ["absent_key"]


def test_generate_example_marks_secrets_and_never_leaks_values(tmp_path):
    packs = load_packs({"skill_packs": str(Path(__file__).resolve().parent.parent / "packs")})
    example = generate_example_config(packs)
    # Every secret-typed key is a !secret ref, never a value.
    assert example["codegen"]["api_key"] == "!secret codegen_api_key"
    assert example["bridges"]["nextcloud"]["app_password"] == "!secret nextcloud_app_password"
    assert example["bridges"]["nextcloud"]["token"] == "!secret nextcloud_token"
    # The pack's non-secret defaults are present.
    assert example["bridges"]["nextcloud"]["port"] == 5230
    assert example["packs"]["essentials"]["enabled"] is True


def test_validate_config_reports_types_unknowns_and_secrets(tmp_path):
    _write(tmp_path / "secrets.json", {})
    config = {
        "tau": "not-a-number",
        "totally_unknown_key": 1,
        "codegen": {"api_key": "!secret codegen_api_key"},
    }
    problems = validate_config(config, packs=[])
    assert any("tau: expected float" in p for p in problems)
    assert any("warning: totally_unknown_key: unknown config key" in p for p in problems)
    assert "codegen_api_key" in missing_secret_refs(config, load_secrets(tmp_path / "secrets.json"))


def test_split_then_load_roundtrip(tmp_path):
    packs = load_packs({"skill_packs": str(Path(__file__).resolve().parent.parent / "packs")})
    merged = generate_example_config(packs)
    written = write_split_config(tmp_path, merged, packs)
    assert any(p.endswith("config/core.json") for p in written)
    assert any(p.endswith("config/packs/nextcloud.json") for p in written)
    # The written index reloads to the same core values.
    reloaded = load_config_file(tmp_path / "config.json")
    assert reloaded["tau"] == merged["tau"]
    assert reloaded["bridges"]["nextcloud"]["port"] == 5230


def test_split_config_places_shared_bridge_outside_packs(tmp_path):
    packs = load_packs({"skill_packs": str(Path(__file__).resolve().parent.parent / "packs")})
    merged = generate_example_config(packs)
    files = split_config(merged, packs)
    assert "bridges" in files["config/bridges.json"]
    assert "llm" in files["config/bridges.json"]["bridges"]
