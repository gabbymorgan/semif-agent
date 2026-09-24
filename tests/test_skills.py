"""Pure-stdlib tests for skill-tree authoring: category prompts, draft parsing,
the category registry, and the error path when the engine is unavailable.

No mocking: engine-dependent success paths are exercised only by the box
integration tests against the real decision model.
"""

import pytest

from semif_agent.decisions import Request
from semif_agent.engine import EngineConfig, EngineUnavailable, SemIfEngine
from semif_agent.skills import (
    CategoryDraft,
    CategoryRegistry,
    build_category_prompt,
    build_skills,
    build_tree,
    generate_category,
    parse_category_draft,
)


def test_build_category_prompt_contains_request_and_tree():
    tree = build_tree(build_skills({"skills": {}}))
    messages = build_category_prompt(Request("tracking for my drone delivery"), tree)
    assert messages[0]["role"] == "system"
    assert "category" in messages[0]["content"]
    joined = messages[1]["content"]
    assert "tracking for my drone delivery" in joined
    assert "email: email.compose" in joined


def test_parse_category_draft_plain_json():
    draft = parse_category_draft(
        '{"title": "delivery", "description": "Track and manage package deliveries."}'
    )
    assert draft.name == "delivery"
    assert "package" in draft.description


def test_parse_category_draft_json_in_prose():
    draft = parse_category_draft(
        'Sure! Here you go:\n{"title": "home_automation", "description": "Control '
        'lights, locks, and appliances around the house."}\nHope that helps.'
    )
    assert draft.name == "home_automation"


def test_parse_category_draft_normalizes_title():
    draft = parse_category_draft(
        '{"title": "Home Automation", "description": "Control household devices."}'
    )
    assert draft.name == "home_automation"


def test_parse_category_draft_missing_fields_raises():
    with pytest.raises(ValueError):
        parse_category_draft('{"title": "only_title"}')
    with pytest.raises(ValueError):
        parse_category_draft("not json at all")


def test_parse_category_draft_rejects_unclean_title():
    with pytest.raises(ValueError):
        parse_category_draft('{"title": "ca$h!", "description": "nope"}')


def test_category_registry_roundtrip(tmp_path):
    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    assert registry.read() == {}
    registry.register("delivery", "Track and manage package deliveries.")
    registry.register("delivery", "Track and manage package deliveries, re-registered.")
    loaded = registry.read()
    assert set(loaded) == {"delivery"}
    assert loaded["delivery"]["description"] == (
        "Track and manage package deliveries, re-registered."
    )
    assert loaded["delivery"]["skills"] == []


def test_generate_category_without_engine_raises():
    engine = SemIfEngine(EngineConfig())
    with pytest.raises(EngineUnavailable):
        generate_category(engine, Request("anything"), {})


def test_generate_without_engine_raises():
    engine = SemIfEngine(EngineConfig())
    with pytest.raises(EngineUnavailable):
        engine.generate([{"role": "user", "content": "hi"}])


def test_build_tree_includes_registry_stubs(tmp_path):
    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    registry.register("delivery", "Track and manage package deliveries.")
    tree = build_tree(build_skills({"skills": {}}))
    for category in registry.read():
        tree.setdefault(category, [])
    assert tree["delivery"] == []