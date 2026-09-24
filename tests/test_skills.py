"""Pure-stdlib tests for skill-tree authoring: category and skill prompts, draft
parsing, the category registry, and the error path when the engine is unavailable.

No mocking: engine-dependent success paths are exercised only by the box
integration tests against the real decision model.
"""

import pytest

from semif_agent.decisions import Request
from semif_agent.engine import EngineConfig, EngineUnavailable, SemIfEngine
from semif_agent.skills import (
    CategoryDraft,
    CategoryRegistry,
    SkillDraft,
    build_category_prompt,
    build_skill_prompt,
    build_skills,
    build_tree,
    generate_category,
    generate_skill,
    merge_registry,
    parse_category_draft,
    parse_skill_draft,
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
    merge_registry(tree, registry.read())
    assert tree["delivery"] == []


def test_build_skill_prompt_contains_request_category_and_skills():
    tree = build_tree(build_skills({"skills": {}}))
    messages = build_skill_prompt(Request("tracking for my drone delivery"), "tracking", tree)
    assert messages[0]["role"] == "system"
    assert "tracking" in messages[0]["content"]
    joined = messages[1]["content"]
    assert "tracking for my drone delivery" in joined
    assert "tracking.check" in joined


def test_parse_skill_draft_plain_json():
    draft = parse_skill_draft(
        '{"title": "track_live", "description": "Follow a package in real time."}'
    )
    assert draft.name == "track_live"
    assert "real time" in draft.description


def test_parse_skill_draft_json_in_prose():
    draft = parse_skill_draft(
        'Here you go:\n{"title": "resend_email", "description": "Re-send a failed '
        'email draft."}\nHope that helps.'
    )
    assert draft.name == "resend_email"


def test_parse_skill_draft_normalizes_title():
    draft = parse_skill_draft(
        '{"title": "Live Tracking", "description": "Follow a package in real time."}'
    )
    assert draft.name == "live_tracking"


def test_parse_skill_draft_missing_fields_raises():
    with pytest.raises(ValueError):
        parse_skill_draft('{"title": "only_title"}')
    with pytest.raises(ValueError):
        parse_skill_draft("not json at all")


def test_parse_skill_draft_rejects_unclean_title():
    with pytest.raises(ValueError):
        parse_skill_draft('{"title": "ca$h!", "description": "nope"}')


def test_generate_skill_without_engine_raises():
    engine = SemIfEngine(EngineConfig())
    with pytest.raises(EngineUnavailable):
        generate_skill(engine, Request("anything"), "tracking", {})


def test_category_registry_register_skill(tmp_path):
    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    registry.register("delivery", "Track and manage package deliveries.")
    registry.register_skill("delivery", "track_live", "Follow a package in real time.")
    registry.register_skill("delivery", "track_live", "Duplicate, ignored.")
    registry.register_skill("brand_new", "ping", "Probe the service.")
    loaded = registry.read()
    assert loaded["delivery"]["skills"] == [
        {"name": "track_live", "description": "Follow a package in real time."}
    ]
    assert loaded["brand_new"] == {
        "description": "",
        "skills": [{"name": "ping", "description": "Probe the service."}],
    }


def test_merge_registry_loads_categories_and_skills(tmp_path):
    registry = CategoryRegistry(str(tmp_path / "categories.json"))
    registry.register_skill("delivery", "track_live", "Follow a package in real time.")
    tree = build_tree(build_skills({"skills": {}}))
    merge_registry(tree, registry.read())
    names = [s.name for s in tree["delivery"]]
    assert names == ["track_live"]
    assert tree["delivery"][0].category == "delivery"