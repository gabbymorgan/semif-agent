"""Pure-stdlib tests for the retained JSON helper on the dormant LLM client.

The LLM no longer participates in any decision path (assessment and fidelity
are SemIf decisions), so only `_parse_json` — which the authoring parsers
borrow — is exercised here.
"""

from semif_agent.llm import LLMClient


def test_parse_json_extracts_object_from_prose():
    parsed = LLMClient._parse_json('Sure, here: {"a": 1, "b": "two"} — done.')
    assert parsed == {"a": 1, "b": "two"}


def test_parse_json_handles_bare_object():
    assert LLMClient._parse_json('{"ok": true}') == {"ok": True}
