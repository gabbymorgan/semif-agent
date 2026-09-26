"""Self-assessment LLM client.

Talks to a local OpenAI-compatible server (e.g. ollama, llama.cpp server) for
the observe -> assess phase of the skill loop. Real, not mocked; the endpoint
must be reachable. Uses only the stdlib HTTP client so the core stays
dependency-free.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Assessment:
    success: bool
    summary: str
    updated_request: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FidelityReview:
    """Whether a generated skill body really performs its action.

    `performs_real_action` is False when the body simulates the action (canned
    result, fabricated data, draft-by-default, or asks the human for operational
    data instead of acting). Degrades to True when the review endpoint is
    unreachable so a broken reviewer never blocks authoring.
    """

    performs_real_action: bool
    reason: str


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(
            {"model": self.model, "messages": messages, "temperature": temperature}
        ).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"LLM endpoint unreachable at {url}: {exc}. Is your local server running?"
            ) from exc
        return payload["choices"][0]["message"]["content"]

    def assess(self, skill: str, request_text: str, action_log: str) -> Assessment:
        system = (
            "You are the self-assessment step of an agent skill run. Decide whether "
            "the skill achieved its goal. Reply with JSON only: "
            '{"success": true|false, "summary": "<brief>", "updated_request": '
            '"<requeued request text or null>"}. success is true only if the goal was met.'
        )
        user = (
            f"Skill: {skill}\n"
            f"Goal request: {request_text}\n"
            f"What was done:\n{action_log}\n"
        )
        try:
            raw = self._chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
            parsed = self._parse_json(raw)
        except Exception as exc:
            return Assessment(success=False, summary=f"assessment failed: {exc}")
        success = bool(parsed.get("success"))
        summary = str(parsed.get("summary", ""))
        updated = parsed.get("updated_request")
        return Assessment(
            success=success,
            summary=summary,
            updated_request=None if updated is None else str(updated),
        )

    def review_skill_body(
        self,
        request_text: str,
        description: str,
        code: str,
        requirements: dict[str, str] | None = None,
        integration: dict | None = None,
    ) -> FidelityReview:
        """Judge whether a generated body performs its action for real.

        The agent exists to do the user's task against their actual service; a
        body that fakes the outcome is broken even when its hermetic test
        passes. Any review failure degrades to `performs_real_action=True` so
        the reviewer never blocks authoring.
        """
        system = (
            "You review a generated agent skill body against the user's request. "
            "Decide whether `act` really performs the requested operation against "
            "the real service, or merely simulates it — a canned result, "
            "fabricated data, a draft written when the user asked for the action, "
            "or asking the human for operational data it should read from "
            "config. A purely local/computational task is real when it actually "
            "computes the result. Reply with JSON only: "
            '{"performs_real_action": true|false, "reason": "<brief>"}'
        )
        requirement_lines = ""
        if requirements:
            requirement_lines = "Requirements answers:\n" + "\n".join(
                f"- {question} -> {answer}" for question, answer in requirements.items()
            ) + "\n"
        user = (
            f"Request: {request_text}\n"
            f"Skill description: {description}\n"
            f"Integration declaration: {json.dumps(integration or {}, sort_keys=True)}\n"
            f"{requirement_lines}"
            f"Skill body:\n```python\n{code}\n```\n"
        )
        try:
            raw = self._chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
            parsed = self._parse_json(raw)
        except Exception as exc:
            return FidelityReview(True, f"review failed: {exc}")
        return FidelityReview(
            performs_real_action=bool(parsed.get("performs_real_action")),
            reason=str(parsed.get("reason", "")),
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return json.loads(text)
