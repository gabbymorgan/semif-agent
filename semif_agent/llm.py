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

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return json.loads(text)
