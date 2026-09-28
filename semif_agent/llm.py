"""OpenAI-compatible LLM client — retained but dormant.

Decision-making (assessment, requeue, fidelity) now belongs to SemIf; this
client no longer participates in any decision path. The `llm` endpoint stays
configured for now (retirement is backlogged), so the class remains as the
stdlib transport plus `_parse_json`, which the authoring parsers borrow. Uses
only the stdlib HTTP client so the core stays dependency-free.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


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

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return json.loads(text)
