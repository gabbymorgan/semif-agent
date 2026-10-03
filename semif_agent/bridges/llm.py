"""LLM bridge: the agent's language model behind the standard bridge surface.

Skill bodies are stdlib-only and must not speak a model's native
OpenAI-compatible protocol directly. This bridge exposes one generic chat call
over the same local, token-guarded JSON surface every other bridge uses, so a
body can ask for short generated text (e.g. extract an event title and
description from a request) without importing a client or knowing the endpoint.

It does **not** own the model connection: it reuses the scheduler's already
configured `llm` client, passed in at construction, so the endpoint/model/
sampler live in one place (the top-level `llm` block). Constructed without a
client (tests, standalone) it has no model and every call reports 502 rather
than fabricating an answer.
"""

from __future__ import annotations

from .base import BridgeInfo, BridgeService


class LLMBridge(BridgeService):
    name = "llm"
    INFO = BridgeInfo(
        name="llm",
        service="llm",
        description=(
            "Generate short text with the agent's configured language model, "
            "e.g. extract a title and a description from a user's request."
        ),
        url_config_var="llm_bridge_url",
        endpoints=(
            'GET /health -> 200 {"ok": true, "platform": "llm"} — the bridge is up',
            'POST /chat {"messages": [{"role", "content"}, ...], "max_tokens"?: int} '
            '-> 200 {"text": "<model reply>"}; 400 {"error": "..."} on a '
            'missing/invalid messages list; 502 {"error": "..."} when the model '
            "call fails or no model is configured",
            'any request -> 401 {"error": "unauthorized"} when the token is '
            "configured and the auth header is missing or wrong",
        ),
        config_vars=("llm_bridge_url", "llm_bridge_token"),
        config_var_docs=(
            (
                "llm_bridge_url",
                "Base URL of the local LLM bridge (e.g. http://127.0.0.1:5229); "
                "never hardcode it in the body.",
            ),
            (
                "llm_bridge_token",
                "Shared secret for the bridge, if one is configured; sent as the "
                "X-Semif-Token header. Leave blank when the bridge requires no auth.",
            ),
        ),
        auth_header="X-Semif-Token",
        auth_config_var="llm_bridge_token",
    )

    def __init__(self, config: dict | None = None, trace=None, client=None):
        super().__init__(config, trace)
        #: The scheduler's configured `llm` client (or a test double). None when
        #: the bridge is built without one; calls then report 502, never a guess.
        self.client = client

    # ---- lifecycle ----

    def check_requirements(self) -> tuple[bool, str | None]:
        # Stdlib only: no missing dependency can make the bridge unstartable. A
        # missing model surfaces as a 502 on the call, not as a startup failure.
        return True, None

    # ---- routes ----

    def handle_get(self, path: str, query: dict) -> tuple[int, dict]:
        if path == "/health":
            return 200, {"ok": True, "platform": self.name}
        return 404, {"error": "not found"}

    def handle_post(self, path: str, payload: dict) -> tuple[int, dict]:
        if path != "/chat":
            return 404, {"error": "not found"}
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return 400, {"error": "messages is required"}
        if not all(
            isinstance(message, dict)
            and isinstance(message.get("role"), str)
            and isinstance(message.get("content"), str)
            for message in messages
        ):
            return 400, {"error": "each message needs a string role and content"}
        max_tokens = payload.get("max_tokens")
        if max_tokens is not None and not isinstance(max_tokens, int):
            return 400, {"error": "max_tokens must be an integer"}
        if self.client is None:
            return 502, {"error": "no language model is configured"}
        try:
            text = self.client.chat(messages, max_tokens=max_tokens)
        except Exception as exc:  # the model client's error type + anything it raises
            return 502, {"error": f"llm call failed: {exc}"[:300]}
        return 200, {"text": text}
