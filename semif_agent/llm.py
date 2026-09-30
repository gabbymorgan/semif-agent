"""OpenAI-compatible provider for skill title + description authoring.

New categories and skills are named by a small, local OpenAI-compatible model
on the `llm` endpoint, deliberately separate from the larger `codegen` model
that writes runnable bodies. It uses the same provider machinery as
`CodegenClient` (`provider.OpenAICompatClient`): stdlib-only HTTP, a real SSE
stream, a token budget, and an idle watchdog — just with its own endpoint,
model, sampler defaults, and error type. `_parse_json` is also borrowed by the
authoring parsers.
"""

from __future__ import annotations

import json

from .provider import OpenAICompatClient, ProviderError


class LLMError(ProviderError):
    """The configured `llm` endpoint could not be reached or stalled."""


class LLMClient(OpenAICompatClient):
    """Small-model author of skill/category titles + descriptions.

    Short JSON replies, so the defaults are conservative: low temperature, no
    presence penalty, a modest timeout, and single-shot generation. Every value
    is overridable from the `llm` block in `config.json`.
    """

    error_class = LLMError
    degeneration_error_class = LLMError
    label = "llm"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 600.0,
        stream: bool = False,
        idle_warn: float = 30.0,
        idle_timeout: float = 120.0,
        context_window: float = 0.0,
        smart_limit: int = 250000,
        warn_limit: int = 500000,
        max_fill_ratio: float = 0.9,
        warn_fill_ratio: float = 0.7,
        max_output: float = 0.85,
        chars_per_token: float = 4.0,
        degeneration_interval: int = 8000,
        degeneration_window: int = 2000,
        degeneration_min_chars: int = 4000,
        temperature: float = 0.2,
        top_p: float = 0.9,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ):
        super().__init__(
            base_url=base_url,
            model=model,
            timeout=timeout,
            stream=stream,
            idle_warn=idle_warn,
            idle_timeout=idle_timeout,
            context_window=context_window,
            smart_limit=smart_limit,
            warn_limit=warn_limit,
            max_fill_ratio=max_fill_ratio,
            warn_fill_ratio=warn_fill_ratio,
            max_output=max_output,
            chars_per_token=chars_per_token,
            degeneration_interval=degeneration_interval,
            degeneration_window=degeneration_window,
            degeneration_min_chars=degeneration_min_chars,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return json.loads(text)
