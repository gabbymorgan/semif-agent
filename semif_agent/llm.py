"""OpenAI-compatible provider for skill title + description authoring.

New categories and skills are named by a small, local OpenAI-compatible model
on the `llm` endpoint, deliberately separate from the larger `codegen` model
that writes runnable bodies. It uses the same provider machinery as
`CodegenClient` (`provider.OpenAICompatClient`): stdlib-only HTTP, a real SSE
stream, a token budget, and an idle watchdog — just with its own endpoint,
model, sampler parameters, and error type. `_parse_json` is also borrowed by
the authoring parsers.
"""

from __future__ import annotations

import json

from .provider import OpenAICompatClient, ProviderError, SemIfEngineClient


class LLMError(ProviderError):
    """The configured `llm` endpoint could not be reached or stalled."""


class LLMClient(OpenAICompatClient):
    """Small-model author of skill/category titles + descriptions.

    Short JSON replies, so the timeout is modest and generation is single-shot.
    No sampler parameters are sent unless the `llm` block in `config.json` sets
    them: an unset `temperature`/`top_p`/`presence_penalty`/`frequency_penalty`
    is omitted so the server's model default applies. Every value is overridable
    from config.

    `disable_thinking` defaults **on**: the small local model is often a
    reasoning model whose hidden chain-of-thought consumes
    the short reply budget and leaves `content` empty — the same cap problem
    codegen documents, but here the reply is a ~30-token JSON object, so the
    thinking is pure overhead. Set `llm.disable_thinking: false` for a
    non-reasoning endpoint that rejects the parameter.
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
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        disable_thinking: bool = True,
        api_key: str = "",
        extra_headers: dict[str, str] | None = None,
        user_agent: str | None = None,
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
            disable_thinking=disable_thinking,
            api_key=api_key,
            extra_headers=extra_headers,
            user_agent=user_agent,
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return json.loads(text)


class SemIfLLMClient(SemIfEngineClient):
    """`llm` authoring backed by the loaded SemIf engine model.

    The `llm.provider = "semif"` option: the same in-process GGUF the decision
    engine already loaded generates the title/description (and any other `llm`
    text) instead of calling an HTTP endpoint. Same `chat(...)` surface, same
    graceful error contract — raises `LLMError` so the scheduler traces a
    `draft_failed` and the LLM bridge reports 502 rather than crashing.
    """

    error_class = LLMError
    label = "semif:llm"
