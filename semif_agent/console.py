"""OpenCode Console provider (hosted, authenticated OpenAI-compatible API).

The `OpenCode Console <https://opencode.ai/console>`_ serves an
OpenAI-compatible Chat Completions API at
``https://opencode.ai/inference/openai/v1``. It is wire-identical to
``provider.OpenAICompatClient`` (SSE ``delta.content`` +
``delta.reasoning_content``, an ``include_usage`` chunk, then ``[DONE]``), so
this module only adds what the Console needs on top of the shared transport:

* a bearer service-account key (``Authorization: Bearer <key>``), and
* ``query_context = False`` — the Console has no ollama ``/api/show``, so the
  context window comes from config or the default instead of a stray probe.

Only the **Chat Completions** model family is supported (GLM, DeepSeek, Kimi,
Qwen3.8 Max, MiniMax, and the free models). Claude / OpenAI-Responses / Gemini
models use different wire formats and would need their own adapters.

``build_scheduler`` selects these classes from ``llm.provider`` /
``codegen.provider`` (``"opencode"`` vs the default ``"ollama"``) and passes
``api_key`` (or ``api_key_env``). The error contracts are preserved: the `llm`
variant still raises ``LLMError`` and the codegen variant ``CodegenError`` /
``DegenerationError``, so existing callers keep catching their own types.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .codegen import CodegenClient, CodegenError, DegenerationError
from .llm import LLMClient, LLMError
from .provider import OpenAICompatClient, ProviderError

OPENCODE_BASE_URL = "https://opencode.ai/inference/openai/v1"
OPENCODE_MODELS_URL = "https://opencode.ai/inference/v1/models"
OPENCODE_USER_AGENT = "semif-agent/0.1"


class OpenCodeError(ProviderError):
    """The OpenCode Console endpoint could not be reached."""


class OpenCodeConsoleClient(OpenAICompatClient):
    """Authenticated OpenAI-compatible client for the OpenCode Console.

    Pins the Console defaults: bearer auth, the Console base URL, a Console
    label, and ``query_context = False`` (no ollama ``/api/show`` probe). The
    endpoint subclasses below reuse it via cooperative multiple inheritance so
    they also keep ``LLMClient``/``CodegenClient`` sampler parameters (opt-in:
    none are sent unless configured).

    A default ``User-Agent`` is always sent: the Console gateway rejects
    urllib's default ``Python-urllib/<ver>`` with HTTP 403. Pass ``user_agent``
    to override it (the Go docs ask clients to identify themselves).
    """

    label = "opencode"
    error_class = OpenCodeError
    degeneration_error_class = OpenCodeError
    query_context = False

    def __init__(self, base_url: str = OPENCODE_BASE_URL, **kwargs):
        if not kwargs.get("user_agent"):
            kwargs["user_agent"] = OPENCODE_USER_AGENT
        super().__init__(base_url=base_url, **kwargs)

    def list_models(self, url: str | None = None) -> list[str]:
        """Fetch the Console model catalog (``GET /inference/v1/models``).

        Returns the model ids; a fresh service-account key is not required for
        the free models. Raises the client's own error type when unreachable.
        """
        target = url or OPENCODE_MODELS_URL
        request = urllib.request.Request(target, headers=self._headers())
        try:
            with urllib.request.urlopen(
                request, timeout=min(self.timeout, 30.0)
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise self.error_class(
                f"{self.label} model list unreachable at {target}: {exc}"
            ) from exc
        return [
            item["id"]
            for item in (data.get("data") or [])
            if isinstance(item, dict) and item.get("id")
        ]


class ConsoleLLMClient(OpenCodeConsoleClient, LLMClient):
    """Console-backed `llm` author (skill/category title + description).

    Same opt-in sampler behavior as ``LLMClient`` (none sent unless configured);
    the Console transport adds the bearer key and skips the ollama probe. Raises
    ``LLMError`` (unchanged contract).

    ``disable_thinking`` is forced **off**: `LLMClient` sends
    ``reasoning_effort: "none"`` (an ollama-compat knob) and the Console
    gateway rejects that parameter with HTTP 400. Console models are not the
    local reasoning model the knob exists for, so the parameter is simply not
    sent. Passing ``disable_thinking=True`` to the constructor still ends up
    off here.
    """

    label = "opencode:llm"
    error_class = LLMError
    degeneration_error_class = LLMError

    def __init__(self, *args, **kwargs):
        kwargs["disable_thinking"] = False
        super().__init__(*args, **kwargs)


class ConsoleCodegenClient(OpenCodeConsoleClient, CodegenClient):
    """Console-backed `codegen` writer (runnable skill bodies).

    Console defaults on the codegen transport. Raises ``CodegenError`` /
    ``DegenerationError`` (unchanged contract).
    """

    label = "opencode:codegen"
    error_class = CodegenError
    degeneration_error_class = DegenerationError
