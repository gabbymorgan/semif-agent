"""Shared OpenAI-compatible chat provider.

Both the small title/description author (`LLMClient` in `llm.py`) and the big
skill-body writer (`CodegenClient` in `codegen.py`) use the same transport
machinery: stdlib-only HTTP, a real SSE stream, a layered token budget, an idle
watchdog, and an optional SemIf degeneration hook. This module holds that
provider logic once; subclasses set `base_url`/`model`, their sampler
parameters, and the error classes they raise so existing callers keep catching
their own types.

Stdlib-only and imports nothing from the agent's own modules, so `codegen.py`
(which imports `skills.py`, which imports `llm.py`) can depend on it without a
cycle.
"""

from __future__ import annotations

import json
import select
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, NamedTuple

DEFAULT_CONTEXT_WINDOW = 100000


class ProviderError(RuntimeError):
    """The configured OpenAI-compatible endpoint could not be reached."""


class TokenBudget(NamedTuple):
    """Per-request token limits derived from the detected context window.

    `total_limit` caps prompt + output; `output_limit` caps just the streamed
    output (the enforced one — it already bakes in the prompt estimate);
    `warn_point` is the total fill at which a one-shot console warning fires.
    An `output_limit` of 0 means no cap is configured.
    """

    window: int
    total_limit: int
    output_limit: int
    warn_point: int | None
    prompt_est: int


class OpenAICompatClient:
    """OpenAI-compatible chat provider shared by the authoring clients.

    Subclasses set `base_url`/`model` (and their sampler parameters) and the
    error classes they raise: `error_class` for transport/parse/budget failures
    and `degeneration_error_class` for a watchdog abort, so existing callers
    keep catching their own types. `label` prefixes console diagnostics.

    The transport always reads the response as an SSE token stream so the
    layered token budget, the idle watchdog, and the total wall-clock budget
    can abort a runaway generation in real time — never via a server-side
    `max_tokens` cap, because Qwen3-style models reason first and a cap
    truncates the hidden reasoning, leaving `content` empty.

    `stream=True` additionally echoes tokens (including the chain-of-thought)
    to stdout as they arrive; echo is console-only and the returned content
    is identical either way. A silent stream (no bytes for `idle_warn`
    seconds) prints a warning, and one that stays silent for `idle_timeout`
    seconds raises CodegenError instead of blocking on the total timeout — a
    wedged generation is surfaced in minutes, not ~20.

    The context window is lazily queried once from ollama's `/api/show`
    (`parameters.num_ctx`, falling back to `model_info.<arch>.context_length`,
    then the `context_window` config, then 100000); the per-request budget
    caps output at `max_output` (absolute tokens if >= 1, else a fraction of
    the window) and total fill at `smart_limit`/`warn_limit`.

    An optional `degeneration_check` callback (passed per `chat` call) is fed
    the last `degeneration_window` chars of content+reasoning every
    `degeneration_interval` chars once `degeneration_min_chars` have
    accumulated; a non-None return aborts the stream with DegenerationError.
    It lets a SemIf continue/stop decision cut off a generation that is
    looping instead of converging, before it fills the context window.

    Sampler parameters are **opt-in**: `temperature`/`top_p`/
    `presence_penalty`/`frequency_penalty` default to `None` and a `None`
    parameter is omitted from the request payload entirely, so the server's
    model/Modelfile default applies. A caller that knows the right value for its
    model sets it (constructor or per-`chat` override); this provider never
    guesses a value on the user's behalf. `max_tokens` and `reasoning_effort`
    work the same way: sent only when explicitly set.

    `disable_thinking` sends `reasoning_effort: "none"` so a reasoning model
    answers directly instead of spending the (short) reply budget on hidden
    chain-of-thought. It is off by default because codegen *wants* the
    reasoning; the `llm` title/description author turns it on. Ollama honors it
    on the `/v1` compat endpoint (its native `think:false` is not plumbed
    through `/v1`).

    `api_key` (when set) is sent as `Authorization: Bearer <key>` on every
    request — the OpenCode Console inference API and most hosted
    OpenAI-compatible gateways require it. `extra_headers` are merged on top
    (custom auth schemes, routing hints) and `user_agent` overrides the default
    client string. `query_context` gates the ollama `/api/show` probe: a
    hosted/non-ollama endpoint (see `console.OpenCodeConsoleClient`) sets it
    False so no stray `/api/show` request is sent and the window comes from
    `context_window`/the default.
    """

    error_class = ProviderError
    degeneration_error_class = ProviderError
    label = "provider"
    query_context = True

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 1200.0,
        stream: bool = False,
        idle_warn: float = 60.0,
        idle_timeout: float = 180.0,
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
        max_attempts: int = 3,
        disable_thinking: bool = False,
        api_key: str = "",
        extra_headers: dict[str, str] | None = None,
        user_agent: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.stream = stream
        self.idle_warn = idle_warn
        self.idle_timeout = idle_timeout
        self.context_window = context_window
        self.smart_limit = smart_limit
        self.warn_limit = warn_limit
        self.max_fill_ratio = max_fill_ratio
        self.warn_fill_ratio = warn_fill_ratio
        self.max_output = max_output
        self.chars_per_token = chars_per_token
        self.degeneration_interval = degeneration_interval
        self.degeneration_window = degeneration_window
        self.degeneration_min_chars = degeneration_min_chars
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_attempts = max_attempts
        self.disable_thinking = disable_thinking
        self.api_key = api_key
        self.extra_headers = dict(extra_headers or {})
        self.user_agent = user_agent
        self._window: int | None = None

    def _headers(self) -> dict[str, str]:
        """Request headers: JSON content type plus optional auth/extra/UA.

        A non-empty `api_key` becomes `Authorization: Bearer <key>` (required by
        the OpenCode Console inference API and most hosted gateways);
        `extra_headers` merge on top (last writer wins) and `user_agent`
        overrides the library default. The ollama context probe uses the same
        headers so an authenticated gateway never sees an unauthenticated
        request.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.user_agent:
            headers["User-Agent"] = self.user_agent
        if self.extra_headers:
            headers.update(self.extra_headers)
        return headers

    def chat(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        timeout: float | None = None,
        degeneration_check: Callable[[str], str | None] | None = None,
    ) -> str:
        # A per-call `timeout` overrides the constructor value for this request
        # only (used by short auxiliary calls such as the gateway's result
        # cleanup, which must not inherit the authoring client's long budget).
        deadline = self.timeout if timeout is None else timeout
        budget = self._compute_budget(messages)
        if self.stream:
            self._print_budget(budget)
        url = f"{self.base_url}/chat/completions"
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Sampler parameters are opt-in: a per-call override wins, otherwise the
        # constructor value; a None value is omitted entirely so the server's
        # model/Modelfile default applies.
        sampler = {
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "presence_penalty": (
                self.presence_penalty if presence_penalty is None else presence_penalty
            ),
            "frequency_penalty": (
                self.frequency_penalty if frequency_penalty is None else frequency_penalty
            ),
        }
        payload.update({k: v for k, v in sampler.items() if v is not None})
        if self.disable_thinking:
            payload["reasoning_effort"] = "none"
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers=self._headers()
        )
        try:
            with urllib.request.urlopen(request, timeout=deadline) as response:
                return self._read_stream(
                    response, budget, degeneration_check, timeout=deadline
                )
        except urllib.error.URLError as exc:
            raise self.error_class(
                f"{self.label} endpoint unreachable at {url}: {exc}. Is your local server running?"
            ) from exc
        except TimeoutError as exc:
            raise self.error_class(
                f"{self.label} request timed out after {deadline:.0f}s at {url}"
            ) from exc

    def _native_base_url(self) -> str:
        """Ollama's `/api/show` lives on the native API, not the /v1 compat."""
        if self.base_url.endswith("/v1"):
            return self.base_url[: -len("/v1")]
        return self.base_url

    def _query_context_window(self) -> int | None:
        """POST /api/show {model} for the runtime context window. Any failure
        (unreachable, non-ollama endpoint, malformed reply) returns None so the
        caller falls back to config or the default. Uses a short request
        timeout so a slow show endpoint never wedges the actual generation."""
        url = f"{self._native_base_url()}/api/show"
        request = urllib.request.Request(
            url,
            data=json.dumps({"model": self.model}).encode("utf-8"),
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(
                request, timeout=min(self.timeout, 10.0)
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
            return None
        params = data.get("parameters")
        if isinstance(params, dict):
            num_ctx = params.get("num_ctx")
            if isinstance(num_ctx, int) and num_ctx > 0:
                return num_ctx
        elif isinstance(params, str):
            # ollama serves parameters as modelfile text, not a dict:
            # "num_ctx 100000\n..." — pull the num_ctx line if present.
            for line in params.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] == "num_ctx":
                    try:
                        num_ctx = int(parts[1])
                    except ValueError:
                        break
                    if num_ctx > 0:
                        return num_ctx
        model_info = data.get("model_info") or {}
        for key, value in model_info.items():
            if key.endswith(".context_length") and isinstance(value, int) and value > 0:
                return value
        return None

    def _context_window(self) -> int:
        """Detected context window, queried once and cached.

        `query_context=False` (hosted/non-ollama endpoints) skips the
        `/api/show` probe entirely: the window is taken from `context_window`
        config or `DEFAULT_CONTEXT_WINDOW`, so no stray request is sent.
        """
        if self._window is not None:
            return self._window
        window = self._query_context_window() if self.query_context else None
        if window is None:
            window = (
                int(self.context_window)
                if self.context_window > 0
                else DEFAULT_CONTEXT_WINDOW
            )
        self._window = window
        return window

    def _compute_budget(self, messages: list[dict]) -> TokenBudget:
        window = self._context_window()
        max_out = (
            int(self.max_output)
            if self.max_output >= 1
            else int(self.max_output * window)
        )
        total_limit = int(min(self.smart_limit, window * self.max_fill_ratio))
        prompt_est = int(
            sum(len(str(m.get("content") or "")) for m in messages)
            / self.chars_per_token
        )
        if total_limit > 0 and prompt_est >= total_limit:
            raise self.error_class(
                f"{self.label} prompt (~{prompt_est} tokens) already exceeds the "
                f"total budget of {total_limit} tokens"
            )
        if total_limit > 0 and max_out > 0:
            output_limit = min(max_out, total_limit - prompt_est)
        else:
            output_limit = 0
        warn_point = None
        if self.warn_limit > 0 and self.warn_fill_ratio > 0:
            warn_point = min(self.warn_limit, int(window * self.warn_fill_ratio))
        return TokenBudget(window, total_limit, output_limit, warn_point, prompt_est)

    def _print_budget(self, budget: TokenBudget) -> None:
        if budget.window and budget.output_limit > 0:
            print(
                f"[{self.label}] context window {budget.window} tokens · "
                f"output cap {budget.output_limit} tokens "
                f"({budget.output_limit / budget.window * 100:.0f}%) · "
                f"peak total fill ~{budget.total_limit} tokens"
            )
        elif budget.window:
            print(
                f"[{self.label}] context window {budget.window} tokens · "
                f"peak total fill ~{budget.total_limit} tokens"
            )
        sys.stdout.flush()

    def _log_usage(self, budget: TokenBudget, usage: dict) -> None:
        """Report exact token usage once `include_usage` yields a usage chunk.

        Logs real prompt/completion/total tokens, % of the detected window, and
        the zone (SMART < smart_limit, WARN < warn_limit, else DUMB) measured
        against absolute total-token thresholds.
        """
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if not isinstance(total, int):
            return
        if not isinstance(prompt, int):
            prompt = total - (completion if isinstance(completion, int) else 0)
        if not isinstance(completion, int):
            completion = total - prompt
        pct = total / budget.window * 100 if budget.window else 0.0
        if self.smart_limit > 0 and total < self.smart_limit:
            zone = "SMART"
        elif self.warn_limit > 0 and total < self.warn_limit:
            zone = "WARN"
        else:
            zone = "DUMB"
        print(
            f"[{self.label}] usage: prompt {prompt} · completion {completion} · "
            f"total {total} tokens · {pct:.0f}% of window · {zone}"
        )
        sys.stdout.flush()

    def _enforce_budget(self, budget: TokenBudget, out_chars: int, warned: bool) -> bool:
        """Abort when the output cap is exceeded; one-shot warn near the fill
        threshold. Returns the updated warned state."""
        output_est = out_chars / self.chars_per_token
        if budget.output_limit > 0 and output_est >= budget.output_limit:
            raise self.error_class(
                f"{self.label} output exceeded {budget.output_limit} token budget "
                f"(~{out_chars} chars at {self.chars_per_token:g} chars/token)"
            )
        if budget.warn_point is not None and not warned:
            total = budget.prompt_est + output_est
            if total >= budget.warn_point:
                sys.stdout.write(
                    f"\n[{self.label}] total fill ~{int(total)} tokens — "
                    f"near the {budget.output_limit} token output budget\n"
                )
                sys.stdout.flush()
                return True
        return warned

    def _read_stream(
        self,
        response,
        budget: TokenBudget,
        degeneration_check: Callable[[str], str | None] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Read an OpenAI-compatible SSE stream; echo tokens if `stream`.

        Only `content` deltas are accumulated into the returned body;
        reasoning (chain-of-thought) is echoed to the console but never part
        of the result. Backends disagree on the field name: ollama streams it
        as `reasoning`, DeepSeek/vllm-style as `reasoning_content`, so read both.

        Reads are gated by `select` so the socket never blocks-and-times-out:
        a socket that delivers no bytes for `idle_warn` seconds prints a
        warning, and `idle_timeout` seconds of silence raises CodegenError.
        The total wall-clock budget `self.timeout` is enforced against the
        whole stream and checked on every loop iteration, so even a
        continuously-streaming generation that never sends `[DONE]` is cut
        short. The token budget aborts as soon as the output cap is reached.
        Thresholds <= 0 disable the idle checks. If the underlying socket
        can't be reached, falls back to a plain blocking read
        (`_read_stream_blocking`, which enforces the same budgets).

        An optional `degeneration_check` (last `window` chars of content +
        reasoning, every `interval` chars past `min_chars`) can abort a
        generation that is looping instead of converging — a non-None return
        is the abort reason.
        """
        parts: list[str] = []
        sock = self._stream_socket(response)
        if sock is None:
            return self._read_stream_blocking(
                response, budget, degeneration_check, timeout=timeout
            )
        deadline = self.timeout if timeout is None else timeout
        start = time.monotonic()
        last_activity = start
        warned = False
        warned_fill = False
        out_chars = 0
        usage: dict | None = None
        recent: list[str] = []
        recent_len = 0
        last_check: int | None = None
        while True:
            if time.monotonic() - start >= deadline:
                raise self.error_class(
                    f"{self.label} request exceeded {deadline:.0f}s total budget"
                )
            ready, _, _ = select.select([sock], [], [], 1.0)
            now = time.monotonic()
            if not ready:
                idle = now - last_activity
                if self.idle_warn > 0 and not warned and idle >= self.idle_warn:
                    sys.stdout.write(
                        f"\n[{self.label}] no tokens for {idle:.0f}s — still waiting, "
                        f"will fail after {self.idle_timeout:.0f}s of silence\n"
                    )
                    sys.stdout.flush()
                    warned = True
                if self.idle_timeout > 0 and idle >= self.idle_timeout:
                    raise self.error_class(
                        f"{self.label} stream stalled: no tokens for {idle:.0f}s "
                        f"(idle_timeout={self.idle_timeout:.0f}s)"
                    )
                continue
            try:
                raw = response.readline()
            except OSError as exc:
                raise self.error_class(f"{self.label} stream read failed: {exc}") from exc
            if not raw:
                break
            last_activity = time.monotonic()
            continues, text, reasoning, frame_usage = self._consume_frame(
                raw.decode("utf-8").strip()
            )
            if frame_usage is not None:
                usage = frame_usage
            if self.stream and (text or reasoning):
                sys.stdout.write(text + reasoning)
                sys.stdout.flush()
            if text:
                parts.append(text)
            out_chars += len(text) + len(reasoning)
            if text or reasoning:
                recent.append(text + reasoning)
                recent_len = self._trim_recent(recent, recent_len)
            warned_fill = self._enforce_budget(budget, out_chars, warned_fill)
            reason, last_check = self._check_degeneration(
                degeneration_check, recent, out_chars, last_check
            )
            if reason:
                raise self.degeneration_error_class(f"{self.label} degeneration detected: {reason}")
            if not continues:
                break
        if usage is not None:
            self._log_usage(budget, usage)
        if parts and self.stream:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return "".join(parts)

    def _read_stream_blocking(
        self,
        response,
        budget: TokenBudget,
        degeneration_check: Callable[[str], str | None] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Fallback reader when the response socket can't be located.

        A plain blocking read relies on the per-read socket timeout for
        silence, but that bounds individual reads, not the whole stream — so
        enforce the same total wall-clock budget (the per-call `timeout`, else
        `self.timeout`) and the token budget in the loop, or a
        continuously-streaming runaway would never be cut short.
        """
        parts: list[str] = []
        deadline = self.timeout if timeout is None else timeout
        start = time.monotonic()
        warned_fill = False
        out_chars = 0
        usage: dict | None = None
        recent: list[str] = []
        recent_len = 0
        last_check: int | None = None
        for raw in response:
            if time.monotonic() - start >= deadline:
                raise self.error_class(
                    f"{self.label} request exceeded {deadline:.0f}s total budget"
                )
            continues, text, reasoning, frame_usage = self._consume_frame(
                raw.decode("utf-8").strip()
            )
            if frame_usage is not None:
                usage = frame_usage
            if self.stream and (text or reasoning):
                sys.stdout.write(text + reasoning)
                sys.stdout.flush()
            if text:
                parts.append(text)
            out_chars += len(text) + len(reasoning)
            if text or reasoning:
                recent.append(text + reasoning)
                recent_len = self._trim_recent(recent, recent_len)
            warned_fill = self._enforce_budget(budget, out_chars, warned_fill)
            reason, last_check = self._check_degeneration(
                degeneration_check, recent, out_chars, last_check
            )
            if reason:
                raise self.degeneration_error_class(f"{self.label} degeneration detected: {reason}")
            if not continues:
                break
        if usage is not None:
            self._log_usage(budget, usage)
        if parts and self.stream:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return "".join(parts)

    def _trim_recent(self, recent: list[str], recent_len: int) -> int:
        """Keep the rolling (content+reasoning) buffer bounded.

        Only the last `degeneration_window + degeneration_interval` chars are
        ever needed: the check runs every `interval` chars against the last
        `window` chars, so older text can be dropped. Returns the new length.
        """
        budget = self.degeneration_window + self.degeneration_interval
        while recent and recent_len > budget:
            recent_len -= len(recent.pop(0))
        return recent_len

    def _check_degeneration(
        self,
        check: Callable[[str], str | None] | None,
        recent: list[str],
        out_chars: int,
        last_check: int | None,
    ) -> tuple[str | None, int | None]:
        """Run the degeneration callback on the last `window` chars of
        content+reasoning, every `interval` chars once `min_chars` have
        accumulated. Returns (reason, new_last_check); a non-None reason means
        the caller should abort."""
        if check is None or out_chars < self.degeneration_min_chars:
            return None, last_check
        if last_check is None or out_chars - last_check >= self.degeneration_interval:
            window = self.degeneration_window or len(recent)
            reason = check("".join(recent)[-window:])
            return reason, out_chars
        return None, last_check

    @staticmethod
    def _consume_frame(line: str) -> tuple[bool, str, str, dict | None]:
        """Process one SSE line.

        Returns (continue, content, reasoning, usage). `content` feeds the
        returned body; `reasoning` is only ever echoed; `usage` is the exact
        token usage carried on the final chunk when `include_usage` was
        requested (None otherwise). (False, "", "", None) on [DONE].
        """
        if not line.startswith("data:"):
            return True, "", "", None
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return False, "", "", None
        try:
            chunk = json.loads(data)
        except ValueError:
            return True, "", "", None
        usage = chunk.get("usage")
        if not isinstance(usage, dict):
            usage = None
        choices = chunk.get("choices") or []
        choice = choices[0] if choices else {}
        delta = choice.get("delta", {}) or {}
        text = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        return True, text, reasoning, usage

    @staticmethod
    def _stream_socket(response):
        """Reach the underlying socket across Python-version response shapes."""
        fp = getattr(response, "fp", response)
        raw = getattr(fp, "raw", None)
        sock = getattr(raw, "_sock", None) if raw is not None else None
        if sock is None:
            sock = getattr(fp, "sock", None)
        if sock is None:
            sock = getattr(response, "sock", None)
        return sock
