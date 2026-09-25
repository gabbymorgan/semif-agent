"""Skill code-body generation through an OpenAI-compatible API.

The decision engine authors a skill's title + description with the small model
(engine.generate). Writing the runnable body is a separate step: a larger
OpenAI-compatible model (e.g. qwen38-iq3s on ollama) is prompted with the
SKILL.md contract plus the request and the existing tree, and must reply with
valid Python implementing `predict` / `act`. Stdlib-only HTTP, mirroring
`llm.py`.
"""

from __future__ import annotations

import ast
import json
import select
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, NamedTuple

from .decisions import Request
from .skills import SkillDraft, tree_summary

DEFAULT_CONTRACT = Path(__file__).resolve().parent.parent / "SKILL.md"
DEFAULT_CONTEXT_WINDOW = 100000


class CodegenError(RuntimeError):
    """The codegen endpoint could not be reached."""


class DegenerationError(CodegenError):
    """A SemIf degeneration watchdog aborted the stream mid-generation.

    Subclass of CodegenError so the scheduler's graceful-stub path catches it;
    distinct so callers can choose to retry a degenerated stream later without
    retrying genuine endpoint failures.
    """


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


def read_skill_contract(path: str | None = None) -> str:
    contract = Path(path) if path else DEFAULT_CONTRACT
    if not contract.is_file():
        raise CodegenError(f"skill contract not found: {contract}")
    return contract.read_text()


def skill_contract_ref(path: str | None = None) -> dict:
    """Provenance pointer for the SKILL.md revision at authoring time.

    Returns {"ref": str | None, "dirty": bool | None}. `ref` is the short git
    commit sha the contract was read under — revivable with
    `git show <ref>:SKILL.md` — and `dirty` records whether the working-tree
    contract differed from that commit. Both are None when the contract is not
    inside a git checkout (the pointer degrades to nothing rather than a
    non-revivable hash). Any subprocess failure degrades the same way; only a
    missing contract file raises, mirroring `read_skill_contract`.
    """
    contract = Path(path) if path else DEFAULT_CONTRACT
    if not contract.is_file():
        raise CodegenError(f"skill contract not found: {contract}")
    directory = str(contract.parent)
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if rev.returncode != 0:
            return {"ref": None, "dirty": None}
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", contract.name],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return {"ref": None, "dirty": None}
    ref = rev.stdout.strip() or None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return {"ref": ref, "dirty": dirty}


class CodegenClient:
    """Minimal OpenAI-compatible chat client for writing skill bodies.

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

    Sampler defaults follow the Qwen3.8 model card's instruct-mode guidance
    (unsloth/Qwen3.8-27B-GGUF): the anti-repetition cure is a high
    `presence_penalty`, not greedy temperature. `temperature`/`top_p`/
    `presence_penalty`/`frequency_penalty` are per-`chat` overridable; the
    `generate_skill_body` escalation ladder bumps presence toward the card's
    max (2.0) and lowers temperature on retries. `max_attempts` bounds that
    ladder (default 3).
    """

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
        temperature: float = 0.7,
        top_p: float = 0.85,
        presence_penalty: float = 1.5,
        frequency_penalty: float = 0.2,
        max_attempts: int = 3,
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
        self._window: int | None = None

    def chat(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        degeneration_check: Callable[[str], str | None] | None = None,
    ) -> str:
        budget = self._compute_budget(messages)
        if self.stream:
            self._print_budget(budget)
        url = f"{self.base_url}/chat/completions"
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "presence_penalty": (
                self.presence_penalty if presence_penalty is None else presence_penalty
            ),
            "frequency_penalty": (
                self.frequency_penalty if frequency_penalty is None else frequency_penalty
            ),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return self._read_stream(response, budget, degeneration_check)
        except urllib.error.URLError as exc:
            raise CodegenError(
                f"codegen endpoint unreachable at {url}: {exc}. Is your local server running?"
            ) from exc
        except TimeoutError as exc:
            raise CodegenError(
                f"codegen request timed out after {self.timeout}s at {url}"
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
            headers={"Content-Type": "application/json"},
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
        """Detected context window, queried once and cached."""
        if self._window is not None:
            return self._window
        window = self._query_context_window()
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
            raise CodegenError(
                f"codegen prompt (~{prompt_est} tokens) already exceeds the "
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
                f"[codegen] context window {budget.window} tokens · "
                f"output cap {budget.output_limit} tokens "
                f"({budget.output_limit / budget.window * 100:.0f}%) · "
                f"peak total fill ~{budget.total_limit} tokens"
            )
        elif budget.window:
            print(
                f"[codegen] context window {budget.window} tokens · "
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
            f"[codegen] usage: prompt {prompt} · completion {completion} · "
            f"total {total} tokens · {pct:.0f}% of window · {zone}"
        )
        sys.stdout.flush()

    def _enforce_budget(self, budget: TokenBudget, out_chars: int, warned: bool) -> bool:
        """Abort when the output cap is exceeded; one-shot warn near the fill
        threshold. Returns the updated warned state."""
        output_est = out_chars / self.chars_per_token
        if budget.output_limit > 0 and output_est >= budget.output_limit:
            raise CodegenError(
                f"codegen output exceeded {budget.output_limit} token budget "
                f"(~{out_chars} chars at {self.chars_per_token:g} chars/token)"
            )
        if budget.warn_point is not None and not warned:
            total = budget.prompt_est + output_est
            if total >= budget.warn_point:
                sys.stdout.write(
                    f"\n[codegen] total fill ~{int(total)} tokens — "
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
            return self._read_stream_blocking(response, budget, degeneration_check)
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
            if time.monotonic() - start >= self.timeout:
                raise CodegenError(
                    f"codegen request exceeded {self.timeout:.0f}s total budget"
                )
            ready, _, _ = select.select([sock], [], [], 1.0)
            now = time.monotonic()
            if not ready:
                idle = now - last_activity
                if self.idle_warn > 0 and not warned and idle >= self.idle_warn:
                    sys.stdout.write(
                        f"\n[codegen] no tokens for {idle:.0f}s — still waiting, "
                        f"will fail after {self.idle_timeout:.0f}s of silence\n"
                    )
                    sys.stdout.flush()
                    warned = True
                if self.idle_timeout > 0 and idle >= self.idle_timeout:
                    raise CodegenError(
                        f"codegen stream stalled: no tokens for {idle:.0f}s "
                        f"(idle_timeout={self.idle_timeout:.0f}s)"
                    )
                continue
            try:
                raw = response.readline()
            except OSError as exc:
                raise CodegenError(f"codegen stream read failed: {exc}") from exc
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
                raise DegenerationError(f"codegen degeneration detected: {reason}")
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
    ) -> str:
        """Fallback reader when the response socket can't be located.

        A plain blocking read relies on the per-read socket timeout for
        silence, but that bounds individual reads, not the whole stream — so
        enforce the same total wall-clock budget `self.timeout` and the token
        budget in the loop, or a continuously-streaming runaway would never be
        cut short.
        """
        parts: list[str] = []
        start = time.monotonic()
        warned_fill = False
        out_chars = 0
        usage: dict | None = None
        recent: list[str] = []
        recent_len = 0
        last_check: int | None = None
        for raw in response:
            if time.monotonic() - start >= self.timeout:
                raise CodegenError(
                    f"codegen request exceeded {self.timeout:.0f}s total budget"
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
                raise DegenerationError(f"codegen degeneration detected: {reason}")
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


def build_skill_body_prompt(
    request: Request,
    category: str,
    draft: SkillDraft,
    tree: dict,
    contract: str,
) -> list[dict]:
    """Messages for the code-generation model.

    The small model already chose the title + description; the big model only
    writes the runnable body against the SKILL.md contract, informed by the
    request, the category, and the existing skills so it avoids duplication.
    """
    existing = ", ".join(s.name for s in tree.get(category, [])) or "(none)"
    system = (
        "You write runnable skill bodies for a local agent. The contract below "
        "is authoritative: follow it exactly.\n\n"
        f"{contract}"
    )
    user = (
        f"Request: {request.text}\n"
        f"Category: {category}\n"
        f"Skill name: {draft.name}\n"
        f"Skill description: {draft.description}\n"
        f"Existing skills in this category: {existing}\n"
        f"Existing categories:\n{tree_summary(tree)}\n"
        "This body will be reused across many requests: give the skill its own "
        "internal/mock data model and never ask the human for operational data — "
        "ask only to clarify intent.\n"
        "Write the Python module body now. Reply with ONLY valid Python code "
        "defining `predict` and `act`. No prose, no markdown fences, no JSON."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_skill_body(raw: str) -> str:
    """Extract and validate a Python skill body from the model's reply.

    Accepts bare code, ```fenced``` code, or JSON {"code": "..."}. The body
    must parse and must define module-level `predict` and `act` functions.
    Returns the cleaned source. Raises ValueError otherwise.
    """
    text = raw.strip()
    if "code" in text[:400] and "{" in text and "}" in text:
        start, end = text.find("{"), text.rfind("}")
        try:
            parsed = json.loads(text[start : end + 1])
            candidate = parsed.get("code")
            if isinstance(candidate, str):
                text = candidate.strip()
        except (ValueError, AttributeError):
            pass
    for fence in ("```python", "```py", "```"):
        start = text.find(fence)
        if start != -1:
            text = text[start + len(fence):]
            close = text.rfind("```")
            if close != -1:
                text = text[:close]
            break
    text = text.strip()
    if not text:
        raise ValueError("skill body is empty")
    try:
        module = ast.parse(text, filename="<generated>")
    except SyntaxError as exc:
        raise ValueError(f"skill body is not valid Python: {exc}") from exc
    names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    for required in ("predict", "act"):
        if required not in names:
            raise ValueError(f"skill body must define a module-level `{required}` function")
    return text


ESCALATED_SAMPLER = {
    "temperature": 0.5,
    "top_p": 0.85,
    "presence_penalty": 2.0,
    "frequency_penalty": 0.3,
}


def _retry_prompt(
    request: Request,
    category: str,
    draft: SkillDraft,
    contract: str,
    error: Exception,
) -> list[dict]:
    """Fresh short prompt for an escalated retry.

    Not a growing conversation: the context resets to SMART (new request, tiny
    prompt) and the model is told in one line to stop looping and emit the
    final Python.
    """
    system = (
        "You write runnable skill bodies for a local agent. The contract below "
        "is authoritative: follow it exactly.\n\n"
        f"{contract}"
    )
    user = (
        f"The previous attempt to write the skill body for {category}.{draft.name} "
        f"was rejected: {error}. You are looping; emit the final Python now.\n"
        f"Request: {request.text}\n"
        f"Category: {category}\n"
        f"Skill name: {draft.name}\n"
        f"Skill description: {draft.description}\n"
        "This body will be reused across many requests: give the skill its own "
        "internal/mock data model and never ask the human for operational data — "
        "ask only to clarify intent.\n"
        "Reply with ONLY valid Python defining `predict` and `act`. No prose, "
        "no markdown fences, no JSON."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_skill_body(
    client: CodegenClient,
    request: Request,
    category: str,
    draft: SkillDraft,
    tree: dict,
    contract: str | None = None,
    max_tokens: int | None = None,
    degeneration_check: Callable[[str], str | None] | None = None,
) -> str:
    """Author a skill body with the big model; escalate up to `max_attempts`.

    Attempt 1 uses the client's sampler defaults. Attempts >= 2 use the
    escalated sampler (higher presence_penalty to suppress repeated thinking,
    lower temperature for a decisive final answer) plus a fresh short prompt
    that resets the context to SMART and tells the model to emit the final
    Python now. Only an invalid parse (ValueError) escalates; a
    DegenerationError or other CodegenError propagates immediately so the
    scheduler can stub out gracefully.
    """
    contract_text = contract if contract is not None else read_skill_contract()
    if client.stream:
        print(f"[codegen] writing body for {category}.{draft.name}...")
    last_error: Exception | None = None
    attempts = max(client.max_attempts, 1)
    for attempt in range(1, attempts + 1):
        if attempt == 1:
            messages = build_skill_body_prompt(
                request, category, draft, tree, contract_text
            )
        else:
            messages = _retry_prompt(
                request, category, draft, contract_text, last_error
            )
        try:
            raw = client.chat(
                messages,
                max_tokens=max_tokens,
                degeneration_check=degeneration_check,
                **(ESCALATED_SAMPLER if attempt > 1 else {}),
            )
            return parse_skill_body(raw)
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"skill body rejected {attempts} times: {last_error}")