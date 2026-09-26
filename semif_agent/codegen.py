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
import re
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


BODY_DIRECTIVES = (
    "This body is a reusable module executed across many requests. It owns no "
    "working data: every operational value is provided by the runner through "
    "`ctx.config` under a clear snake_case name — request data from the runner, "
    "never embed or fabricate working values, and never ask the human for "
    "operational data. Ask the human only to refine the product goal and "
    "requirements.\n"
    "Perform the real action: if this skill interacts with an external system, "
    "`act` must call the user's configured service with a stdlib transport "
    "(urllib.request/http.client for HTTP, imaplib/smtplib/poplib for mail, "
    "subprocess for a configured local CLI) using endpoint, account, "
    "credential, and CLI-path values read from `ctx.config`. Never simulate "
    "success, never return canned output, and never default to writing a draft "
    "unless the requirements explicitly ask for draft-only. The auto-run test "
    "is a hermetic mechanics check and will not exercise the live service, so "
    "no localhost defaults, test ports, or fixture values may appear in the "
    "body.\n"
    "Declare the integration as a module-level `INTEGRATION` dict exactly as "
    "SKILL.md specifies: service, transport, config_vars.\n"
)


def _requirements_block(requirements: dict[str, str] | None) -> str:
    if not requirements:
        return ""
    lines = "\n".join(
        f"- {question} -> {answer}" for question, answer in requirements.items()
    )
    return f"Requirements gathered from the product owner:\n{lines}\n"


def build_skill_body_prompt(
    request: Request,
    category: str,
    draft: SkillDraft,
    tree: dict,
    contract: str,
    requirements: dict[str, str] | None = None,
) -> list[dict]:
    """Messages for the code-generation model.

    The small model already chose the title + description; the big model only
    writes the runnable body against the SKILL.md contract, informed by the
    request, the category, and the existing skills so it avoids duplication.
    `requirements` carries the answers to elicitation questions asked of the
    human during authoring.
    """
    existing = ", ".join(s.name for s in tree.get(category, [])) or "(none)"
    system = (
        "You write runnable skill bodies for a local agent. The contract below "
        "is authoritative: follow it exactly.\n\n"
        f"{contract}"
    )
    req_block = _requirements_block(requirements)
    user = (
        f"Request: {request.text}\n"
        f"Category: {category}\n"
        f"Skill name: {draft.name}\n"
        f"Skill description: {draft.description}\n"
        f"Existing skills in this category: {existing}\n"
        f"Existing categories:\n{tree_summary(tree)}\n"
        f"{req_block}"
        f"{BODY_DIRECTIVES}"
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


# ---- integration declaration ----

INTEGRATION_TRANSPORTS = (
    "http",
    "caldav",
    "imap",
    "smtp",
    "pop",
    "subprocess",
    "file",
    "compute",
)
_SERVICE_RE = re.compile(r"[a-z0-9_]+")


def _const_str(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_ctx_config(node) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "config"
        and isinstance(node.value, ast.Name)
        and node.value.id == "ctx"
    )


def _config_reads(code: str) -> list[str]:
    """Names the body pulls out of `ctx.config` (subscripts and .get calls)."""
    try:
        module = ast.parse(code, filename="<integration>")
    except SyntaxError:
        return []
    reads: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Subscript) and _is_ctx_config(node.value):
            key = _const_str(node.slice)
            if key:
                reads.append(key)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_ctx_config(node.func.value)
            and node.args
        ):
            key = _const_str(node.args[0])
            if key:
                reads.append(key)
    return reads


def _import_names(module: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _transport_evidence(code: str) -> set[str]:
    """Stdlib transports the body visibly uses, from imports and calls."""
    try:
        module = ast.parse(code, filename="<integration>")
    except SyntaxError:
        return set()
    names = _import_names(module)
    calls = {
        node.func.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    calls.update(
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )
    evidence: set[str] = set()
    if (
        any(name.startswith(("urllib.request", "urllib.error", "http.client")) for name in names)
        or "urllib" in names
        or "http" in names
    ):
        evidence.add("http")
    if "imaplib" in names:
        evidence.add("imap")
    if "smtplib" in names:
        evidence.add("smtp")
    if "poplib" in names:
        evidence.add("pop")
    if "subprocess" in names:
        evidence.add("subprocess")
    if "pathlib" in names or calls & {"open", "write_text", "write_bytes"}:
        evidence.add("file")
    return evidence


def _validate_integration(integration: dict) -> None:
    service = integration.get("service")
    if not isinstance(service, str) or not _SERVICE_RE.fullmatch(service):
        raise ValueError("INTEGRATION['service'] must be a lowercase snake_case id")
    transport = integration.get("transport")
    if transport not in INTEGRATION_TRANSPORTS:
        raise ValueError(
            f"INTEGRATION['transport'] must be one of {', '.join(INTEGRATION_TRANSPORTS)}"
        )
    config_vars = integration.get("config_vars")
    if not isinstance(config_vars, list) or not all(
        isinstance(item, str) for item in config_vars
    ):
        raise ValueError("INTEGRATION['config_vars'] must be a list of strings")


def parse_integration(code: str) -> dict:
    """Validate the body's module-level INTEGRATION declaration.

    Returns {"service": str, "transport": str, "config_vars": [str, ...]}.
    Raises ValueError when the constant is missing or malformed.
    """
    try:
        module = ast.parse(code, filename="<generated>")
    except SyntaxError as exc:
        raise ValueError(f"skill body is not valid Python: {exc}") from exc
    for node in module.body:
        value = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "INTEGRATION"
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "INTEGRATION"
        ):
            value = node.value
        if value is None:
            continue
        if not isinstance(value, ast.Dict):
            raise ValueError("INTEGRATION must be a flat dict literal")
        integration: dict = {}
        for key_node, value_node in zip(value.keys, value.values):
            key = _const_str(key_node) if key_node is not None else None
            if not key:
                raise ValueError("INTEGRATION keys must be string literals")
            if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                integration[key] = value_node.value
            elif key == "config_vars" and isinstance(value_node, (ast.List, ast.Tuple)):
                parsed_vars = []
                for element in value_node.elts:
                    item = _const_str(element)
                    if item is None:
                        raise ValueError(
                            "INTEGRATION['config_vars'] must be a list of string literals"
                        )
                    parsed_vars.append(item)
                integration[key] = parsed_vars
            else:
                raise ValueError(
                    f"INTEGRATION[{key!r}] must be a string or a list of strings"
                )
        _validate_integration(integration)
        return integration
    raise ValueError("skill body must define a module-level INTEGRATION dict")


def infer_integration(code: str) -> dict:
    """Best-effort integration for a body without a valid declaration.

    Used so an undeclared body is still inspectable and reviewable; the caller
    records the inference as `unverified` rather than trusting it.
    """
    evidence = _transport_evidence(code)
    for transport in ("http", "imap", "smtp", "pop", "subprocess", "file"):
        if transport in evidence:
            selected = transport
            break
    else:
        selected = "compute"
    return {
        "service": "unknown",
        "transport": selected,
        "config_vars": _config_reads(code),
    }


def extract_integration(code: str) -> tuple[dict, str]:
    """The body's integration plus its source: `declared` or `inferred`."""
    try:
        return parse_integration(code), "declared"
    except ValueError:
        return infer_integration(code), "inferred"


def integration_findings(integration: dict, source: str, code: str) -> list[str]:
    """Static mismatches between the declaration and the body.

    Empty list means the declaration is consistent with what the code does;
    findings feed the fidelity review's corrective regen reason.
    """
    findings: list[str] = []
    if source != "declared":
        findings.append("the body does not define a valid module-level INTEGRATION dict")
    transport = integration.get("transport")
    if transport and transport != "compute":
        evidence = _transport_evidence(code)
        accepted = {transport, "http"} if transport == "caldav" else {transport}
        if not (evidence & accepted):
            findings.append(
                f"INTEGRATION declares transport {transport!r} but the body shows no "
                f"{transport} calls"
            )
    declared_vars = integration.get("config_vars") or []
    reads = set(_config_reads(code))
    missing = sorted(set(declared_vars) - reads)
    if missing:
        findings.append(
            f"INTEGRATION.config_vars {missing} are never read from ctx.config"
        )
    return findings


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
    requirements: dict[str, str] | None = None,
) -> list[dict]:
    """Fresh short prompt for an escalated retry.

    Not a growing conversation: the context resets to SMART (new request, tiny
    prompt) and the model is told in one line to stop looping and emit the
    final Python. The elicitation answers are repeated so a retry cannot fall
    back to a toy body.
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
        f"{_requirements_block(requirements)}"
        f"{BODY_DIRECTIVES}"
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
    requirements: dict[str, str] | None = None,
) -> str:
    """Author a skill body with the big model; escalate up to `max_attempts`.

    Attempt 1 uses the client's sampler defaults. Attempts >= 2 use the
    escalated sampler (higher presence_penalty to suppress repeated thinking,
    lower temperature for a decisive final answer) plus a fresh short prompt
    that resets the context to SMART and tells the model to emit the final
    Python now. Only an invalid parse (ValueError) escalates; a
    DegenerationError or other CodegenError propagates immediately so the
    scheduler can stub out gracefully. `requirements` (elicitation answers) are
    fed to the first-attempt prompt only.
    """
    contract_text = contract if contract is not None else read_skill_contract()
    if client.stream:
        print(f"[codegen] writing body for {category}.{draft.name}...")
    last_error: Exception | None = None
    attempts = max(client.max_attempts, 1)
    for attempt in range(1, attempts + 1):
        if attempt == 1:
            messages = build_skill_body_prompt(
                request,
                category,
                draft,
                tree,
                contract_text,
                requirements=requirements,
            )
        else:
            messages = _retry_prompt(
                request, category, draft, contract_text, last_error, requirements
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


# ---- requirements elicitation ----

ELICITATION_EXAMPLES = [
    "Should this skill really send/query/execute against the live service, or produce a reviewable draft first?",
    "Which service or account should it use, and how should it connect (HTTP API base URL, IMAP/SMTP host, CalDAV URL, local CLI command)?",
    "How does it authenticate with the service, and should that credential come from the app's config?",
    "What does a successful run look like — what should it report back to you?",
    "If the service is unreachable or refuses the action, should the run fail loudly or record 'could not complete' as the result?",
]


def build_elicitation_prompt(
    request: Request,
    category: str,
    draft: SkillDraft,
    tree: dict,
    max_questions: int = 4,
) -> list[dict]:
    """Messages asking the big model to propose implementation questions.

    The agent does real tasks against the user's actual services, so success
    depends on pinning down how the new skill will connect. The questions
    refine the product goal and the integration — never operational data values
    (the runner collects those after the body is written) and never
    config-vs-input cadence questions (the config step decides that at first
    fire). The examples are deliberately concrete so a weaker model can mirror
    them.
    """
    examples = "\n".join(f"- \"{q}\"" for q in ELICITATION_EXAMPLES)
    existing = ", ".join(s.name for s in tree.get(category, [])) or "(none)"
    system = (
        "You are eliciting requirements for a new skill in a local agent. The "
        "agent performs real tasks for an everyday person against their actual "
        "services, so success depends on understanding HOW the skill will "
        "connect. You ask the human (the product owner) up to "
        f"{max_questions} questions that refine the product goal and the "
        "integration. Ask about product behavior and integration, not code.\n\n"
        "Good example questions:\n"
        f"{examples}\n\n"
        "Never ask:\n"
        "- for operational data values (passwords, API keys, addresses, "
        "tracking numbers, hostnames) — the runner collects those after the "
        "body is written;\n"
        "- config-vs-input cadence questions like \"should the sender address "
        "change?\" — the config step owns that decision at first fire;\n"
        "- trivia you can decide yourself (variable names, formatting, "
        "internal wording).\n\n"
        "If the request and description already make the integration clear, "
        "return an empty list rather than inventing questions.\n\n"
        "Also give your best guess of the integration the questions are aiming "
        "at, so the body writer implements exactly that. Reply with JSON only: "
        '{"questions": ["...", "..."], "integration": {"service": '
        '"<lowercase_snake_case or unknown>", "transport": '
        '"<http|caldav|imap|smtp|pop|subprocess|file|compute>"}}'
    )
    user = (
        f"Request: {request.text}\n"
        f"Category: {category}\n"
        f"Skill name: {draft.name}\n"
        f"Skill description: {draft.description}\n"
        f"Existing skills in this category: {existing}\n"
        f"Existing categories:\n{tree_summary(tree)}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class ElicitationResult(NamedTuple):
    """Questions for the human plus the integration they are aiming at."""

    questions: list[str]
    integration: dict


def _parse_elicitation_object(raw: str) -> dict:
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"elicitation reply is not a JSON object: {raw!r}")
    try:
        return json.loads(text[start : end + 1])
    except ValueError as exc:
        raise ValueError(f"elicitation reply is not valid JSON: {exc}") from exc


def parse_elicitation(raw: str) -> list[str]:
    """Extract a JSON {"questions": [...]} list of non-empty strings."""
    parsed = _parse_elicitation_object(raw)
    questions = parsed.get("questions")
    if not isinstance(questions, list):
        raise ValueError("elicitation reply must have a `questions` list")
    cleaned = []
    for question in questions:
        if isinstance(question, str) and question.strip():
            cleaned.append(question.strip())
    return cleaned


def parse_elicitation_result(raw: str) -> ElicitationResult:
    """Questions plus the model's integration hint (service + transport).

    A bad or absent hint is dropped, never fatal: the questions are the point
    and the body writer can still declare the integration itself.
    """
    parsed = _parse_elicitation_object(raw)
    integration: dict = {}
    hint = parsed.get("integration")
    if isinstance(hint, dict):
        service = hint.get("service")
        transport = hint.get("transport")
        if (
            isinstance(service, str)
            and _SERVICE_RE.fullmatch(service)
            and transport in INTEGRATION_TRANSPORTS
        ):
            integration = {"service": service, "transport": transport}
    return ElicitationResult(parse_elicitation(raw), integration)


def _retry_elicitation_prompt(
    request: Request,
    category: str,
    draft: SkillDraft,
    tree: dict,
    error: Exception,
    max_questions: int = 4,
) -> list[dict]:
    messages = build_elicitation_prompt(
        request, category, draft, tree, max_questions=max_questions
    )
    messages[1]["content"] += (
        f"\nYour previous reply was rejected: {error}. Reply with JSON only: "
        '{"questions": ["...", "..."], "integration": {"service": "...", '
        '"transport": "..."}}'
    )
    return messages


def generate_elicitation(
    client: CodegenClient,
    request: Request,
    category: str,
    draft: SkillDraft,
    tree: dict,
    max_questions: int = 4,
) -> ElicitationResult:
    """Propose implementation questions + an integration hint with the big model.

    Returns an empty result when the model produced none or the reply failed to
    parse after the escalation ladder — elicitation must never block authoring.
    """
    attempts = max(client.max_attempts, 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if attempt == 1:
            messages = build_elicitation_prompt(
                request, category, draft, tree, max_questions=max_questions
            )
        else:
            messages = _retry_elicitation_prompt(
                request, category, draft, tree, last_error, max_questions=max_questions
            )
        try:
            raw = client.chat(
                messages,
                **(ESCALATED_SAMPLER if attempt > 1 else {}),
            )
            result = parse_elicitation_result(raw)
            return ElicitationResult(
                result.questions[: max(0, max_questions)], result.integration
            )
        except ValueError as exc:
            last_error = exc
    return ElicitationResult([], {})


def generate_requirements(
    client: CodegenClient,
    request: Request,
    category: str,
    draft: SkillDraft,
    tree: dict,
    max_questions: int = 4,
) -> list[str]:
    """Question-only view of generate_elicitation (kept for callers/tests)."""
    return generate_elicitation(
        client, request, category, draft, tree, max_questions=max_questions
    ).questions


# ---- data contract + test ----

TESTGEN_CONTRACT = Path(__file__).resolve().parent.parent / "TESTGEN.md"
CONTRACT_VARIABLE_RE = re.compile(r"[a-z0-9_]+")


def read_testgen_contract(path: str | None = None) -> str:
    contract = Path(path) if path else TESTGEN_CONTRACT
    if not contract.is_file():
        raise CodegenError(f"testgen contract not found: {contract}")
    return contract.read_text()


def build_testgen_base_prompt(
    request: Request, category: str, name: str, skill_code: str, contract_md: str
) -> list[dict]:
    """The shared context for the contract + test generation calls.

    Seeded with the TESTGEN.md contract and the finished skill body; the
    contract call appends its ask, and the testgen call appends the contract
    output (as the assistant turn) plus its own ask — the two calls share this
    context but produce distinct artifacts.
    """
    system = (
        "You produce the data contract and test for a skill body of a "
        "local agent. The contract below is authoritative: follow it exactly.\n\n"
        f"{contract_md}"
    )
    user = (
        f"Skill: {category}.{name}\n"
        f"Request: {request.text}\n"
        f"Finished skill body:\n```python\n{skill_code}\n```"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_contract_request_prompt(
    base: list[dict], note: str | None = None
) -> list[dict]:
    user = "Generate the data contract now: a single JSON object."
    if note:
        user += f"\nNote: {note}"
    user += "\nReply with ONLY the JSON object. No prose, no markdown fences."
    return base + [{"role": "user", "content": user}]


def parse_data_contract(raw: str) -> dict:
    """Validate a flat semantic contract: one JSON object whose keys are
    snake_case variable names and whose values are non-empty descriptions."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"data contract is not a JSON object: {raw!r}")
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError as exc:
        raise ValueError(f"data contract is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("data contract must be a single JSON object")
    for key, value in parsed.items():
        if not CONTRACT_VARIABLE_RE.fullmatch(key):
            raise ValueError(f"contract variable {key!r} must be lowercase snake_case")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"contract value for {key!r} must be a semantic description string")
    return parsed


def generate_data_contract(
    client: CodegenClient,
    request: Request,
    category: str,
    draft: SkillDraft,
    skill_code: str,
    contract_md: str | None = None,
    reason: str | None = None,
) -> dict:
    """Derive the flat data contract from the finished skill body.

    Reads every `ctx.config[...]` operational access in the body and expresses
    each as a variable name -> semantic description. Escalates like the body
    writer on parse failure; `reason` (a failing-test error) is fed on regen.
    """
    contract_text = contract_md if contract_md is not None else read_testgen_contract()
    attempts = max(client.max_attempts, 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if attempt == 1:
            note = reason
        else:
            note = str(last_error)
        messages = build_contract_request_prompt(
            build_testgen_base_prompt(
                request, category, draft.name, skill_code, contract_text
            ),
            note=note,
        )
        try:
            raw = client.chat(
                messages,
                **(ESCALATED_SAMPLER if attempt > 1 else {}),
            )
            return parse_data_contract(raw)
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"data contract rejected {attempts} times: {last_error}")


def build_testgen_request_prompt(
    base: list[dict],
    contract: dict,
    note: str | None = None,
    target: str | None = None,
) -> list[dict]:
    user = (
        "Generate the test now: skill.test.py, a stdlib-only Python script, "
        "runnable as `python skill.test.py` from the skill folder (exit 0 on "
        "pass, non-zero on failure). It is a HERMETIC mechanics check: no "
        "external network, no real services, fixture data embedded directly in "
        "the test as inline Python literals, no external files, no writes "
        "outside the folder. For a body that performs HTTP in `act`, stand up a "
        "real loopback `http.server` on 127.0.0.1 (ephemeral port), pass its URL "
        "through the fixture config under the same key the body reads, and "
        "assert the server received the expected request. For non-HTTP "
        "transports, exercise config resolution, argument construction, and "
        "error paths without real I/O. Never simulate the action to make the "
        "test pass; the test does not verify the live service."
    )
    if target:
        user += f"\nFocus the fix on the {target.removeprefix('regen_')}."
    if note:
        user += f"\nNote: {note}"
    user += (
        "\nReply with ONLY valid Python for skill.test.py. No prose, no "
        "markdown fences, no JSON."
    )
    return base + [
        {"role": "assistant", "content": json.dumps(contract)},
        {"role": "user", "content": user},
    ]


def parse_skill_test(raw: str) -> str:
    """Extract and validate skill.test.py from the model's reply.

    Accepts bare code, ```fenced``` code, or JSON {"code": "..."}. The test
    must parse as Python. Returns the cleaned source. Raises ValueError
    otherwise.
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
        raise ValueError("skill test is empty")
    try:
        ast.parse(text, filename="<generated test>")
    except SyntaxError as exc:
        raise ValueError(f"skill test is not valid Python: {exc}") from exc
    return text


def generate_skill_tests(
    client: CodegenClient,
    request: Request,
    category: str,
    draft: SkillDraft,
    skill_code: str,
    contract: dict,
    contract_md: str | None = None,
    reason: str | None = None,
    target: str | None = None,
) -> str:
    """Generate skill.test.py against the finished body.

    Shares the base context with the contract call (TESTGEN.md + skill.py) and
    continues it with the accepted contract, so the test stays compatible with
    the body and the contract. Fixture data is embedded inline in the test.
    Escalates on parse failure; `reason` + `target` drive a corrective
    regeneration.
    """
    contract_text = contract_md if contract_md is not None else read_testgen_contract()
    attempts = max(client.max_attempts, 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if attempt == 1:
            note = reason
        else:
            note = str(last_error)
        messages = build_testgen_request_prompt(
            build_testgen_base_prompt(
                request, category, draft.name, skill_code, contract_text
            ),
            contract,
            note=note,
            target=target,
        )
        try:
            raw = client.chat(
                messages,
                **(ESCALATED_SAMPLER if attempt > 1 else {}),
            )
            return parse_skill_test(raw)
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"testgen reply rejected {attempts} times: {last_error}")


def _regen_body_prompt(
    request: Request,
    category: str,
    draft: SkillDraft,
    contract: str,
    previous_code: str,
    note: str,
    requirements: dict[str, str] | None = None,
    reason_kind: str = "test",
) -> list[dict]:
    system = (
        "You write runnable skill bodies for a local agent. The contract below "
        "is authoritative: follow it exactly.\n\n"
        f"{contract}"
    )
    headers = {
        "test": (
            f"The auto-run test for {category}.{draft.name} failed. Rewrite "
            "the body so it passes."
        ),
        "fidelity": (
            f"A review of {category}.{draft.name} rejected the body: it does "
            "not really perform the requested action. Rewrite it so `act` "
            "performs the real operation against the configured service."
        ),
        "run_failure": (
            f"A real run of {category}.{draft.name} failed against the "
            "service. Rewrite the body so it handles the observed failure "
            "correctly."
        ),
    }
    failure_labels = {
        "fidelity": "Review finding",
        "run_failure": "Run failure",
    }
    header = headers.get(reason_kind, headers["test"])
    failure_label = failure_labels.get(reason_kind, "Test failure")
    user = (
        f"{header}\n"
        f"Request: {request.text}\n"
        f"Skill description: {draft.description}\n"
        f"{_requirements_block(requirements)}"
        f"{failure_label}: {note}\n"
        f"Previous body:\n```python\n{previous_code}\n```\n"
        f"{BODY_DIRECTIVES}"
        "Reply with ONLY valid Python defining `predict` and `act`. No prose, "
        "no markdown fences, no JSON."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def regenerate_skill_body(
    client: CodegenClient,
    request: Request,
    category: str,
    draft: SkillDraft,
    previous_code: str,
    reason: str,
    contract: str | None = None,
    requirements: dict[str, str] | None = None,
    reason_kind: str = "test",
) -> str:
    """Rewrite a skill body whose test, review, or real run failed.

    Fresh prompt (context resets) carrying the failure and the previous body;
    escalation ladder identical to generate_skill_body. `reason_kind` picks the
    corrective framing (`test`, `fidelity`, or `run_failure`); the elicitation
    answers ride along so the repair cannot regress to a toy. A
    DegenerationError or other CodegenError propagates immediately.
    """
    contract_text = contract if contract is not None else read_skill_contract()
    attempts = max(client.max_attempts, 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        note = reason if attempt == 1 else str(last_error)
        messages = _regen_body_prompt(
            request, category, draft, contract_text, previous_code, note,
            requirements=requirements, reason_kind=reason_kind,
        )
        try:
            raw = client.chat(
                messages,
                **(ESCALATED_SAMPLER if attempt > 1 else {}),
            )
            return parse_skill_body(raw)
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"skill body rejected {attempts} times: {last_error}")


def run_skill_test(skill_dir: str | Path, timeout: float = 30.0) -> tuple[bool, str]:
    """Run skill.test.py in its folder as a subprocess.

    cwd is the skill folder so the test can import the `skill` module; its
    fixture data is embedded inline, so no external files are needed. Returns
    (passed, captured output). A non-zero exit or a timeout is a failure; the
    output feeds the regen decision and corrective prompts.
    """
    cwd = Path(skill_dir)
    script = cwd / "skill.test.py"
    if not script.is_file():
        return False, "no skill.test.py found in the skill folder"
    try:
        proc = subprocess.run(
            [sys.executable, "skill.test.py"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=max(timeout, 1.0),
        )
    except subprocess.TimeoutExpired:
        return False, f"skill test timed out after {timeout:.0f}s"
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output