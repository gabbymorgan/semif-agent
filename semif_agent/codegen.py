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
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .decisions import Request
from .skills import SkillDraft, tree_summary

DEFAULT_CONTRACT = Path(__file__).resolve().parent.parent / "SKILL.md"


class CodegenError(RuntimeError):
    """The codegen endpoint could not be reached."""


def read_skill_contract(path: str | None = None) -> str:
    contract = Path(path) if path else DEFAULT_CONTRACT
    if not contract.is_file():
        raise CodegenError(f"skill contract not found: {contract}")
    return contract.read_text()


class CodegenClient:
    """Minimal OpenAI-compatible chat client for writing skill bodies.

    No token cap by default: Qwen3-style models reason first and the cap
    truncates the hidden reasoning, leaving `content` empty. Omit `max_tokens`
    so the model runs to completion; the reasoning is filtered automatically
    because only `content` is read.

    With `stream=True` the response is read as an SSE token stream and echoed
    to stdout as it arrives — including the chain-of-thought — so a long
    body write shows live progress. Echoing is console-only; the returned
    content is identical either way. A silent stream (no bytes for
    `idle_warn` seconds) prints a warning, and one that stays silent for
    `idle_timeout` seconds raises CodegenError instead of blocking on the
    total timeout — a wedged generation is surfaced in minutes, not ~20.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 1200.0,
        stream: bool = False,
        idle_warn: float = 60.0,
        idle_timeout: float = 180.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.stream = stream
        self.idle_warn = idle_warn
        self.idle_timeout = idle_timeout

    def chat(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": self.stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if self.stream:
                    return self._read_stream(response)
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise CodegenError(
                f"codegen endpoint unreachable at {url}: {exc}. Is your local server running?"
            ) from exc
        except TimeoutError as exc:
            raise CodegenError(
                f"codegen request timed out after {self.timeout}s at {url}"
            ) from exc
        return payload["choices"][0]["message"]["content"]

    def _read_stream(self, response) -> str:
        """Read an OpenAI-compatible SSE stream, echo tokens to stdout.

        Only `content` deltas are accumulated into the returned body;
        reasoning (chain-of-thought) is echoed to the console but never part
        of the result. Backends disagree on the field name: ollama streams it
        as `reasoning`, DeepSeek/vllm-style as `reasoning_content`, so read both.

        Reads are gated by `select` so the socket never blocks-and-times-out:
        a socket that delivers no bytes for `idle_warn` seconds prints a
        warning, and `idle_timeout` seconds of silence raises CodegenError.
        The total wall-clock budget is `self.timeout`, so a slow-but-streaming
        generation is never cut short. Thresholds <= 0 disable that check. If
        the underlying socket can't be reached, falls back to a plain blocking
        read (relying on the outer TimeoutError handling).
        """
        parts: list[str] = []
        sock = self._stream_socket(response)
        if sock is None:
            return self._read_stream_blocking(response)
        start = time.monotonic()
        last_activity = start
        warned = False
        while True:
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
                if now - start >= self.timeout:
                    raise CodegenError(
                        f"codegen request exceeded {self.timeout:.0f}s total budget"
                    )
                continue
            try:
                raw = response.readline()
            except OSError as exc:
                raise CodegenError(f"codegen stream read failed: {exc}") from exc
            if not raw:
                break
            last_activity = time.monotonic()
            if not self._consume_frame(raw.decode("utf-8").strip(), parts):
                break
        if parts:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return "".join(parts)

    def _read_stream_blocking(self, response) -> str:
        """Fallback reader when the response socket can't be located."""
        parts: list[str] = []
        for raw in response:
            if not self._consume_frame(raw.decode("utf-8").strip(), parts):
                break
        if parts:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return "".join(parts)

    @staticmethod
    def _consume_frame(line: str, parts: list[str]) -> bool:
        """Process one SSE line. Returns False on [DONE] (stop reading)."""
        if not line.startswith("data:"):
            return True
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return False
        try:
            chunk = json.loads(data)
        except ValueError:
            return True
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {}) or {}
        text = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if text or reasoning:
            sys.stdout.write(text + reasoning)
            sys.stdout.flush()
        parts.append(text)
        return True

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


def generate_skill_body(
    client: CodegenClient,
    request: Request,
    category: str,
    draft: SkillDraft,
    tree: dict,
    contract: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Author a skill body with the big model; retries once on invalid output."""
    contract_text = contract if contract is not None else read_skill_contract()
    messages = build_skill_body_prompt(request, category, draft, tree, contract_text)
    if client.stream:
        print(f"[codegen] writing body for {category}.{draft.name}...")
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            raw = client.chat(messages, max_tokens=max_tokens)
            return parse_skill_body(raw)
        except ValueError as exc:
            last_error = exc
            messages = messages + [
                {
                    "role": "user",
                    "content": (
                        f"That was rejected: {exc}. Reply with ONLY the Python code "
                        "now — no prose, no fences, no JSON."
                    ),
                }
            ]
    raise ValueError(f"skill body rejected twice: {last_error}")