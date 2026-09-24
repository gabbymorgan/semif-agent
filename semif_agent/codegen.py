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
    """Minimal OpenAI-compatible chat client for writing skill bodies."""

    def __init__(self, base_url: str, model: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict], max_tokens: int = 2048, temperature: float = 0.0) -> str:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise CodegenError(
                f"codegen endpoint unreachable at {url}: {exc}. Is your local server running?"
            ) from exc
        return payload["choices"][0]["message"]["content"]


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
    max_tokens: int = 2048,
) -> str:
    """Author a skill body with the big model; retries once on invalid output."""
    contract_text = contract if contract is not None else read_skill_contract()
    messages = build_skill_body_prompt(request, category, draft, tree, contract_text)
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