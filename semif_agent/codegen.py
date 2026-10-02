"""Skill code-body generation through an OpenAI-compatible API.

The small `llm` provider authors a skill's title + description; writing the
runnable body is a separate step: a larger OpenAI-compatible model (a
code-capable model on your configured codegen endpoint) is prompted with the SKILL.md contract plus the request
and the existing tree, and must reply with valid Python implementing `act` plus
the `INTEGRATION` and `CONTRACT` module constants. The shared provider transport
lives in `provider.py`.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, NamedTuple

from .bridges.registry import describe_bridges
from .decisions import Request
from .provider import OpenAICompatClient, ProviderError
from .skills import SkillDraft, tree_summary

DEFAULT_CONTRACT = Path(__file__).resolve().parent.parent / "SKILL.md"


class CodegenError(ProviderError):
    """The codegen endpoint could not be reached."""


class DegenerationError(CodegenError):
    """A SemIf degeneration watchdog aborted the stream mid-generation.

    Subclass of CodegenError so the scheduler's graceful-stub path catches it;
    distinct so callers can choose to retry a degenerated stream later without
    retrying genuine endpoint failures.
    """


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


class CodegenClient(OpenAICompatClient):
    """OpenAI-compatible chat client that writes skill bodies.

    The provider machinery (real SSE transport, layered token budget, idle
    watchdog, degeneration hook, `/api/show` context detection) lives in
    `provider.OpenAICompatClient`; this subclass pins the codegen error types
    and console label. See the base class docstring for full behavior.
    """

    error_class = CodegenError
    degeneration_error_class = DegenerationError
    label = "codegen"


BODY_DIRECTIVES = (
    "This body is a reusable module executed across many requests. It owns no "
    "working data: every required operational value is provided by the runner "
    "through `ctx.config` under a clear snake_case name — request data from the "
    "runner, never embed or fabricate working values, and never ask the human "
    "for operational data. Ask the human only to refine the product goal and "
    "requirements.\n"
    "Write a single `act(ctx, request)` function. Declare the required "
    "operational values as a module-level `CONTRACT` dict exactly as SKILL.md "
    "specifies: a flat map of variable name -> semantic description. Every key "
    "in CONTRACT must be read from `ctx.config`; do not declare optional "
    "values with safe defaults — read those with `ctx.config.get(...)` instead.\n"
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
    "If this skill interacts with one of the available bridge services listed "
    "below, call that bridge over HTTP with the URL from its config variable — "
    "never speak the service's native protocol directly (no WebSocket to "
    "simplex-chat, no direct daemon access). Whenever you have a set of "
    "candidates to choose from (contacts, conversations, calendars, targets), "
    "employ a `ctx.engine` SemIf sub-decision over them and return the "
    "`(DecisionRequest, DecisionResult)` pairs on `ActionResult.decisions`; "
    "send only when the request or requirements ask for it.\n"
)


def _requirements_block(requirements: dict[str, str] | None) -> str:
    if not requirements:
        return ""
    lines = "\n".join(
        f"- {question} -> {answer}" for question, answer in requirements.items()
    )
    return f"Requirements gathered from the product owner:\n{lines}\n"


def _integration_hint_block(draft: SkillDraft) -> str:
    """Advisory integration hint from elicitation: service + transport.

    Never overrides the request, requirements, or SKILL.md contract — it only
    biases the body writer toward the integration the questions were aiming at.
    """
    integration = getattr(draft, "integration", None) or {}
    service = integration.get("service")
    transport = integration.get("transport")
    if not service and not transport:
        return ""
    return (
        "Elicitation pointed at this integration (advisory only: follow the "
        "request, requirements, and contract if they disagree): "
        f"service={service or 'unknown'}, transport={transport or 'unknown'}.\n"
    )


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
        f"{_integration_hint_block(draft)}"
        f"{describe_bridges()}\n"
        f"{BODY_DIRECTIVES}"
        "Write the Python module body now. Reply with ONLY valid Python code "
        "defining `act`, `INTEGRATION`, and `CONTRACT`. No prose, no markdown "
        "fences, no JSON."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_skill_body(raw: str) -> str:
    """Extract and validate a Python skill body from the model's reply.

    Accepts bare code, ```fenced``` code, or JSON {"code": "..."}. The body
    must parse, define a module-level `act` function, and declare a valid flat
    `CONTRACT` whose keys are all read from `ctx.config`. Returns the cleaned
    source. Raises ValueError otherwise.
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
    if "act" not in names:
        raise ValueError("skill body must define a module-level `act` function")
    contract = parse_contract(text)
    reads = set(_config_reads(text))
    dead = sorted(set(contract) - reads)
    if dead:
        raise ValueError(
            f"CONTRACT declares {dead} but the body never reads them from ctx.config"
        )
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
    findings feed the fidelity gate's reconsider regen evidence.
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


def _retry_prompt(
    request: Request,
    category: str,
    draft: SkillDraft,
    contract: str,
    error: Exception,
    requirements: dict[str, str] | None = None,
) -> list[dict]:
    """Fresh short prompt for a corrective retry.

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
        f"{_integration_hint_block(draft)}"
        f"{describe_bridges()}\n"
        f"{BODY_DIRECTIVES}"
        "Reply with ONLY valid Python defining `act`, `INTEGRATION`, and "
        "`CONTRACT`. No prose, no markdown fences, no JSON."
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
    """Author a skill body with the big model; retry up to `max_attempts`.

    Every attempt uses the client's configured sampler parameters (or omits
    them when unset). A rejected parse (ValueError) retries with a fresh short
    prompt that resets the context to SMART and tells the model to emit the
    final Python now. Only an invalid parse retries; a DegenerationError or
    other CodegenError propagates immediately so the scheduler can stub out
    gracefully. `requirements` (elicitation answers) are fed to the
    first-attempt prompt only.
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
    "Should this skill read or send through one of the available bridge services (see the catalog below), and if so which conversation or target?",
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
        f"{describe_bridges()}\n"
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
    parse after the retry ladder — elicitation must never block authoring.
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
            raw = client.chat(messages)
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
        f"Finished skill body:\n```python\n{skill_code}\n```\n"
        f"{describe_bridges()}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _validate_contract_dict(contract: dict) -> dict:
    """Validate a flat semantic contract: snake_case keys -> non-empty
    description strings."""
    if not isinstance(contract, dict):
        raise ValueError("data contract must be a single flat object")
    for key, value in contract.items():
        if not isinstance(key, str) or not CONTRACT_VARIABLE_RE.fullmatch(key):
            raise ValueError(f"contract variable {key!r} must be lowercase snake_case")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"contract value for {key!r} must be a semantic description string")
    return contract


def parse_contract(code: str) -> dict:
    """Validate the body's module-level `CONTRACT` declaration.

    The contract is a flat `{variable: description}` dict literal that the body
    declares itself; it is the source of truth for what the runner must supply.
    Returns the parsed dict. Raises ValueError when the constant is missing or
    malformed.
    """
    try:
        module = ast.parse(code, filename="<generated>")
    except SyntaxError as exc:
        raise ValueError(f"skill body is not valid Python: {exc}") from exc
    for node in module.body:
        value = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CONTRACT"
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "CONTRACT"
        ):
            value = node.value
        if value is None:
            continue
        if not isinstance(value, ast.Dict):
            raise ValueError("CONTRACT must be a flat dict literal")
        contract: dict = {}
        for key_node, value_node in zip(value.keys, value.values):
            key = _const_str(key_node) if key_node is not None else None
            if not key:
                raise ValueError("CONTRACT keys must be string literals")
            description = _const_str(value_node)
            if not description or not description.strip():
                raise ValueError(
                    f"CONTRACT[{key!r}] must be a non-empty description string"
                )
            contract[key] = description
        return _validate_contract_dict(contract)
    raise ValueError("skill body must define a module-level CONTRACT dict")


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
            raw = client.chat(messages)
            return parse_skill_test(raw)
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"testgen reply rejected {attempts} times: {last_error}")


def render_evidence(evidence: dict) -> str:
    """Render a raw evidence bundle as-is for a corrective prompt.

    The codegen model gets the raw bundle verbatim — no prose diagnosis and no
    `or`-chain collapse of the failure fields. Nested dict/list values are
    JSON-serialized with sorted keys so the rendering is deterministic;
    `previous_body` is fenced Python. A missing value renders as `null` rather
    than disappearing, so the model can see the absence.
    """
    lines: list[str] = []
    for key, value in evidence.items():
        if key == "previous_body":
            lines.append(f"previous_body:\n```python\n{value}\n```")
        elif isinstance(value, (dict, list)):
            lines.append(
                f"{key}: {json.dumps(value, sort_keys=True, ensure_ascii=False)}"
            )
        elif value is None:
            lines.append(f"{key}: null")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _regen_body_prompt(
    request: Request,
    category: str,
    draft: SkillDraft,
    contract: str,
    previous_code: str,
    evidence: dict,
    reason_kind: str = "test",
) -> list[dict]:
    """Fresh corrective prompt carrying the raw evidence bundle, not a prose
    diagnosis. `reason_kind` labels the corrective context in one word."""
    system = (
        "You write runnable skill bodies for a local agent. The contract below "
        "is authoritative: follow it exactly.\n\n"
        f"{contract}"
    )
    user = (
        f"Correct a skill body for {category}.{draft.name} ({reason_kind}).\n"
        f"Request: {request.text}\n"
        f"Skill description: {draft.description}\n"
        f"Evidence:\n{render_evidence(evidence)}\n"
        f"Previous body:\n```python\n{previous_code}\n```\n"
        f"{_integration_hint_block(draft)}"
        f"{describe_bridges()}\n"
        f"{BODY_DIRECTIVES}"
        "Reply with ONLY valid Python defining `act`, `INTEGRATION`, and "
        "`CONTRACT`. No prose, no markdown fences, no JSON."
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
    evidence: dict,
    contract: str | None = None,
    reason_kind: str = "test",
) -> str:
    """Rewrite a skill body whose test, review, or real run failed.

    Fresh prompt (context resets) carrying the raw evidence bundle and the
    previous body; retry ladder identical to generate_skill_body.
    `reason_kind` labels the corrective context (`test`, `fidelity`, or
    `run_failure`). The bundle is handed through as-is — the caller includes
    requirements and the raw failure fields, never a paraphrase. A
    DegenerationError or other CodegenError propagates immediately.
    """
    contract_text = contract if contract is not None else read_skill_contract()
    attempts = max(client.max_attempts, 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        attempt_evidence = dict(evidence)
        if attempt > 1:
            attempt_evidence["previous_parse_error"] = str(last_error)
        messages = _regen_body_prompt(
            request, category, draft, contract_text, previous_code,
            attempt_evidence, reason_kind=reason_kind,
        )
        try:
            raw = client.chat(messages)
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