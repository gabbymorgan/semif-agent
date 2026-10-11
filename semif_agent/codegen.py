"""Skill code-body generation through an OpenAI-compatible API.

The small `llm` provider authors a skill's title + description; writing the
runnable body is a separate step: a larger OpenAI-compatible model (a
code-capable model on your configured codegen endpoint) is prompted with the CODEGEN.md contract plus the request
and the existing tree, and must reply with valid Python implementing `act` plus
the `INTEGRATION` and `CONTRACT` module constants. The shared provider transport
lives in `provider.py`.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
import subprocess
import symtable
import sys
from pathlib import Path
from typing import Callable, NamedTuple

from .bridges.registry import describe_bridges
from .provider import OpenAICompatClient, ProviderError
from .skills import SkillDraft, tree_summary

DEFAULT_CONTRACT = Path(__file__).resolve().parent.parent / "CODEGEN.md"


class CodegenError(ProviderError):
    """The codegen endpoint could not be reached."""


class DegenerationError(CodegenError):
    """A SemIf degeneration watchdog aborted the stream mid-generation.

    Subclass of CodegenError so the scheduler's graceful-stub path catches it;
    distinct so callers can choose to retry a degenerated stream later without
    retrying genuine endpoint failures.
    """


class ArtifactTooLong(ValueError):
    """A generated artifact exceeded the size budget for a decision.

    A hard ceiling, not a warning: the artifact is ingested *whole* by a
    decision (the fidelity gate reads the entire skill body), so it must fit
    the decision engine's context with headroom for everything else in the
    state. The budget is derived from the active engine's context; the codegen
    retry ladder asks the model for a shorter artifact, and a body that still
    cannot fit stubs out rather than shipping something the gate cannot read.
    """


def read_skill_contract(path: str | None = None) -> str:
    contract = Path(path) if path else DEFAULT_CONTRACT
    if not contract.is_file():
        raise CodegenError(f"skill contract not found: {contract}")
    return contract.read_text()


def skill_contract_ref(path: str | None = None) -> dict:
    """Provenance pointer for the CODEGEN.md revision at authoring time.

    Returns {"ref": str | None, "dirty": bool | None}. `ref` is the short git
    commit sha the contract was read under — revivable with
    `git show <ref>:CODEGEN.md` — and `dirty` records whether the working-tree
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
    "through `ctx.config` under a clear snake_case name — never embed or "
    "fabricate working values, and never assume a specific user, account, "
    "address, or value.\n"
    "Write a single `act(ctx, request)` function. Declare the required "
    "operational values as a module-level `CONTRACT` dict exactly as CODEGEN.md "
    "specifies: a flat map of variable name -> semantic description. Every key "
    "in CONTRACT must be read from `ctx.config`; do not declare optional "
    "values with safe defaults — read those with `ctx.config.get(...)` instead.\n"
    "Read `request.text` and derive the action's per-request arguments from it "
    "(recipient, subject/body, date, target, search terms); never declare a "
    "query argument in CONTRACT. Return the real result in `new_state` and a "
    "truthful `action_log` — the fetched data or the service's real response, "
    "never an empty or placeholder output.\n"
    "Perform the real action: if this skill interacts with an external system, "
    "`act` must call the user's configured service with a stdlib transport "
    "(urllib.request/http.client for HTTP, imaplib/smtplib/poplib for mail, "
    "subprocess for a configured local CLI) using endpoint, account, "
    "credential, and CLI-path values read from `ctx.config`. Never simulate "
    "success, never return canned output, and never default to writing a draft "
    "unless the reusable goal explicitly asks for draft-only. The auto-run test "
    "is a hermetic mechanics check and will not exercise the live service, so "
    "no localhost defaults, test ports, or fixture values may appear in the "
    "body.\n"
    "Declare the integration as a module-level `INTEGRATION` dict exactly as "
    "CODEGEN.md specifies: service, transport, config_vars.\n"
    "If this skill interacts with one of the available bridge services listed "
    "below, call that bridge over HTTP with the URL from its config variable — "
    "never speak the service's native protocol directly (no WebSocket to "
    "simplex-chat, no direct daemon access). Whenever you have a set of "
    "candidates to choose from (contacts, conversations, calendars, targets), "
    "employ a `ctx.engine` SemIf sub-decision over them and return the "
    "`(DecisionRequest, DecisionResult)` pairs on `ActionResult.decisions`; "
    "send only when the reusable goal calls for it.\n"
    "Keep the body small: it is ingested whole by the decision engine's "
    "fidelity check, so delegate protocol details to a bridge instead of "
    "hand-rolling them, and keep `act` to one purpose.\n"
)


def _integration_hint_block(draft: SkillDraft) -> str:
    """Advisory integration hint from the de-identified skill spec: service +
    transport. Never overrides the reusable goal or the CODEGEN.md contract — it
    only biases the body writer toward the integration the spec named."""
    integration = getattr(draft, "integration", None) or {}
    service = integration.get("service")
    transport = integration.get("transport")
    if not service and not transport:
        return ""
    return (
        "The skill spec points at this integration (advisory only: follow the "
        "reusable goal and the contract if they disagree): "
        f"service={service or 'unknown'}, transport={transport or 'unknown'}.\n"
    )


def build_skill_body_prompt(
    category: str,
    draft: SkillDraft,
    tree: dict,
    contract: str,
) -> list[dict]:
    """Messages for the code-generation model.

    Built ONLY from the de-identified spec (goal + generic service/transport),
    the category, the existing tree, and the CODEGEN.md contract. It never
    carries the user's request, question answers, or any run output: codegen
    must be able to write a reusable body from the generalized goal alone.
    """
    existing = ", ".join(s.name for s in tree.get(category, [])) or "(none)"
    system = (
        "You write runnable skill bodies for a local agent. The contract below "
        "is authoritative: follow it exactly.\n\n"
        f"{contract}"
    )
    user = (
        f"Category: {category}\n"
        f"Skill name: {draft.name}\n"
        f"Reusable goal (de-identified; never embed or ask for user-specific "
        f"values): {draft.goal}\n"
        f"Skill description: {draft.description}\n"
        f"Existing skills in this category: {existing}\n"
        f"Existing categories:\n{tree_summary(tree)}\n"
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


_MODULE_SCOPE_NAMES = frozenset(
    {
        "__name__",
        "__file__",
        "__doc__",
        "__package__",
        "__spec__",
        "__loader__",
        "__builtins__",
        "__annotations__",
        "__cached__",
    }
)


def _unresolved_names(code: str) -> list[str]:
    """Names the body references but never binds (and that aren't builtins).

    Catches a body that calls `urllib.request.urlopen` or returns
    `ActionResult` without importing them: the generated body must import
    everything it uses, since the runner imports the file as-is. Scope-aware via
    stdlib `symtable` and purely static — it never executes the body, so it
    cannot catch a wrong module path (the import statement binds the name
    regardless); the real import in `materialize_skill` catches that.
    """
    try:
        table = symtable.symtable(code, "<generated>", "exec")
    except SyntaxError:
        return []
    bound = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    unresolved: set[str] = set()

    def visit(scope: symtable.SymbolTable) -> None:
        for symbol in scope.get_symbols():
            name = symbol.get_name()
            if not symbol.is_referenced():
                continue
            if (
                symbol.is_local()
                or symbol.is_parameter()
                or symbol.is_assigned()
                or symbol.is_imported()
            ):
                continue
            if symbol.is_free():
                # Closure over an enclosing function's local — bound, not free.
                continue
            if name in bound or name in _MODULE_SCOPE_NAMES or hasattr(builtins, name):
                continue
            unresolved.add(name)
        for child in scope.get_children():
            visit(child)

    visit(table)
    return sorted(unresolved)


def parse_skill_body(raw: str, max_chars: int | None = None) -> str:
    """Extract and validate a Python skill body from the model's reply.

    Accepts bare code, ```fenced``` code, or JSON {"code": "..."}. The body
    must parse, define a module-level `act` function, declare a valid flat
    `CONTRACT` whose keys are all read from `ctx.config`, and import every name
    it uses. Returns the cleaned source. Raises ValueError otherwise.

    `max_chars` is the hard size ceiling (the scheduler derives it from the
    active decision engine's context). A body over it raises `ArtifactTooLong`
    so the retry ladder asks for a shorter one.
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
    if max_chars is not None and len(text) > max_chars:
        raise ArtifactTooLong(
            f"skill body is {len(text)} chars, over the {max_chars}-char limit "
            "for a decision-ingestible body; keep it small and delegate "
            "protocol details to a bridge"
        )
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
    unresolved = _unresolved_names(text)
    if unresolved:
        raise ValueError(
            "skill body references names it never imports or defines: "
            f"{unresolved}; import everything you use "
            "(e.g. `from semif_agent.skills import ActionResult`, "
            "`import urllib.request`)"
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


_HTTP_VERBS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
_SUBPROCESS_FUNCS = frozenset({"run", "Popen", "call", "check_call", "check_output"})
_FILE_WRITE_FUNCS = frozenset({"write_text", "write_bytes", "write"})


def _call_name(node) -> str:
    """Best-effort dotted name of a call target (e.g. `urllib.request.urlopen`)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def operation_inventory(code: str) -> list[str]:
    """A compact, static inventory of the side-effecting operations in a body.

    The fidelity gate reads the whole body; this inventory is its deterministic
    companion. It names the concrete operations (HTTP verb + path, CLI run,
    mail send, file write, timer) so the gate and its regen evidence see *what
    the body does* at a glance, and a body that performs no operation at all is
    legible immediately. Best-effort and purely static: a dynamic target
    renders as `?`, an unparseable body yields nothing. Never raises.
    """
    try:
        module = ast.parse(code, filename="<inventory>")
    except SyntaxError:
        return []
    operations: list[str] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        leaf = name.rsplit(".", 1)[-1]
        args = list(node.args) + [kw.value for kw in node.keywords]
        strings = [s for a in args if (s := _const_str(a)) is not None]
        verbs = [s for s in strings if s.upper() in _HTTP_VERBS]
        if verbs:
            paths = [s for s in strings if s.startswith("/")]
            target = paths[0] if paths else "?"
            operations.append(f"HTTP {verbs[0].upper()} {target} (via {name or '?'})")
            continue
        if name in {
            "urllib.request.urlopen",
            "urlopen",
            "http.client.HTTPConnection",
            "http.client.HTTPSConnection",
        }:
            operations.append(f"HTTP request (via {name})")
        elif "subprocess" in name and leaf in _SUBPROCESS_FUNCS:
            operations.append(f"run CLI (via {name})")
        elif "smtplib" in name and leaf in {"sendmail", "send_message"}:
            operations.append(f"send mail (via {name})")
        elif "imaplib" in name and leaf in {"append", "store", "copy", "expunge"}:
            operations.append(f"mail mutation (via {name})")
        elif "poplib" in name and leaf in {"dele", "quit"}:
            operations.append(f"mail mutation (via {name})")
        elif leaf in _FILE_WRITE_FUNCS or (
            name == "open"
            and any(s in {"w", "a", "x", "wb", "ab", "xb", "w+", "a+", "wt", "at"} for s in strings)
        ):
            operations.append(f"write file (via {name})")
        elif name.startswith("ctx.timers"):
            operations.append(f"schedule timer (via {name})")
    if not operations:
        operations.append("no side-effecting operation found")
    return operations


def render_operation_inventory(operations: list[str]) -> str:
    """Render `operation_inventory` as a bulleted block for a decision state."""
    return "\n".join(f"- {operation}" for operation in operations)


def _retry_prompt(
    category: str,
    draft: SkillDraft,
    contract: str,
    error: Exception,
) -> list[dict]:
    """Fresh short prompt for a corrective retry.

    Not a growing conversation: the context resets to SMART (new request, tiny
    prompt) and the model is told in one line to stop looping and emit the
    final Python. The de-identified goal is repeated so a retry cannot fall
    back to a toy body.
    """
    system = (
        "You write runnable skill bodies for a local agent. The contract below "
        "is authoritative: follow it exactly.\n\n"
        f"{contract}"
    )
    if isinstance(error, ArtifactTooLong):
        guidance = (
            "The previous attempt was too long to fit the decision engine's "
            "context. Emit a SHORTER body: keep `act` minimal and delegate "
            "protocol details to a bridge.\n"
        )
    else:
        guidance = "You are looping; emit the final Python now.\n"
    user = (
        f"The previous attempt to write the skill body for {category}.{draft.name} "
        f"was rejected: {error}. {guidance}"
        f"Category: {category}\n"
        f"Skill name: {draft.name}\n"
        f"Reusable goal (de-identified): {draft.goal}\n"
        f"Skill description: {draft.description}\n"
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
    category: str,
    draft: SkillDraft,
    tree: dict,
    contract: str | None = None,
    max_tokens: int | None = None,
    degeneration_check: Callable[[str], str | None] | None = None,
    max_chars: int | None = None,
) -> str:
    """Author a skill body with the big model; retry up to `max_attempts`.

    Built only from the de-identified spec (goal + generic integration), the
    category, the tree, and the CODEGEN.md contract — never the user's request.
    Every attempt uses the client's configured sampler parameters (or omits
    them when unset). A rejected parse (ValueError) retries with a fresh short
    prompt that resets the context to SMART and tells the model to emit the
    final Python now. Only an invalid parse retries; a DegenerationError or
    other CodegenError propagates immediately so the scheduler can stub out
    gracefully.
    """
    contract_text = contract if contract is not None else read_skill_contract()
    if client.stream:
        print(f"[codegen] writing body for {category}.{draft.name}...")
    last_error: Exception | None = None
    attempts = max(client.max_attempts, 1)
    for attempt in range(1, attempts + 1):
        if attempt == 1:
            messages = build_skill_body_prompt(
                category,
                draft,
                tree,
                contract_text,
            )
        else:
            messages = _retry_prompt(
                category, draft, contract_text, last_error
            )
        try:
            raw = client.chat(
                messages,
                max_tokens=max_tokens,
                degeneration_check=degeneration_check,
            )
            return parse_skill_body(raw, max_chars=max_chars)
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"skill body rejected {attempts} times: {last_error}")


# ---- data contract + test ----

TESTGEN_CONTRACT = Path(__file__).resolve().parent.parent / "TESTGEN.md"
CONTRACT_VARIABLE_RE = re.compile(r"[a-z0-9_]+")


def read_testgen_contract(path: str | None = None) -> str:
    contract = Path(path) if path else TESTGEN_CONTRACT
    if not contract.is_file():
        raise CodegenError(f"testgen contract not found: {contract}")
    return contract.read_text()


def build_testgen_base_prompt(
    category: str, draft: SkillDraft, skill_code: str, contract_md: str
) -> list[dict]:
    """The shared context for the contract + test generation calls.

    Seeded with the TESTGEN.md contract, the de-identified spec, and the
    finished skill body; the contract call appends its ask, and the testgen call
    appends the contract output (as the assistant turn) plus its own ask — the
    two calls share this context but produce distinct artifacts. Never carries
    the user's request.
    """
    system = (
        "You produce the data contract and test for a skill body of a "
        "local agent. The contract below is authoritative: follow it exactly.\n\n"
        f"{contract_md}"
    )
    user = (
        f"Skill: {category}.{draft.name}\n"
        f"Reusable goal (de-identified): {draft.goal}\n"
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


def parse_skill_test(raw: str, max_chars: int | None = None) -> str:
    """Extract and validate skill.test.py from the model's reply.

    Accepts bare code, ```fenced``` code, or JSON {"code": "..."}. The test
    must parse as Python and (when `max_chars` is set) fit the artifact size
    budget. Returns the cleaned source. Raises ValueError otherwise.
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
    if max_chars is not None and len(text) > max_chars:
        raise ArtifactTooLong(
            f"skill test is {len(text)} chars, over the {max_chars}-char limit "
            "for a codegen artifact; keep fixtures compact"
        )
    try:
        ast.parse(text, filename="<generated test>")
    except SyntaxError as exc:
        raise ValueError(f"skill test is not valid Python: {exc}") from exc
    return text


def generate_skill_tests(
    client: CodegenClient,
    category: str,
    draft: SkillDraft,
    skill_code: str,
    contract: dict,
    contract_md: str | None = None,
    reason: str | None = None,
    target: str | None = None,
    max_chars: int | None = None,
) -> str:
    """Generate skill.test.py against the finished body.

    Shares the base context with the contract call (TESTGEN.md + skill.py) and
    continues it with the accepted contract, so the test stays compatible with
    the body and the contract. Built only from the de-identified spec — never
    the user's request. Fixture data is embedded inline in the test.
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
                category, draft, skill_code, contract_text
            ),
            contract,
            note=note,
            target=target,
        )
        try:
            raw = client.chat(messages)
            return parse_skill_test(raw, max_chars=max_chars)
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
    category: str,
    draft: SkillDraft,
    contract: str,
    previous_code: str,
    evidence: dict,
    reason_kind: str = "test",
) -> list[dict]:
    """Fresh corrective prompt carrying the raw evidence bundle, not a prose
    diagnosis. `reason_kind` labels the corrective context in one word. Built
    only from the de-identified spec and the evidence — never the user's
    request."""
    system = (
        "You write runnable skill bodies for a local agent. The contract below "
        "is authoritative: follow it exactly.\n\n"
        f"{contract}"
    )
    user = (
        f"Correct a skill body for {category}.{draft.name} ({reason_kind}).\n"
        f"Reusable goal (de-identified): {draft.goal}\n"
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
    category: str,
    draft: SkillDraft,
    previous_code: str,
    evidence: dict,
    contract: str | None = None,
    reason_kind: str = "test",
    max_chars: int | None = None,
) -> str:
    """Rewrite a skill body whose test, review, or real run failed.

    Fresh prompt (context resets) carrying the raw evidence bundle and the
    previous body; retry ladder identical to generate_skill_body.
    `reason_kind` labels the corrective context (`test`, `fidelity`, or
    `run_failure`). The evidence bundle must be de-identified by the caller: it
    carries only the generalized failure kind, integration, goal, and previous
    body — never the user's request or any run output. A
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
            category, draft, contract_text, previous_code,
            attempt_evidence, reason_kind=reason_kind,
        )
        try:
            raw = client.chat(messages)
            return parse_skill_body(raw, max_chars=max_chars)
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