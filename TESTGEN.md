# TESTGEN.md — the contract for skill test artifacts

This file is the authoritative spec for the **testing** side of a skill. It is
fed verbatim (alongside the finished `skill.py`) to the model that produces the
test artifact. The runnable body contract lives in `CODEGEN.md`; this file is
deliberately separate so the body writer never embeds test data and the test
writer never worries about body semantics.

A finished skill leaf is a folder
`data/skills/<category>/<name>/` containing (among others) two artifacts:

| File              | Meaning                                                            |
| ----------------- | ------------------------------------------------------------------ |
| `contract.json`   | A mirror of the body's `CONTRACT` constant: what the runner must provide. Derived from `skill.py`, not authored here. |
| `skill.test.py`   | A hermetic mechanics test, runnable as `python skill.test.py`.     |

## What the test is for (and is not)

The test is a **hermetic mechanics check**: it proves the body's code runs, its
config resolution works, its request construction is sane, and its error paths
return proper results. It is deliberately cut off from the outside world.

It is **not** proof that the real integration works. Skill bodies act on the
user's actual services; only a real run against that service can verify the
integration, and the agent offers to fix the skill when that real run fails. A
test must never be "fixed" by simulating the action or by weakening an
assertion, and a simulated body is broken even if every test passes.

## contract.json

A **single JSON object**. Each key is the name of a variable the skill needs;
each value is a plain-language, semantic description of the expected values for
that variable (what it is, what good values look like).

The body declares this itself as a module-level `CONTRACT` constant (see
`CODEGEN.md`); `contract.json` is a persisted mirror of it, provided to the test
writer as context. The test writer does not author or change the contract.

This contract exists for exactly two consumers:

1. **User input** — the config step asks the human for these variables when
   they are not already configured.
2. **SemIf** — the config search and the record-vs-ask decision route on the
   variable names.

Therefore the contract must be flat and semantic. Do NOT put type declarations,
validation logic, or nested data structures in it — the skill body and the test
implement whatever type safety and validation they need on their own. The test
writer consumes `skill.py` in context and keeps the artifact compatible, so the
contract never has to carry that burden. Convoluted structure here is
pointless: it only confuses the human and the decision model.

Rules:

- Top level is exactly one JSON object; no arrays, no nesting.
- Each key is the snake_case name of a variable the body reads from
  `ctx.config`.
- Each value is a non-empty string describing the expected values in plain
  language (e.g. `"The email address the message is sent from."`).
- The keys are exactly the body's `CONTRACT` constant; every declared key is
  read from `ctx.config`. Do not invent keys the body does not use.
- Optional values with safe defaults are read with `ctx.config.get(...)` and
  are not contract keys.

## skill.test.py

A stdlib-only Python script that:

- Runs as `python skill.test.py` from the skill folder (its cwd is the folder,
  so the `skill` module is importable at `import skill`).
- Is hermetic: it never contacts an external network, never touches the user's
  real services, and never writes outside the folder.
- Embeds its fixture data **inline** as Python literals — no external files.
- For a body that performs HTTP in `act`, starts a real loopback server
  (`http.server` on `127.0.0.1`, ephemeral port) and passes its URL to the body
  through the fixture config — the same `ctx.config` key the body reads.
  Assert the server actually received the expected request (method, path,
  headers, body) where practical. The loopback address exists only in the test;
  the body must read its endpoint from config.
- For a messaging skill (a body that talks to a bridge service), stand up a
  loopback server implementing the bridge endpoints the body uses (the bridge
  catalog injected into this prompt lists them) and assert the body built the
  real requests. If the catalog names an auth header, set the fixture token in
  the fixture config and assert the body sent that header. Never connect to a
  real service daemon or a live bridge.
- For non-HTTP transports (IMAP, SMTP, subprocess, ...), exercise config
  resolution, argument construction, and error paths (missing or misconfigured
  values) without performing the real I/O.
- Exercises the skill's `act` against those fixtures — call it directly with a
  real-ish `ActionContext` (a fake `engine` is acceptable **in the test only**;
  the test is the exception to the no-mocking rule for the decision engine).
- Exits 0 on success and non-zero on failure (use `assert`/`sys.exit(1)` with a
  message). No third-party imports.
- Covers at least the happy path (through the loopback server where the body
  does HTTP) and one edge case (missing input, unclear request, unreachable
  target).

## Acceptance criteria

The test artifact is accepted only if:

1. `skill.test.py` parses as Python, imports only the stdlib and the agent
   package, embeds its fixtures inline (no external files), is hermetic, and
   runs to exit 0 when executed from the skill folder.
2. The test stays compatible with the body's `CONTRACT` (the same variable
   names the body reads from `ctx.config`).

Produce one artifact: `skill.test.py` (derived from `skill.py` plus its
`CONTRACT`, with fixture data embedded inline).

## Worked example

A body that checks a service over HTTP with `service_url` read from
`ctx.config`. The test stands up a loopback `http.server` and points the fixture
config at it, so the body's real HTTP call is exercised without leaving the
machine:

```python
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_agent.decisions import DecisionResult, Request
from semif_agent.skills import ActionContext


class Handler(BaseHTTPRequestHandler):
    seen = []

    def do_GET(self):
        Handler.seen.append(self.path)
        body = b'{"status": "ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class FakeEngine:
    def call(self, decision):
        return DecisionResult(
            request=decision,
            option_ids=[o.id for o in decision.options],
            probabilities=[1.0] + [0.0] * (len(decision.options) - 1),
        )


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/health"
    try:
        ctx = ActionContext(engine=FakeEngine(), config={"service_url": url})
        req = Request("check the service")
        skill = __import__("skill")
        result = skill.act(ctx, req)
        assert result.action_log, "act must log what it did"
        assert Handler.seen == ["/health"], f"body must really call the service: {Handler.seen}"
        print(result.action_log)
        print("ok")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
```
