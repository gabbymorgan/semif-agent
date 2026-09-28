# TESTGEN.md — the contract for skill test artifacts

This file is the authoritative spec for the **testing** side of a skill. It is
fed verbatim (alongside the finished `skill.py`) to the model that produces the
test artifact. The runnable body contract lives in `SKILL.md`; this file is
deliberately separate so the body writer never embeds test data and the test
writer never worries about body semantics.

A finished skill leaf is a folder
`data/skills/<category>/<name>/` containing (among others) two artifacts
produced here:

| File              | Meaning                                                            |
| ----------------- | ------------------------------------------------------------------ |
| `contract.json`   | The skill's data contract: what the runner must provide.           |
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
- Derive the keys by reading the finished `skill.py`: every `ctx.config[...]`
  access that carries operational data is a contract key. Do not invent keys
  the body does not use.
- Do not include keys for purely internal/derived values.

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
  loopback server implementing the bridge endpoints the body uses (`GET /inbox`,
  `GET /inbox/next?contact=`, `GET /address`, `POST /send`) and assert the body
  built the real requests. Never connect to a real `simplex-chat` daemon or a
  live bridge.
- For non-HTTP transports (IMAP, SMTP, subprocess, ...), exercise config
  resolution, argument construction, and error paths (missing or misconfigured
  values) without performing the real I/O.
- Exercises the skill's `predict` and `act` against those fixtures — call them
  directly with a real-ish `ActionContext` (a fake `engine` is acceptable
  **in the test only**; the test is the exception to the no-mocking rule for
  the decision engine).
- Exits 0 on success and non-zero on failure (use `assert`/`sys.exit(1)` with a
  message). No third-party imports.
- Covers at least the happy path (through the loopback server where the body
  does HTTP) and one edge case (missing input, unclear request, unreachable
  target).

## Acceptance criteria

The test artifact is accepted only if:

1. `contract.json` is a single flat object of snake_case keys to semantic
   descriptions, and every `ctx.config[...]` operational access in `skill.py`
   has a matching key.
2. `skill.test.py` parses as Python, imports only the stdlib and the agent
   package, embeds its fixtures inline (no external files), is hermetic, and
   runs to exit 0 when executed from the skill folder.

Produce the two artifacts in this order: first `contract.json` (derived from
`skill.py`), then `skill.test.py` (derived from the same `skill.py` plus the
contract, with fixture data embedded inline).

## Worked example

A body that checks a service over HTTP with `service_url` read from
`ctx.config`. The test stands up a loopback `http.server` and points the fixture
config at it, so the body's real HTTP call is exercised without leaving the
machine:

```python
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_agent.skills import ActionContext, DecisionResult, Request


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
        pred = skill.predict(ctx, req)
        assert pred.text, "predict must produce a forecast"
        result = skill.act(ctx, req, pred)
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
