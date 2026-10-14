# PACKGEN.md — the contract for new packs

This file is the authoritative spec for what a **pack** is and how to author a
new one. It is the third sibling of the skill contracts: `CODEGEN.md` governs a
skill's runnable **body**, `TESTGEN.md` governs its **test**, and this file
governs the **pack** that bundles a concern, its skills, its bridge reference,
and its settings into one thing.

Read this before adding a pack. A pack is what makes a new
product/concern a first-class part of the agent instead of scattered edits to
`skills.py` and `config.py`.

## What a pack is

Semif-agent ships a set of starter integrations. Each is a **pack**: a folder
`packs/<name>/` holding a `pack.json` manifest plus a `skills/` store of leaves
in the exact generated-skill format. Each pack lives in its **own repository**
(`pack.<name>`, listed in the `semif-packs` pack-index repo); `scripts/bootstrap.sh`
clones them into the gitignored `packs/<name>/`. The manifest binds the pieces
that would otherwise be scattered by convention:

- the **concern** (what product/job this is) and its **title**;
- its **categories** (the tree buckets it owns, each with a scope description);
- the **bridge(s)** it talks to, referenced by name — and **ships** the bridge
  code itself under `bridge/` (see "Authoring a new bridge");
- its **runtime requirements** (optional heavy deps);
- its **enable flag**; and
- its **config schema** (the defaults and validation for its settings).

A pack is loaded by `semif_agent/packs.py` (`PackRegistry` discovers packs,
`merge_packs` loads every **enabled** pack into the tree at startup) and its
`config` schema is read by `semif_agent/config.py` to generate
`config.example.json` and to validate a live config.

Three boundaries must stay crisp:

- A **skill** is one leaf action (`packs/<name>/skills/<category>/<leaf>/`),
  governed by `CODEGEN.md`/`TESTGEN.md`. A pack contains skills; it is not one.
- A **bridge** is a standalone process in front of a third-party system,
  warranted when the system needs state (especially auth state), a native or
  heavy protocol, or one sanitized surface; a system with a plain HTTP API needs
  no bridge — its skill calls the API directly. A bridge **ships with its pack**
  (`packs/<name>/bridge/`); the core keeps only the shared contract
  (`semif_agent/bridges/base.py`) and the discovery registry
  (`semif_agent/bridges/registry.py`). The one shared, pack-less bridge is the
  **LLM bridge** (a core built-in many packs call). A pack's `bridges` list names
  the bridge(s) it ships; a pack never edits core bridge code.
- A pack is **declarative structure**, committed in its own repository.
  Per-machine values are the split config; credentials live in `secrets.json`.
  A pack repo never holds a secret.

## Folder layout

```
packs/<name>/
  pack.json                       # the manifest (this file's subject)
  skills/<category>/<leaf>/
    skill.py                      # the body — CODEGEN.md
    manifest.json                 # {"description": "..."} navigation text
    contract.json                 # mirror of the body's CONTRACT — TESTGEN.md
    skill.test.py                 # hermetic mechanics test — TESTGEN.md
  bridge/                         # only for a bridge-backed pack (see below)
    __init__.py                   # the pack's bridge package
    <bridge>.py                   # the BridgeService subclass (bridge name)
    <bridge>_client.py            # optional protocol client (e.g. nextcloud_client.py)
    test_<bridge>_bridge.py       # hermetic bridge test (loopback HTTP)
  scripts/                        # optional concern helpers (e.g. an auth or
                                  # setup script), run from the checkout
```

- The `<category>` directory name is the tree bucket; the `<leaf>` directory
  name is the leaf's `name`. The runner joins them as `category.name` only in
  run summaries.
- Every leaf carries the four artifacts above. Their contents are governed by
  `CODEGEN.md` (body + `CONTRACT` + `INTEGRATION`) and `TESTGEN.md`
  (test + `contract.json` mirror); do not restate those rules here.
- `bridge/` ships the pack's bridge (one module per name in `bridges`, plus
  optional sibling modules it imports relatively). It is absent for a
  bridge-less pack. See "Authoring a new bridge".
- `manifest.json` holds a single non-empty `description` — the navigation text
  SemIf routes on. It is the seed analog of the registry entry a generated
  skill gets from the `llm` provider.
- A root with no `pack.json` is treated as **one implicit, manifest-less pack**
  whose store is the root itself (a plain skills directory still works), but a
  pack repo should always have a manifest.

The shipped packs are the authoritative worked examples: `essentials` is
the minimal bridge-less pack; `nextcloud`, `simplex`, `outlook`, and `browser`
are bridge-backed packs.

## `pack.json` — the manifest

A single JSON object. Fields (from `packs._parse_pack`):

| Field                | Type            | Required | Meaning |
| -------------------- | --------------- | -------- | ------- |
| `name`               | string          | no\*     | Pack id (snake_case). Defaults to the folder name. |
| `title`              | string          | no       | Human display name. Defaults to `name`. |
| `concern`            | string          | **yes**  | One- to two-sentence statement of the product/job the pack serves. (`description` is accepted as a legacy alias.) |
| `categories`         | object          | **yes**  | `{category_id: scope_description}` — the tree buckets this pack owns. |
| `bridges`            | array of string | no       | Bridge names this pack ships and talks to. Each name is a `bridge/<name>.py` module in the pack; `simplex`/`llm`/`nextcloud`/`outlook`/`browser` are the shipped names. Empty for a bridge-less pack. |
| `requires`           | array of string | no       | Optional heavy deps (e.g. `playwright`). Informational; the code's `check_requirements` is what enforces them. |
| `enabled_by_default` | bool            | no       | Default `true`. Whether the pack loads when nothing else decides. |
| `config`             | object          | no       | The pack's config schema (see below). |

\* `name` is optional in the parser but should always be set explicitly.

Rules:

- **`concern` and `categories` are required** (`tests/test_packs.py` fails
  without them). A pack with no declared concern is not a pack.
- **Category ids are the tree keys.** They are the same ids the leaf folders
  live under (`skills/<category>/<leaf>/`). When `categories` is omitted, the
  pack's categories are inferred from the `skills/` subdirectories — but declare
  them, because the description is load-bearing.
- **A category description is navigation-critical.** It is what SemIf reads to
  route a request into the bucket; a bare id with no description routes badly.
  Write one to two sentences naming the concrete jobs the category does.
- **`bridges[0]` owns the config block.** For a bridge pack, the pack's `config`
  schema maps onto `bridges.<bridges[0]>`; only the first bridge gets the block
  (a pack normally fronts one bridge). Each name in `bridges` must have a
  `bridge/<name>.py` module that defines its `BridgeService` subclass.
- Keep the manifest minimal: it binds metadata and settings, nothing executable.

## The config schema

`config` is a nested schema. A **leaf** is a dict with a `default`; a **block**
is a dict without one (recurse). Each leaf spec:

| Key       | Type   | Meaning |
| --------- | ------ | ------- |
| `type`    | string | `bool` \| `int` \| `float` \| `str` \| `list` \| `dict`. Checked by `check-config` (`int`/`float` are interchangeable). |
| `default` | any    | The value shipped in `config.example.json` and seeded by `config-init`. Required on a leaf. |
| `doc`     | string | One-line plain-language description. Shown in the generated example. |
| `secret`  | bool   | `true` marks a credential: the example emits `"!secret <pack>_<key>"` instead of a value. |

Example (a bridge-backed pack, mirroring `packs/simplex/pack.json`):

```json
"config": {
  "enabled": {"type": "bool", "default": true, "doc": "Load this pack's skills and run its bridge."},
  "host": {"type": "str", "default": "127.0.0.1", "doc": "Bridge bind address (localhost only)."},
  "port": {"type": "int", "default": 5227, "doc": "Bridge listen port."},
  "token": {"type": "str", "default": "", "secret": true, "doc": "Shared secret required in the X-Semif-Token header when set."},
  "reconnect": {
    "initial": {"type": "float", "default": 1.0, "doc": "Initial reconnect delay (seconds)."},
    "max": {"type": "float", "default": 60.0, "doc": "Maximum reconnect delay (seconds)."}
  }
}
```

How the schema maps onto the live config (`config.default_config` /
`generate_example_config`):

- A **bridge pack** (`bridges` non-empty) maps its whole `config` schema onto
  the `bridges.<bridges[0]>` block.
- A **bridge-less pack** maps only its `enabled` flag onto
  `packs.<name>.enabled`; its other keys are not carried (a bridge-less pack is
  configured through its skills' own config, not a pack block).
- Secret leaves become `"!secret <pack>_<key>"` in `config.example.json`, where
  `<pack>_` is the pack name prefix (nested keys are prefixed with their path,
  e.g. `simplex_reconnect_initial`). Never write a real credential into the
  manifest.

Rules:

- Every leaf needs a `default` and a `doc`. A key with no `default` is treated
  as a nested block.
- Put **operational** settings here (endpoints, accounts, defaults, timeouts) —
  the values a skill body reads from `ctx.config`. Per-request arguments never
  belong in config (see `CODEGEN.md`).
- `config.json`, `config/`, and `secrets.json` are gitignored and per-machine.
  The manifest is committed structure only.

After changing a schema, regenerate and validate:

```sh
python -m semif_agent.cli check-config --write-example config.example.json
python -m semif_agent.cli check-config
```

`config-init` seeds the split `config/` layout from the schemas (bootstrap runs
it once, only when `config.json` is absent); afterwards only a human edits it.
The generated example and the live config are never hand-edited.

## The enable switch

One switch governs the pack's skills **and** its bridge, so a host never offers
a skill whose bridge is off. `packs.pack_enabled` resolves in order:

1. an explicit `packs.<name>.enabled` in config (wins);
2. else, for a bridge pack, `bridges.<bridges[0]>.enabled` (the historical
   switch);
3. else `enabled_by_default`.

Disabling a pack drops its categories entirely — including any generated body
authored into one — and stops its bridge. The switch lives in the pack's
`config/packs/<name>.json` fragment (written by `config-init`/`check-config`).

## Referencing a bridge by name

A pack whose skills talk to a third-party system lists that bridge in
`bridges`, **ships** it under `bridge/`, and carries its config block in
`config`. (A pack whose skills reach a plain HTTP API directly needs no bridge:
`bridges` stays empty and each leaf declares the endpoint in its own
`CONTRACT`.) The bridge itself is a `BridgeService` subclass in
`packs/<name>/bridge/<name>.py`; the core registry imports it by convention and
its `BridgeInfo` (name, service, description, URL config var, other config vars,
auth header, endpoints) is injected into every codegen prompt via
`describe_bridges()`, so bodies learn how to reach it without the pack restating
protocol details. The one exception is the shared, pack-less **LLM bridge**
(a core built-in).

Skill-facing config is **derived**, never duplicated: `derived_config_vars`
turns the bridge's `bridges.<name>` block (`host`/`port`/`token`) into the
`<url_config_var>` and `<auth_config_var>` values a body reads from
`ctx.config` (e.g. `simplex_bridge_url`, `simplex_bridge_token`). Bodies call
the bridge over the ordinary `http` transport; their hermetic tests are
loopback HTTP like any other HTTP body.

Rules:

- The bridge block's `enabled`/`host`/`port`/`token` are the pack's config
  schema; the URL/token config vars are derived and must not be declared again.
- A pack ships its own bridge and never imports or edits another pack's or the
  core's. Gateway isolation is non-negotiable — read/send/address capability is
  a bridge, never a gateway feature.

## Authoring a new bridge

A **bridge** is a standalone process that stands up a simple, secure localhost
HTTP layer in front of one third-party system. Author one when a skill body
cannot sanely talk to the system itself — because it holds auth/session state,
needs a native or heavy protocol, or should be reached through one sanitized
surface. A system with a plain HTTP API needs no bridge: its skill calls the API
directly. Adding a bridge means:

1. **Write the bridge class** in the pack's `bridge/<name>.py`, subclassing
   `BridgeService` (`semif_agent/bridges/base.py`). Set the `name` attribute and
   a module-level/class `INFO = BridgeInfo(...)`. Ship `bridge/__init__.py` (the
   package marker) and any sibling modules the bridge imports relatively (e.g.
   `bridge/<name>_client.py` for the protocol client):

   ```python
   from semif_agent.bridges.base import BridgeInfo, BridgeService

   class WidgetBridge(BridgeService):
       name = "widget"
       INFO = BridgeInfo(
           name="widget",
           service="widget_service",
           description="The user's Widget account: widgets and their status.",
           url_config_var="widget_bridge_url",
           endpoints=(
               "GET /widgets — list widgets",
               "POST /widgets — create a widget",
           ),
           config_vars=("widget_bridge_token",),
           config_var_docs=(("widget_bridge_token", "Shared secret for the X-Semif-Token header."),),
           auth_header="X-Semif-Token",
           auth_config_var="widget_bridge_token",
       )

       def check_requirements(self) -> tuple[bool, str | None]:
           return True, None  # stdlib-only

       def handle_get(self, path, query): ...
       def handle_post(self, path, payload): ...
   ```

   The base class owns the HTTP/JSON/auth plumbing (`start`/`stop`, the
   `X-Semif-Token` guard); a subclass implements only its routes. Override
   `start`/`stop`/`_stop_transport` only if the bridge owns a service daemon.
   Anchor checkout-relative paths (e.g. a token store under `.runtime/`) with
   `semif_agent.bridges.base.checkout_root()`, never from the bridge's own
   `__file__` — the bridge ships in its pack, so its path is pack-relative.

2. **Declare it** in the pack's `bridges` list and ship its module at
   `bridge/<name>.py`. That is what makes it discoverable, describable to
   codegen, and runnable via `python -m semif_agent.cli bridge` — there is no
   core `CATALOG` to edit.

3. **Carry its config block** in the pack's `config` schema
   (`enabled`/`host`/`port`/`token` plus service vars). The schema is what gives
   the bridge its config defaults and validation.

4. **Override `BridgeService.from_config`** only when the bridge needs more than
   its own block: the scheduler's configured `llm` client (the shared LLM
   bridge) or top-level fallback values (like Nextcloud's `nextcloud_*`). A
   plain bridge uses the default and needs no override.

5. **Heavy/optional deps:** `check_requirements()` must **never raise** and must
   return `(ok, install_hint)`. `run_bridges` aborts **all** bridges when one
   reports `ok=False`, so a bridge with an optional dependency (like the
   browser) must return `(True, None)` and surface a per-call `503` with an
   install hint instead — a missing browser must not take down simplex/llm/
   nextcloud. Lazy-import every heavy dep so the core stays stdlib-only.

6. **Report real failures:** a transport/HTTP/bad-reply failure is a `502`, a
   missing connection/dep is a `503`, bad input a `400` — never a fabricated
   success. The bridge owns its own state (cursors, daemons); bodies keep none.

7. **Test it hermetically:** add `bridge/test_<name>_bridge.py` driving the real
   routes over a loopback `http.server` (inject a fake driver where a real
   service would be needed). No external network, no real daemon. Import the
   bridge through the core registry
   (`load_pack_bridge_module("<pack>", "<name>")`) so the test also proves the
   pack's bridge is discoverable; `tests/test_packs.py` runs it from the
   checkout. `describe_bridges()` picks the new bridge up automatically.

8. **Regenerate** `config.example.json` and run `check-config` (see above), and
   document the bridge in `AGENTS.md` (bridges section). Never widen the
   gateway for a new capability.

## Acceptance criteria

A new pack is accepted only if:

1. `packs/<name>/pack.json` parses as JSON, declares a non-empty `concern` and
   a non-empty `categories` map, and every `bridges` name has a
   `bridge/<name>.py` module defining its `BridgeService` subclass.
2. Every leaf under `skills/<category>/<leaf>/` is a complete skill folder
   (`skill.py`, `manifest.json`, `contract.json`, `skill.test.py`) satisfying
   `CODEGEN.md` and `TESTGEN.md`: a module-level `act`, a flat `CONTRACT` whose
   every key is read from `ctx.config`, an `INTEGRATION` consistent with the
   code, a non-empty `manifest.json` `description`, and `contract.json` an exact
   mirror of the body's `CONTRACT`.
3. The leaf's hermetic mechanics test passes
   (`python skill.test.py` from the leaf folder). This proves mechanics only —
   only a real run against the user's service proves the integration.
4. Each category id in the manifest is the id its leaves live under, and each
   category has a navigation-grade scope description.
5. The `config` schema is well-formed (every leaf has `default` + `doc`; secrets
   marked `secret`), maps onto `bridges.<bridges[0]>` (or `packs.<name>` for a
   bridge-less pack), and the regenerated `config.example.json` + `check-config`
   are clean.
6. A bridge-backed pack ships its bridge test (`bridge/test_<bridge>_bridge.py`)
   and it passes hermetically (loopback HTTP, no external network, no daemon).
7. Nothing deployment-specific (hostname, address, username, absolute home
   path, local model name) appears in any committed file —
   `tests/test_no_env_leaks.py` enforces this for the agent repo; keep a pack
   repo clean too. Secrets live in `secrets.json`.

`tests/test_packs.py` enforces items 1–6 mechanically for every shipped pack
(valid manifest, concern + categories, integration matches code, contract flat
and mirrored, manifest description, hermetic skill test, hermetic bridge test)
and keeps the shipped pack ids present.

## Worked example — a minimal bridge-less pack

The smallest complete pack needs a manifest and one leaf. `packs/essentials/`
is the committed model; a hypothetical `packs/hello/`:

```
packs/hello/
  pack.json
  skills/hello/greet/
    skill.py
    manifest.json
    contract.json
    skill.test.py
```

`pack.json`:

```json
{
  "name": "hello",
  "title": "Hello",
  "concern": "A trivial local greeting utility that needs no account, network, or bridge.",
  "categories": {
    "hello": "Greetings: producing a friendly greeting for the user."
  },
  "bridges": [],
  "requires": [],
  "enabled_by_default": true,
  "config": {
    "enabled": {"type": "bool", "default": true, "doc": "Load the hello skills."}
  }
}
```

`skills/hello/greet/manifest.json`:

```json
{ "description": "Produce a friendly greeting." }
```

The leaf's `skill.py` (a `compute`-transport body — no external action, empty
`CONTRACT`) and `skill.test.py` follow `CODEGEN.md`/`TESTGEN.md`; the shipped
`packs/essentials/skills/time/now/` is the reference for exactly this shape.

For a bridge-backed pack, start from `packs/simplex/pack.json` (bridge
reference + `host`/`port`/`token` + service vars), `packs/simplex/bridge/` (the
bridge module, its package marker, and its hermetic test), and
`packs/simplex/skills/simplex/send_message/` (a body that calls the bridge over
HTTP). For a bridge with top-level fallbacks, see `packs/nextcloud/` (its
`bridge/nextcloud.py` overrides `from_config`).

## Checklist

1. Decide the concern and the category ids; write navigation-grade category
   descriptions.
2. If a third-party system is involved and it needs state/auth, a native or
   heavy protocol, or one sanitized surface, author a bridge first (see
   "Authoring a new bridge") and ship it under `bridge/`. A system with a plain
   HTTP API needs no bridge — the skill calls it directly.
3. Create the pack (its own repository, `pack.<name>`): `packs/<name>/pack.json`
   with `name`, `title`, `concern`, `categories`, `bridges` (if any), `requires`,
   `enabled_by_default`, and the `config` schema.
4. Add each leaf under `skills/<category>/<leaf>/` following `CODEGEN.md` and
   `TESTGEN.md` (`skill.py`, `manifest.json`, `contract.json`, `skill.test.py`),
   and each bridge under `bridge/<name>.py` with its `bridge/test_<name>_bridge.py`.
5. Regenerate and validate config:
   `python -m semif_agent.cli check-config --write-example config.example.json`
   then `python -m semif_agent.cli check-config`.
6. Run the tests:
   `python3 -m pytest tests/test_packs.py tests/test_codegen.py -q`
   (it runs every pack's skill and bridge tests, including the new ones).
7. Cross-link the pack/bridge in `AGENTS.md` (and `README.md` for a shipped
   pack).
8. Verify a real run against the user's actual service — the only proof the
   integration works.
9. Register the pack in the `semif-packs` index repo — add an entry (name + clone
   URL, plus title/description for the interview) to its `packs.json` — then push
   the pack's own repository (`pack.<name>`); `scripts/bootstrap.sh` reads the
   index and clones the pack into `packs/`.
