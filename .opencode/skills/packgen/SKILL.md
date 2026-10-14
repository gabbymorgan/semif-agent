---
name: Pack Generation
description: Author a new semif-agent skill pack (its own packs/<name>/ repository) — the pack.json manifest, its skill leaves, its bridge reference, and its config schema. Use when adding a new concern/product bundle as its own pack repo, or when adding a new bridge service to back one.
---

# Pack Generation

This skill walks you through adding a **pack** to semif-agent. The full contract
is the repository file `PACKGEN.md` (at the repo root, i.e. `../../../PACKGEN.md`
relative to this skill). Read it before making changes — it is the source of
truth and this skill is only the workflow.

## When to use

- The user wants a new concern/product bundle (its own `packs/<name>/` repo).
- The user wants a new third-party service backed by a bridge that ships with
  its pack (`packs/<name>/bridge/`).
- The user is adding skills that should ship as a pack rather than be
  authored at runtime.

## Workflow

1. **Read `PACKGEN.md`** in full.
2. **Decide the concern and categories.** Write one- to two-sentence,
   navigation-grade category descriptions — SemIf routes on them.
3. **Bridge check.** If the pack talks to a third-party system, confirm a bridge
   already fronts it (a shipped pack's `bridge/`, or the shared LLM bridge). If
   not, author the bridge first (see the "Authoring a new bridge" section of
   `PACKGEN.md`), ship it under `packs/<name>/bridge/`, and give it a pack config
   block.
4. **Create the pack** (its own repository, `pack.<name>`):
   `packs/<name>/pack.json` with `name`, `title`, `concern`, `categories`,
   `bridges`, `requires`, `enabled_by_default`, and the `config` schema. Model
   it on `packs/essentials/pack.json` (bridge-less) or
   `packs/simplex/pack.json` (bridge-backed).
5. **Add each leaf** under `skills/<category>/<leaf>/` — `skill.py`,
   `manifest.json`, `contract.json`, `skill.test.py` — following `CODEGEN.md`
   and `TESTGEN.md`. Do not restate those rules; follow them. Ship each bridge
   under `bridge/<name>.py` with its `bridge/test_<name>_bridge.py`.
6. **Regenerate and validate config:**
   `python -m semif_agent.cli check-config --write-example config.example.json`
   then `python -m semif_agent.cli check-config`.
7. **Run the tests:**
   `python3 -m pytest tests/test_packs.py tests/test_codegen.py -q`
   (it runs every pack's skill and bridge tests, including the new ones).
8. **Cross-link** the pack/bridge in `AGENTS.md` (and `README.md` for a shipped
   pack).
9. **Verify a real run** against the user's actual service — a passing hermetic
   test proves mechanics only, never the live integration.
10. **Register and publish:** add the pack to the `semif-packs` pack-index repo
    (an entry in its `packs.json`: name + clone URL, plus title/description for
    the interview), then commit and push the pack's own repository
    (`pack.<name>`). `scripts/bootstrap.sh` reads the index and clones it into
    `packs/`.

## Guardrails

- A pack is declarative, committed in its own repository; never put a credential
  in it (secrets live in the gitignored `secrets.json`).
- A bridge **ships with its pack** (`packs/<name>/bridge/`); the core keeps only
  the shared contract + discovery. Never widen the gateway.
- `concern` and `categories` are required; every category needs a description.
- Regenerate `config.example.json` from the schema after any schema change; the
  example and the live config are never hand-edited.
