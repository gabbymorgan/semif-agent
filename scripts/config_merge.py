#!/usr/bin/env python3
"""Print the merged config.json as JSON, following its `include` list.

Stdlib-only on purpose: bootstrap.sh reads per-machine values from the config
before the venv exists, so it cannot import `semif_agent.config`. This is the
same include/deep-merge semantics (`semif_agent/config.py`), reduced to the
minimum bootstrap needs (secrets are not resolved — bootstrap only reads
non-secret values like ports and endpoints).

Usage: python3 scripts/config_merge.py [config.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _deep_merge(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def merge(path: Path, seen: set[Path] | None = None) -> dict:
    seen = seen if seen is not None else set()
    resolved = path.resolve()
    if resolved in seen:
        raise SystemExit(f"config include cycle at {path}")
    seen.add(resolved)
    data = json.loads(path.read_text())
    includes = data.pop("include", None)
    merged: dict = {}
    if includes:
        for entry in includes:
            inc = Path(entry)
            if not inc.is_absolute():
                inc = path.parent / inc
            _deep_merge(merged, merge(inc, seen))
    return _deep_merge(merged, data)


if __name__ == "__main__":
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "config.json")
    print(json.dumps(merge(source)))
