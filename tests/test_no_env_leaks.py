"""Guard: no deployment-specific artifacts in tracked files.

The repo is generic. Hostnames, addresses, usernames, absolute home paths, and
the local model choices of any one deployment must not leak into tracked files —
they belong in the gitignored ``local/`` folder (see ``local/ENVIRONMENT.md``).
This test fails if a denylisted term reappears anywhere in the tree.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Generated/ignored trees and binary assets. ``local/`` is the intended home
# for deployment specifics and is gitignored, so it is exempt.
SKIP_DIRS = {".git", ".runtime", "local", "data", ".pytest_cache", "__pycache__"}
SKIP_SUFFIXES = {
    ".pyc", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico",
    ".woff", ".woff2", ".ttf", ".pdf", ".zip", ".gz",
}
SKIP_NAMES = {Path(__file__).name}
# Fallback (non-git) walk: skip gitignored per-machine artifacts by name/pattern.
SKIP_GLOBS = ("config.json", "*.bak.*", "*.log")

# label -> regex. Kept specific so a legitimate generic term is not flagged.
DENYLIST = {
    "deployment hostname": r"jarvis|guppy",
    "operator username": r"\babby\b",
    "ssh key comment": r"shitass",
    "git owner": r"gabby",
    "git host": r"manyworlds",
    "lan address": r"192\.168\.8\.",
    "local codegen model": r"qwen38",
    "local llm model": r"qwen3\.5:4b",
    "stale llm default": r"qwen2\.5:3b",
    "absolute home path": r"/home/abby",
    "old venv path": r"semif-venv",
    "machine descriptor": r"\b(?:dev box|dev machine|staging box)\b",
}


def _git_tracked_files() -> list[Path] | None:
    """Tracked files from git (so gitignored per-machine files are exempt).

    Returns ``None`` when git is unavailable or this is not a checkout, so the
    guard still works on an extracted tarball via the walk fallback.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    names = [n for n in out.decode("utf-8", "surrogateescape").split("\0") if n]
    return [REPO_ROOT / n for n in names]


def _iter_tracked_text_files():
    """Yield candidate files, pruning ignored trees and binary assets."""
    tracked = _git_tracked_files()
    if tracked is not None:
        candidates = tracked
    else:
        candidates = []
        for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                path = Path(dirpath) / name
                if any(path.match(g) for g in SKIP_GLOBS):
                    continue
                candidates.append(path)
    for path in candidates:
        if path.name in SKIP_NAMES or path.suffix in SKIP_SUFFIXES:
            continue
        yield path


def test_no_deployment_specific_artifacts():
    hits: list[str] = []
    for path in _iter_tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in DENYLIST.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(
                    f"{path.relative_to(REPO_ROOT)}:{line}: {label} "
                    f"-> {match.group(0)!r}"
                )
    assert not hits, "deployment-specific artifacts in tracked files:\n" + "\n".join(
        hits
    )
