#!/usr/bin/env python3
"""Prime Playwright's browser cache over IPv4 (a fallback for `playwright install`).

`playwright install` downloads through Playwright's own Node fetcher, which
forces `autoSelectFamily` with a custom dual-stack resolver that prefers IPv6.
On a host that has an IPv6 address (often a ULA) but no working global IPv6
route, that first IPv6 connection hangs instead of falling back to IPv4, so the
download stalls until the socket timeout and `playwright install` fails even
though the CDN is reachable over IPv4.

This helper downloads the exact pinned artifacts Playwright expects, forcing
IPv4, and lays them out under `PLAYWRIGHT_BROWSERS_PATH` with the
`INSTALLATION_COMPLETE` marker Playwright checks, so the normal install (and the
browser bridge) then works. It reads revisions/versions from the installed
Playwright's own `browsers.json`, so it tracks version bumps automatically.

Idempotent: an artifact with its marker present is skipped. Only Chromium (plus
its headless shell and ffmpeg) is handled; Firefox/WebKit have no such helper.

Usage:
  scripts/playwright-browsers.py --browsers-path .runtime/browser
  scripts/playwright-browsers.py --browsers-path .runtime/browser --engine chromium
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

#: The CDN mirror Playwright's fetcher falls back to and that serves the exact
#: Chrome-for-Testing layout over a redirect (IPv4 works).
CDN = "https://cdn.playwright.dev"

#: Executables that must stay executable after a pure-Python unzip (which does
#: not preserve permissions). Relative to each browser's directory.
_EXECUTABLES = {
    "chromium": ("chrome-linux64/chrome",),
    "chromium_headless_shell": ("chrome-headless-shell-linux64/chrome-headless-shell",),
    "ffmpeg": ("ffmpeg-linux",),
}


def _arch_suffix() -> str:
    machine = platform.machine().lower()
    return "arm64" if machine in ("aarch64", "arm64") else "x64"


def _cft_paths(arch: str) -> dict[str, str]:
    if arch == "arm64":
        return {
            "chromium": "linux-arm64/chrome-linux-arm64.zip",
            "chromium_headless_shell": "linux-arm64/chrome-headless-shell-linux-arm64.zip",
            "ffmpeg": "ffmpeg-linux-arm64.zip",
        }
    return {
        "chromium": "linux64/chrome-linux64.zip",
        "chromium_headless_shell": "linux64/chrome-headless-shell-linux64.zip",
        "ffmpeg": "ffmpeg-linux.zip",
    }


def _browsers_json() -> dict:
    import playwright  # the package the venv already has

    path = Path(playwright.__file__).resolve().parent / "driver" / "package" / "browsers.json"
    with open(path) as handle:
        return json.load(handle)


def _artifacts(browsers_json: dict, arch: str) -> list[tuple[str, str]]:
    """List of `(directory_name, download_path)` Playwright expects."""
    suffixes = _cft_paths(arch)
    wanted = {
        "chromium": ("chromium", "chromium"),
        "chromium-headless-shell": ("chromium_headless_shell", "chromium_headless_shell"),
        "ffmpeg": ("ffmpeg", "ffmpeg"),
    }
    out: list[tuple[str, str]] = []
    for browser in browsers_json.get("browsers", []):
        name = browser.get("name")
        if name not in wanted or not browser.get("installByDefault", True):
            continue
        slug, key = wanted[name]
        revision = browser["revision"]
        if key == "ffmpeg":
            path = f"builds/ffmpeg/{revision}/{suffixes['ffmpeg']}"
        else:
            path = f"builds/cft/{browser['browserVersion']}/{suffixes[key]}"
        out.append((f"{slug}-{revision}", path))
    return out


def _download(url: str, dest: str) -> None:
    if shutil.which("curl"):
        subprocess.run(
            ["curl", "-4", "-fL", "--retry", "3", "-m", "600", "-o", dest, url],
            check=True,
        )
        return
    # No curl: urllib may pick IPv6 on a broken host, but it is the best we have.
    print(f"    (curl not found; using urllib for {url})", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "semif-bootstrap"})
    with urllib.request.urlopen(request, timeout=600) as response, open(dest, "wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract(archive: str, target: Path, directory_name: str) -> None:
    if shutil.which("unzip"):
        subprocess.run(["unzip", "-q", "-o", archive, "-d", str(target)], check=True)
    else:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        for relative in _EXECUTABLES.get(directory_name.split("-")[0], ()):
            candidate = target / relative
            if candidate.is_file():
                candidate.chmod(candidate.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _install_one(browsers_path: Path, directory_name: str, download_path: str) -> bool:
    target = browsers_path / directory_name
    marker = target / "INSTALLATION_COMPLETE"
    if marker.is_file():
        print(f"  = {directory_name} already installed")
        return True
    url = f"{CDN}/{download_path}"
    print(f"  + {directory_name} <- {url}")
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="semif-browser-") as tmp:
        archive = os.path.join(tmp, "download.zip")
        _download(url, archive)
        _extract(archive, target, directory_name)
    marker.write_text("")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prime Playwright's browser cache over IPv4.")
    parser.add_argument("--browsers-path", required=True, help="PLAYWRIGHT_BROWSERS_PATH to populate")
    parser.add_argument("--engine", default="chromium", help="only 'chromium' is supported by this helper")
    args = parser.parse_args(argv)

    if args.engine != "chromium":
        print(f"helper only primes chromium, not {args.engine!r}; nothing to do", file=sys.stderr)
        return 0

    browsers_path = Path(args.browsers_path).expanduser().resolve()
    browsers_path.mkdir(parents=True, exist_ok=True)
    arch = _arch_suffix()
    artifacts = _artifacts(_browsers_json(), arch)
    if not artifacts:
        print("no chromium artifacts found in playwright's browsers.json", file=sys.stderr)
        return 1

    print(f"priming Playwright browsers into {browsers_path} ({arch}, IPv4)")
    for directory_name, download_path in artifacts:
        _install_one(browsers_path, directory_name, download_path)
    print("done; 'playwright install chromium' should now be a no-op")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
