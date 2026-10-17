"""Long-form spoken reading: bookmark positions and speakable chunking.

A skill that resolves a long text (a book) asks the voice front end to read it
aloud; the front end streams it with text-to-speech one speakable chunk at a
time. The position reached is recorded in a ``bookmarks.json`` file beside the
saved text so a later request resumes where the listener stopped.

This module is neutral and stdlib-only: it knows nothing about the scheduler,
the gateway, or audio. The chunker is deterministic — the same text and start
offset always yield the same chunks — so a saved offset is always a valid
resume point.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator

#: The file (beside the saved books) that maps a book key -> last read offset.
BOOKMARKS_FILE = "bookmarks.json"

#: Default characters per spoken chunk. Kept below the voice front end's
#: `max_speak_chars` so a chunk is spoken whole, never truncated.
DEFAULT_CHUNK_CHARS = 300

_WHITESPACE = " \t\r\n"
_SENTENCE_END = ".!?"


def bookmarks_file(directory) -> Path:
    """The bookmarks path inside a books directory."""
    return Path(directory) / BOOKMARKS_FILE


def load_bookmarks(path) -> dict:
    """Read the bookmarks map (``{key: {"offset", "title", "updated_at"}}``).

    A missing or unparseable file is an empty map — bookmarks are an
    optimization, never a reason a read fails.
    """
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_offset(path, key) -> int:
    """The last read offset recorded for `key` (0 when unset/unknown)."""
    entry = load_bookmarks(path).get(str(key))
    if isinstance(entry, dict):
        try:
            return max(0, int(entry.get("offset", 0)))
        except (TypeError, ValueError):
            return 0
    if isinstance(entry, int):  # tolerate a bare-offset legacy shape
        return max(0, entry)
    return 0


def save_bookmark(path, key, offset, title: str = "") -> None:
    """Record the read offset for `key`, atomically.

    Written to a temp file and renamed so a reader never sees a half-written
    map. Any filesystem failure is swallowed: a failed bookmark must not break
    the reading itself.
    """
    target = Path(path)
    data = load_bookmarks(target)
    data[str(key)] = {
        "offset": max(0, int(offset)),
        "title": str(title or ""),
        "updated_at": time.time(),
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp, target)
    except OSError:
        pass


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in _WHITESPACE:
        index += 1
    return index


def _break_point(window: str) -> int:
    """How many characters of `window` to keep for one chunk.

    Prefer a paragraph break, then a sentence end, then a whitespace break —
    but only past the window's midpoint, so a break never yields a tiny chunk.
    Falls back to the whole window when it holds no good boundary.
    """
    half = len(window) // 2
    paragraph = window.rfind("\n\n")
    if paragraph >= half:
        return paragraph + 2
    for index in range(len(window) - 1, -1, -1):
        if window[index] in _SENTENCE_END and (
            index + 1 == len(window) or window[index + 1] in _WHITESPACE + "\"')]"
        ):
            return index + 1
    space = window.rfind(" ")
    if space >= half:
        return space + 1
    return len(window)


def chunk_text(text: str, start: int = 0, max_chars: int = DEFAULT_CHUNK_CHARS) -> Iterator[tuple[int, str]]:
    """Yield ``(offset, chunk)`` speakable chunks of `text` from `start`.

    `offset` is the absolute index of the chunk's first character in `text`, so
    a front end can persist it as the resume point. Deterministic for a given
    `text`, `start`, and `max_chars`.
    """
    text = text or ""
    total = len(text)
    limit_chars = max(1, int(max_chars))
    index = _skip_whitespace(text, max(0, int(start)))
    while index < total:
        window_end = min(index + limit_chars, total)
        if window_end >= total:
            end = total
        else:
            end = index + _break_point(text[index:window_end])
        chunk = text[index:end].strip()
        if chunk:
            yield index, chunk
        index = _skip_whitespace(text, end)
