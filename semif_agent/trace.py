"""Run lifecycle tracing: the events around each request's trip through the
scheduler.

Decision rows live in `decisions.jsonl` (SemIf-compatible). Lifecycle events
(submit, dropped, queued, preempted, assessed, requeued, ...) live here in
`runs.jsonl`, keyed by `run_id` (the request id) so the dashboard can rebuild
each run's flow without touching the decision-log contract. Both files are
append-only; replays read them in order.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class TraceLog:
    def __init__(self, path: str = "data/runs.jsonl"):
        self.path = Path(path)

    def append(self, kind: str, run_id: str, **fields) -> None:
        row = {"kind": kind, "run_id": run_id, "ts": time.time(), **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        rows = []
        with self.path.open("r") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def runs(self) -> dict[str, list[dict]]:
        """Group events by run_id, preserving insertion order of first sighting."""
        grouped: dict[str, list[dict]] = {}
        for row in self.read():
            grouped.setdefault(row.get("run_id", ""), []).append(row)
        return grouped