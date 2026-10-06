"""Decision logging: every SemIf call is written as a labeled training row.

Rows match the SemIf `decisions.jsonl` shape plus the prediction-observation
cost fields. `observed_outcome` is the label for the cost function; by default
it is the option that was actually selected (self-consistent), and a human
override can relabel a row to the correct outcome with a higher weight.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .decisions import DecisionRequest, DecisionResult


class DecisionLog:
    def __init__(self, path: str = "data/decisions.jsonl"):
        self.path = Path(path)

    def append(
        self,
        request: DecisionRequest,
        result: DecisionResult,
        label: str | None = None,
        extra: dict | None = None,
    ) -> None:
        """Append one decision row. `label` overrides observed_outcome."""
        observed = label if label is not None else result.selected
        source = "human" if label is not None else "self"
        row = {
            "id": request.id,
            "ts": time.time(),
            "state": request.state,
            "question": request.question,
            "options": [{"id": o.id, "description": o.description} for o in request.options],
            "predicted_probs": result.probs,
            "selected": result.selected,
            "observed_outcome": observed,
            "label_source": source,
        }
        if extra:
            row["extra"] = extra
        result_extra = getattr(result, "extra", None) or {}
        timing = {
            key: result_extra[key]
            for key in ("input_tokens", "forward_seconds", "total_seconds")
            if key in result_extra
        }
        if timing:
            row.setdefault("extra", {})["timing"] = timing
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def relabel(self, decision_id: str, observed: str) -> bool:
        """Human override: set a corrected observed outcome for one row."""
        rows = self.read()
        found = False
        for row in rows:
            if row["id"] == decision_id:
                row["observed_outcome"] = observed
                row["label_source"] = "human"
                found = True
        if not found:
            return False
        self._write(rows)
        return True

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

    def _write(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
