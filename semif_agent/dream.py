"""The dream pass: compute the prediction-observation cost over the decision log.

Replays `decisions.jsonl` and reports the training signal for the fine-tuning
step: per-row negative log likelihood (NLL) of the observed outcome under the
predicted distribution, overall cross-entropy, accuracy, and binned expected
calibration error (ECE). Human-override labels are weighted higher, matching
the design (self-assessment + human overrides). This computes the signal; the
actual fine-tuning and CI/CD model swap are a separate v2 step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .log import DecisionLog

HUMAN_WEIGHT = 3.0
SELF_WEIGHT = 1.0


@dataclass
class RowCost:
    decision_id: str
    predicted: float
    observed: str
    selected: str
    nll: float
    weight: float
    correct: bool


@dataclass
class DreamReport:
    rows: list[RowCost] = field(default_factory=list)
    skipped: int = 0
    human_overrides: int = 0

    @property
    def cross_entropy(self) -> float | None:
        if not self.rows:
            return None
        total = sum(r.weight for r in self.rows)
        return sum(r.weight * r.nll for r in self.rows) / total

    @property
    def accuracy(self) -> float | None:
        if not self.rows:
            return None
        return sum(r.correct for r in self.rows) / len(self.rows)

    @property
    def ece(self) -> float | None:
        """Binned expected calibration error over confidence of the selected option."""
        if not self.rows:
            return None
        bins: dict[int, list[RowCost]] = {index: [] for index in range(10)}
        for row in self.rows:
            bin_index = min(int(row.predicted * 10), 9)
            bins[bin_index].append(row)
        total_weight = sum(r.weight for r in self.rows)
        if total_weight <= 0:
            return None
        error = 0.0
        for index, members in bins.items():
            if not members:
                continue
            weight = sum(r.weight for r in members)
            confidence = sum(r.predicted * r.weight for r in members) / weight
            accuracy = sum(r.correct * r.weight for r in members) / weight
            error += abs(confidence - accuracy) * weight / total_weight
        return error

    def render(self) -> str:
        lines = [
            f"decision rows: {len(self.rows)}  (skipped {self.skipped})",
            f"human overrides: {self.human_overrides}",
        ]
        ce = self.cross_entropy
        acc = self.accuracy
        ece = self.ece
        lines.append(f"cross-entropy (weighted): {ce:.4f}" if ce is not None else "cross-entropy: n/a")
        lines.append(f"accuracy (selected==observed): {acc:.3f}" if acc is not None else "accuracy: n/a")
        lines.append(f"ECE (10 bins): {ece:.4f}" if ece is not None else "ECE: n/a")
        if self.rows:
            worst = sorted(self.rows, key=lambda r: r.nll, reverse=True)[:5]
            lines.append("highest-cost rows:")
            for row in worst:
                lines.append(
                    f"  {row.decision_id}  pred={row.predicted:.3f} selected={row.selected} "
                    f"observed={row.observed} nll={row.nll:.3f}"
                )
        return "\n".join(lines)


def dream(log: DecisionLog) -> DreamReport:
    report = DreamReport()
    for row in log.read():
        probs = row.get("predicted_probs") or {}
        observed = row.get("observed_outcome")
        selected = row.get("selected")
        if observed is None or selected is None or observed not in probs:
            report.skipped += 1
            continue
        predicted = float(probs[observed])
        if predicted <= 0:
            predicted = 1e-9
        nll = -math.log(predicted)
        weight = HUMAN_WEIGHT if row.get("label_source") == "human" else SELF_WEIGHT
        if row.get("label_source") == "human":
            report.human_overrides += 1
        report.rows.append(
            RowCost(
                decision_id=row["id"],
                predicted=predicted,
                observed=observed,
                selected=selected,
                nll=nll,
                weight=weight,
                correct=(selected == observed),
            )
        )
    return report