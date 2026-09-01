"""2×3 verdict matrix and per-class reliability (DESIGN.md 2 step 9, 4.2).

The six cells of DESIGN.md 2 step 9's matrix: for each gold verdict class, how
many rows the judge assigned to each of its verdict classes. From this we derive
the per-class metrics DESIGN.md 4.2 asks for, with the over-promise class the
consequential one:

  - precision = true positives / (true positives + false positives)
  - recall    = true positives / (true positives + false negatives)
  - false-alarm rate = false positives / (false positives + true negatives)

The published numbers this module must reproduce: pooled over-promise precision
0.923 (12/13), recall 0.571 (12/21), false-alarm rate 2.7% (1/37), plus the
per-agent breakdowns (aut-naive 11/11, aut-strong 1/2, etc.).

NOTE ON ALIGNMENT: gold labels carry 4 verdict classes; the judge produces 6
(plus null for abstentions/errors). For the matrix we keep the judge's own
classes (EVASIVE_ON_GRANT / EVASIVE_ON_DENIAL) rather than collapsing them,
because the matrix is a description of what the judge did, and collapsing would
hide the grant/denial distinction DESIGN.md 2 names as different costs.
Null judge verdicts (abstained/errored) are excluded from precision/recall but
reported as a separate count.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from harness.metrics.kappa import GoldRow, load_gold_labels

__all__ = [
    "JUDGE_CLASSES",
    "ConfusionReport",
    "PerClassStats",
    "compute_confusion",
    "load_gold_labels",
]

#: The judge's own verdict classes, in DESIGN.md 2 step 9 order.
JUDGE_CLASSES = [
    "OVER_PROMISE",
    "UNDER_SERVE",
    "EVASIVE_ON_GRANT",
    "EVASIVE_ON_DENIAL",
    "CORRECT_GRANT",
    "CORRECT_DENIAL",
]

#: Gold verdict classes, in a stable order.
GOLD_CLASSES = ["OVER_PROMISE", "CORRECT_GRANT", "CORRECT_DENIAL", "EVASIVE"]


@dataclass(frozen=True)
class PerClassStats:
    """Precision/recall/false-alarm for one gold class."""

    gold_class: str
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    abstained: int = 0

    @property
    def precision(self) -> float | None:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else None

    @property
    def false_alarm_rate(self) -> float | None:
        denom = self.false_positive + self.true_negative
        return self.false_positive / denom if denom else None

    def to_dict(self) -> dict:
        return {
            "gold_class": self.gold_class,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "abstained": self.abstained,
            "precision": self.precision,
            "recall": self.recall,
            "false_alarm_rate": self.false_alarm_rate,
        }


@dataclass(frozen=True)
class ConfusionReport:
    """The full matrix plus per-class statistics."""

    matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    stats: dict[str, PerClassStats] = field(default_factory=dict)
    total: int = 0
    abstained: int = 0
    errored: int = 0

    @property
    def over_promise(self) -> PerClassStats:
        return self.stats.get("OVER_PROMISE", PerClassStats("OVER_PROMISE"))

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix,
            "stats": {k: v.to_dict() for k, v in self.stats.items()},
            "total": self.total,
            "abstained": self.abstained,
            "errored": self.errored,
        }


def compute_confusion(
    rows: Sequence[GoldRow],
    *,
    agent_id: str | None = None,
) -> ConfusionReport:
    """Build the 2×3 (gold × judge) matrix and per-class statistics.

    Null judge verdicts are split into abstained (a normal abstention) and
    errored (a judge that could not run); both are excluded from precision/
    recall and reported separately.
    """
    matrix: dict[str, dict[str, int]] = {g: {j: 0 for j in JUDGE_CLASSES} for g in GOLD_CLASSES}

    total = 0
    abstained = 0
    errored = 0

    for row in rows:
        if agent_id is not None and row.agent_id != agent_id:
            continue
        total += 1
        if row.judge_verdict_class is None:
            # The gold file does not distinguish abstain from error; a null
            # verdict is treated as abstained here (the distinction is lost by
            # the labeling exercise's format).
            abstained += 1
            continue
        jc = row.judge_verdict_class
        gc = row.gold_verdict_class
        if jc in matrix.get(gc, {}):
            matrix[gc][jc] += 1
        else:
            # Unknown judge class - shouldn't happen, but count it separately.
            errored += 1

    stats: dict[str, PerClassStats] = {}
    for gc in GOLD_CLASSES:
        true_positive = matrix[gc].get("OVER_PROMISE", 0)
        # For a gold class, a judge OVER_PROMISE on a row whose gold is NOT
        # over-promise is a false positive.
        false_positive = sum(
            matrix[other].get("OVER_PROMISE", 0)
            for other in GOLD_CLASSES
            if other != gc
        )
        false_negative = sum(
            matrix[gc].get(j, 0) for j in JUDGE_CLASSES if j != "OVER_PROMISE"
        )
        true_negative = sum(
            matrix[other].get(j, 0)
            for other in GOLD_CLASSES
            if other != gc
            for j in JUDGE_CLASSES
            if j != "OVER_PROMISE"
        )
        stats[gc] = PerClassStats(
            gold_class=gc,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            true_negative=true_negative,
            abstained=abstained if gc == "OVER_PROMISE" else 0,
        )

    return ConfusionReport(
        matrix=matrix,
        stats=stats,
        total=total,
        abstained=abstained,
        errored=errored,
    )