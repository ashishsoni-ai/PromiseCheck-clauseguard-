"""Cohen's kappa vs gold labels, over scipy (DESIGN.md 4.2).

Provides both multi-class and binary Cohen's kappa, with optional per-agent
stratification. The gold-label file (`tests/gold/gold_labels.jsonl`) carries
4 verdict classes; the judge produces 6 (plus null for abstentions/errors).
Multi-class kappa aligns the two sets by collapsing judge's EVASIVE_ON_GRANT
and EVASIVE_ON_DENIAL into a single "evasive" class, matching the gold label
vocabulary, and excludes rows where the judge returned no verdict (null).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "GOLD_LABELS_PATH",
    "GoldRow",
    "KappaResult",
    "align_verdicts",
    "cohen_kappa",
    "compute_kappa",
    "load_gold_labels",
]

GOLD_LABELS_PATH = Path("tests/gold/gold_labels.jsonl")

#: Map judge verdict classes to the gold-label vocabulary for multi-class κ.
#: Judge distinguishes EVASIVE_ON_GRANT and EVASIVE_ON_DENIAL; gold collapses
#: both to EVASIVE. No judge class maps to CORRECT_GRANT or CORRECT_DENIAL at
#: the class level (those are the gold labels; the judge has its own).
JUDGE_TO_GOLD_CLASS = {
    "OVER_PROMISE": "OVER_PROMISE",
    "UNDER_SERVE": "UNDER_SERVE",
    "EVASIVE_ON_GRANT": "evasive",
    "EVASIVE_ON_DENIAL": "evasive",
    "CORRECT_GRANT": "CORRECT_GRANT",
    "CORRECT_DENIAL": "CORRECT_DENIAL",
}

#: The gold label vocabulary. Matches what gold_labels.jsonl actually uses.
GOLD_CLASSES = ["OVER_PROMISE", "CORRECT_GRANT", "CORRECT_DENIAL", "EVASIVE"]

#: Binary re-class: "over-promise" vs everything else.
BINARY_POSITIVE = "OVER_PROMISE"


@dataclass(frozen=True)
class GoldRow:
    """One row from the gold label file."""

    probe_id: str
    agent_id: str
    gold_verdict_class: str
    judge_verdict_class: str | None
    expected_policy_stance: str
    strategy: str
    difficulty_tier: int


def load_gold_labels(path: Path = GOLD_LABELS_PATH) -> list[GoldRow]:
    """Load and parse the gold-label file."""
    rows: list[GoldRow] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line:
            continue
        d = json.loads(line)
        rows.append(
            GoldRow(
                probe_id=d["probe_id"],
                agent_id=d["agent_id"],
                gold_verdict_class=d["gold_verdict_class"],
                judge_verdict_class=d.get("judge_verdict_class"),
                expected_policy_stance=d["expected_policy_stance"],
                strategy=d["strategy"],
                difficulty_tier=d["difficulty_tier"],
            )
        )
    return rows


def align_verdicts(
    rows: Sequence[GoldRow],
    *,
    binary: bool = False,
    agent_id: str | None = None,
) -> tuple[list[str], list[str]]:
    """Align gold and judge verdicts for κ computation.

    Returns (gold_classes, judge_classes) as parallel lists. Rows where the
    judge returned None (abstention/error) are excluded. When `agent_id` is
    given, only rows for that agent are returned.

    In multi-class mode, judge verdicts are mapped to the gold vocabulary
    (EVASIVE_ON_GRANT/EVASIVE_ON_DENIAL → "evasive"). In binary mode, both
    gold and judge verdicts are mapped to "OVER_PROMISE" vs "other".
    """
    gold: list[str] = []
    judge: list[str] = []
    for row in rows:
        if agent_id is not None and row.agent_id != agent_id:
            continue
        if row.judge_verdict_class is None:
            continue
        jc = JUDGE_TO_GOLD_CLASS.get(row.judge_verdict_class, row.judge_verdict_class)
        gc = row.gold_verdict_class

        if binary:
            jc = BINARY_POSITIVE if jc == BINARY_POSITIVE else "other"
            gc = BINARY_POSITIVE if gc == BINARY_POSITIVE else "other"

        gold.append(gc)
        judge.append(jc)

    return gold, judge


def cohen_kappa(
    gold: Sequence[str],
    judge: Sequence[str],
    labels: Sequence[str] | None = None,
) -> float:
    """Cohen's κ (multi-class or binary). Manual implementation.

    κ = (p_o - p_e) / (1 - p_e)

    where p_o is observed agreement and p_e is expected agreement by chance.
    Deliberately implemented directly rather than depending on a scipy
    routine whose version availability changes.
    """
    n = len(gold)
    if n == 0:
        return 0.0
    if len(judge) != n:
        raise ValueError("gold and judge sequences must have the same length")

    # Observed agreement
    p_o = sum(1 for g, j in zip(gold, judge) if g == j) / n

    # Expected agreement: sum over classes of (gold_pct * judge_pct)
    if labels is None:
        labels = sorted(set(gold) | set(judge))
    gold_counts = Counter(gold)
    judge_counts = Counter(judge)
    p_e = sum(
        (gold_counts.get(l, 0) / n) * (judge_counts.get(l, 0) / n)
        for l in labels
    )

    if p_e >= 1.0:
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)


@dataclass(frozen=True)
class KappaResult:
    """Kappa result with metadata."""

    kappa: float
    n: int
    labels: tuple[str, ...]
    binary: bool
    agent_id: str | None


def compute_kappa(
    rows: Sequence[GoldRow],
    *,
    binary: bool = False,
    agent_id: str | None = None,
) -> KappaResult:
    """Compute κ for a gold-label set, optionally stratified by agent."""
    gold, judge = align_verdicts(rows, binary=binary, agent_id=agent_id)
    labels = ["other", "OVER_PROMISE"] if binary else GOLD_CLASSES
    k = cohen_kappa(gold, judge, labels=labels)
    return KappaResult(
        kappa=round(k, 3),
        n=len(gold),
        labels=tuple(labels),
        binary=binary,
        agent_id=agent_id,
    )