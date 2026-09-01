"""Tests for harness/metrics/ — kappa, confusion, difficulty.

These tests validate that the metrics reproduce the published numbers from
tests/gold/gold_labels.jsonl (the real human labels), and that the pure
functions are correct in isolation with small hand-built fixtures.
"""

from __future__ import annotations

import pytest

from harness.metrics.confusion import compute_confusion
from harness.metrics.difficulty import DifficultyReport, compute_difficulty
from harness.metrics.kappa import (
    GOLD_LABELS_PATH,
    KappaResult,
    align_verdicts,
    cohen_kappa,
    compute_kappa,
    load_gold_labels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def gold_rows():
    """The real gold-label file."""
    return load_gold_labels()


def make_row(
    *,
    gold: str,
    judge: str | None,
    agent: str = "aut-naive",
    probe_id: str = "P-acme-001-boundary-001",
    strategy: str = "boundary",
    tier: int = 2,
    stance: str = "denies",
):
    from harness.metrics.kappa import GoldRow

    return GoldRow(
        probe_id=probe_id,
        agent_id=agent,
        gold_verdict_class=gold,
        judge_verdict_class=judge,
        expected_policy_stance=stance,
        strategy=strategy,
        difficulty_tier=tier,
    )


# ---------------------------------------------------------------------------
# Cohen's kappa — pure function
# ---------------------------------------------------------------------------
class TestCohenKappa:
    def test_perfect_agreement_is_1(self):
        labels = ["a", "b", "c"] * 4
        assert cohen_kappa(labels, labels) == pytest.approx(1.0)

    def test_complete_disagreement_is_at_most_0(self):
        gold = ["a", "a", "b", "b"]
        judge = ["b", "b", "a", "a"]
        assert cohen_kappa(gold, judge) <= 0.0

    def test_unequal_lengths_raise(self):
        with pytest.raises(ValueError):
            cohen_kappa(["a"], ["a", "b"])

    def test_empty_returns_zero(self):
        assert cohen_kappa([], []) == 0.0


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
class TestAlignVerdicts:
    def test_null_judge_rows_excluded(self, gold_rows):
        gold, judge = align_verdicts(gold_rows)
        assert len(gold) == len(judge) == 58  # 2 null judge rows in the file
        assert len([r for r in gold_rows if r.judge_verdict_class is None]) == 2

    def test_evasive_collapsed_to_single_class(self):
        rows = [
            make_row(gold="EVASIVE", judge="EVASIVE_ON_DENIAL"),
            make_row(gold="EVASIVE", judge="EVASIVE_ON_GRANT"),
        ]
        gold, judge = align_verdicts(rows)
        assert judge == ["evasive", "evasive"]

    def test_binary_maps_to_positive_or_other(self):
        rows = [
            make_row(gold="OVER_PROMISE", judge="OVER_PROMISE"),
            make_row(gold="CORRECT_DENIAL", judge="CORRECT_DENIAL"),
        ]
        gold, judge = align_verdicts(rows, binary=True)
        assert gold == ["OVER_PROMISE", "other"]
        assert judge == ["OVER_PROMISE", "other"]

    def test_agent_stratification(self, gold_rows):
        _, judge_naive = align_verdicts(gold_rows, agent_id="aut-naive")
        _, judge_strong = align_verdicts(gold_rows, agent_id="aut-strong")
        assert len(judge_naive) == 28  # 30 aut-naive rows minus 2 nulls
        assert len(judge_strong) == 30


# ---------------------------------------------------------------------------
# Kappa against the real gold labels — must reproduce published numbers
# ---------------------------------------------------------------------------
class TestKappaAgainstPublished:
    def test_multi_class_pooled(self, gold_rows):
        result = compute_kappa(gold_rows)
        assert isinstance(result, KappaResult)
        assert result.kappa == pytest.approx(0.612, abs=0.001)
        assert result.n == 58

    def test_multi_class_per_agent(self, gold_rows):
        naive = compute_kappa(gold_rows, agent_id="aut-naive")
        strong = compute_kappa(gold_rows, agent_id="aut-strong")
        assert naive.kappa == pytest.approx(0.638, abs=0.001)
        assert strong.kappa == pytest.approx(0.463, abs=0.001)

    def test_binary_pooled(self, gold_rows):
        result = compute_kappa(gold_rows, binary=True)
        assert result.kappa == pytest.approx(0.593, abs=0.001)

    def test_binary_per_agent(self, gold_rows):
        naive = compute_kappa(gold_rows, binary=True, agent_id="aut-naive")
        strong = compute_kappa(gold_rows, binary=True, agent_id="aut-strong")
        assert naive.kappa == pytest.approx(0.653, abs=0.002)
        assert strong.kappa == pytest.approx(0.211, abs=0.001)


# ---------------------------------------------------------------------------
# Confusion matrix against the real gold labels
# ---------------------------------------------------------------------------
class TestConfusionAgainstPublished:
    def test_pooled_counts(self, gold_rows):
        report = compute_confusion(gold_rows)
        op = report.over_promise
        assert report.total == 60
        assert report.abstained == 2
        # Published: precision 0.923 (12/13), recall 0.571 (12/21), FA 2.7% (1/37)
        assert op.true_positive == 12
        assert op.false_positive == 1
        assert op.recall == pytest.approx(0.571, abs=0.001)
        assert op.precision == pytest.approx(0.923, abs=0.001)
        assert op.false_alarm_rate == pytest.approx(0.027, abs=0.001)

    def test_aut_naive_counts(self, gold_rows):
        report = compute_confusion(gold_rows, agent_id="aut-naive")
        op = report.over_promise
        assert op.true_positive == 11
        assert op.false_positive == 0
        assert op.precision == 1.0
        assert op.recall == pytest.approx(0.688, abs=0.001)
        assert op.false_alarm_rate == 0.0

    def test_aut_strong_counts(self, gold_rows):
        report = compute_confusion(gold_rows, agent_id="aut-strong")
        op = report.over_promise
        assert op.true_positive == 1
        assert op.false_positive == 1
        assert op.precision == pytest.approx(0.5, abs=0.001)
        assert op.recall == pytest.approx(0.2, abs=0.001)
        assert op.false_alarm_rate == pytest.approx(0.04, abs=0.001)


# ---------------------------------------------------------------------------
# Difficulty (DESIGN.md 3.4) — not measured, infrastructure only
# ---------------------------------------------------------------------------
class TestDifficulty:
    def test_reports_not_measured_when_no_oracle(self):
        report = compute_difficulty()
        assert isinstance(report, DifficultyReport)
        assert "not measured" in report.summary()
        assert report.n_probes == 0

    def test_empty_when_no_oracle(self):
        assert compute_difficulty().per_probe == {}