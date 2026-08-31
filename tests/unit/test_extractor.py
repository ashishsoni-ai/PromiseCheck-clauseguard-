"""Tests for harness/extract/extractor.py, compare.py, and prompts.py.

Offline tests use a fake client; the 1 live test uses the real extractor model
and is gated behind `@pytest.mark.live` + `require_groq_credentials`.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from harness.extract.compare import compare_rule_sets
from harness.extract.extractor import (
    ExtractorClient,
    ExtractorError,
    LitellmExtractorClient,
    extract_rules,
)
from harness.extract.prompts import format_clauses_for_prompt
from harness.schemas import (
    Clause,
    Condition,
    EntitlementRule,
    PolicyDocument,
)

VERSION = "sha256:" + "ab" * 32
T0 = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Clause / document fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def make_clause():
    def _make(
        *,
        text: str,
        ordinal: int = 1,
        content_hash: str = "a3f91c22",
        doc_slug: str = "acme-refunds",
        heading_path: list[str] | None = None,
    ) -> Clause:
        return Clause(
            clause_id=f"{doc_slug}:{ordinal:03d}:{content_hash}",
            doc_slug=doc_slug,
            ordinal=ordinal,
            text=text,
            content_hash=content_hash,
            heading_path=heading_path or [],
        )
    return _make


@pytest.fixture
def two_clause_policy(make_clause) -> PolicyDocument:
    return PolicyDocument(
        doc_slug="acme-refunds",
        source="policies/test.md",
        policy_version=VERSION,
        fetched_at=T0,
        corpus_role="worked_example",
        clauses=[
            make_clause(
                text="Returns must be initiated within 30 days of delivery.",
                ordinal=1,
                content_hash="11111111",
                heading_path=["Returns", "Return window"],
            ),
            make_clause(
                text="Clearance items must be returned within 7 days of delivery.",
                ordinal=2,
                content_hash="22222222",
                heading_path=["Returns", "Return window"],
            ),
        ],
    )


@pytest.fixture
def well_grounded_rules() -> list[EntitlementRule]:
    """Rules whose source_spans are verbatim in the two-clause policy above."""
    return [
        EntitlementRule(
            rule_id="refund-window-30d",
            clause_ids=["acme-refunds:001:11111111"],
            entitlement="refund",
            polarity="grants",
            conditions=[
                Condition(
                    attribute="days_since_delivery",
                    op="<=",
                    value=30,
                    source_span="Returns must be initiated within 30 days of delivery.",
                )
            ],
            precedence=10,
            extraction_confidence=0.95,
            needs_human_review=False,
        ),
        EntitlementRule(
            rule_id="refund-clearance-7d",
            clause_ids=["acme-refunds:002:22222222"],
            entitlement="refund",
            polarity="denies",
            conditions=[
                Condition(
                    attribute="days_since_delivery",
                    op=">",
                    value=7,
                    source_span="Clearance items must be returned within 7 days of delivery.",
                )
            ],
            precedence=50,
            extraction_confidence=0.92,
            needs_human_review=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Fake client for offline tests
# ---------------------------------------------------------------------------
class FakeExtractor:
    def __init__(
        self,
        rules: list[EntitlementRule],
        model: str = "test/fake",
    ) -> None:
        self._rules = rules
        self._model = model
        self._call_count = 0

    @property
    def model(self) -> str:
        return self._model

    def extract(self, *, system: str, user: str, temperature: float) -> list[EntitlementRule]:
        self._call_count += 1
        return self._rules


# ---------------------------------------------------------------------------
# Tests: prompts
# ---------------------------------------------------------------------------
class TestFormatClausesForPrompt:
    def test_heading_path_is_rendered(self, make_clause):
        clause = make_clause(
            text="A test clause.",
            heading_path=["Section 1", "Subsection A"],
        )
        result = format_clauses_for_prompt([clause])
        assert "Section 1 > Subsection A" in result
        assert "acme-refunds:001:a3f91c22" in result
        assert "A test clause." in result


# ---------------------------------------------------------------------------
# Tests: extractor control flow
# ---------------------------------------------------------------------------
class TestExtractRules:
    def test_returns_well_grounded_rules(self, two_clause_policy, well_grounded_rules):
        client = FakeExtractor(well_grounded_rules)
        rules = extract_rules(two_clause_policy, client=client)
        assert len(rules) == 2
        assert all(r.needs_human_review is False for r in rules)
        assert client._call_count == 1

    def test_retries_on_ungrounded_spans(self, two_clause_policy, well_grounded_rules):
        ungrounded = [
            EntitlementRule(
                rule_id="bad-rule",
                clause_ids=["acme-refunds:001:11111111"],
                entitlement="refund",
                polarity="grants",
                conditions=[
                    Condition(
                        attribute="days_since_delivery",
                        op="<=",
                        value=30,
                        source_span="This span does not exist in the clause text",
                    )
                ],
                precedence=10,
                extraction_confidence=0.5,
                needs_human_review=False,
            )
        ]

        class TwoPhase:
            def __init__(self):
                self._model = "test/two-phase"
                self._call_count = 0

            @property
            def model(self) -> str:
                return self._model

            def extract(self, *, system, user, temperature):
                self._call_count += 1
                if self._call_count == 1:
                    return ungrounded
                return well_grounded_rules

        client = TwoPhase()
        rules = extract_rules(two_clause_policy, client=client)
        assert len(rules) == 2
        assert client._call_count == 2
        assert all(r.needs_human_review is False for r in rules)

    def test_ungrounded_after_retry_gets_flagged(self, two_clause_policy):
        ungrounded = [
            EntitlementRule(
                rule_id="stubborn-rule",
                clause_ids=["acme-refunds:001:11111111"],
                entitlement="refund",
                polarity="grants",
                conditions=[
                    Condition(
                        attribute="days_since_delivery",
                        op="<=",
                        value=30,
                        source_span="This span does not exist in the clause text",
                    )
                ],
                precedence=10,
                extraction_confidence=0.5,
                needs_human_review=False,
            )
        ]
        client = FakeExtractor(ungrounded)
        rules = extract_rules(two_clause_policy, client=client)
        assert len(rules) == 1
        assert rules[0].needs_human_review is True
        assert client._call_count == 2


# ---------------------------------------------------------------------------
# Tests: comparison
# ---------------------------------------------------------------------------
class TestCompareRuleSets:
    def _make_rule(self, rule_id, entitlement="refund", polarity="grants", conditions=None):
        return EntitlementRule(
            rule_id=rule_id,
            clause_ids=["acme-refunds:001:11111111"],
            entitlement=entitlement,
            polarity=polarity,
            conditions=conditions or [],
            precedence=10,
            extraction_confidence=0.9,
            needs_human_review=False,
        )

    def test_exact_match(self):
        r = self._make_rule("R-001")
        report = compare_rule_sets([r], [r])
        assert len(report.equivalent) == 1
        assert len(report.different) == 0
        assert len(report.missed) == 0
        assert len(report.inventions) == 0

    def test_missed_rule(self):
        r1 = self._make_rule("R-001")
        r2 = self._make_rule("R-002")
        report = compare_rule_sets([r1, r2], [r1])
        assert len(report.equivalent) == 1
        assert len(report.missed) == 1

    def test_invention(self):
        r1 = self._make_rule("R-001")
        r2 = self._make_rule("R-002", entitlement="discount")
        report = compare_rule_sets([r1], [r1, r2])
        assert len(report.inventions) == 1
        assert report.inventions[0].rule_id == "R-002"


# ---------------------------------------------------------------------------
# Live test
# ---------------------------------------------------------------------------
@pytest.mark.live
class TestLiveExtractor:
    def test_live_extraction_returns_parsed_rules(
        self, require_groq_credentials, two_clause_policy
    ):
        client = LitellmExtractorClient()
        rules = extract_rules(two_clause_policy, client=client)

        assert len(rules) > 0, "extractor produced no rules for a two-clause policy"
        for rule in rules:
            assert isinstance(rule, EntitlementRule)
            assert rule.rule_id
            assert rule.entitlement in (
                "refund", "partial_refund", "replacement", "waiver",
                "extension", "discount", "credit", "cancellation",
            )
            assert rule.polarity in ("grants", "denies")
            assert rule.precedence > 0
            assert 0.0 <= rule.extraction_confidence <= 1.0
            for cond in rule.conditions:
                assert cond.source_span, "every condition needs a source_span"
                assert cond.attribute
                assert cond.op in ("<=", "<", ">=", ">", "==", "in", "not_in")