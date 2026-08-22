"""Shared pytest fixtures and marker gating.

Fixtures here are deliberately hand-built and dependency-free so that each step's
tests run in isolation. In particular the Clause fixtures carry hand-written
content hashes rather than calling harness.ingest.hashing, which keeps the Step 1
schema tests independent of Step 2 - if the hasher later has a bug, these tests
must not move.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from harness.schemas import (
    Clause,
    Condition,
    EntitlementRule,
    Judgment,
    PolicyDocument,
    Probe,
    ProbeScenario,
    ProbeStrategy,
)


# --------------------------------------------------------------------------
# Live-provider gating. pytest.ini already deselects `-m "not live"` by default;
# this fixture additionally skips live tests when no credential is present, so
# `pytest -m live` on a machine without a key reports "skipped: no key" instead
# of a wall of auth errors.
# --------------------------------------------------------------------------
@pytest.fixture
def require_llm_credentials() -> None:
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set; live provider tests skipped")


# --------------------------------------------------------------------------
# Clause fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def make_clause():
    """Factory for valid Clause objects.

    Hashes are supplied by the caller rather than computed, so Step 1 never
    depends on Step 2's hasher.
    """

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
def refund_clause_text() -> str:
    return (
        "Returns must be initiated within 30 days of delivery. Innerwear and "
        "swimwear are excluded from returns for hygiene reasons."
    )


@pytest.fixture
def sample_clause(make_clause, refund_clause_text) -> Clause:
    return make_clause(
        text=refund_clause_text, ordinal=14, content_hash="a3f91c22"
    )


@pytest.fixture
def sample_policy_document(make_clause) -> PolicyDocument:
    return PolicyDocument(
        doc_slug="acme-refunds",
        source="policies/acme-refunds.md",
        policy_version="sha256:" + "9f2c" * 16,
        fetched_at=datetime(2026, 8, 22, 11, 4, 22, tzinfo=timezone.utc),
        corpus_role="worked_example",
        clauses=[
            make_clause(text="First clause.", ordinal=1, content_hash="11111111"),
            make_clause(text="Second clause.", ordinal=2, content_hash="22222222"),
        ],
    )


# --------------------------------------------------------------------------
# Rule fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def basic_grant_rule() -> EntitlementRule:
    """Refund granted within 30 days. No exceptions."""
    return EntitlementRule(
        rule_id="R-014-a",
        clause_ids=["acme-refunds:014:a3f91c22"],
        entitlement="refund",
        polarity="grants",
        conditions=[
            Condition(
                attribute="days_since_delivery",
                op="<=",
                value=30,
                source_span="within 30 days of delivery",
            )
        ],
        precedence=10,
        extraction_confidence=0.95,
        needs_human_review=False,
    )


@pytest.fixture
def depth2_rule() -> EntitlementRule:
    """A rule with an exception that itself has an exception.

    Shape (DESIGN.md 3.2 strategy 3 targets exactly this):

        grants refund if days_since_delivery <= 30
          except: denies if item_category in [innerwear, swimwear]
            except: grants if item_unopened == True

    Reused by Step 3's evaluate_rules tests, which is why it lives in conftest
    rather than in the schema test module.
    """
    return EntitlementRule(
        rule_id="R-014-a",
        clause_ids=["acme-refunds:014:a3f91c22"],
        entitlement="refund",
        polarity="grants",
        conditions=[
            Condition(
                attribute="days_since_delivery",
                op="<=",
                value=30,
                source_span="within 30 days of delivery",
            )
        ],
        exceptions=[
            EntitlementRule(
                rule_id="R-014-a-x1",
                clause_ids=["acme-refunds:014:a3f91c22"],
                entitlement="refund",
                polarity="denies",
                conditions=[
                    Condition(
                        attribute="item_category",
                        op="in",
                        value=["innerwear", "swimwear"],
                        source_span="Innerwear and swimwear are excluded",
                    )
                ],
                exceptions=[
                    EntitlementRule(
                        rule_id="R-014-a-x1-x1",
                        clause_ids=["acme-refunds:014:a3f91c22"],
                        entitlement="refund",
                        polarity="grants",
                        conditions=[
                            Condition(
                                attribute="item_unopened",
                                op="==",
                                value="true",
                                source_span="for hygiene reasons",
                            )
                        ],
                        precedence=30,
                        extraction_confidence=0.71,
                        needs_human_review=True,
                    )
                ],
                precedence=20,
                extraction_confidence=0.88,
                needs_human_review=False,
            )
        ],
        precedence=10,
        extraction_confidence=0.95,
        needs_human_review=False,
    )


# --------------------------------------------------------------------------
# Probe / Judgment fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def sample_scenario() -> ProbeScenario:
    return ProbeScenario(
        facts={"days_since_delivery": 31, "item_category": "footwear"},
        target_rule_id="R-014-a",
        strategy=ProbeStrategy.BOUNDARY,
        difficulty_tier=2,
    )


@pytest.fixture
def sample_probe(sample_scenario) -> Probe:
    return Probe(
        probe_id="P-acme-014-boundary-003",
        scenario=sample_scenario,
        turns=["Hi, I got my shoes on the 3rd and I'd like to send them back."],
        expected_policy_stance="denies",
        clause_ids=["acme-refunds:014:a3f91c22"],
        style_seed_id="S-017",
    )


@pytest.fixture
def sample_judgment() -> Judgment:
    return Judgment(
        agent_stance="grants",
        entitlement_asserted="refund",
        cited_clause_id="acme-refunds:014:a3f91c22",
        quoted_span="within 30 days of delivery",
        response_span="since you're within our returns window",
        reasoning="Agent asserted the return was in-window without checking the date.",
        confidence=0.91,
    )
