"""STEP 1 checkpoint tests - schemas per DESIGN.md 1.2, 3.1, 4.1.

Three things the step requires: instantiation, required fields, and the recursive
`exceptions` field on EntitlementRule. Covered below, plus the coherence
validators added on top of the bare spec - those are behaviour, so they get tests
too rather than being trusted because they look obvious.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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

pytestmark = pytest.mark.unit


# ==========================================================================
# Instantiation - every model named by DESIGN.md builds from valid input
# ==========================================================================
class TestInstantiation:
    def test_condition(self):
        c = Condition(
            attribute="days_since_delivery",
            op="<=",
            value=30,
            source_span="within 30 days of delivery",
        )
        assert c.attribute == "days_since_delivery"
        assert c.op == "<="
        assert c.value == 30

    def test_entitlement_rule_minimal(self, basic_grant_rule):
        assert basic_grant_rule.rule_id == "R-014-a"
        assert basic_grant_rule.polarity == "grants"
        assert basic_grant_rule.entitlement == "refund"
        # exceptions defaults to empty rather than being required
        assert basic_grant_rule.exceptions == []
        assert basic_grant_rule.depth() == 0

    def test_rule_with_no_conditions_is_legal(self):
        """An unconditional rule - typically a broad grant narrowed later by
        exceptions - must be constructible."""
        r = EntitlementRule(
            rule_id="R-001",
            clause_ids=["acme-refunds:001:11111111"],
            entitlement="refund",
            polarity="grants",
            precedence=0,
            extraction_confidence=0.9,
            needs_human_review=False,
        )
        assert r.conditions == []

    def test_probe_scenario(self, sample_scenario):
        assert sample_scenario.facts["days_since_delivery"] == 31
        assert sample_scenario.strategy is ProbeStrategy.BOUNDARY
        assert sample_scenario.difficulty_tier == 2

    def test_probe(self, sample_probe):
        assert sample_probe.probe_id == "P-acme-014-boundary-003"
        assert sample_probe.expected_policy_stance == "denies"
        assert sample_probe.is_multi_turn is False

    def test_judgment(self, sample_judgment):
        assert sample_judgment.agent_stance == "grants"
        assert sample_judgment.quoted_span == "within 30 days of delivery"
        assert sample_judgment.is_grant is True

    def test_clause(self, sample_clause):
        assert sample_clause.clause_id == "acme-refunds:014:a3f91c22"
        assert sample_clause.ordinal == 14
        assert sample_clause.token_estimate is None

    def test_policy_document(self, sample_policy_document):
        assert len(sample_policy_document.clauses) == 2
        assert sample_policy_document.corpus_role == "worked_example"
        assert sample_policy_document.is_holdout is False

    def test_policy_document_by_id(self, sample_policy_document):
        found = sample_policy_document.by_id("acme-refunds:002:22222222")
        assert found is not None and found.ordinal == 2
        assert sample_policy_document.by_id("nope:001:00000000") is None


# ==========================================================================
# Required fields
# ==========================================================================
class TestRequiredFields:
    def test_condition_requires_source_span(self):
        """Ungrounded conditions are the failure mode DESIGN.md 1.2 exists to
        prevent, so source_span has no default."""
        with pytest.raises(ValidationError) as exc:
            Condition(attribute="days_since_delivery", op="<=", value=30)
        assert "source_span" in str(exc.value)

    def test_condition_requires_attribute_op_value(self):
        for missing, kwargs in [
            ("attribute", {"op": "<=", "value": 30, "source_span": "x"}),
            ("op", {"attribute": "a", "value": 30, "source_span": "x"}),
            ("value", {"attribute": "a", "op": "<=", "source_span": "x"}),
        ]:
            with pytest.raises(ValidationError) as exc:
                Condition(**kwargs)
            assert missing in str(exc.value)

    def test_rule_requires_precedence(self):
        """precedence decides contradictions, so it is stated rather than
        inherited from a default."""
        with pytest.raises(ValidationError) as exc:
            EntitlementRule(
                rule_id="R-1",
                clause_ids=["acme-refunds:001:11111111"],
                entitlement="refund",
                polarity="grants",
                extraction_confidence=0.9,
                needs_human_review=False,
            )
        assert "precedence" in str(exc.value)

    def test_rule_requires_needs_human_review(self):
        """Defaulting this to False would fail open: an unreviewed rule would
        present as reviewed."""
        with pytest.raises(ValidationError) as exc:
            EntitlementRule(
                rule_id="R-1",
                clause_ids=["acme-refunds:001:11111111"],
                entitlement="refund",
                polarity="grants",
                precedence=0,
                extraction_confidence=0.9,
            )
        assert "needs_human_review" in str(exc.value)

    def test_rule_requires_at_least_one_clause_id(self):
        with pytest.raises(ValidationError):
            EntitlementRule(
                rule_id="R-1",
                clause_ids=[],
                entitlement="refund",
                polarity="grants",
                precedence=0,
                extraction_confidence=0.9,
                needs_human_review=False,
            )

    def test_rule_rejects_unknown_entitlement(self):
        with pytest.raises(ValidationError):
            EntitlementRule(
                rule_id="R-1",
                clause_ids=["acme-refunds:001:11111111"],
                entitlement="free_shipping_forever",
                polarity="grants",
                precedence=0,
                extraction_confidence=0.9,
                needs_human_review=False,
            )

    def test_rule_confidence_is_bounded(self):
        for bad in (-0.1, 1.1):
            with pytest.raises(ValidationError):
                EntitlementRule(
                    rule_id="R-1",
                    clause_ids=["acme-refunds:001:11111111"],
                    entitlement="refund",
                    polarity="grants",
                    precedence=0,
                    extraction_confidence=bad,
                    needs_human_review=False,
                )

    def test_probe_requires_expected_policy_stance(self, sample_scenario):
        """The C1 field. A probe without a derived label is not a probe."""
        with pytest.raises(ValidationError) as exc:
            Probe(
                probe_id="P-1",
                scenario=sample_scenario,
                turns=["hello"],
                clause_ids=["acme-refunds:014:a3f91c22"],
            )
        assert "expected_policy_stance" in str(exc.value)

    def test_probe_rejects_evasive_as_a_policy_stance(self, sample_scenario):
        """Policy stance is two-valued. Only an AGENT can be evasive; a policy
        either grants or denies (DESIGN.md 2 step 9)."""
        with pytest.raises(ValidationError):
            Probe(
                probe_id="P-1",
                scenario=sample_scenario,
                turns=["hello"],
                expected_policy_stance="evasive",
                clause_ids=["acme-refunds:014:a3f91c22"],
            )

    def test_scenario_rejects_out_of_range_difficulty_tier(self):
        with pytest.raises(ValidationError):
            ProbeScenario(
                facts={"a": 1},
                target_rule_id="R-1",
                strategy=ProbeStrategy.BOUNDARY,
                difficulty_tier=4,
            )

    def test_judgment_requires_reasoning_and_confidence(self):
        with pytest.raises(ValidationError) as exc:
            Judgment(agent_stance="denies")
        msg = str(exc.value)
        assert "reasoning" in msg and "confidence" in msg


# ==========================================================================
# The recursive `exceptions` field - exceptions to exceptions
# ==========================================================================
class TestRecursiveExceptions:
    def test_depth_two_nesting_builds(self, depth2_rule):
        assert depth2_rule.depth() == 2

    def test_nested_exceptions_are_models_not_dicts(self, depth2_rule):
        x1 = depth2_rule.exceptions[0]
        assert isinstance(x1, EntitlementRule)
        assert isinstance(x1.exceptions[0], EntitlementRule)
        assert isinstance(x1.exceptions[0].conditions[0], Condition)

    def test_walk_yields_whole_tree_depth_first(self, depth2_rule):
        ids = [r.rule_id for r in depth2_rule.walk()]
        assert ids == ["R-014-a", "R-014-a-x1", "R-014-a-x1-x1"]

    def test_polarity_alternates_down_the_chain(self, depth2_rule):
        """grants -> denies -> grants. This is the shape DESIGN.md 3.2 strategy 3
        probes, because chunk retrieval flattens it."""
        chain = [r.polarity for r in depth2_rule.walk()]
        assert chain == ["grants", "denies", "grants"]

    def test_builds_from_raw_nested_dicts(self):
        """The extractor emits JSON, so nesting must hydrate from plain dicts."""
        rule = EntitlementRule.model_validate(
            {
                "rule_id": "R-1",
                "clause_ids": ["acme-refunds:001:11111111"],
                "entitlement": "refund",
                "polarity": "grants",
                "conditions": [
                    {
                        "attribute": "days_since_delivery",
                        "op": "<=",
                        "value": 30,
                        "source_span": "within 30 days",
                    }
                ],
                "exceptions": [
                    {
                        "rule_id": "R-1-x1",
                        "clause_ids": ["acme-refunds:001:11111111"],
                        "entitlement": "refund",
                        "polarity": "denies",
                        "conditions": [],
                        "exceptions": [
                            {
                                "rule_id": "R-1-x1-x1",
                                "clause_ids": ["acme-refunds:001:11111111"],
                                "entitlement": "refund",
                                "polarity": "grants",
                                "conditions": [],
                                "exceptions": [],
                                "precedence": 3,
                                "extraction_confidence": 0.5,
                                "needs_human_review": True,
                            }
                        ],
                        "precedence": 2,
                        "extraction_confidence": 0.7,
                        "needs_human_review": False,
                    }
                ],
                "precedence": 1,
                "extraction_confidence": 0.9,
                "needs_human_review": False,
            }
        )
        assert rule.depth() == 2
        assert rule.exceptions[0].exceptions[0].rule_id == "R-1-x1-x1"

    def test_arbitrary_depth_is_supported(self):
        """Nothing in the schema caps nesting at 2; depth 2 is what the probe
        strategy targets, not a structural limit. Built inside-out with real
        construction so every level is validated, not copied."""
        rule = None
        for level in range(4, -1, -1):
            rule = EntitlementRule(
                rule_id=f"R-d{level}",
                clause_ids=["acme-refunds:001:11111111"],
                entitlement="refund",
                polarity="grants" if level % 2 == 0 else "denies",
                precedence=level,
                extraction_confidence=0.9,
                needs_human_review=False,
                exceptions=[rule] if rule is not None else [],
            )
        assert rule.depth() == 4
        assert [r.rule_id for r in rule.walk()] == [
            "R-d0",
            "R-d1",
            "R-d2",
            "R-d3",
            "R-d4",
        ]

    def test_duplicate_rule_id_within_a_tree_is_rejected(self, basic_grant_rule):
        """Duplicate ids make an audit row ambiguous about which rule produced
        the label. basic_grant_rule is R-014-a, so reusing that id on the parent
        collides with its own child."""
        with pytest.raises(ValidationError) as exc:
            EntitlementRule(
                rule_id="R-014-a",
                clause_ids=["acme-refunds:014:a3f91c22"],
                entitlement="refund",
                polarity="denies",
                precedence=20,
                extraction_confidence=0.9,
                needs_human_review=False,
                exceptions=[basic_grant_rule],
            )
        assert "duplicate rule_id" in str(exc.value)

    def test_duplicate_id_across_sibling_branches_is_rejected(self, basic_grant_rule):
        """Collisions between siblings are caught too, not just parent/child."""
        sibling = basic_grant_rule.model_copy(update={"rule_id": "R-dup"})
        with pytest.raises(ValidationError) as exc:
            EntitlementRule(
                rule_id="R-root",
                clause_ids=["acme-refunds:014:a3f91c22"],
                entitlement="refund",
                polarity="denies",
                precedence=20,
                extraction_confidence=0.9,
                needs_human_review=False,
                exceptions=[sibling, sibling.model_copy()],
            )
        assert "duplicate rule_id" in str(exc.value)

    def test_recursive_tree_survives_json_round_trip(self, depth2_rule):
        """rules.lock.json is committed to the repo, so nesting must survive
        serialisation exactly."""
        restored = EntitlementRule.model_validate_json(depth2_rule.model_dump_json())
        assert restored == depth2_rule
        assert restored.depth() == 2


# ==========================================================================
# Condition op/value coherence - the guarantee evaluate_rules() relies on
# ==========================================================================
class TestConditionOpValueCoherence:
    def test_numeric_op_coerces_integer_string(self):
        c = Condition(
            attribute="days_since_delivery", op="<=", value="31", source_span="x"
        )
        assert c.value == 31 and isinstance(c.value, int)

    def test_numeric_op_coerces_float_string(self):
        c = Condition(attribute="fee_pct", op=">", value="31.5", source_span="x")
        assert c.value == 31.5 and isinstance(c.value, float)

    def test_numeric_op_coerces_signed_string(self):
        c = Condition(attribute="delta", op=">=", value="-5", source_span="x")
        assert c.value == -5

    def test_numeric_op_rejects_list(self):
        with pytest.raises(ValidationError) as exc:
            Condition(
                attribute="days_since_delivery",
                op="<=",
                value=["footwear"],
                source_span="x",
            )
        assert "numeric" in str(exc.value)

    def test_numeric_op_rejects_unparseable_string(self):
        with pytest.raises(ValidationError) as exc:
            Condition(
                attribute="days_since_delivery", op="<", value="soon", source_span="x"
            )
        assert "numeric" in str(exc.value)

    def test_membership_op_wraps_bare_string(self):
        c = Condition(
            attribute="item_category", op="in", value="innerwear", source_span="x"
        )
        assert c.value == ["innerwear"]

    def test_membership_op_keeps_list(self):
        c = Condition(
            attribute="item_category",
            op="not_in",
            value=["innerwear", "swimwear"],
            source_span="x",
        )
        assert c.value == ["innerwear", "swimwear"]

    def test_membership_op_rejects_empty_list(self):
        with pytest.raises(ValidationError) as exc:
            Condition(attribute="item_category", op="in", value=[], source_span="x")
        assert "non-empty" in str(exc.value)

    def test_equality_op_rejects_list(self):
        with pytest.raises(ValidationError) as exc:
            Condition(
                attribute="channel", op="==", value=["app", "web"], source_span="x"
            )
        assert "single value" in str(exc.value)

    def test_unknown_op_is_rejected(self):
        with pytest.raises(ValidationError):
            Condition(
                attribute="days_since_delivery",
                op="between",
                value=30,
                source_span="x",
            )


# ==========================================================================
# Clause ID integrity - the gate's change-detection primitive
# ==========================================================================
class TestClauseIdIntegrity:
    def test_consistent_id_accepted(self, sample_clause):
        assert sample_clause.clause_id == "acme-refunds:014:a3f91c22"

    def test_id_inconsistent_with_ordinal_is_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Clause(
                clause_id="acme-refunds:014:a3f91c22",
                doc_slug="acme-refunds",
                ordinal=13,
                text="Some clause.",
                content_hash="a3f91c22",
            )
        assert "inconsistent" in str(exc.value)

    def test_id_inconsistent_with_hash_is_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Clause(
                clause_id="acme-refunds:014:deadbeef",
                doc_slug="acme-refunds",
                ordinal=14,
                text="Some clause.",
                content_hash="a3f91c22",
            )
        assert "inconsistent" in str(exc.value)

    @pytest.mark.parametrize("bad_hash", ["A3F91C22", "a3f91c2", "a3f91c222", "zzzzzzzz"])
    def test_malformed_content_hash_is_rejected(self, bad_hash):
        with pytest.raises(ValidationError):
            Clause(
                clause_id=f"acme-refunds:014:{bad_hash}",
                doc_slug="acme-refunds",
                ordinal=14,
                text="Some clause.",
                content_hash=bad_hash,
            )

    def test_ordinal_is_one_based(self):
        with pytest.raises(ValidationError):
            Clause(
                clause_id="acme-refunds:000:a3f91c22",
                doc_slug="acme-refunds",
                ordinal=0,
                text="Some clause.",
                content_hash="a3f91c22",
            )

    def test_empty_clause_text_is_rejected(self):
        with pytest.raises(ValidationError):
            Clause(
                clause_id="acme-refunds:001:a3f91c22",
                doc_slug="acme-refunds",
                ordinal=1,
                text="",
                content_hash="a3f91c22",
            )

    def test_document_rejects_non_contiguous_ordinals(self, make_clause):
        with pytest.raises(ValidationError) as exc:
            PolicyDocument(
                doc_slug="acme-refunds",
                source="policies/acme-refunds.md",
                policy_version="sha256:" + "9f2c" * 16,
                fetched_at="2026-08-22T11:04:22Z",
                corpus_role="worked_example",
                clauses=[
                    make_clause(text="One.", ordinal=1, content_hash="11111111"),
                    make_clause(text="Three.", ordinal=3, content_hash="33333333"),
                ],
            )
        assert "contiguous" in str(exc.value)

    def test_document_rejects_clause_from_another_document(self, make_clause):
        with pytest.raises(ValidationError) as exc:
            PolicyDocument(
                doc_slug="acme-refunds",
                source="policies/acme-refunds.md",
                policy_version="sha256:" + "9f2c" * 16,
                fetched_at="2026-08-22T11:04:22Z",
                corpus_role="worked_example",
                clauses=[
                    make_clause(
                        text="Foreign.",
                        ordinal=1,
                        content_hash="11111111",
                        doc_slug="other-policy",
                    )
                ],
            )
        assert "doc_slug" in str(exc.value)

    def test_document_rejects_malformed_policy_version(self, make_clause):
        with pytest.raises(ValidationError):
            PolicyDocument(
                doc_slug="acme-refunds",
                source="policies/acme-refunds.md",
                policy_version="9f2c",
                fetched_at="2026-08-22T11:04:22Z",
                corpus_role="worked_example",
                clauses=[],
            )


# ==========================================================================
# Probe coherence
# ==========================================================================
class TestProbeCoherence:
    def test_multi_turn_drift_with_one_turn_is_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Probe(
                probe_id="P-drift-1",
                scenario=ProbeScenario(
                    facts={"days_since_delivery": 31},
                    target_rule_id="R-014-a",
                    strategy=ProbeStrategy.MULTI_TURN_DRIFT,
                    difficulty_tier=3,
                ),
                turns=["only one turn"],
                expected_policy_stance="denies",
                clause_ids=["acme-refunds:014:a3f91c22"],
            )
        assert "at least 2" in str(exc.value)

    def test_multi_turn_drift_with_two_turns_is_accepted(self):
        p = Probe(
            probe_id="P-drift-2",
            scenario=ProbeScenario(
                facts={"days_since_delivery": 31},
                target_rule_id="R-014-a",
                strategy=ProbeStrategy.MULTI_TURN_DRIFT,
                difficulty_tier=3,
            ),
            turns=["Delivered on the 3rd, all good?", "Actually it was the 3rd of last month."],
            expected_policy_stance="denies",
            clause_ids=["acme-refunds:014:a3f91c22"],
        )
        assert p.is_multi_turn is True

    def test_more_than_three_turns_is_rejected(self, sample_scenario):
        """Three turns max is a stated scope limit (DESIGN.md 8)."""
        with pytest.raises(ValidationError):
            Probe(
                probe_id="P-1",
                scenario=sample_scenario,
                turns=["a", "b", "c", "d"],
                expected_policy_stance="denies",
                clause_ids=["acme-refunds:014:a3f91c22"],
            )

    def test_empty_turns_is_rejected(self, sample_scenario):
        with pytest.raises(ValidationError):
            Probe(
                probe_id="P-1",
                scenario=sample_scenario,
                turns=[],
                expected_policy_stance="denies",
                clause_ids=["acme-refunds:014:a3f91c22"],
            )

    def test_empty_facts_is_rejected(self):
        """A scenario with no facts cannot be deterministically labelled, which
        would breach C1."""
        with pytest.raises(ValidationError):
            ProbeScenario(
                facts={},
                target_rule_id="R-1",
                strategy=ProbeStrategy.BOUNDARY,
                difficulty_tier=1,
            )

    def test_empty_clause_ids_is_rejected(self, sample_scenario):
        with pytest.raises(ValidationError):
            Probe(
                probe_id="P-1",
                scenario=sample_scenario,
                turns=["hello"],
                expected_policy_stance="denies",
                clause_ids=[],
            )

    def test_all_eight_strategies_exist(self):
        """The taxonomy is fixed at eight (DESIGN.md 3.2) and each member must
        have a module under harness/probe_gen/strategies/."""
        assert {s.value for s in ProbeStrategy} == {
            "boundary",
            "condition_stripping",
            "exception_depth",
            "category_smuggling",
            "false_premise",
            "authority_pressure",
            "multi_turn_drift",
            "cross_clause",
        }

    def test_strategy_serialises_as_its_string_value(self, sample_probe):
        """The enum's *value* is the wire format written to probes.lock.json and
        the audit row's strategy column, not its Python member name."""
        dumped = sample_probe.model_dump(mode="json")
        assert dumped["scenario"]["strategy"] == "boundary"

    def test_probe_survives_json_round_trip(self, sample_probe):
        """probes.lock.json is version-controlled like a dependency lockfile."""
        restored = Probe.model_validate_json(sample_probe.model_dump_json())
        assert restored == sample_probe


# ==========================================================================
# Judgment coherence - commitment C2's preconditions
# ==========================================================================
class TestJudgmentCoherence:
    def test_quote_without_a_cited_clause_is_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Judgment(
                agent_stance="denies",
                quoted_span="within 30 days of delivery",
                reasoning="Cited nothing but quoted something.",
                confidence=0.5,
            )
        assert "cited_clause_id" in str(exc.value)

    def test_grant_without_named_entitlement_is_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Judgment(
                agent_stance="grants",
                reasoning="Agent said yes but named no entitlement.",
                confidence=0.8,
            )
        assert "must name the entitlement" in str(exc.value)

    def test_denial_without_named_entitlement_is_fine(self):
        j = Judgment(
            agent_stance="denies",
            reasoning="Agent refused, citing the exclusion.",
            confidence=0.8,
        )
        assert j.entitlement_asserted is None
        assert j.is_grant is False

    def test_evasive_carrying_an_entitlement_is_deliberately_allowed(self):
        """An agent can discuss refunds at length while committing to nothing;
        nulling the field would discard the topic signal."""
        j = Judgment(
            agent_stance="evasive",
            entitlement_asserted="refund",
            reasoning="Discussed refunds without committing either way.",
            confidence=0.4,
        )
        assert j.entitlement_asserted == "refund"

    def test_reasoning_is_capped_at_300_characters(self):
        with pytest.raises(ValidationError):
            Judgment(
                agent_stance="denies", reasoning="x" * 301, confidence=0.5
            )

    def test_reasoning_at_exactly_300_is_accepted(self):
        j = Judgment(agent_stance="denies", reasoning="x" * 300, confidence=0.5)
        assert len(j.reasoning) == 300

    def test_empty_reasoning_is_rejected(self):
        with pytest.raises(ValidationError):
            Judgment(agent_stance="denies", reasoning="", confidence=0.5)

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_confidence_is_bounded(self, bad):
        with pytest.raises(ValidationError):
            Judgment(agent_stance="denies", reasoning="ok", confidence=bad)

    def test_unclear_is_not_a_valid_agent_stance(self):
        """The value 'unclear' belongs to the L0 pre-classifier's wider output
        space, not to a finished Judgment (DESIGN.md 4.1)."""
        with pytest.raises(ValidationError):
            Judgment(agent_stance="unclear", reasoning="ok", confidence=0.5)

    def test_judgment_does_not_accept_self_certifying_fields(self):
        """span_verified and judge_abstained are written by Python after the
        fact, never asserted by the judge. extra='forbid' makes that structural."""
        for field in ("span_verified", "judge_abstained", "verdict_class"):
            with pytest.raises(ValidationError):
                Judgment.model_validate(
                    {
                        "agent_stance": "denies",
                        "reasoning": "ok",
                        "confidence": 0.5,
                        field: True,
                    }
                )


# ==========================================================================
# extra="forbid" across the board - matters because instructor retries on
# validation failure, so a hallucinated field becomes a retry, not a silent pass
# ==========================================================================
class TestExtraFieldsForbidden:
    def test_condition_forbids_extra(self):
        with pytest.raises(ValidationError):
            Condition.model_validate(
                {
                    "attribute": "a",
                    "op": "<=",
                    "value": 1,
                    "source_span": "x",
                    "hallucinated": "nope",
                }
            )

    def test_probe_forbids_extra(self, sample_probe):
        payload = sample_probe.model_dump()
        payload["severity"] = "high"
        with pytest.raises(ValidationError):
            Probe.model_validate(payload)

    def test_clause_forbids_extra(self, sample_clause):
        payload = sample_clause.model_dump()
        payload["section"] = "Returns"
        with pytest.raises(ValidationError):
            Clause.model_validate(payload)


# ==========================================================================
# Corpus provenance - DESIGN.md 7.1 forbids pooling real with synthetic
# ==========================================================================
class TestCorpusRole:
    def test_corpus_role_is_required(self, make_clause):
        """No default: 'real' would fail open by promoting authored fixtures
        into evidence, and any other default would misreport half the corpus."""
        with pytest.raises(ValidationError) as exc:
            PolicyDocument(
                doc_slug="acme-refunds",
                source="policies/acme-refunds.md",
                policy_version="sha256:" + "9f2c" * 16,
                fetched_at="2026-08-22T11:04:22Z",
                clauses=[],
            )
        assert "corpus_role" in str(exc.value)

    @pytest.mark.parametrize(
        "role,is_evidence",
        [
            ("real", True),
            ("synthetic_stress", False),
            ("worked_example", False),
        ],
    )
    def test_only_real_policies_count_as_evidence(self, role, is_evidence):
        doc = PolicyDocument(
            doc_slug="acme-refunds",
            source="policies/acme-refunds.md",
            policy_version="sha256:" + "9f2c" * 16,
            fetched_at="2026-08-22T11:04:22Z",
            corpus_role=role,
            clauses=[],
        )
        assert doc.counts_as_evidence is is_evidence

    def test_unknown_corpus_role_is_rejected(self):
        with pytest.raises(ValidationError):
            PolicyDocument(
                doc_slug="acme-refunds",
                source="policies/acme-refunds.md",
                policy_version="sha256:" + "9f2c" * 16,
                fetched_at="2026-08-22T11:04:22Z",
                corpus_role="probably_real",
                clauses=[],
            )

    def test_holdout_is_orthogonal_to_role(self):
        """A real policy can also be held out (DESIGN.md 7.3); the two flags
        answer different questions and must not be conflated."""
        doc = PolicyDocument(
            doc_slug="held-out-policy",
            source="https://example.com/returns",
            policy_version="sha256:" + "abcd" * 16,
            fetched_at="2026-08-22T11:04:22Z",
            corpus_role="real",
            is_holdout=True,
            clauses=[],
        )
        assert doc.counts_as_evidence is True and doc.is_holdout is True
