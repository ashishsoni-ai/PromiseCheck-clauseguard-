"""STEP 6 checkpoint tests - the audit row schema per DESIGN.md 5.1.

Four things this module is trying to pin down, in descending order of how much
damage their absence would do:

  1. That an error, an abstention and a judgment are mutually exclusive states.
     DESIGN.md 4.2 publishes the abstain rate; the judge was built so that only a
     twice-rejected span books an abstention, and everything else raises
     `JudgeError`. `TestAnErrorIsNotAnAbstention` is where that stops being a
     convention in one module and becomes a property of the stored data.
  2. That `verdict_class` is implied by the two stances rather than asserted. It is
     the field the headline number is read off.
  3. That the validated model and the table model have not drifted apart, which the
     two-model split makes possible and duplication makes likely.
  4. That the *premise* of the split is real - `TestTheRecordDoesNotValidate`
     demonstrates SQLModel skipping validation on a `table=True` model rather than
     taking the docstring's word for it. If that ever stops being true the split is
     merely harmless; if it were never true, every invariant here was theatre.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.audit.models import (
    AGENT_STANCES,
    JSON_COLUMNS,
    POLICY_STANCES,
    AuditRow,
    AuditRowRecord,
    VerdictClass,
    classify_verdict,
    new_row_id,
    new_run_id,
    utc_now_iso,
)

pytestmark = pytest.mark.unit

#: The five fields this project adds to DESIGN.md 5.1, each argued for in the
#: `harness/audit/models.py` docstring. Listed here so that adding a sixth is a
#: test failure and therefore a decision, not a drift.
FIELDS_BEYOND_THE_SPEC = frozenset(
    {
        "row_id",
        "judge_error",
        "agent_backend",
        "judge_temperature",
        "judge_completions",
    }
)

DESIGN_MD = Path(__file__).resolve().parents[2] / "docs" / "DESIGN.md"


def spec_row_keys() -> list[str]:
    """The keys of the JSON row in DESIGN.md 5.1, in the order the spec lists them.

    Read out of the document rather than transcribed. A transcribed copy of a
    specification is a second specification, and the two disagree eventually.

    Parsed by regex rather than `json.loads` because the block contains `...`
    elisions in several values and is therefore not valid JSON - which is fine, the
    keys are what is being checked.
    """
    text = DESIGN_MD.read_text(encoding="utf-8")
    match = re.search(r"### 5\.1 Row schema\s*\n+```json\n(.*?)\n```", text, re.S)
    assert match, "could not find the JSON row block under DESIGN.md 5.1"
    keys = re.findall(r'^\s*"([a-z_]+)":', match.group(1), re.M)
    assert keys, "found the 5.1 block but no keys in it"
    return keys


def model_field_names(model) -> list[str]:
    """Declared field names, in declaration order."""
    return list(model.model_fields)


# ==========================================================================
# The row against the specification it implements
# ==========================================================================
class TestTheRowMatchesTheSpecification:
    def test_every_spec_field_exists_on_the_row(self):
        missing = sorted(set(spec_row_keys()) - set(model_field_names(AuditRow)))
        assert missing == [], (
            f"DESIGN.md 5.1 lists {missing} but AuditRow does not carry them"
        )

    def test_the_only_extra_fields_are_the_five_that_were_argued_for(self):
        """The docstring in models.py claims 33 spec fields plus 5 additions.

        This is the test that keeps that claim true. A field added without being
        added to `FIELDS_BEYOND_THE_SPEC` fails here, which forces the question
        "does this belong in DESIGN.md?" to be answered rather than skipped.
        """
        extra = set(model_field_names(AuditRow)) - set(spec_row_keys())
        assert extra == FIELDS_BEYOND_THE_SPEC

    def test_the_counts_in_the_docstring_are_the_counts_in_the_files(self):
        assert len(spec_row_keys()) == 33
        assert len(model_field_names(AuditRow)) == 38

    def test_the_spec_lists_no_field_twice(self):
        keys = spec_row_keys()
        assert len(keys) == len(set(keys))

    def test_field_order_follows_the_spec(self):
        """AuditRow's docstring says the field order follows 5.1's JSON block so the
        two can be read side by side. Interleaving the five additions is allowed;
        reordering the spec's own fields relative to each other is not."""
        spec = spec_row_keys()
        ours = [f for f in model_field_names(AuditRow) if f in set(spec)]
        assert ours == spec


# ==========================================================================
# The two-model split
# ==========================================================================
class TestTheTwoModelsCannotDriftApart:
    """The cost of the split is a duplicated field list. This is the pin."""

    def test_same_field_names(self):
        row = set(model_field_names(AuditRow))
        record = set(model_field_names(AuditRowRecord))
        assert sorted(row - record) == [], "on AuditRow but not on the table"
        assert sorted(record - row) == [], "on the table but not on AuditRow"

    def test_same_field_order(self):
        """Not required for correctness - required for reviewability. Two 38-field
        classes in the same file are only checkable side by side if they are in the
        same order."""
        assert model_field_names(AuditRowRecord) == model_field_names(AuditRow)

    def test_the_json_columns_are_text_on_the_table_and_containers_on_the_row(self):
        for column in JSON_COLUMNS:
            assert AuditRowRecord.model_fields[column].annotation is str
            assert AuditRow.model_fields[column].annotation is not str

    def test_row_id_is_the_primary_key(self):
        """`supersedes_id` points at `row_id`, so it has to be the unique one."""
        assert AuditRowRecord.__table__.primary_key.columns.keys() == ["row_id"]

    def test_the_table_is_named_for_a_human_with_a_sql_browser(self):
        assert AuditRowRecord.__tablename__ == "audit_rows"


class TestTheRecordDoesNotValidate:
    """Demonstrates the premise of the split instead of asserting it.

    SQLModel skips pydantic validation on `table=True` models. That is the entire
    reason `AuditRow` exists separately, and it is a fact about a third-party
    library rather than about this code - so it is the kind of fact that should be
    checked by a test that fails loudly if the library changes.
    """

    def test_the_table_model_accepts_a_row_no_validator_would_allow(self):
        """The sharp version of the claim: a *type* violation gets through.

        `difficulty_tier` is annotated `int` on both classes. A plain pydantic model
        would reject the string, or coerce it and fail; the table model stores it
        as-is. Cross-field nonsense gets through too - `judge_error` together with
        `judge_abstained` is the one combination `AuditRow` refuses hardest - which
        is the concrete demonstration that AuditRow's validators are not inherited
        by the table class and cannot be relied on at the storage boundary.
        """
        record = AuditRowRecord(
            row_id="r1",
            run_id="run1",
            probe_id="p1",
            ts="not a timestamp at all",
            policy_doc="d",
            policy_version="v",
            clause_ids="[]",
            rule_id="r",
            rule_version="v",
            strategy="boundary",
            difficulty_tier="ninety-nine",  # not an int, and not rejected
            scenario_facts="{}",
            probe_turns="[]",
            expected_policy_stance="sideways",  # outside the Literal on AuditRow
            agent_id="a",
            agent_model="m",
            agent_commit_sha="s",
            agent_response="x",
            agent_latency_ms=-5,
            judge_model="j",
            git_sha="g",
            judge_error="boom",
            judge_abstained=True,
        )
        assert record.difficulty_tier == "ninety-nine"
        assert record.agent_latency_ms == -5
        assert record.expected_policy_stance == "sideways"
        assert record.judge_error == "boom" and record.judge_abstained is True

    def test_the_same_input_is_refused_by_the_validated_model(self):
        """The control for the test above. Without it, "the table model accepted
        this" would not establish that anything validates the input anywhere."""
        with pytest.raises(ValidationError):
            AuditRow(
                run_id="run1",
                probe_id="p1",
                ts="not a timestamp at all",
                policy_doc="d",
                policy_version="v",
                clause_ids=["c1"],
                rule_id="r",
                rule_version="v",
                strategy="boundary",
                difficulty_tier="ninety-nine",
                scenario_facts={"a": 1},
                probe_turns=["t"],
                expected_policy_stance="sideways",
                agent_id="a",
                agent_model="m",
                agent_commit_sha="s",
                agent_response="x",
                agent_latency_ms=-5,
                judge_model="j",
                git_sha="g",
                judge_error="boom",
                judge_abstained=True,
            )

    def test_but_reading_it_back_through_to_row_does_raise(self):
        """Which is what makes a hand-edited runs.db detectable."""
        record = AuditRowRecord(
            row_id="r1",
            run_id="run1",
            probe_id="p1",
            ts="not a timestamp at all",
            policy_doc="d",
            policy_version="v",
            clause_ids='["c1"]',
            rule_id="r",
            rule_version="v",
            strategy="boundary",
            difficulty_tier=99,
            scenario_facts='{"a": 1}',
            probe_turns='["t"]',
            expected_policy_stance="denies",
            agent_id="a",
            agent_model="m",
            agent_commit_sha="s",
            agent_response="x",
            agent_latency_ms=-5,
            judge_model="j",
            git_sha="g",
            judge_error="boom",
        )
        with pytest.raises(ValidationError):
            record.to_row()


# ==========================================================================
# The 2x3 matrix (DESIGN.md 2 step 9)
# ==========================================================================
class TestTheMatrixIsExhaustive:
    def test_every_policy_agent_pair_has_a_cell(self):
        """A missing cell would raise on a legitimate pair, and would do so on the
        rarest combination - the worst possible time to find out."""
        for policy in sorted(POLICY_STANCES):
            for agent in sorted(AGENT_STANCES):
                assert isinstance(classify_verdict(policy, agent), VerdictClass)

    def test_the_matrix_is_two_by_three(self):
        assert len(POLICY_STANCES) == 2
        assert len(AGENT_STANCES) == 3

    def test_every_verdict_class_is_reachable_exactly_once(self):
        """Six cells, six classes, no class used twice: the mapping is a bijection,
        so the matrix can be reconstructed from the column alone."""
        produced = [
            classify_verdict(p, a) for p in sorted(POLICY_STANCES) for a in sorted(AGENT_STANCES)
        ]
        assert len(produced) == 6
        assert sorted(set(produced), key=lambda v: v.value) == sorted(
            VerdictClass, key=lambda v: v.value
        )

    def test_the_cell_that_matters(self):
        assert classify_verdict("denies", "grants") is VerdictClass.OVER_PROMISE

    def test_its_mirror(self):
        assert classify_verdict("grants", "denies") is VerdictClass.UNDER_SERVE

    def test_evasive_is_not_collapsed_across_the_two_policy_stances(self):
        """§5.2 displays a single EVASIVE bucket. Storing one would throw away the
        difference between evading a question the policy grants and one it denies,
        which are different merchant-side costs - and collapsing at write time is
        not reversible."""
        assert classify_verdict("grants", "evasive") is not classify_verdict(
            "denies", "evasive"
        )

    def test_an_unknown_stance_is_refused_with_the_vocabulary_in_the_message(self):
        with pytest.raises(ValueError, match="no verdict cell"):
            classify_verdict("denies", "maybe")

    def test_swapped_arguments_do_not_silently_produce_a_verdict(self):
        """`classify_verdict(agent, policy)` with agent='evasive' has no cell, so the
        commonest way to misuse a two-string function fails instead of returning a
        plausible wrong answer."""
        with pytest.raises(ValueError):
            classify_verdict("evasive", "denies")

    def test_member_names_equal_values(self):
        """So nothing depends on whether a serialiser writes .name or .value."""
        for member in VerdictClass:
            assert member.name == member.value


class TestVerdictClassIsRederivedNotTrusted:
    def test_a_mismatched_verdict_class_is_rejected(self, make_audit_row):
        with pytest.raises(ValidationError, match="implies OVER_PROMISE"):
            make_audit_row(verdict_class=VerdictClass.CORRECT_GRANT)

    def test_each_cell_round_trips(self, make_audit_row):
        for policy in sorted(POLICY_STANCES):
            for agent in sorted(AGENT_STANCES):
                row = make_audit_row(
                    expected_policy_stance=policy,
                    agent_stance=agent,
                    verdict_class=classify_verdict(policy, agent),
                    entitlement_asserted="refund" if agent == "grants" else None,
                    quoted_span="returns must be initiated within 7 days of delivery"
                    if agent == "grants"
                    else None,
                    span_verified=agent == "grants",
                )
                assert row.matrix_cell == (policy, agent)

    def test_a_row_with_a_stance_must_carry_a_verdict(self, make_audit_row):
        with pytest.raises(ValidationError, match="no verdict_class"):
            make_audit_row(verdict_class=None)


# ==========================================================================
# THE LOAD-BEARING INVARIANT
# ==========================================================================
class TestAnErrorIsNotAnAbstention:
    """DESIGN.md 4.2's abstain rate may mean only one kind of thing: the judge was
    asked and what came back could not be believed.

    The design's own warning is that "a bad API key would look like judicial
    humility". Four unrelated live failures have tested the separation - a
    decommissioned model returning 404, a local CUDA crash, a Groq rate limit, and a
    Groq `tool_use_failed` - and each raised `JudgeError` rather than booking an
    abstention. Reviewing what it caught is what showed the fourth had been misfiled:
    a malformed tool call is the judge failing to answer, not the harness failing to
    run, so on 2026-08-24 it moved to the abstention side after retries. The other
    three did not move, and these tests are that separation expressed as a constraint
    on what can be stored.
    """

    def test_a_clean_judgment_row_is_valid(self, make_audit_row):
        row = make_audit_row()
        assert row.is_scorable
        assert row.is_over_promise
        assert row.judge_error is None
        assert row.judge_abstained is False

    def test_an_error_row_is_valid_and_not_scorable(self, make_audit_row):
        row = make_audit_row(judge_error="RateLimitError: 429")
        assert row.judge_error == "RateLimitError: 429"
        assert row.judge_abstained is False
        assert row.agent_stance is None
        assert row.verdict_class is None
        assert row.is_scorable is False
        assert row.is_over_promise is False
        assert row.matrix_cell is None

    def test_an_abstention_row_is_valid_and_not_scorable(self, make_audit_row):
        row = make_audit_row(judge_abstained=True)
        assert row.judge_abstained is True
        assert row.judge_error is None
        assert row.agent_stance is None
        assert row.is_scorable is False

    def test_a_row_cannot_be_both_an_error_and_an_abstention(self, make_audit_row):
        with pytest.raises(ValidationError, match="an error is not an abstention"):
            make_audit_row(judge_error="boom", judge_abstained=True)

    def test_an_error_row_cannot_carry_a_judgment(self, make_audit_row):
        with pytest.raises(ValidationError, match="produced no judgment to record"):
            make_audit_row(
                judge_error="boom",
                agent_stance="grants",
                entitlement_asserted="refund",
                verdict_class=VerdictClass.OVER_PROMISE,
            )

    def test_an_error_row_cannot_claim_a_verified_span(self, make_audit_row):
        with pytest.raises(ValidationError, match="never returned"):
            make_audit_row(judge_error="boom", span_verified=True)

    def test_an_abstention_cannot_carry_a_judgment(self, make_audit_row):
        """A row that is both would be counted and excluded at once."""
        with pytest.raises(ValidationError, match="an abstention is the absence"):
            make_audit_row(
                judge_abstained=True,
                agent_stance="denies",
                verdict_class=VerdictClass.CORRECT_DENIAL,
            )

    def test_an_abstention_cannot_claim_a_verified_span(self, make_audit_row):
        """An abstention is what happens when L2 rejected the span twice."""
        with pytest.raises(ValidationError, match="span_verified is True"):
            make_audit_row(judge_abstained=True, span_verified=True)

    def test_a_lost_result_cannot_masquerade_as_a_clean_one(self, make_audit_row):
        """No stance, and no error or abstention explaining why - the single most
        dangerous row shape, because it silently shrinks every denominator."""
        with pytest.raises(ValidationError, match="must carry agent_stance"):
            make_audit_row(
                agent_stance=None,
                entitlement_asserted=None,
                verdict_class=None,
                quoted_span=None,
                span_verified=None,
            )

    def test_a_completed_judgment_must_say_whether_the_span_was_checked(
        self, make_audit_row
    ):
        """C2 is the claim that every judgment was mechanically checked; None
        records that the outcome of the check is unknown."""
        with pytest.raises(ValidationError, match="commitment C2"):
            make_audit_row(span_verified=None)

    def test_a_denial_with_no_span_is_honest_rather_than_incomplete(
        self, make_audit_row
    ):
        """The asymmetry worth keeping: a `denies` judgment legitimately quotes
        nothing, and span_verified=False is the honest record of that. If this were
        rejected, the harness would have to fabricate a span to store a denial."""
        row = make_audit_row(
            agent_stance="denies",
            entitlement_asserted=None,
            verdict_class=VerdictClass.CORRECT_DENIAL,
            quoted_span=None,
            span_verified=False,
        )
        assert row.is_scorable
        assert row.span_verified is False


class TestSpanAndEntitlementCoherence:
    def test_a_quote_needs_a_clause_to_have_come_from(self, make_audit_row):
        with pytest.raises(ValidationError, match="no clause to verify the quote"):
            make_audit_row(cited_clause_id=None)

    def test_verified_without_a_span_is_refused(self, make_audit_row):
        with pytest.raises(ValidationError, match="no span"):
            make_audit_row(
                agent_stance="denies",
                entitlement_asserted=None,
                verdict_class=VerdictClass.CORRECT_DENIAL,
                quoted_span=None,
                span_verified=True,
            )

    def test_a_grant_must_name_what_it_granted(self, make_audit_row):
        with pytest.raises(ValidationError, match="must name the entitlement"):
            make_audit_row(entitlement_asserted=None)

    def test_an_evasive_row_may_still_carry_an_entitlement(self, make_audit_row):
        """An agent can discuss refunds at length while committing to nothing;
        nulling the field would throw away the topic signal. Same asymmetry as
        `Judgment`'s."""
        row = make_audit_row(
            agent_stance="evasive",
            entitlement_asserted="refund",
            verdict_class=VerdictClass.EVASIVE_ON_DENIAL,
        )
        assert row.entitlement_asserted == "refund"


class TestAgreementBelongsToAVote:
    def test_agreement_at_k_of_1_is_refused(self, make_audit_row):
        """Storing 1.0 for a single vote would read as unanimity - the same failure
        as scoring L0's designed escalations as wrong answers."""
        with pytest.raises(ValidationError, match="single vote has no agreement"):
            make_audit_row(judge_k=1, judge_agreement=1.0)

    def test_agreement_at_k_of_3_is_fine(self, make_audit_row):
        row = make_audit_row(judge_k=3, judge_agreement=2 / 3)
        assert row.judge_k == 3

    def test_k_of_3_without_agreement_is_allowed(self, make_audit_row):
        """k=3 with no agreement is a real L3 outcome, not just a stub artefact.

        `harness/judge/consistency.py` emits exactly this shape when three samples
        were drawn and none returned a stance: the samples are paid for, so `judge_k`
        is 3, but there was nothing to agree or disagree about, so agreement stays
        null rather than becoming a 0.0 that would read as measured disagreement.
        """
        assert make_audit_row(judge_k=3).judge_agreement is None

    def test_agreement_at_k_of_0_is_refused(self, make_audit_row):
        """No votes at all has even less agreement than one vote."""
        with pytest.raises(ValidationError, match="no votes has no agreement"):
            make_audit_row(judge_k=0, judge_agreement=1.0)

    def test_k_must_not_be_negative(self, make_audit_row):
        with pytest.raises(ValidationError):
            make_audit_row(judge_k=-1)


class TestZeroSamplesMeansNoModelRan:
    """`judge_k=0` is how an L0 termination and a failed call are recorded.

    DESIGN.md 5.1 is silent on this - its only example is `judge_k: 3` - so the
    field was relaxed from `ge=1` to `ge=0` deliberately. These tests pin what the
    relaxation is allowed to mean, because the alternative was leaving the default
    at 1 and having every L0 row claim a sample it never took.
    """

    def test_an_l0_row_is_writable_at_all(self, make_audit_row):
        row = make_audit_row(
            judge_k=0,
            judge_model="deterministic-prefilter-L0",
            agent_stance="denies",
            verdict_class=VerdictClass.CORRECT_DENIAL,
            cited_clause_id=None,
            quoted_span=None,
            response_span=None,
            span_verified=None,
            entitlement_asserted=None,
        )
        assert row.judge_k == 0
        assert row.is_scorable
        assert row.judge_confidence is None

    def test_confidence_at_k_of_0_is_refused(self, make_audit_row):
        """A lexicon has no calibrated confidence, and 4.2 reports this field."""
        with pytest.raises(ValidationError, match="judge_confidence is set"):
            make_audit_row(judge_k=0, judge_confidence=0.91)

    def test_completions_at_k_of_0_is_refused(self, make_audit_row):
        """'0 samples, 1 completion' reads as a completion that was thrown away,
        which is what an abstention is - and abstentions are k=1."""
        with pytest.raises(ValidationError, match="judge_completions is set"):
            make_audit_row(judge_k=0, judge_completions=1)

    def test_temperature_at_k_of_0_is_refused_without_an_error(self, make_audit_row):
        """An L0 row made no call, so a temperature on it is the default
        masquerading as provenance."""
        with pytest.raises(ValidationError, match="judge_temperature is set"):
            make_audit_row(judge_k=0, judge_temperature=0.0)

    def test_temperature_at_k_of_0_is_allowed_with_an_error(self, make_audit_row):
        """A failed call was still made, and at a temperature. Recording it is a
        fact about the request rather than a claim about a reply."""
        row = make_audit_row(
            judge_k=0,
            judge_error="JudgeError: RateLimitError from groq",
            judge_temperature=0.0,
        )
        assert row.judge_temperature == 0.0
        assert not row.is_scorable

    def test_the_two_zero_sample_shapes_are_distinguished_by_judge_error(
        self, make_audit_row
    ):
        """The invariant leans on L0 never erroring, so that is asserted here
        rather than left as a comment three files away."""
        l0 = make_audit_row(
            judge_k=0,
            judge_model="deterministic-prefilter-L0",
            agent_stance="denies",
            verdict_class=VerdictClass.CORRECT_DENIAL,
            entitlement_asserted=None,
            cited_clause_id=None,
            quoted_span=None,
            response_span=None,
            span_verified=None,
        )
        errored = make_audit_row(judge_k=0, judge_error="JudgeError: boom")
        assert l0.judge_error is None
        assert errored.judge_error is not None
        assert l0.judge_k == errored.judge_k == 0
        # Only the L0 row counts toward a headline metric: it produced a verdict.
        assert l0.is_scorable
        assert not errored.is_scorable


# ==========================================================================
# `ts`
# ==========================================================================
class TestTimestampShape:
    def test_the_factory_default_satisfies_the_validator(self):
        """`utc_now_iso` and `_ts_must_be_the_shape_the_spec_prints` are written in
        the same file and could still disagree."""
        value = utc_now_iso()
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value), value

    def test_the_default_is_used_when_ts_is_not_given(self, make_audit_row):
        row = AuditRow(**make_audit_row().model_dump(exclude={"ts"}))
        assert row.ts.endswith("Z")

    def test_the_spec_example_is_accepted_verbatim(self, make_audit_row):
        assert make_audit_row(ts="2026-08-28T11:04:22.118Z").ts.endswith("Z")

    def test_the_offset_form_is_refused_even_though_it_means_the_same_instant(
        self, make_audit_row
    ):
        """A timestamp column that mixes formats cannot be range-queried, and the
        lexicographic ordering this schema relies on quietly stops holding."""
        with pytest.raises(ValidationError, match="'Z' suffix"):
            make_audit_row(ts="2026-08-28T11:04:22.118+00:00")

    def test_a_naive_timestamp_is_refused(self, make_audit_row):
        with pytest.raises(ValidationError, match="'Z' suffix"):
            make_audit_row(ts="2026-08-28T11:04:22.118")

    def test_a_non_timestamp_ending_in_z_is_refused(self, make_audit_row):
        with pytest.raises(ValidationError, match="not an ISO-8601 timestamp"):
            make_audit_row(ts="last Tuesday-ishZ")

    def test_lexicographic_order_is_chronological_order(self):
        """The property the ISO-string column is chosen for. Without it, ORDER BY ts
        in a panelist's SQL browser silently returns the wrong order."""
        base = datetime(2026, 8, 28, 11, 4, 22, 118000, tzinfo=timezone.utc)
        stamps = []
        for delta in (timedelta(0), timedelta(milliseconds=1), timedelta(days=400)):
            moment = base + delta
            stamps.append(
                moment.strftime("%Y-%m-%dT%H:%M:%S.")
                + f"{moment.microsecond // 1000:03d}Z"
            )
        assert stamps == sorted(stamps)


# ==========================================================================
# Identifiers
# ==========================================================================
class TestIdentifiers:
    def test_row_ids_are_unique(self):
        assert len({new_row_id() for _ in range(1000)}) == 1000

    def test_row_ids_increase_within_a_process(self):
        """uuid6 2025.0.1's `uuid7` bumps its millisecond field when the clock has
        not advanced, so ids are strictly increasing in one process - which is what
        lets `AuditStore.rows()` order by row_id instead of carrying a sequence
        column. Cross-process ordering is NOT claimed; see `new_row_id`'s docstring.
        """
        ids = [new_row_id() for _ in range(200)]
        assert ids == sorted(ids)

    def test_run_ids_are_uuid7_shaped(self):
        value = new_run_id()
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            value,
        ), value

    def test_a_row_gets_an_id_without_being_given_one(self, make_audit_row):
        assert make_audit_row().row_id


# ==========================================================================
# Persistence round-trip
# ==========================================================================
class TestRoundTripThroughTheTable:
    def _assert_round_trips(self, row: AuditRow) -> None:
        restored = AuditRowRecord.from_row(row).to_row()
        assert restored.model_dump() == row.model_dump()

    def test_a_judgment_row_round_trips(self, make_audit_row):
        self._assert_round_trips(make_audit_row())

    def test_an_abstention_row_round_trips(self, make_audit_row):
        self._assert_round_trips(make_audit_row(judge_abstained=True))

    def test_an_error_row_round_trips(self, make_audit_row):
        self._assert_round_trips(make_audit_row(judge_error="CUDA error: out of memory"))

    def test_a_multi_turn_row_round_trips(self, make_audit_row):
        self._assert_round_trips(
            make_audit_row(
                strategy="multi_turn_drift",
                probe_turns=["turn one", "turn two", "turn three"],
            )
        )

    def test_the_json_columns_are_stored_as_text(self, make_audit_row):
        record = AuditRowRecord.from_row(make_audit_row())
        for column in JSON_COLUMNS:
            stored = getattr(record, column)
            assert isinstance(stored, str)
            json.loads(stored)  # and it is real JSON

    def test_the_enum_is_stored_as_its_value_and_returns_as_the_enum(
        self, make_audit_row
    ):
        record = AuditRowRecord.from_row(make_audit_row())
        assert record.verdict_class == "OVER_PROMISE"
        assert record.to_row().verdict_class is VerdictClass.OVER_PROMISE

    def test_scenario_facts_survive_mixed_value_types(self, make_audit_row):
        """§5.1's example has an int and a string; probes also produce floats and
        bools, and JSON text is only lossless if nothing coerces on the way back."""
        facts = {
            "days_since_delivery": 8,
            "item_category": "footwear",
            "order_value": 1299.5,
            "item_unopened": True,
        }
        restored = AuditRowRecord.from_row(make_audit_row(scenario_facts=facts)).to_row()
        assert restored.scenario_facts == facts

    def test_non_ascii_survives(self, make_audit_row):
        """Real merchant policies are not ASCII, and `json.dumps` escapes non-ASCII
        by default - which round-trips, but is worth proving rather than assuming."""
        row = make_audit_row(
            probe_turns=["Mera order ₹1,299 ka tha — kya refund milega?"]
        )
        assert AuditRowRecord.from_row(row).to_row().probe_turns == row.probe_turns

    def test_the_json_text_is_key_sorted_so_two_equal_rows_store_identically(
        self, make_audit_row
    ):
        """Dict insertion order would otherwise leak into the stored bytes, and two
        rows that compare equal in Python would differ in the file - which makes
        diffing two runs report changes that are not changes."""
        a = AuditRowRecord.from_row(make_audit_row(scenario_facts={"b": 1, "a": 2}))
        b = AuditRowRecord.from_row(make_audit_row(scenario_facts={"a": 2, "b": 1}))
        assert a.scenario_facts == b.scenario_facts == '{"a": 2, "b": 1}'


class TestExtraFieldsForbidden:
    def test_an_unknown_field_is_refused(self, make_audit_row):
        """`extra="forbid"` matters more here than on most models: a typo'd field
        name on a row that is never read back would be a silently discarded
        measurement."""
        with pytest.raises(ValidationError):
            make_audit_row(judge_aggreement=1.0)
