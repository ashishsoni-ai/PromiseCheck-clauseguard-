"""`source_span` grounding - the check that makes provenance mechanical (task #47).

DESIGN.md says it three times (lines 77, 94, 194): every condition's `source_span`
must appear verbatim in the clause, by "the same mechanical check as C2".

WHAT THESE TESTS ARE AND ARE NOT ABOUT
--------------------------------------
It is tempting to file this under "C1 has a hole", and the imprecision matters
because it points at the wrong assertions. C1 is that ground-truth labels are
derived deterministically from rules in Python. An ungrounded span does not dent
that: `evaluate_rules()` reads `attribute`, `op` and `value` and never reads
`source_span`, so a rule with a fabricated span labels identically on every run.

What an ungrounded span breaks is the only mechanical link between a rule and the
policy text it claims to encode - so ground truth can be deterministically derived
from a rule no clause supports, and it looks exactly as authoritative as a correct
one. The failure mode is not "the numbers move", it is "the numbers are stable,
confident, and about a policy nobody wrote". So nothing here asserts about labels;
these tests are about whether a rule set is allowed to run at all.

THE THREE-WAY RULE IS THE SPEC'S, NOT A CHOICE
----------------------------------------------
DESIGN.md 94 contrasts flagging against *silence*, not against *stopping*:
extraction that cannot ground itself "gets `needs_human_review=True` rather than
being silently accepted". Hence:

    grounded                              -> runs
    ungrounded, needs_human_review=True   -> runs, and is reported every time
    ungrounded, needs_human_review=False  -> refuses

`TestTheThreeWayRule` pins each arm. Refusing the middle arm would make line 194's
retry-then-flag path unusable: an extractor would flag a rule exactly as instructed
and then find the harness would not run it.
"""

from datetime import datetime, timezone

import pytest

from harness.execution.grounding import (
    UngroundedSpanError,
    assert_spans_grounded,
    check_spans_grounded,
)
from harness.schemas.clause import PolicyDocument
from harness.schemas.rule import Condition, EntitlementRule

#: Split across two clauses so the multi-clause union semantics has something to
#: bite on. Clause 1 carries the window, clause 2 the hygiene exclusion.
WINDOW_TEXT = "Returns must be initiated within 30 days of delivery."
HYGIENE_TEXT = "Innerwear and swimwear are excluded from returns for hygiene reasons."

IN_CLAUSE_1 = "within 30 days of delivery"
IN_CLAUSE_2 = "excluded from returns"
IN_NEITHER = "refunds are available for ninety days after purchase"

CLAUSE_1 = "acme-refunds:001:11111111"
CLAUSE_2 = "acme-refunds:002:22222222"
#: Well-formed and absent from the document, which is the hand-edited-lockfile case.
CLAUSE_ABSENT = "acme-refunds:099:ffffffff"


@pytest.fixture
def policy(make_clause) -> PolicyDocument:
    """A two-clause document. Ordinals 1 and 2, so `PolicyDocument` is satisfied."""
    return PolicyDocument(
        doc_slug="acme-refunds",
        source="policies/acme-refunds.md",
        policy_version="sha256:" + "9f2c" * 16,
        fetched_at=datetime(2026, 8, 22, 11, 4, 22, tzinfo=timezone.utc),
        corpus_role="worked_example",
        clauses=[
            make_clause(text=WINDOW_TEXT, ordinal=1, content_hash="11111111"),
            make_clause(text=HYGIENE_TEXT, ordinal=2, content_hash="22222222"),
        ],
    )


def rule(
    *,
    span: str,
    clause_ids: list[str] | None = None,
    rule_id: str = "R-001-a",
    needs_human_review: bool = False,
    exceptions: list[EntitlementRule] | None = None,
    precedence: int = 10,
) -> EntitlementRule:
    """One rule with one condition carrying `span`. Everything else is scenery.

    A factory rather than fixtures-per-shape: these tests vary exactly two things,
    the span and the citation, and a reader should be able to see which one a given
    test moved without diffing two fixture bodies.
    """
    return EntitlementRule(
        rule_id=rule_id,
        clause_ids=clause_ids or [CLAUSE_1],
        entitlement="refund",
        polarity="grants",
        conditions=[
            Condition(
                attribute="days_since_delivery", op="<=", value=30, source_span=span
            )
        ],
        exceptions=exceptions or [],
        precedence=precedence,
        extraction_confidence=0.95,
        needs_human_review=needs_human_review,
    )


class TestTheThreeWayRule:
    """The spec's own outcomes for grounded, flagged and unflagged-ungrounded."""

    def test_a_grounded_span_passes(self, policy):
        report = check_spans_grounded([rule(span=IN_CLAUSE_1)], policy)

        assert report.ok
        assert report.checked == 1
        assert report.grounded == 1
        assert report.refused == ()
        assert report.flagged == ()

    def test_an_ungrounded_span_on_a_settled_rule_refuses(self, policy):
        report = check_spans_grounded([rule(span=IN_NEITHER)], policy)

        assert not report.ok
        assert report.grounded == 0
        assert [f.source_span for f in report.refused] == [IN_NEITHER]
        assert report.flagged == ()

    def test_an_ungrounded_span_on_a_flagged_rule_is_allowed_but_reported(self, policy):
        """DESIGN.md 194's retry-then-flag path has to remain runnable.

        `ok` is True here and that is the whole point: the report still carries the
        failure, so `execute_run` can surface it as a warning on every run rather
        than only in the run where somebody thought to look.
        """
        report = check_spans_grounded(
            [rule(span=IN_NEITHER, needs_human_review=True)], policy
        )

        assert report.ok
        assert report.refused == ()
        assert [f.source_span for f in report.flagged] == [IN_NEITHER]
        assert report.flagged[0].needs_human_review is True
        # Neither refused nor grounded - it is a third category, and a `grounded`
        # that counted it would overstate how much provenance was verified.
        assert report.grounded == 0
        assert report.checked == 1


class TestAssertRaisesOnlyForRefusals:
    def test_it_raises_and_names_the_rule_the_span_and_the_remedy(self, policy):
        with pytest.raises(UngroundedSpanError) as exc:
            assert_spans_grounded([rule(span=IN_NEITHER)], policy, source="rules.json")

        message = str(exc.value)
        assert "rules.json" in message
        assert IN_NEITHER in message
        assert "R-001-a" in message
        # The remedy has to be in the message. Someone hitting this at the start of
        # a run needs to know the alternative to editing the span is declaring the
        # rule unreviewed, not deleting the check.
        assert "needs_human_review=True" in message

    def test_it_returns_the_report_when_only_flagged_spans_are_present(self, policy):
        report = assert_spans_grounded(
            [rule(span=IN_NEITHER, needs_human_review=True)], policy
        )

        assert report.ok
        assert len(report.flagged) == 1


class TestTheFlagIsReadPerNodeNotPerTree:
    """An exception carries its own `needs_human_review`.

    Inheriting it either way would be wrong in a way that matters: a flagged parent
    would license ungrounded spans in every exception under it, and a flagged
    exception would license one in the parent. Both let a single flag launder a
    whole subtree's provenance.
    """

    def test_a_flagged_exception_does_not_license_its_parent(self, policy):
        tree = rule(
            span=IN_NEITHER,
            needs_human_review=False,
            exceptions=[
                rule(
                    span=IN_NEITHER,
                    rule_id="R-001-a-x1",
                    needs_human_review=True,
                    precedence=20,
                )
            ],
        )

        report = check_spans_grounded([tree], policy)

        assert report.checked == 2
        assert [f.rule_id for f in report.refused] == ["R-001-a"]
        assert [f.rule_id for f in report.flagged] == ["R-001-a-x1"]

    def test_a_flagged_parent_does_not_license_its_exception(self, policy):
        tree = rule(
            span=IN_NEITHER,
            needs_human_review=True,
            exceptions=[
                rule(
                    span=IN_NEITHER,
                    rule_id="R-001-a-x1",
                    needs_human_review=False,
                    precedence=20,
                )
            ],
        )

        report = check_spans_grounded([tree], policy)

        assert not report.ok
        assert [f.rule_id for f in report.refused] == ["R-001-a-x1"]
        assert [f.rule_id for f in report.flagged] == ["R-001-a"]


class TestMultiClauseCitationsUseUnionSemantics:
    """Grounded in AT LEAST ONE cited clause - weaker than the authoring check.

    `scripts/author_rules.py` checks a span against the one clause the human named.
    The harness cannot: `Condition` has no clause pointer, and the clause link lives
    on `EntitlementRule.clause_ids`, which is a list. So union is the strongest
    question available here, and it is *required* rather than merely tolerated - 5 of
    the 19 rule nodes in the committed lockfile deliberately draw spans from
    different clauses among the several they cite, and a "must be in the first cited
    clause" rule would wrongly refuse them.

    The resulting gap - a span grounded in the rule's clause 001 while the human
    meant 002 - is recorded in `docs/limitations.md`, and pinned here so that a later
    tightening has to change a test that says what was given up.
    """

    def test_a_span_in_the_second_cited_clause_grounds(self, policy):
        report = check_spans_grounded(
            [rule(span=IN_CLAUSE_2, clause_ids=[CLAUSE_1, CLAUSE_2])], policy
        )

        assert report.ok
        assert report.grounded == 1

    def test_a_span_in_a_clause_the_rule_does_not_cite_does_not_ground(self, policy):
        """The check is against cited clauses, not against the whole document.

        Without this the check would degrade into "does this text appear anywhere in
        the policy", which a paraphrase lifted from an unrelated clause would pass -
        and that is precisely the mis-citation C2 exists to catch elsewhere.
        """
        report = check_spans_grounded(
            [rule(span=IN_CLAUSE_2, clause_ids=[CLAUSE_1])], policy
        )

        assert not report.ok
        assert report.refused[0].source_span == IN_CLAUSE_2


class TestCitationsThePolicyDoesNotContain:
    """The hand-edited-lockfile case, and the reason unknown ids are not skipped.

    Skipping unknown clause ids would have been convenient - it would have left the
    existing lockfile fixtures untouched - and it would have handed anyone editing
    `rules.lock.json` a one-line way to defeat the whole check: cite a clause id that
    does not exist and no span is ever compared against anything.
    """

    def test_citing_only_an_absent_clause_cannot_ground(self, policy):
        report = check_spans_grounded(
            [rule(span=IN_CLAUSE_1, clause_ids=[CLAUSE_ABSENT])], policy
        )

        assert not report.ok
        assert report.refused[0].unknown_clause_ids == (CLAUSE_ABSENT,)

    def test_an_absent_clause_alongside_a_real_one_still_grounds(self, policy):
        """A dangling citation is not by itself a grounding failure.

        It is a different defect - `assert_clauses_resolve` and the staleness checks
        own it - and conflating the two would report "ungrounded span" for a rule
        whose span is demonstrably in the policy.
        """
        report = check_spans_grounded(
            [rule(span=IN_CLAUSE_1, clause_ids=[CLAUSE_ABSENT, CLAUSE_1])], policy
        )

        assert report.ok

    def test_the_message_names_the_absent_clause_separately(self, policy):
        with pytest.raises(UngroundedSpanError) as exc:
            assert_spans_grounded(
                [rule(span=IN_CLAUSE_1, clause_ids=[CLAUSE_ABSENT])], policy
            )

        assert "NOT IN POLICY" in str(exc.value)
        assert CLAUSE_ABSENT in str(exc.value)


class TestNormalisationMatchesC2AndNotMore:
    """`collapse_whitespace`, never `normalize()`.

    `harness/judge/span_verify.py` documents why at length: `normalize()` casefolds
    and turns punctuation into spaces, so under it a re-cased, re-punctuated
    paraphrase would pass the check whose entire job is to catch paraphrase.
    """

    def test_line_wrapping_in_the_span_is_folded(self, policy):
        wrapped = "within 30 days\n   of delivery"
        report = check_spans_grounded([rule(span=wrapped)], policy)

        assert report.ok, "a span re-wrapped by an editor should still ground"

    def test_a_recased_span_does_not_ground(self, policy):
        report = check_spans_grounded([rule(span="Within 30 Days Of Delivery")], policy)

        assert not report.ok

    def test_a_span_missing_the_clause_punctuation_does_not_ground(self, policy):
        """Punctuation is content here. The clause has no comma, so a span with one
        is not verbatim - `collapse_whitespace` folds layout and nothing else."""
        report = check_spans_grounded(
            [rule(span="within 30 days, of delivery")], policy
        )

        assert not report.ok

    def test_a_whitespace_only_span_does_not_ground(self, policy):
        """The empty-span hole in `contains_verbatim`, closed by arithmetic.

        `Condition.source_span` has `min_length=1`, so "" is unconstructable - but " "
        is not, and `collapse_whitespace(" ")` is "". `contains_verbatim` returns
        False for an empty needle rather than the vacuous True that `"" in text`
        would give, so a whitespace-only span refuses instead of grounding against
        every clause in the document.
        """
        report = check_spans_grounded([rule(span="   ")], policy)

        assert not report.ok
        assert report.refused[0].source_span == "   "


class TestCountingAndTraversal:
    def test_every_condition_on_every_node_is_checked(self, policy):
        tree = rule(
            span=IN_CLAUSE_1,
            exceptions=[
                rule(
                    span=IN_CLAUSE_1,
                    rule_id="R-001-a-x1",
                    precedence=20,
                    exceptions=[
                        rule(span=IN_CLAUSE_1, rule_id="R-001-a-x1-x1", precedence=30)
                    ],
                )
            ],
        )

        report = check_spans_grounded([tree], policy)

        assert report.checked == 3
        assert report.grounded == 3

    def test_a_rule_with_no_conditions_checks_nothing_and_passes(self, policy):
        """An unconditional rule is legitimate (`conditions` defaults to empty).

        There is no span to ground, so there is nothing to refuse. Worth pinning
        because the alternative reading - "a rule with no grounded span is
        ungrounded" - would refuse the broad-grant-narrowed-by-exceptions shape the
        schema explicitly blesses.
        """
        unconditional = EntitlementRule(
            rule_id="R-001-b",
            clause_ids=[CLAUSE_1],
            entitlement="refund",
            polarity="grants",
            conditions=[],
            precedence=5,
            extraction_confidence=0.9,
            needs_human_review=False,
        )

        report = check_spans_grounded([unconditional], policy)

        assert report.ok
        assert report.checked == 0
        assert report.grounded == 0

    def test_an_empty_rule_set_is_vacuously_grounded(self, policy):
        report = check_spans_grounded([], policy)

        assert report.ok
        assert report.checked == 0
