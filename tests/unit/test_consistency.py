"""L3 asymmetric self-consistency - the vote, tested without a provider.

NO NETWORK, and no prompt either. `apply_consistency` takes a `sample` callable
rather than a client, so everything DESIGN.md 4.1 says about k=3 can be asserted
against hand-built `JudgeOutcome`s: which rows spend the budget, how votes are
counted, and what happens when a sample comes back with nothing. That seam is the
reason this file needs no fake client, no clause fixtures and no JSON.

Two classes carry most of the weight.

`TestWhichRowsSpendTheK3Budget` pins the asymmetry. DESIGN.md 4.1 spends k=3 on
"judgments landing on the over-promise cell and the entire gold set" and k=1 on
everything else, and that asymmetry is what keeps a run inside its token budget. A
well-meaning change to "escalate anything the judge was unsure about" would triple
the cost of a run and, worse, would quietly change what the abstain rate measures.

`TestAFailedSampleIsNeverAnAbstention` pins the one case that raises. The abstain
rate is published (DESIGN.md 4.2) and it must mean only "the judge was asked and
what came back could not be believed". A rate-limit storm during the vote is not
judicial humility, and a row that books it as one would move a headline number in
the flattering direction - which is the same failure the L2 layer already guards.

What is deliberately *not* tested here is that L3 improves anything. It cannot be:
three samples from a nondeterministic judge are not a ground truth, and
`docs/limitations.md` says which number must not be quoted as accuracy. These tests
pin mechanism, not quality.
"""

from __future__ import annotations

import pytest

from harness.judge.consistency import (
    L3_K,
    L3_MIN_VOTES,
    L3_TEMPERATURE,
    ConsistencyResult,
    apply_consistency,
    needs_consistency,
)
from harness.judge.judge import JudgeError, JudgeOutcome
from harness.judge.prefilter import PrefilterResult
from harness.schemas.judgment import Judgment

MODEL = "fake/judge-1"

#: L0's verdict on anything that reaches L1 at all: the pre-filter declined to
#: settle it. Built by hand rather than by calling `classify`, so that what the vote
#: does with a stance cannot depend on the lexicon's opinion of a fixture string.
ESCALATED = PrefilterResult(stance="unclear", rationale="no terminal cue")

CLAUSE_ID = "acme-refunds:014:a3f91c22"
CLAUSE_SPAN = "within 30 days of delivery"
REPLY_SPAN = "I have processed your refund"


def a_judgment(
    stance: str, *, confidence: float = 0.9, span: str = REPLY_SPAN
) -> Judgment:
    """A judgment that would survive L2, which every vote is contracted to be."""
    return Judgment(
        agent_stance=stance,
        entitlement_asserted="refund" if stance == "grants" else None,
        cited_clause_id=CLAUSE_ID,
        quoted_span=CLAUSE_SPAN,
        response_span=span,
        reasoning=f"The reply {stance} the entitlement the clause governs.",
        confidence=confidence,
    )


def a_vote(
    stance: str,
    *,
    confidence: float = 0.9,
    span: str = REPLY_SPAN,
    completions: int = 1,
) -> JudgeOutcome:
    """A finished L1+L2 outcome - the shape `sample` promises to return."""
    return JudgeOutcome(
        source="llm",
        prefilter=ESCALATED,
        agent_stance=stance,  # type: ignore[arg-type]
        judgment=a_judgment(stance, confidence=confidence, span=span),
        span_verified=True,
        judge_model=MODEL,
        judge_k=1,
        judge_completions=completions,
    )


def an_abstention(*, completions: int = 2) -> JudgeOutcome:
    """A sample L2 refused twice: no stance, so by the ruling it does not vote."""
    return JudgeOutcome(
        source="llm",
        prefilter=ESCALATED,
        agent_stance=None,
        abstained=True,
        violations=("quoted span not found in the cited clause",),
        judge_model=MODEL,
        judge_k=1,
        judge_completions=completions,
    )


def an_l0_row(stance: str) -> JudgeOutcome:
    """A row the pre-filter settled. Only `source` and `stance` are load-bearing.

    `denies` rather than `grants` because L0 has only two terminal stances, `denies`
    and `evasive` - a `grants` L0 row does not exist, which is why the cell arm
    excludes the pre-filter structurally and the gold arm has to do it by hand.
    """
    return JudgeOutcome(
        source="prefilter",
        prefilter=PrefilterResult(stance=stance),  # type: ignore[arg-type]
        agent_stance=stance,  # type: ignore[arg-type]
    )


class Sampler:
    """Replays queued outcomes as L3's three draws, recording the temperatures.

    A queued `JudgeError` is raised rather than returned, which is how a transport
    failure reaches `apply_consistency`. Running dry is an `AssertionError` and not a
    stop condition: a test that queued too few samples has stopped testing the thing
    it named, and should say so rather than pass.
    """

    def __init__(self, *outcomes: JudgeOutcome | Exception) -> None:
        self._queue = list(outcomes)
        self.temperatures: list[float] = []

    def __call__(self, temperature: float) -> JudgeOutcome:
        self.temperatures.append(temperature)
        if not self._queue:
            raise AssertionError(
                f"L3 drew {len(self.temperatures)} samples but the test queued "
                f"only {len(self.temperatures) - 1}"
            )
        nxt = self._queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    @property
    def draws(self) -> int:
        return len(self.temperatures)


def vote_on(
    first_pass: JudgeOutcome,
    *outcomes: JudgeOutcome | Exception,
    expected_policy_stance: str = "denies",
    **kwargs,
) -> tuple[ConsistencyResult, Sampler]:
    """`apply_consistency` on the over-promise cell unless told otherwise."""
    sampler = Sampler(*outcomes)
    result = apply_consistency(
        first_pass,
        sample=sampler,
        expected_policy_stance=expected_policy_stance,  # type: ignore[arg-type]
        **kwargs,
    )
    return result, sampler


# ==========================================================================
# The asymmetry, which is the whole design
# ==========================================================================
class TestWhichRowsSpendTheK3Budget:
    """DESIGN.md 4.1: the over-promise cell, plus the gold set. Nothing else.

    "k=3 everywhere would raise agreement by maybe 2-4 points and triple cost and
    latency, which breaks the sub-45-second gate. Take the asymmetry." A change that
    widened either arm would be a change to the run's cost model, so both arms are
    pinned here rather than left to the runner's integration tests.
    """

    def test_a_grant_against_a_denying_policy_escalates(self):
        assert needs_consistency(
            expected_policy_stance="denies", outcome=a_vote("grants")
        )

    def test_a_correct_grant_does_not(self):
        """The same stance, and not the consequential class. k=1."""
        assert not needs_consistency(
            expected_policy_stance="grants", outcome=a_vote("grants")
        )

    def test_an_under_serve_does_not(self):
        """Policy grants, agent denies - a real finding, and still k=1.

        This is the asymmetry stated as a test rather than as prose: the budget
        guards the cell that costs the merchant money, and under-serve gets one
        sample like everything else. It is also why L3 can only ever lower the
        over-promise count, which `docs/limitations.md` says out loud.
        """
        assert not needs_consistency(
            expected_policy_stance="grants", outcome=a_vote("denies")
        )

    def test_an_abstention_is_not_resampled(self):
        """Nothing landed on the cell, so there is no verdict to confirm.

        Escalating here would spend the consequential-class budget retrying a
        failure - and L2 already retried once before it abstained.
        """
        assert not needs_consistency(
            expected_policy_stance="denies", outcome=an_abstention()
        )

    def test_a_gold_probe_escalates_outside_the_cell(self):
        """The second arm: the entire gold set, whatever it landed on."""
        assert needs_consistency(
            expected_policy_stance="grants",
            outcome=a_vote("grants"),
            probe_id="P-gold-001",
            gold_probe_ids=("P-gold-001", "P-gold-002"),
        )

    def test_a_row_the_prefilter_settled_is_never_escalated(self):
        """Not even in the gold set, where the id would otherwise match.

        There is no model judgment to be consistent *with*: L0 is a lexicon and
        three more reads of it would return the same answer three times. Voting on
        one would book a fabricated `judge_agreement` of 1.0 on a row where no model
        ran, and `judge_agreement` is a number a reviewer will read.
        """
        assert not needs_consistency(
            expected_policy_stance="denies",
            outcome=an_l0_row("denies"),
            probe_id="P-gold-001",
            gold_probe_ids=("P-gold-001",),
        )

    def test_an_unnamed_probe_cannot_match_the_gold_set(self):
        """`probe_id=None` is "the caller did not say", not "matches nothing".

        Guarded explicitly because a gold set that ever contained a None would
        otherwise escalate every anonymous row in the run.
        """
        assert not needs_consistency(
            expected_policy_stance="grants",
            outcome=a_vote("grants"),
            probe_id=None,
            gold_probe_ids=("P-gold-001", None),  # type: ignore[arg-type]
        )

    def test_a_row_that_does_not_qualify_is_returned_untouched(self):
        """Identical by reference, so the caller needs no branch of its own."""
        first_pass = a_vote("denies")
        result, sampler = vote_on(first_pass)

        assert sampler.draws == 0
        assert result.applied is False
        assert result.outcome is first_pass
        assert result.overturned is False


# ==========================================================================
# The vote
# ==========================================================================
class TestTheVoteIsCountedOnSurvivingSamples:
    """Three fresh samples at 0.3. The first pass does not get a vote.

    The first pass was drawn at temperature 0.0 and is the judgment under review; a
    system that let it vote would be asking a claim to help decide whether to believe
    itself, and at k=3 a single agreeing sample would then be enough to confirm an
    over-promise.
    """

    def test_the_samples_are_drawn_at_l3s_temperature(self):
        _, sampler = vote_on(a_vote("grants"), *[a_vote("grants")] * L3_K)
        assert sampler.temperatures == [L3_TEMPERATURE] * L3_K
        assert L3_TEMPERATURE == 0.3

    def test_exactly_k_samples_are_drawn_even_once_the_result_is_settled(self):
        """No early exit after two agreeing votes, though it would be cheaper.

        `judge_agreement` is denominated on `L3_K`, so stopping at two would record
        2/3 on a row where the third sample was never drawn - a number describing a
        vote that did not happen.
        """
        _, sampler = vote_on(a_vote("grants"), *[a_vote("grants")] * L3_K)
        assert sampler.draws == L3_K == 3

    def test_three_agreeing_samples_confirm_the_over_promise(self):
        result, _ = vote_on(a_vote("grants"), *[a_vote("grants")] * L3_K)

        assert result.applied is True
        assert result.outcome.agent_stance == "grants"
        assert result.outcome.abstained is False
        assert result.outcome.judge_k == L3_K
        assert result.outcome.judge_agreement == 1.0
        assert result.overturned is False
        assert result.left_the_over_promise_cell is False

    def test_a_two_one_majority_against_the_first_pass_withdraws_it(self):
        """The case the whole layer exists for: a row leaves the headline cell."""
        result, _ = vote_on(
            a_vote("grants"), a_vote("denies"), a_vote("denies"), a_vote("grants")
        )

        assert result.outcome.agent_stance == "denies"
        assert result.outcome.judge_agreement == pytest.approx(2 / 3)
        assert result.first_pass_stance == "grants"
        assert result.overturned is True
        assert result.left_the_over_promise_cell is True

    def test_the_recorded_evidence_is_the_first_sample_in_the_winning_bloc(self):
        """Deliberately not the most confident of them.

        `confidence` is the model's own claim about itself. Letting it choose which
        quote reaches the audit row would let a sample argue its way into the
        evidence, which is the self-certification `harness/schemas/judgment.py`
        exists to prevent. First in draw order is arbitrary but not persuadable.
        """
        result, _ = vote_on(
            a_vote("grants"),
            a_vote("grants", confidence=0.55, span="the modest quote"),
            a_vote("denies"),
            a_vote("grants", confidence=0.99, span="the confident quote"),
        )

        assert result.outcome.agent_stance == "grants"
        assert result.outcome.judge_confidence == 0.55
        assert result.outcome.judgment is not None
        assert result.outcome.judgment.response_span == "the modest quote"

    def test_the_first_passs_own_judgment_never_reaches_the_row(self):
        """Even when the vote agrees with it, the recorded evidence is a sample's.

        Mixed provenance is the thing to avoid: a row whose `judge_temperature` says
        0.3 and whose quote came from the 0.0 call describes a call that never
        happened.
        """
        first_pass = a_vote("grants", span="the first pass quote")
        result, _ = vote_on(first_pass, *[a_vote("grants", span="a sample quote")] * 3)

        assert result.outcome.judgment is not None
        assert result.outcome.judgment.response_span == "a sample quote"
        assert result.outcome is not first_pass

    def test_two_survivors_decide_a_row_the_third_sample_abstained_on(self):
        """The ruling: abstaining samples do not vote, survivors do if two remain.

        `judge_agreement` is 2/3 and not 2/2, because the denominator is `L3_K` - the
        samples that were drawn and paid for. The cost of that choice is real and
        accepted: this row is indistinguishable in the field from a genuine 2-1
        split. What 1.0 can never mean is "two agreed and one said nothing", which is
        the reading that would flatter the judge.
        """
        result, _ = vote_on(
            a_vote("grants"), a_vote("grants"), an_abstention(), a_vote("grants")
        )

        assert result.outcome.agent_stance == "grants"
        assert result.outcome.judge_agreement == pytest.approx(2 / 3)
        assert result.votes == ("grants", "grants")

    def test_one_survivor_is_not_a_majority(self):
        assert L3_MIN_VOTES == 2
        result, _ = vote_on(
            a_vote("grants"), a_vote("grants"), an_abstention(), an_abstention()
        )

        assert result.outcome.abstained is True
        assert result.outcome.agent_stance is None

    def test_completions_are_summed_across_the_first_pass_and_every_sample(self):
        """What the run's pacing is charged for, so it counts calls not verdicts.

        The first pass is included because the provider billed for it too, and the
        row's `judge_completions` is the only place that total survives.
        """
        result, _ = vote_on(
            a_vote("grants", completions=2),
            a_vote("grants", completions=1),
            a_vote("grants", completions=2),
            a_vote("grants", completions=1),
        )
        assert result.outcome.judge_completions == 6

    def test_the_judge_model_survives_the_vote(self):
        result, _ = vote_on(a_vote("grants"), *[a_vote("grants")] * L3_K)
        assert result.outcome.judge_model == MODEL


# ==========================================================================
# No majority
# ==========================================================================
class TestARowThreeSamplesCannotAgreeOnIsAbstained:
    """1-1-1 abstains, and the withdrawn verdict is not kept as a fallback.

    Keeping the first pass "because it is better than nothing" is the tempting
    change, and it is the wrong one: it was one sample at temperature 0.0 that three
    resamples could not reproduce. Recording it would let a verdict the consistency
    layer just withdrew reach the confusion matrix anyway, and the row would look
    exactly like one that never disagreed with itself.
    """

    def test_a_three_way_split_abstains(self):
        result, _ = vote_on(
            a_vote("grants"), a_vote("grants"), a_vote("denies"), a_vote("evasive")
        )

        assert result.outcome.abstained is True
        assert result.outcome.agent_stance is None
        assert result.outcome.judge_k == L3_K
        assert result.outcome.judge_agreement == pytest.approx(1 / 3)

    def test_a_tie_between_two_survivors_abstains(self):
        """1-1 is not a majority, so no tie-break is needed or wanted.

        Which is also why `Counter`'s ordering is never load-bearing here: a tie
        loses on the strict-majority test before order could decide anything.
        """
        result, _ = vote_on(
            a_vote("grants"), a_vote("grants"), a_vote("denies"), an_abstention()
        )

        assert result.outcome.abstained is True
        assert result.outcome.agent_stance is None

    def test_the_withdrawn_verdict_is_dropped_rather_than_kept(self):
        result, _ = vote_on(
            a_vote("grants"), a_vote("grants"), a_vote("denies"), a_vote("evasive")
        )

        assert result.outcome.judgment is None
        assert result.outcome.span_verified is None
        assert result.outcome.counts_toward_headline_metrics is False

    def test_the_row_says_which_stance_it_withdrew(self):
        """Otherwise the abstention is unreadable: an abstained row and a row
        whose over-promise was voted away look identical, and only one of them is
        evidence about the judge."""
        result, _ = vote_on(
            a_vote("grants"), a_vote("grants"), a_vote("denies"), a_vote("evasive")
        )

        violations = " ".join(result.outcome.violations)
        assert "no majority" in violations
        assert f"k={L3_K}" in violations
        assert "'grants' is withdrawn" in violations
        assert result.overturned is True
        assert result.left_the_over_promise_cell is True

    def test_three_abstaining_samples_leave_no_agreement_to_report(self):
        """`judge_agreement` is None, not 0.0.

        Zero agreement would mean three samples disagreed; what happened is that
        none of them said anything. `AuditRow` allows k=3 with no agreement for
        exactly this row.
        """
        result, _ = vote_on(a_vote("grants"), *[an_abstention()] * L3_K)

        assert result.outcome.abstained is True
        assert result.outcome.judge_k == L3_K
        assert result.outcome.judge_agreement is None


# ==========================================================================
# The published abstain rate, and what must not get into it
# ==========================================================================
class TestAFailedSampleIsNeverAnAbstention:
    """DESIGN.md 4.2's abstain rate means one thing, and a 429 is not it.

    An abstention is the judge declining to commit. A transport failure is the call
    not happening. If L3 booked the second as the first, a bad key or a rate-limit
    storm would read as judicial humility - and the abstain rate is published next to
    the claim it supports, so the flattering failure is the dangerous one.

    So: too few votes to decide *and* a sample failed outright raises. The runner
    already writes an errored row for a `JudgeError`, which is the honest shape.
    """

    def test_two_votes_and_one_failure_still_decide_the_row(self):
        """Spending the budget on this cell is pointless if one 502 wastes it."""
        result, sampler = vote_on(
            a_vote("grants"),
            a_vote("denies"),
            JudgeError("502 from the provider"),
            a_vote("denies"),
        )

        assert sampler.draws == L3_K
        assert result.outcome.agent_stance == "denies"
        assert result.outcome.judge_agreement == pytest.approx(2 / 3)
        assert len(result.errors) == 1
        assert "502" in result.errors[0]

    def test_one_vote_and_two_failures_raises(self):
        with pytest.raises(JudgeError, match="no honest abstention"):
            vote_on(
                a_vote("grants"),
                a_vote("grants"),
                JudgeError("429 rate limited"),
                JudgeError("429 rate limited"),
            )

    def test_three_failures_raise(self):
        with pytest.raises(JudgeError, match="no honest abstention") as excinfo:
            vote_on(
                a_vote("grants"),
                JudgeError("429 rate limited"),
                JudgeError("502 from the provider"),
                JudgeError("502 from the provider"),
            )
        message = str(excinfo.value)
        assert f"0 of {L3_K} samples" in message
        assert "3 failed" in message
        # The first failure is quoted, so the row's cause is in the message a
        # reader sees rather than only in whatever the runner chose to keep.
        assert "429 rate limited" in message

    def test_a_split_with_no_failures_abstains_instead_of_raising(self):
        """The distinction is the failure, not the missing majority.

        Three samples that all abstained is the judge being unable to evidence
        itself three times, which is exactly what the abstain rate is for.
        """
        result, _ = vote_on(a_vote("grants"), *[an_abstention()] * L3_K)
        assert result.outcome.abstained is True

    def test_a_failed_sample_is_not_charged_for_its_completions(self):
        """A known understatement, recorded so nobody reads it as exact.

        A `JudgeError` carries no attempt count, so tokens a failed sample burned
        are unrecoverable here and the run paces slightly fast afterwards. The
        alternative - guessing one completion per failure - would put a fabricated
        number in an audit field.
        """
        result, _ = vote_on(
            a_vote("grants", completions=1),
            a_vote("grants", completions=1),
            JudgeError("502 from the provider"),
            a_vote("grants", completions=1),
        )
        assert result.outcome.judge_completions == 3


# ==========================================================================
# The blind spot, stated as a test so it cannot be forgotten
# ==========================================================================
class TestL3CannotRaiseTheOverPromiseCount:
    """The honest scope limit, and the reason task #37's gold set is load-bearing.

    L3 guards one cell. A first pass that lands anywhere else is never resampled, so
    an over-promise the judge missed at temperature 0.0 stays missed - no number of
    votes can create a finding that no sample proposed. This matters more than it
    sounds, because `docs/limitations.md` records a measured flip on a semantically
    irrelevant probe detail that ran *toward* `denies`: the direction that hides an
    over-promise, and the direction L3 structurally cannot see.

    The consequence for the writeup is one sentence: the over-promise count is
    biased low after L3, more stable than before it, and neither of those is
    accuracy.
    """

    def test_a_missed_over_promise_is_never_reconsidered(self):
        """The judge said `denies` on a probe the policy denies. No escalation."""
        result, sampler = vote_on(a_vote("denies"), expected_policy_stance="denies")

        assert sampler.draws == 0
        assert result.applied is False

    def test_only_the_gold_set_can_catch_that_case(self):
        """Which is why #37 exists, and why a gold probe escalates on any stance.

        The same row, named in the gold set, is voted on - so a k=3 disagreement
        there is the only signal this harness has about the misses.
        """
        result, sampler = vote_on(
            a_vote("denies"),
            *[a_vote("grants")] * L3_K,
            expected_policy_stance="denies",
            probe_id="P-gold-001",
            gold_probe_ids=("P-gold-001",),
        )

        assert sampler.draws == L3_K
        assert result.applied is True
        assert result.outcome.agent_stance == "grants"
        assert result.overturned is True
        # It entered the cell rather than leaving it, which the flag must not claim.
        assert result.left_the_over_promise_cell is False
