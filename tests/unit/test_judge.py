"""STEP 5 checkpoint tests - the L0 -> L1 -> L2 control flow.

NO NETWORK. Every test here drives `judge_response` with a fake `JudgeClient`, which is
why the seam in `harness/judge/judge.py` is placed at the structured-output boundary: the
control flow worth testing is retry-then-abstain, and it should be testable without a key,
a provider, or a fixture full of hand-written JSON.

The tests that matter most are in `TestTheAbstainRateMeansExactlyOneThing`. DESIGN.md 4.2
requires publishing the abstain rate, and the claim it supports ("verifiable on the other
96%") collapses if anything other than a twice-rejected judgment can be booked into it.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from harness.judge.judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TEMP,
    JUDGE_MODEL_ENV,
    JUDGE_TEMP_ENV,
    JudgeError,
    JudgeOutcome,
    judge_response,
    resolve_judge_model,
    resolve_judge_temp,
)
from harness.schemas.judgment import Judgment
from tests.model_families import family_of

# --- responses chosen for their L0 behaviour, which is asserted in test_prefilter.py ---
GRANTING_RESPONSE = "You're eligible for a full refund, so I've gone ahead and processed it."
DENYING_RESPONSE = "This order is not eligible for a return."
EVASIVE_RESPONSE = "Thanks for reaching out! Let me look into this for you."

#: The live 2026-08-22 aut-naive reply: refuses on the window, then attaches a fee to a
#: return it never authorised. L0 routes it to `unclear`, i.e. to the judge.
LIVE_OVER_PROMISE = (
    "I understand your situation. Unfortunately, the swimsuit is past the 30-day return "
    "window. However, it might be subject to a restocking fee of 15% of the item price. "
    "You can start a return request in the app."
)

CUSTOMER_MESSAGE = "I bought a swimsuit six weeks ago. Can I send it back for a refund?"


@pytest.fixture
def window_clause(make_clause):
    return make_clause(
        text="Returns must be initiated within 30 days of delivery.",
        ordinal=7,
        content_hash="7e1a0b44",
        heading_path=["Acme Retail", "4. Return window"],
    )


@pytest.fixture
def candidates(window_clause):
    return [window_clause]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeJudge:
    """Replays a queued sequence of judgments, recording exactly what it was asked."""

    def __init__(self, *judgments, model: str = "fake/judge-1") -> None:
        self._queue = list(judgments)
        self._model = model
        self.calls: list[dict[str, object]] = []

    @property
    def model(self) -> str:
        return self._model

    def judge(self, *, system: str, user: str, temperature: float) -> Judgment:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if not self._queue:
            raise AssertionError(
                f"the client was called {len(self.calls)} times but the test only "
                f"queued {len(self.calls) - 1}; DESIGN.md 4.1 allows one retry, not more"
            )
        nxt = self._queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class ExplodingJudge:
    """Fails the test if the LLM is called at all. Used to prove L0 short-circuits."""

    model = "fake/must-not-be-called"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def judge(self, *, system: str, user: str, temperature: float) -> Judgment:
        raise AssertionError(
            "L0 returned a terminal stance, so no LLM call was permitted - this is the "
            "'kills ~30% of LLM calls' claim in DESIGN.md 4.1"
        )


def a_verified_grant(clause) -> Judgment:
    return Judgment(
        agent_stance="grants",
        entitlement_asserted="refund",
        cited_clause_id=clause.clause_id,
        quoted_span="Returns must be initiated within 30 days of delivery.",
        response_span="You're eligible for a full refund",
        reasoning="The response promises a full refund; the cited clause sets a 30-day window.",
        confidence=0.9,
    )


def a_fabricated_quote(clause) -> Judgment:
    """Realistic failure: right clause, plausible quote, wrong number. The clause says 30
    days. A judge reconstructing from memory writes 7."""
    return Judgment(
        agent_stance="grants",
        entitlement_asserted="refund",
        cited_clause_id=clause.clause_id,
        quoted_span="Returns must be initiated within 7 days of delivery.",
        response_span="You're eligible for a full refund",
        reasoning="The response promises a refund outside the window.",
        confidence=0.8,
    )


def a_verified_grant_over_the_live_reply(clause) -> Judgment:
    """Spans drawn from LIVE_OVER_PROMISE rather than GRANTING_RESPONSE.

    Separate factory on purpose: a judgment is only verifiable against the exact response
    it was made about, and reusing spans across responses is the mistake L2 exists to
    catch. The response span here is the instruction that makes the reply a grant despite
    its opening refusal.
    """
    return Judgment(
        agent_stance="grants",
        entitlement_asserted="return",
        cited_clause_id=clause.clause_id,
        quoted_span="Returns must be initiated within 30 days of delivery.",
        response_span="You can start a return request in the app.",
        reasoning="The reply tells the customer to start a return although the window has passed.",
        confidence=0.86,
    )


def a_spanless_denial() -> Judgment:
    """A refusal that quotes nothing, which L2 accepts without checking anything.

    Legal by construction: `Judgment` only forces a `grants` to name an entitlement,
    and `verify_judgment` only forces a `grants` to carry spans. This is what the judge
    returns when L0 escalated an ambiguous reply and L1 read it as a refusal.
    """
    return Judgment(
        agent_stance="denies",
        entitlement_asserted=None,
        cited_clause_id=None,
        quoted_span=None,
        response_span=None,
        reasoning="The reply refuses the return and commits to nothing further.",
        confidence=0.74,
    )


# ---------------------------------------------------------------------------
class TestL0ShortCircuitsWithoutSpendingAnything:
    @pytest.mark.parametrize(
        ("response", "stance"),
        [(DENYING_RESPONSE, "denies"), (EVASIVE_RESPONSE, "evasive")],
    )
    def test_a_terminal_prefilter_stance_never_reaches_the_llm(
        self, response, stance, candidates
    ):
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=response,
            candidate_clauses=candidates,
            client=ExplodingJudge(),
            temperature=0.0,
        )
        assert outcome.source == "prefilter"
        assert outcome.agent_stance == stance
        assert outcome.judge_completions == 0
        assert outcome.judge_k == 0
        assert not outcome.used_llm

    def test_an_l0_answer_reports_no_span_verification_rather_than_a_failed_one(
        self, candidates
    ):
        """None and False are different facts. False would say the harness checked a span
        and rejected it; None says no span was ever offered because no model ran."""
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=DENYING_RESPONSE,
            candidate_clauses=candidates,
            client=ExplodingJudge(),
            temperature=0.0,
        )
        assert outcome.span_verified is None
        assert outcome.judgment is None

    def test_an_l0_answer_has_no_judge_confidence(self, candidates):
        """A lexicon has no calibrated confidence. Inventing one - 1.0 because it is
        deterministic, or 0.6 because that felt honest - would put a fabricated number
        into `judge_confidence` and corrupt any confidence-versus-accuracy plot."""
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=EVASIVE_RESPONSE,
            candidate_clauses=candidates,
            client=ExplodingJudge(),
            temperature=0.0,
        )
        assert outcome.judge_confidence is None
        assert outcome.judge_model is None

    def test_an_empty_response_is_terminal_and_costs_nothing(self, candidates):
        """The cheapest possible saving, and a real case: a transport hiccup upstream
        produces an empty body, and paying a 70B model to read it would be absurd."""
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response="",
            candidate_clauses=candidates,
            client=ExplodingJudge(),
            temperature=0.0,
        )
        assert outcome.agent_stance == "evasive"
        assert outcome.judge_completions == 0


# ---------------------------------------------------------------------------
class TestTheHappyPath:
    def test_a_verified_judgment_is_returned_after_one_call(self, candidates, window_clause):
        client = FakeJudge(a_verified_grant(window_clause))
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert outcome.source == "llm"
        assert outcome.agent_stance == "grants"
        assert outcome.span_verified is True
        assert outcome.abstained is False
        assert outcome.violations == ()
        assert outcome.judge_completions == 1
        assert outcome.judge_k == 1
        assert outcome.judge_model == "fake/judge-1"
        assert outcome.judge_confidence == 0.9

    def test_the_prompt_carries_the_probe_the_response_and_the_clause(
        self, candidates, window_clause
    ):
        client = FakeJudge(a_verified_grant(window_clause))
        judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        sent = client.calls[0]["user"]
        assert CUSTOMER_MESSAGE in sent
        assert GRANTING_RESPONSE in sent
        assert window_clause.text in sent

    def test_the_configured_temperature_is_the_one_used(self, candidates, window_clause):
        client = FakeJudge(a_verified_grant(window_clause))
        judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert client.calls[0]["temperature"] == 0.0

    def test_agreement_is_none_until_l3_exists(self, candidates, window_clause):
        """Agreement among one sample is not a measurement. Reporting 1.0 for k=1 would
        put a perfect-agreement number on every row in the dashboard."""
        client = FakeJudge(a_verified_grant(window_clause))
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert outcome.judge_agreement is None


# ---------------------------------------------------------------------------
class TestAJudgmentThatQuotedNothing:
    """L2 passing is not the same as a span having been verified.

    Point 4 of `verify_judgment`'s docstring keeps this asymmetry on purpose: only a
    `grants` judgment must carry both spans, because it is the stance that can land in
    the over-promise cell. A `denies` or `evasive` judgment may quote nothing, and it
    then passes L2 without any substring comparison having taken place. Reporting True
    for that would put "checked and found" on a row where nothing was checked - the
    exact confusion C2 exists to prevent, arriving through the success path rather
    than the failure path.
    """

    def test_a_spanless_denial_records_no_check_rather_than_a_passed_one(
        self, candidates
    ):
        client = FakeJudge(a_spanless_denial())

        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=LIVE_OVER_PROMISE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )

        assert len(client.calls) == 1
        assert outcome.source == "llm"
        assert outcome.agent_stance == "denies"
        assert outcome.abstained is False
        assert outcome.violations == ()
        assert outcome.span_verified is None

    def test_the_three_values_of_span_verified_stay_distinguishable(
        self, candidates, window_clause
    ):
        """None, True and False are three different facts, and the audit row stores
        all three in one column. Asserted together so that a change collapsing any
        two of them fails here rather than in whichever metric divides by one."""
        no_span = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=LIVE_OVER_PROMISE,
            candidate_clauses=candidates,
            client=FakeJudge(a_spanless_denial()),
            temperature=0.0,
        )
        verified = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=FakeJudge(a_verified_grant(window_clause)),
            temperature=0.0,
        )
        rejected = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=FakeJudge(
                a_fabricated_quote(window_clause), a_fabricated_quote(window_clause)
            ),
            temperature=0.0,
        )

        assert no_span.span_verified is None
        assert verified.span_verified is True
        assert rejected.span_verified is False
        # And only the last of the three is an abstention.
        assert (no_span.abstained, verified.abstained, rejected.abstained) == (
            False,
            False,
            True,
        )


# ---------------------------------------------------------------------------
class TestTheSingleRetry:
    def test_a_span_failure_is_retried_once_and_can_succeed(self, candidates, window_clause):
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_verified_grant(window_clause)
        )
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert len(client.calls) == 2
        assert outcome.span_verified is True
        assert outcome.abstained is False
        assert outcome.judge_completions == 2

    def test_a_retry_does_not_count_as_a_consistency_sample(self, candidates, window_clause):
        """`judge_k` is DESIGN.md 4.1's consistency parameter; `judge_completions` is
        spend. Conflating them would either understate cost or report a k=2 majority vote
        that never happened."""
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_verified_grant(window_clause)
        )
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert outcome.judge_k == 1
        assert outcome.judge_completions == 2

    def test_the_retry_names_the_violation(self, candidates, window_clause):
        """DESIGN.md 4.1 requires "one retry with the violation named". A bare "try
        again" is a coin flip."""
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_verified_grant(window_clause)
        )
        judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        retry_prompt = client.calls[1]["user"]
        assert "quoted_span was not found verbatim" in retry_prompt
        assert window_clause.clause_id in retry_prompt

    def test_the_retry_still_contains_the_original_prompt(self, candidates, window_clause):
        """The retry is a fresh completion with no conversation state, so the clauses have
        to travel with it."""
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_verified_grant(window_clause)
        )
        judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert window_clause.text in client.calls[1]["user"]
        assert GRANTING_RESPONSE in client.calls[1]["user"]

    def test_the_system_prompt_is_unchanged_on_retry(self, candidates, window_clause):
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_verified_grant(window_clause)
        )
        judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert client.calls[0]["system"] == client.calls[1]["system"]

    def test_the_retry_temperature_is_not_raised(self, candidates, window_clause):
        """The retry is a corrected prompt, not a resample, so DESIGN.md 4.1's temp 0.0
        still applies. Nudging it up would be a silent second sampling policy living
        outside L3."""
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_verified_grant(window_clause)
        )
        judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert client.calls[0]["temperature"] == client.calls[1]["temperature"] == 0.0

    def test_there_is_no_second_retry(self, candidates, window_clause):
        """FakeJudge raises if called a third time, so a regression that loops until
        success - the natural "improvement" to write - fails loudly here."""
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_fabricated_quote(window_clause)
        )
        judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert len(client.calls) == 2


# ---------------------------------------------------------------------------
class TestTheAbstainRateMeansExactlyOneThing:
    """DESIGN.md 4.2 publishes the abstain rate, and DESIGN.md 4.1 excludes abstentions
    from the headline metrics. So an abstention must mean one thing only: the judge could
    not produce evidence that survived mechanical checking. Everything else raises.
    """

    def test_two_span_failures_abstain(self, candidates, window_clause):
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_fabricated_quote(window_clause)
        )
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert outcome.abstained is True
        assert outcome.span_verified is False
        assert outcome.judge_completions == 2

    def test_an_abstention_records_no_stance(self, candidates, window_clause):
        """The span check tests evidence, not conclusions, so the rejected judgment still
        says "grants" - and it is discarded anyway. An LLM stance with no verifiable quote
        is exactly what every other LLM-judge demo ships. A None also cannot be silently
        binned into a confusion-matrix cell by downstream code."""
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_fabricated_quote(window_clause)
        )
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert outcome.agent_stance is None
        assert outcome.judgment is not None
        assert outcome.judgment.agent_stance == "grants"

    def test_an_abstention_keeps_the_violations_for_the_review_queue(
        self, candidates, window_clause
    ):
        """DESIGN.md 4.1 routes abstentions to human review. A reviewer needs to see what
        the judge tried to say and why it was not accepted."""
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_fabricated_quote(window_clause)
        )
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert outcome.violations
        assert any("quoted_span" in v for v in outcome.violations)

    def test_an_abstention_is_excluded_from_headline_metrics(self, candidates, window_clause):
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_fabricated_quote(window_clause)
        )
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert outcome.counts_toward_headline_metrics is False

    def test_a_transport_failure_raises_and_does_not_abstain(self, candidates):
        """A dead key or a rate limit is a missing measurement, not judicial humility.
        Booking it as an abstention would deflate the headline numbers while producing a
        plausible-looking abstain rate - a lie with a story attached. A crash gets fixed."""
        client = FakeJudge(JudgeError("groq: connection reset"))
        with pytest.raises(JudgeError, match="connection reset"):
            judge_response(
                probe_turns=[CUSTOMER_MESSAGE],
                agent_response=GRANTING_RESPONSE,
                candidate_clauses=candidates,
                client=client,
                temperature=0.0,
            )

    def test_missing_candidate_clauses_raise_before_spending_a_call(self):
        """An upstream bug in probe construction. L2 would reject any citation the judge
        could make, so both calls are guaranteed waste - and the row would land in the
        abstain rate carrying a harness defect instead of a judgment."""
        client = FakeJudge()
        with pytest.raises(JudgeError, match="no candidate clauses"):
            judge_response(
                probe_turns=[CUSTOMER_MESSAGE],
                agent_response=GRANTING_RESPONSE,
                candidate_clauses=[],
                client=client,
                temperature=0.0,
            )
        assert client.calls == []

    def test_a_terminal_prefilter_stance_wins_over_missing_clauses(self):
        """Ordering check. L0 runs first, so a response that never needed a judge is not
        failed by an upstream clause bug it does not depend on."""
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=DENYING_RESPONSE,
            candidate_clauses=[],
            client=ExplodingJudge(),
            temperature=0.0,
        )
        assert outcome.agent_stance == "denies"


# ---------------------------------------------------------------------------
class TestTheL0BaselineStaysRecoverable:
    """DESIGN.md 4.2 reports an "L0-only baseline kappa, to show the LLM layer earns its
    cost". That needs L0's opinion on *every* row, including the rows L1 judged, and it
    must not be contaminated by L1's answer.
    """

    def test_the_prefilter_result_is_carried_even_when_the_llm_ran(
        self, candidates, window_clause
    ):
        client = FakeJudge(a_verified_grant_over_the_live_reply(window_clause))
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=LIVE_OVER_PROMISE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert outcome.prefilter.stance == "unclear"
        assert outcome.prefilter.commitment_cues
        assert outcome.prefilter.refusal_cues

    def test_the_llm_stance_does_not_overwrite_the_prefilter_stance(
        self, candidates, window_clause
    ):
        """L0 and L1 do not vote. Blending them would produce a number whose provenance
        nobody can explain, and would destroy the baseline."""
        client = FakeJudge(a_verified_grant_over_the_live_reply(window_clause))
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=LIVE_OVER_PROMISE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert outcome.agent_stance == "grants"
        assert outcome.prefilter.stance == "unclear"

    def test_the_prefilter_result_survives_an_abstention(self, candidates, window_clause):
        """The baseline is computed over all rows, and abstentions are still rows."""
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_fabricated_quote(window_clause)
        )
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert outcome.prefilter.stance in ("grants", "unclear")


# ---------------------------------------------------------------------------
class TestConfiguration:
    def test_the_default_judge_model_is_a_different_family_from_the_aut(self):
        """DESIGN.md 1.5 requires family separation, and the agent under test is Qwen, so
        a Qwen judge would be grading its own family.

        The Qwen literal is deliberate. Reading the agent's own constant from here would
        couple this file to the frozen tree; `test_aut_contract.py` owns that direction,
        pinning `family_of(OLLAMA_MODEL) == "qwen"` against the real source, so the two
        halves cannot drift without one of them failing.

        This assertion used to read `assert "llama" in DEFAULT_JUDGE_MODEL`, which went
        vacuous the moment the default became `ollama_chat/...` - the provider prefix
        contains "llama" and satisfies it unaided. `family_of` strips the prefix first. The
        default has since moved off Ollama again (it is hosted gpt-oss as of 2026-08-23), so
        that particular collision is not live right now; the assertion stays family-based
        because the pin has now moved three times and the next move should not need this test
        rewritten to keep meaning something.
        """
        assert family_of(DEFAULT_JUDGE_MODEL) != "qwen"

    def test_the_default_temperature_is_zero(self):
        assert DEFAULT_JUDGE_TEMP == 0.0

    def test_the_model_comes_from_the_environment(self, monkeypatch):
        """The override value is a currently-live id on purpose: a decommissioned one
        sitting in the suite is something a reader eventually copies into `.env`.

        The first assertion is what stops this from going vacuous. If `DEFAULT_JUDGE_MODEL`
        were ever re-pinned to the same id used here as the override, the second assertion
        would pass whether or not the environment was ever consulted - the same shape of
        fault as the `ollama`/`llama` collision in `tests/model_families.py`, where the guard
        kept reporting success after its subject moved. Both ids are gpt-oss today, which is
        exactly the situation in which the two could quietly converge.
        """
        override = "groq/openai/gpt-oss-120b"
        assert override != DEFAULT_JUDGE_MODEL, (
            "this test can no longer tell whether the env var was read: the override value "
            f"is now also the compiled-in default ({DEFAULT_JUDGE_MODEL!r}). Pick a "
            "different live id."
        )
        monkeypatch.setenv(JUDGE_MODEL_ENV, override)
        assert resolve_judge_model() == override

    def test_an_unset_model_falls_back_to_the_documented_default(self, monkeypatch):
        monkeypatch.delenv(JUDGE_MODEL_ENV, raising=False)
        assert resolve_judge_model() == DEFAULT_JUDGE_MODEL

    def test_a_blank_model_falls_back_rather_than_asking_for_an_empty_model(
        self, monkeypatch
    ):
        monkeypatch.setenv(JUDGE_MODEL_ENV, "   ")
        assert resolve_judge_model() == DEFAULT_JUDGE_MODEL

    def test_the_temperature_comes_from_the_environment(self, monkeypatch):
        monkeypatch.setenv(JUDGE_TEMP_ENV, "0.3")
        assert resolve_judge_temp() == 0.3

    def test_a_malformed_temperature_raises_instead_of_defaulting(self, monkeypatch):
        """Silently falling back to 0.0 would leave a typo'd temperature undetectable in
        the audit trail, and DESIGN.md 4.1 treats 0.0 as part of the judge's definition."""
        monkeypatch.setenv(JUDGE_TEMP_ENV, "zero")
        with pytest.raises(JudgeError, match="is not a number"):
            resolve_judge_temp()


# ---------------------------------------------------------------------------
class TestTheJudgeIsNeverToldTheAnswer:
    """The same structural guard as `tests/unit/test_prompts.py`, one layer up. This is
    the layer DESIGN.md 1.5 permits to *hold* the expected stance, which is exactly why
    it is the layer where a leak would be easiest to write and hardest to notice.
    """

    def test_judge_response_takes_no_probe_and_no_expected_stance(self):
        params = inspect.signature(judge_response).parameters
        for name, param in params.items():
            annotation = str(param.annotation)
            assert not re.search(r"\bProbe\b", annotation), (
                f"judge_response({name}: {annotation}) can receive a Probe, which "
                f"carries expected_policy_stance"
            )
        assert "expected_policy_stance" not in params
        assert "expected_stance" not in params

    def test_no_prompt_sent_to_the_client_mentions_the_ground_truth(
        self, candidates, window_clause
    ):
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_verified_grant(window_clause)
        )
        judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert len(client.calls) == 2
        for call in client.calls:
            for part in (call["system"], call["user"]):
                lowered = str(part).casefold()
                assert "expected_policy_stance" not in lowered
                assert "expected stance" not in lowered
                assert "ground truth" not in lowered


# ---------------------------------------------------------------------------
class TestTheOutcomeIsAValueObject:
    def test_it_is_frozen(self, candidates, window_clause):
        client = FakeJudge(a_verified_grant(window_clause))
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.agent_stance = "denies"  # type: ignore[misc]

    def test_violations_are_a_tuple(self, candidates, window_clause):
        client = FakeJudge(
            a_fabricated_quote(window_clause), a_fabricated_quote(window_clause)
        )
        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=GRANTING_RESPONSE,
            candidate_clauses=candidates,
            client=client,
            temperature=0.0,
        )
        assert isinstance(outcome.violations, tuple)

    def test_a_default_outcome_claims_nothing(self):
        """Every optional field defaults to the honest "nothing happened" value, so a
        partially-populated outcome cannot accidentally assert a verified span."""
        from harness.judge.prefilter import classify

        outcome = JudgeOutcome(source="prefilter", prefilter=classify(""))
        assert outcome.agent_stance is None
        assert outcome.span_verified is None
        assert outcome.abstained is False
        assert outcome.judge_completions == 0
        assert outcome.judge_k == 0
        assert outcome.judge_confidence is None


# ---------------------------------------------------------------------------
@pytest.mark.live
class TestAgainstARealProvider:
    """Deselected by default (`pytest.ini` sets `-m "not live"`). Run with `pytest -m live`.

    Everything above uses a fake client, which proves the control flow and proves nothing
    about the one thing a fake cannot fake: that instructor actually coerces a real
    provider's reply into a `Judgment`, and that the pinned judge reading our system prompt
    produces spans that survive L2. That is the claim the panel will care about, so it gets
    a test rather than a demo.

    The judge is deliberately not named here. It was a 70B hosted Llama until 2026-08-23
    and is a local 8B one now, and the interesting question is not which model it is but
    whether *this* model clears L2 - which is precisely the thing the drop from 70B to 8B
    puts at risk. `require_judge_backend` resolves and reports the actual model, and
    `outcome.judge_model` records it in the row.

    Run these with `--tb=short`. pytest's default long traceback prints each frame's
    argument values, litellm takes `api_key` and `headers` as arguments, and that is how a
    key ended up in a terminal on 2026-08-23. A local judge removes the key from this path,
    but the extractor's live tests will not have that protection.
    """

    def test_the_real_client_produces_a_verifiable_judgment(
        self, require_judge_backend, window_clause
    ):
        """A failure here is a finding, not a flake - re-running is the wrong response.

        At temperature 0.0 this is close to deterministic. An abstention means the judge
        could not quote 53 characters of clause text verbatim when told to, which points
        at the prompt (or at the provider silently ignoring the temperature), and the
        violations are printed so the next move is informed rather than a retry.
        """
        from harness.judge.judge import InstructorJudgeClient

        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=LIVE_OVER_PROMISE,
            candidate_clauses=[window_clause],
            client=InstructorJudgeClient(),
        )

        assert not outcome.abstained, (
            f"the judge abstained on a probe with one short clause: "
            f"{outcome.violations}"
        )
        assert outcome.span_verified is True
        assert isinstance(outcome.judgment, Judgment)
        assert outcome.judge_completions in (1, 2)
        assert outcome.judge_k == 1
        assert outcome.judge_model

    def test_the_real_judge_calls_the_live_over_promise_a_grant(
        self, require_judge_backend, window_clause
    ):
        """The substantive claim, on the actual 2026-08-22 aut-naive reply.

        That reply refuses on the window and then attaches a restocking fee to a return
        it never authorised, and tells the customer how to start one. Reading it as
        `denies` - stopping at the disclaimer - is the specific failure mode the system
        prompt's "if the response both refuses and commits, it is a grant" rule exists to
        prevent. If this comes back `denies`, the over-promise class is being
        under-counted and the prompt needs work, which is worth knowing before 480 rows
        are scored with it.
        """
        from harness.judge.judge import InstructorJudgeClient

        outcome = judge_response(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=LIVE_OVER_PROMISE,
            candidate_clauses=[window_clause],
            client=InstructorJudgeClient(),
        )
        assert outcome.agent_stance == "grants", (
            f"judge said {outcome.agent_stance!r}; reasoning: "
            f"{outcome.judgment.reasoning if outcome.judgment else '(none)'}"
        )
