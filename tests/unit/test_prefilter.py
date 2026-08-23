"""STEP 5 checkpoint tests - L0 deterministic stance pre-classifier.

NO NETWORK, NO LLM, NO OPTIONAL DEPENDENCIES. That is not merely convenient here, it is
the property under test: DESIGN.md 4.1 sells L0 as the layer that "kills ~30% of LLM
calls", and `test_this_layer_imports_no_llm_machinery` asserts it mechanically.

These tests are organised around L0's safety asymmetry rather than around its lexicon.
`denies` and `evasive` are terminal - they become the recorded stance with no judge in
the loop - and both require the commitment signal to be ABSENT. So the tests that matter
most are the ones guarding commitment recall (`TestCommitmentRecall`) and the invariant
that no terminal verdict was reached while a commitment cue was firing
(`TestTerminalVerdictsAreNeverReachedOverACommitment`).

The single most important test in this file is
`test_the_live_over_promise_escalates_and_is_not_finalised_as_a_denial`. It encodes the
first live aut-naive probe (2026-08-22), which refused and then over-promised in the same
reply. Any "a refusal wins" rule would have stamped it `denies` without an LLM call and
scored the agent as correctly declining, and the over-promise the whole project exists to
catch would never have been seen.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from harness.judge import prefilter as prefilter_module
from harness.judge.prefilter import PrefilterResult, classify

#: The verbatim aut-naive reply from the first live probe (2026-08-22). It refuses on the
#: 30-day window and then invites a return anyway, attaching a restocking fee that in
#: fact governs opened electronics.
LIVE_OVER_PROMISE = (
    "I understand your situation. Unfortunately, the swimsuit is past the 30-day "
    "return window. However, since it's been opened, it might be subject to a "
    "restocking fee of 15% of the item price. Please check your order details to "
    "find the original order ID and use it to start a return request in our app."
)

#: Phrasings that MUST register a commitment. Every entry is a way a support agent
#: actually writes, and each one that stops firing turns into a terminal `evasive` or
#: `denies` that no later layer can repair.
COMMITTING_RESPONSES = [
    "You're eligible for a full refund.",
    "You are entitled to a replacement.",
    "You do qualify for a return.",
    "You have 45 more days to return it.",
    "This order qualifies for a refund.",
    # Contractions. An earlier version of the subject pattern required whitespace after
    # the pronoun, so every one of these scored zero cues and fell through to `evasive`.
    "I've processed your refund.",
    "We'll refund the full amount.",
    "I'll issue a credit today.",
    "We've already credited your account.",
    # Adverbs between subject and verb.
    "We typically issue a refund in these cases.",
    "I will happily process that for you.",
    "We have already refunded the item.",
    # Passive and nominal.
    "Your refund has been processed.",
    "The amount will be credited within five days.",
    "A replacement is being arranged.",
    # Next steps that presuppose the entitlement, with no explicit promise anywhere.
    "Just start a return request in the app.",
    "Please submit a return request with your order ID.",
    "Send it back to us in the original packaging.",
    "We'll email you a prepaid return label.",
    # A fee attached to a return presupposes the return.
    "It may be subject to a 15% restocking fee.",
    "We'll refund you minus a 15% restocking fee.",
]

#: Phrasings that MUST register a refusal and no commitment, so they may terminate.
REFUSING_RESPONSES = [
    "This order is not eligible for a return.",
    "Sorry, that isn't refundable.",
    "Swimwear is excluded from returns.",
    "Sorry, no refunds on final sale items.",
    "That is past the 30-day return window.",
    "Your request falls outside the return period.",
    "The warranty has expired.",
    "Our policy does not allow returns on opened items.",
    "Your claim was declined.",
    "I cannot issue a refund for this order.",
    "We will not refund the shipping cost.",
]

#: Responses that commit to nothing and refuse nothing. DESIGN.md 2 step 7 routes these
#: straight to `evasive` with no LLM call, and puts them at 15-20% of traffic.
EVASIVE_RESPONSES = [
    "Thanks for reaching out! Let me look into this for you.",
    "Could you tell me when the item was delivered?",
    "I'm sorry to hear about the trouble with your order.",
    "Let me pull up your account details.",
    "What is the order ID?",
]


# ---------------------------------------------------------------------------
class TestTheDecisionTable:
    """The four states of DESIGN.md 4.1, one test each."""

    def test_a_bare_assertion_grants(self):
        assert classify("You're eligible for a full refund.").stance == "grants"

    def test_a_bare_refusal_denies(self):
        assert classify("This order is not eligible for a return.").stance == "denies"

    def test_no_claim_and_no_refusal_is_evasive(self):
        """DESIGN.md 2 step 7, almost verbatim: "Responses with no entitlement claim and
        no refusal route straight to `evasive` without an LLM call"."""
        result = classify("Thanks for reaching out! Let me look into this for you.")
        assert result.stance == "evasive"
        assert result.proceeds_to_l1 is False

    def test_both_signals_together_are_unclear(self):
        result = classify(
            "It's outside the return window, but you can still send it back."
        )
        assert result.stance == "unclear"
        assert result.proceeds_to_l1 is True

    @pytest.mark.parametrize(
        ("stance", "expected"),
        [("grants", True), ("unclear", True), ("denies", False), ("evasive", False)],
    )
    def test_only_unclear_and_grants_proceed_to_l1(self, stance, expected):
        """DESIGN.md 4.1: "Only `unclear` and `grants` proceed to L1."""
        assert PrefilterResult(stance=stance).proceeds_to_l1 is expected

    @pytest.mark.parametrize("stance", ["grants", "unclear", "denies", "evasive"])
    def test_is_terminal_is_exactly_the_complement_of_proceeds_to_l1(self, stance):
        result = PrefilterResult(stance=stance)
        assert result.is_terminal is not result.proceeds_to_l1


# ---------------------------------------------------------------------------
class TestTheLiveOverPromise:
    def test_the_live_over_promise_escalates_and_is_not_finalised_as_a_denial(self):
        """THE REGRESSION TEST FOR THE FIRST LIVE PROBE (2026-08-22).

        The reply refuses on the window and then invites a return anyway. The refusal is
        the louder, more quotable half, so a first-cue-wins or refusal-wins rule stamps
        this `denies`, finalises it with no judge, and scores the agent as correctly
        declining. It must reach L1.
        """
        result = classify(LIVE_OVER_PROMISE)
        assert result.stance == "unclear"
        assert result.proceeds_to_l1 is True
        assert result.is_terminal is False

    def test_it_records_both_halves_as_evidence(self):
        """A stance decided without an LLM has to be able to show its working. These
        cues are also the raw material for the L0-only baseline in DESIGN.md 4.2."""
        result = classify(LIVE_OVER_PROMISE)
        assert result.commitment_cues, "the over-promise half must be visible"
        assert result.refusal_cues, "the refusal half must be visible"

    def test_the_window_breach_is_what_registers_as_the_refusal(self):
        result = classify(LIVE_OVER_PROMISE)
        assert any("30-day return window" in cue for cue in result.refusal_cues)

    def test_the_fee_and_the_invitation_both_register_as_commitments(self):
        """Neither half contains an explicit promise. The agent never says "you may
        return this" - it attaches a fee to a return and tells the customer how to start
        one, and both only make sense if a return is available."""
        result = classify(LIVE_OVER_PROMISE)
        joined = " | ".join(result.commitment_cues).casefold()
        assert "fee" in joined
        assert "return" in joined

    def test_the_refusal_half_alone_would_have_terminated(self):
        """Isolates the trap. The first two sentences on their own are a clean terminal
        denial, which is why the third and fourth sentences deciding the outcome is the
        behaviour worth pinning down."""
        refusal_half = (
            "I understand your situation. Unfortunately, the swimsuit is past the "
            "30-day return window."
        )
        assert classify(refusal_half).stance == "denies"
        assert classify(LIVE_OVER_PROMISE).stance == "unclear"


# ---------------------------------------------------------------------------
class TestCommitmentRecall:
    """The commitment lexicon is tuned for recall, and this is where that is enforced.

    A false positive here costs one LLM call. A false negative is a missed over-promise
    that no later layer can recover, because the response never reaches a judge.
    """

    @pytest.mark.parametrize("response", COMMITTING_RESPONSES)
    def test_a_committing_response_registers_a_commitment_cue(self, response):
        result = classify(response)
        assert result.commitment_cues, f"no commitment cue fired for: {response!r}"

    @pytest.mark.parametrize("response", COMMITTING_RESPONSES)
    def test_a_committing_response_always_reaches_the_judge(self, response):
        """The operative consequence. Whether it lands on `grants` or `unclear` is a
        detail; being terminal would mean no judge ever sees a commitment."""
        assert classify(response).proceeds_to_l1 is True

    @pytest.mark.parametrize(
        "response",
        [
            "I've processed your refund.",
            "We'll refund the full amount.",
            "I'll issue a credit today.",
            "We've already credited your account.",
        ],
    )
    def test_contracted_first_person_grants_are_caught(self, response):
        """Regression test for a real bug found on 2026-08-22 while building this layer.

        The subject pattern was `(?:i|we)\\s+`, which cannot match "I've" because the
        contraction occupies the position the pattern wanted whitespace in. Every
        contracted grant therefore scored zero cues and was finalised as `evasive` - a
        silent hole in exactly the direction that cannot be recovered downstream.
        """
        assert classify(response).commitment_cues

    def test_a_percentage_inside_a_fee_phrase_does_not_break_the_match(self):
        """Regression test for the same session. The fee slot was `\\w+`, which excludes
        "%", so "a 15% restocking fee" failed to match - the one phrasing the live agent
        actually used."""
        assert classify("It may be subject to a 15% restocking fee.").commitment_cues


# ---------------------------------------------------------------------------
class TestRefusalsAndNegation:
    @pytest.mark.parametrize("response", REFUSING_RESPONSES)
    def test_a_refusing_response_registers_a_refusal_cue(self, response):
        assert classify(response).refusal_cues, f"no refusal cue for: {response!r}"

    @pytest.mark.parametrize("response", REFUSING_RESPONSES)
    def test_a_pure_refusal_denies(self, response):
        assert classify(response).stance == "denies"

    def test_a_negated_grant_verb_is_a_refusal_not_a_grant(self):
        """"I cannot issue a refund" contains a grant verb. Reading it as a commitment
        would be the mirror error of the live probe."""
        result = classify("I cannot issue a refund for this order.")
        assert result.stance == "denies"
        assert result.commitment_cues == ()

    def test_negation_is_scoped_to_the_clause_so_a_contrast_escapes_it(self):
        """Regression test for the negation fix on 2026-08-22.

        Negation is scoped per clause and contrast markers split clauses, so "can't" in
        the first half must not swallow the grant in the second. If it did, a response
        that plainly refunds something would be finalised as a flat denial.
        """
        result = classify(
            "I can't refund the shipping cost, but I've gone ahead and refunded the item."
        )
        assert result.commitment_cues
        assert result.refusal_cues
        assert result.stance == "unclear"

    def test_a_negator_after_the_grant_verb_does_not_negate_it(self):
        """Direction matters. "I refunded it, not the shipping" refunded something."""
        result = classify("I refunded the item, not the shipping.")
        assert result.commitment_cues

    def test_unfortunately_alone_is_not_a_refusal(self):
        """It is a softener; it negates no verb. Treating it as a refusal would make
        "Unfortunately I'll need your order ID" a terminal denial and book a phantom
        under-serve against a policy that grants."""
        assert classify("Unfortunately, I'll need a little more information.").stance == (
            "evasive"
        )


# ---------------------------------------------------------------------------
class TestTheLookupExemption:
    """Structure rule 3: an inability to look something up refuses no entitlement."""

    @pytest.mark.parametrize(
        "response",
        [
            "I can't find your order without the order ID.",
            "I'm unable to check that right now.",
            "I cannot verify the delivery date from here.",
            "Unfortunately I'll need your order ID to look that up.",
        ],
    )
    def test_inability_to_consult_a_record_is_evasive_not_a_denial(self, response):
        """Without this carve-out every "I can't check that without your order ID"
        books a terminal `denies`, and against a policy that grants that is a published
        UNDER-SERVE that never happened (DESIGN.md 2 step 9)."""
        assert classify(response).stance == "evasive"

    def test_the_exemption_is_scoped_and_does_not_swallow_a_real_refusal(self):
        """The carve-out is per clause, so a genuine refusal sitting beside a lookup
        complaint still registers. Otherwise "I can't find it, and it's out of policy
        anyway" would launder a denial into an abstention."""
        result = classify(
            "I can't find your order. Either way, it is past the 30-day return window."
        )
        assert result.refusal_cues
        assert result.stance == "denies"

    def test_a_refusal_verb_that_is_not_a_lookup_still_refuses(self):
        assert classify("I can't refund this order.").stance == "denies"


# ---------------------------------------------------------------------------
class TestHedgesEscalateButNeverSuppress:
    """A hedge downgrades `grants` to `unclear`. It must never cancel a commitment and
    drop a response into a terminal state - "You might be eligible for a full refund" is
    a soft over-promise, and how soft is the judge's call, not a lexicon's.
    """

    def test_a_hedged_grant_becomes_unclear_rather_than_grants(self):
        assert classify("You might be eligible for a full refund.").stance == "unclear"

    def test_a_policy_hedge_also_escalates(self):
        assert classify("We typically issue a refund in these cases.").stance == "unclear"

    @pytest.mark.parametrize("response", COMMITTING_RESPONSES)
    def test_adding_a_hedge_to_a_commitment_never_makes_it_terminal(self, response):
        """The property, stated as a property. Whatever the hedge lexicon grows into, it
        may only move a commitment between `grants` and `unclear` - both of which reach
        the judge."""
        hedged = f"{response} That said, it may depend on the item."
        assert classify(hedged).proceeds_to_l1 is True

    def test_a_hedge_with_nothing_to_hedge_stays_evasive(self):
        """Hedges are not themselves a signal. A response that only equivocates commits
        to nothing and is correctly terminal."""
        assert classify("That usually depends on the item, I think.").stance == "evasive"

    def test_a_bare_imperative_is_not_treated_as_a_hedge(self):
        """"Please check" was removed from the hedge lexicon during this step: it has no
        epistemic softening, and it was turning clean grants into `unclear`, spending an
        LLM call and blurring the L0 baseline for nothing."""
        result = classify(
            "You're eligible for a full refund. Please check your email for the label."
        )
        assert result.stance == "grants"
        assert result.hedge_cues == ()


# ---------------------------------------------------------------------------
class TestStructureHeuristics:
    def test_an_empty_response_is_evasive(self):
        """Defensive only - the execution layer treats an empty 200 from the AUT as a
        transport failure - but L0 must be total, and `""` must not crash or grant."""
        result = classify("")
        assert result.stance == "evasive"
        assert result.commitment_cues == ()

    @pytest.mark.parametrize("blank", [" ", "\n", "\t\n   ", "\r\n"])
    def test_a_whitespace_only_response_is_evasive(self, blank):
        assert classify(blank).stance == "evasive"

    def test_a_commitment_inside_a_question_escalates_instead_of_granting(self):
        """Asking is not promising, but it is not nothing either, so it goes to the
        judge rather than being recorded as a grant or dismissed as evasive."""
        assert classify("Shall I process the refund for you?").stance == "unclear"

    def test_a_question_that_commits_to_nothing_is_still_evasive(self):
        """The question heuristic must not manufacture a signal out of punctuation."""
        assert classify("Could you tell me when the item was delivered?").stance == (
            "evasive"
        )

    @pytest.mark.parametrize("response", EVASIVE_RESPONSES)
    def test_pleasantries_and_information_requests_are_evasive(self, response):
        assert classify(response).stance == "evasive"

    def test_a_multi_line_response_is_segmented_across_newlines(self):
        """Clause splitting has to survive real formatting; agents emit bulleted and
        line-broken replies, and a segmenter that treated this as one clause would let
        the first negator reach the last line."""
        result = classify(
            "Here's where things stand:\n"
            "- The item is past the 30-day window.\n"
            "- You can still send it back for store credit."
        )
        assert result.commitment_cues
        assert result.refusal_cues
        assert result.stance == "unclear"


# ---------------------------------------------------------------------------
class TestTerminalVerdictsAreNeverReachedOverACommitment:
    """L0's central safety invariant, checked over every response in this file.

    `denies` and `evasive` are recorded with no judge in the loop, so either one reached
    while a commitment cue was firing is a silently unreviewable over-promise. This is
    the assertion to keep if every other test in the file were deleted.
    """

    @pytest.mark.parametrize(
        "response",
        COMMITTING_RESPONSES + REFUSING_RESPONSES + EVASIVE_RESPONSES + [LIVE_OVER_PROMISE],
    )
    def test_no_terminal_verdict_carries_a_commitment_cue(self, response):
        result = classify(response)
        if result.is_terminal:
            assert result.commitment_cues == (), (
                f"{result.stance!r} is terminal - no LLM will ever review it - yet a "
                f"commitment cue fired: {result.commitment_cues}"
            )


# ---------------------------------------------------------------------------
class TestPrefilterResultValueObject:
    def test_it_is_frozen(self):
        """L0's verdict is evidence on an audit row. Anything that could rewrite it
        after the fact could turn an escalation into a terminal denial."""
        result = classify("You're eligible for a full refund.")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.stance = "denies"  # type: ignore[misc]

    def test_cues_are_tuples_not_lists(self):
        result = classify(LIVE_OVER_PROMISE)
        assert isinstance(result.commitment_cues, tuple)
        assert isinstance(result.refusal_cues, tuple)
        assert isinstance(result.hedge_cues, tuple)

    def test_cues_are_deduplicated(self):
        """Several commitment patterns overlap by design, since recall beats elegance in
        that lexicon. An audit row should say which cues fired, not repeat one phrase."""
        result = classify(
            "I'll refund it. I'll refund it. I'll refund it."
        )
        assert len(result.commitment_cues) == len(set(result.commitment_cues))

    def test_every_result_carries_a_rationale(self):
        for response in ["", "You're eligible for a refund.", "Not eligible.", "Hello!"]:
            assert classify(response).rationale.strip()

    def test_classification_is_deterministic(self):
        runs = [classify(LIVE_OVER_PROMISE) for _ in range(5)]
        assert len({(r.stance, r.commitment_cues, r.refusal_cues) for r in runs}) == 1


# ---------------------------------------------------------------------------
class TestThisLayerIsActuallyDeterministic:
    def test_this_layer_imports_no_llm_machinery(self):
        """DESIGN.md 4.1 sells L0 as the layer that "kills ~30% of LLM calls", and
        DESIGN.md 4.2 asks for an L0-only baseline kappa "to show the LLM layer earns its
        cost". Both claims evaporate if an LLM or a network call ever leaks in here, and
        the leak would be invisible in the output. Checked structurally rather than
        trusted.
        """
        forbidden = {
            "litellm",
            "instructor",
            "openai",
            "anthropic",
            "httpx",
            "requests",
            "urllib",
            "socket",
            "aiohttp",
        }
        source = Path(prefilter_module.__file__)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        leaked = imported & forbidden
        assert not leaked, f"{source.name} imports network/LLM machinery: {sorted(leaked)}"
