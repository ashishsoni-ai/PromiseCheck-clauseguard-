"""STEP 5 checkpoint tests - L2 span verification, commitment C2.

L2 is the layer DESIGN.md 4.1 calls "deterministic, non-negotiable", so these tests are
about mechanical properties rather than model behaviour. NO NETWORK, NO LLM, NO OPTIONAL
DEPENDENCIES.

The single most important test in this file is
`test_a_real_span_from_the_wrong_clause_is_void`. It encodes the failure the first live
aut-naive probe actually produced on 2026-08-22: asked about a swimsuit, the agent
quoted a genuine 15%-restocking-fee clause that governs *opened electronics*. The span
was real. Any check that merely asked "does this text appear somewhere in the policy?"
would have passed it and the over-promise would have been scored as well-evidenced. C2
only bites because the quote is anchored to a specific cited clause, and the cited
clause is anchored to the candidate set the judge was shown.
"""

from __future__ import annotations

import dataclasses

import pytest

from harness.judge.span_verify import (
    SpanVerification,
    contains_verbatim,
    verify_judgment,
)
from harness.schemas import Judgment

WINDOW_TEXT = "Returns must be initiated within 30 days of delivery."
SWIM_TEXT = "Swimwear and swim accessories, including goggles and swim caps."

#: The verbatim aut-naive reply from the first live probe (2026-08-22).
REAL_RESPONSE = (
    "I understand your situation. Unfortunately, the swimsuit is past the 30-day "
    "return window. However, since it's been opened, it might be subject to a "
    "restocking fee of 15% of the item price. Please check your order details to "
    "find the original order ID and use it to start a return request in our app."
)


@pytest.fixture
def window_clause(make_clause):
    return make_clause(
        text=WINDOW_TEXT,
        ordinal=7,
        content_hash="7e1a0b44",
        heading_path=["Acme Retail", "4. Return window"],
    )


@pytest.fixture
def swim_clause(make_clause):
    """The clause whose text, in isolation, states no rule at all.

    This is the example named in harness/judge/prompts.py: read alone it is a bare
    noun phrase, which is why the judge is shown `heading_path` as context - and why
    the heading must never be quotable.
    """
    return make_clause(
        text=SWIM_TEXT,
        ordinal=10,
        content_hash="10c5d9f0",
        heading_path=["Acme Retail", "5. Categories excluded from return"],
    )


@pytest.fixture
def candidates(window_clause, swim_clause):
    return [window_clause, swim_clause]


# ---------------------------------------------------------------------------
class TestContainsVerbatim:
    """The substring primitive. Layout folds; nothing else does."""

    def test_an_exact_substring_matches(self):
        assert contains_verbatim(WINDOW_TEXT, "within 30 days of delivery")

    def test_a_line_wrapped_clause_matches_a_one_line_quote(self):
        """Clause text carries newlines wherever the source document wrapped. A
        judge asked to quote across a wrap returns a single space, and refusing
        that would abstain on honest quotes."""
        wrapped = "Returns must be initiated\nwithin 30 days\n  of delivery."
        assert contains_verbatim(wrapped, "initiated within 30 days of delivery")

    def test_a_quote_containing_a_newline_matches_unwrapped_text(self):
        assert contains_verbatim(WINDOW_TEXT, "initiated\nwithin 30 days")

    @pytest.mark.parametrize("empty", ["", " ", "\n", "\t\n   ", "\r\n"])
    def test_an_empty_or_whitespace_only_span_never_verifies(self, empty):
        """`"" in anything` is True in Python, so a naive check would report an
        empty quote as verified - C2 silently switched off, indistinguishable from
        a healthy row in every published metric."""
        assert contains_verbatim(WINDOW_TEXT, empty) is False

    def test_case_is_not_folded(self):
        """`hashing.normalize` casefolds, which is right for hashing and fatal
        here: it would let a re-cased paraphrase pass as verbatim."""
        assert contains_verbatim(WINDOW_TEXT, "RETURNS MUST BE INITIATED") is False

    def test_punctuation_is_not_folded(self):
        """`normalize` maps punctuation to spaces. Under it, a hyphenated
        mangling of a clause would verify."""
        assert contains_verbatim(SWIM_TEXT, "goggles-and-swim-caps") is False

    def test_a_paraphrase_does_not_verify(self):
        assert contains_verbatim(WINDOW_TEXT, "returns are allowed for a month") is False

    def test_a_non_breaking_space_folds_like_ordinary_whitespace(self):
        """U+00A0 is whitespace to `str.split`, so PDF-extracted clauses match
        quotes typed with an ordinary space. This is the permissive direction and
        it is intended.

        Written as an escape, not a literal: a test whose subject is an invisible
        character must not depend on that character surviving an editor.
        """
        assert contains_verbatim("within\u00a030 days", "within 30 days")

    def test_a_zero_width_space_does_not_fold(self):
        """U+200B is a format character, not whitespace, so it survives and the
        quote fails. Conservative in the safe direction - it abstains rather than
        accepting - and recorded here so the behaviour is chosen rather than
        discovered later."""
        assert contains_verbatim("within\u200b30 days", "within30 days") is False

    def test_full_width_digits_do_not_match_ascii_digits(self):
        """No NFKC at this layer, deliberately: NFKC is part of `normalize`, whose
        docstring says it is lossy and never for span checks."""
        assert contains_verbatim("within \uff13\uff10 days", "within 30 days") is False


# ---------------------------------------------------------------------------
class TestCitedClauseMustBeInTheCandidateSet:
    def test_a_clause_id_outside_the_candidate_set_is_void(self, candidates):
        judgment = Judgment(
            agent_stance="denies",
            cited_clause_id="acme-refunds:041:deadbeef",
            quoted_span="restocking fee of 15% of the item price",
            reasoning="cites a clause the judge was never shown",
            confidence=0.8,
        )
        result = verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        )
        assert result.ok is False
        assert any("not one of the candidate clauses" in v for v in result.violations)

    def test_an_unknown_clause_id_produces_exactly_one_violation(self, candidates):
        """One mistake, one complaint. DESIGN.md 4.1 requires the retry to name the
        violation; a retry prompt listing two violations for a single root cause
        invites the judge to correct the wrong half."""
        judgment = Judgment(
            agent_stance="denies",
            cited_clause_id="acme-refunds:041:deadbeef",
            quoted_span="anything at all",
            reasoning="one root cause",
            confidence=0.8,
        )
        result = verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        )
        assert len(result.violations) == 1

    def test_an_empty_candidate_set_is_reported_readably(self):
        judgment = Judgment(
            agent_stance="denies",
            cited_clause_id="acme-refunds:007:7e1a0b44",
            reasoning="no candidates supplied",
            confidence=0.5,
        )
        result = verify_judgment(
            judgment, candidate_clauses=[], agent_response=REAL_RESPONSE
        )
        assert result.ok is False
        assert "none were supplied" in result.violation_text

    def test_citing_nothing_is_legal_for_a_denial(self, candidates):
        judgment = Judgment(
            agent_stance="denies",
            reasoning="a refusal need not rest on a quoted clause",
            confidence=0.6,
        )
        result = verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        )
        assert result.ok is True
        assert result.violations == ()


# ---------------------------------------------------------------------------
class TestQuotedSpanIsCheckedAgainstClauseTextOnly:
    def test_a_verbatim_quote_from_the_cited_clause_verifies(self, candidates):
        judgment = Judgment(
            agent_stance="denies",
            cited_clause_id="acme-refunds:007:7e1a0b44",
            quoted_span="within 30 days of delivery",
            response_span="past the 30-day return window",
            reasoning="correctly grounded refusal",
            confidence=0.93,
        )
        assert verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        ).ok

    def test_a_span_drawn_from_the_heading_path_is_void(self, candidates):
        """The whole point of the CAUTION in harness/judge/prompts.py. The
        breadcrumbs are shown to the judge because `swim_clause` states no rule in
        isolation, but if they were quotable a heading-derived span would verify
        and a fabrication would pass the check built to catch fabrications."""
        judgment = Judgment(
            agent_stance="denies",
            cited_clause_id="acme-refunds:010:10c5d9f0",
            quoted_span="5. Categories excluded from return",
            reasoning="quoted the breadcrumb, not the clause",
            confidence=0.7,
        )
        result = verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        )
        assert result.ok is False
        assert any("not found verbatim in clause" in v for v in result.violations)

    def test_a_quote_from_a_different_candidate_clause_is_void(self, candidates):
        """Right policy, right candidate set, wrong pairing. The text exists in
        `window_clause` but the judgment cited `swim_clause`."""
        judgment = Judgment(
            agent_stance="denies",
            cited_clause_id="acme-refunds:010:10c5d9f0",
            quoted_span="within 30 days of delivery",
            reasoning="quote and citation disagree",
            confidence=0.7,
        )
        assert not verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        ).ok

    def test_a_real_span_from_the_wrong_clause_is_void(self, candidates):
        """THE REGRESSION TEST FOR THE FIRST LIVE PROBE (2026-08-22).

        aut-naive answered a swimwear question by applying a real 15%-restocking-fee
        clause that governs opened electronics. The fee clause is not in the
        candidate set for this probe - L1 is given only "the 2-4 candidate clauses
        (the ones the probe was constructed from, plus their exception parents)" -
        so a judgment resting on it cannot be verified, and is void rather than
        counted as evidence.
        """
        judgment = Judgment(
            agent_stance="grants",
            entitlement_asserted="return subject to a 15% restocking fee",
            cited_clause_id="acme-refunds:041:0e17c3aa",
            quoted_span="restocking fee of 15% of the item price",
            response_span="restocking fee of 15% of the item price",
            reasoning="real span, real policy, wrong clause and wrong category",
            confidence=0.88,
        )
        result = verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        )
        assert result.ok is False
        assert any("not one of the candidate clauses" in v for v in result.violations)


# ---------------------------------------------------------------------------
class TestResponseSpanIsCheckedAgainstTheResponse:
    def test_a_verbatim_response_span_verifies(self, candidates):
        judgment = Judgment(
            agent_stance="denies",
            cited_clause_id="acme-refunds:007:7e1a0b44",
            quoted_span="within 30 days of delivery",
            response_span="it might be subject to a restocking fee",
            reasoning="quoted the agent exactly",
            confidence=0.9,
        )
        assert verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        ).ok

    def test_a_response_span_the_agent_never_wrote_is_void(self, candidates):
        judgment = Judgment(
            agent_stance="denies",
            cited_clause_id="acme-refunds:007:7e1a0b44",
            quoted_span="within 30 days of delivery",
            response_span="you are entitled to a full refund",
            reasoning="describes a response nobody sent",
            confidence=0.9,
        )
        result = verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        )
        assert result.ok is False
        assert any("not found verbatim in the agent response" in v for v in result.violations)


# ---------------------------------------------------------------------------
class TestAGrantMustBeEvidenced:
    """DESIGN.md does not spell this out; it follows from C2 plus DESIGN.md 5.2
    item 4, and `span_verify.verify_judgment` documents it as an inference. `grants`
    is the stance that reaches the over-promise cell, and an unevidenced
    over-promise is a claim about a merchant's liability the harness cannot
    substantiate - so it is routed to retry and then into the published abstain
    rate rather than into the headline number.
    """

    def test_a_grant_without_a_quoted_span_is_void(self, candidates):
        judgment = Judgment(
            agent_stance="grants",
            entitlement_asserted="fee-based return",
            response_span="start a return request in our app",
            reasoning="no clause quoted",
            confidence=0.8,
        )
        result = verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        )
        assert result.ok is False
        assert any("no quoted_span was given" in v for v in result.violations)

    def test_a_grant_without_a_response_span_is_void(self, candidates):
        judgment = Judgment(
            agent_stance="grants",
            entitlement_asserted="fee-based return",
            cited_clause_id="acme-refunds:010:10c5d9f0",
            quoted_span="Swimwear and swim accessories",
            reasoning="no committing words quoted",
            confidence=0.8,
        )
        result = verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        )
        assert result.ok is False
        assert any("no response_span was given" in v for v in result.violations)

    def test_a_fully_evidenced_grant_passes(self, candidates):
        judgment = Judgment(
            agent_stance="grants",
            entitlement_asserted="return subject to a restocking fee",
            cited_clause_id="acme-refunds:010:10c5d9f0",
            quoted_span="Swimwear and swim accessories",
            response_span="use it to start a return request in our app",
            reasoning="agent invited a return the exclusion forbids",
            confidence=0.9,
        )
        assert verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        ).ok

    @pytest.mark.parametrize("stance", ["denies", "evasive"])
    def test_non_grant_stances_may_omit_spans(self, candidates, stance):
        """The asymmetry mirrors Judgment's own validators: a response that
        commits to nothing has nothing to evidence."""
        judgment = Judgment(
            agent_stance=stance,
            reasoning="committed to nothing",
            confidence=0.5,
        )
        assert verify_judgment(
            judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
        ).ok


# ---------------------------------------------------------------------------
class TestTheConftestJudgmentFixtureIsVerifiable:
    def test_the_shared_sample_judgment_verifies_against_its_clause(
        self, sample_judgment, sample_clause
    ):
        """Guards the fixture itself. `sample_judgment` is reused by Step 1's
        schema tests and by later steps; if it ever stops being span-verifiable,
        every downstream test that treats it as a well-formed judgment is quietly
        testing an impossible object."""
        response = "Sure - since you're within our returns window I can refund that."
        result = verify_judgment(
            sample_judgment,
            candidate_clauses=[sample_clause],
            agent_response=response,
        )
        assert result.ok is True, result.violation_text


# ---------------------------------------------------------------------------
class TestSpanVerificationValueObject:
    def test_passed_has_no_violations_and_renders_empty(self):
        result = SpanVerification.passed()
        assert result.ok is True
        assert result.violations == ()
        assert result.violation_text == ""

    def test_failed_requires_at_least_one_violation(self):
        """A failure the retry prompt cannot name is not actionable, and would
        produce a bare "try again" - which DESIGN.md 4.1 specifically rules out by
        requiring the retry to name the violation."""
        with pytest.raises(ValueError, match="at least one violation"):
            SpanVerification.failed([])

    def test_violations_are_numbered_in_order_for_the_retry_prompt(self):
        result = SpanVerification.failed(["first problem", "second problem"])
        assert result.violation_text == "1. first problem\n2. second problem"

    def test_it_is_frozen(self):
        """L2's verdict is evidence. Anything downstream that could rewrite it
        could launder a void judgment into a verified one."""
        result = SpanVerification.passed()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.ok = False  # type: ignore[misc]

    def test_verification_is_deterministic(self, candidates):
        judgment = Judgment(
            agent_stance="grants",
            entitlement_asserted="fee-based return",
            cited_clause_id="acme-refunds:010:10c5d9f0",
            quoted_span="Swimwear and swim accessories",
            response_span="use it to start a return request in our app",
            reasoning="same inputs must give the same answer, always",
            confidence=0.9,
        )
        runs = [
            verify_judgment(
                judgment, candidate_clauses=candidates, agent_response=REAL_RESPONSE
            )
            for _ in range(5)
        ]
        assert len({(r.ok, r.violations) for r in runs}) == 1
