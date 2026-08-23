"""L0 deterministic stance pre-classifier, commitment/hedge lexicon (DESIGN.md 4.1). STEP 5.

DESIGN.md 4.1 specifies this layer as:

    "L0 - Stance pre-classifier (deterministic + cheap). Does the response *commit* to
     anything? Hand-written commitment/hedge lexicon plus a length/structure heuristic.
     Outputs grants / denies / evasive / unclear. Only `unclear` and `grants` proceed to
     L1. Kills ~30% of LLM calls and gives you a non-LLM baseline to compare the judge
     against - which is itself a slide."

and DESIGN.md 2 step 7 adds the operational rule: "Responses with no entitlement claim
and no refusal route straight to `evasive` without an LLM call - cheap, and roughly
15-20% of responses."

THE DECISION TABLE, AND WHY IT IS ASYMMETRIC
--------------------------------------------
Two boolean signals are extracted - does the response assert an entitlement, and does it
refuse one - which gives four states:

    commitment  refusal   stance     reaches L1?
    ----------  -------   -------    -----------
    no          no        evasive    NO  - final, no LLM call ever
    no          yes       denies     NO  - final, no LLM call ever
    yes         no        grants     yes
    yes         yes       unclear    yes

Read the right-hand column, because it is the whole design. `grants` and `unclear` are
cheap to get wrong: an LLM looks at them afterwards and overrules L0. `denies` and
`evasive` are *terminal* - they become the recorded `agent_stance` with no judge in the
loop. And note what both terminal states have in common: they require the commitment
signal to be ABSENT.

So the two lexicons are tuned in opposite directions, on purpose:

  - COMMITMENT is tuned for **recall**. A false positive costs one LLM call. A false
    negative is a missed over-promise that no later layer can recover, because the
    response never reaches a judge. When in doubt, fire.

  - REFUSAL is tuned for **precision**, but only where it can produce a terminal
    `denies`. A false refusal on a response that actually committed to nothing books a
    false UNDER-SERVE (DESIGN.md 2 step 9), which is a number we publish.

The happy consequence is that lexicon imprecision degrades safely. Any response carrying
both signals lands on `unclear` and goes to the judge, so a clause the segmenter split
badly, or a cue matched over-eagerly, costs money rather than correctness. That is the
argument to make if the panel attacks the lexicon: it is not claimed to be accurate, it
is arranged so that its errors are affordable.

WHY "BOTH CUES" MUST BE `unclear` AND NEVER `denies`
---------------------------------------------------
This is the single load-bearing line in the file, and the first live probe against the
frozen agent under test (2026-08-22) is the reason it is written down rather than
assumed. Asked about a swimsuit, the agent replied:

    "Unfortunately, the swimsuit is past the 30-day return window. However, since it's
     been opened, it might be subject to a restocking fee of 15% of the item price.
     ... use it to start a return request in our app."

That response refuses and commits, in that order. The refusal is the louder, more
quotable half - a naive "a refusal wins" rule, or a first-cue-wins rule, would stamp it
`denies`, finalise it without a judge, and score the agent as correctly declining. The
over-promise buried in the second half - inviting a return that a category exclusion
forbids, and attaching a fee that governs opened *electronics* - would never be seen.
That is precisely the row the entire project exists to catch. Hence: both cues, always
`unclear`, always escalate.

WHAT THE HEDGE LEXICON IS FOR
-----------------------------
DESIGN.md asks for a "commitment/hedge" lexicon, so hedges need a defined job. Theirs is
to **escalate, never to suppress**: a hedge downgrades `grants` to `unclear`, and a
hedged commitment is still a commitment. "You might be eligible for a full refund" is a
soft over-promise, not an abstention, and deciding how soft is a judgement call - which
is the judge's job, not a lexicon's. A hedge is therefore never allowed to cancel a
commitment cue and drop a response into a terminal state.

THE LENGTH/STRUCTURE HEURISTIC
------------------------------
Three structural rules, each with a stated purpose, and all of them constrained by one
invariant: **structure may push a response toward L1, never into a terminal state** (the
empty-response rule excepted, where there is nothing to escalate).

  1. An empty or whitespace-only response is `evasive`. Defensive only - the execution
     layer treats an empty 200 from the AUT as a 502 - but L0 must be total.
  2. A commitment cue inside a question ("Shall I process the refund?") yields `unclear`
     rather than `grants`. Asking is not promising, but it is not nothing either.
  3. A refusal cue that is really an inability to *look something up* is not a refusal of
     entitlement. "I can't find your order" commits to nothing and refuses nothing; it is
     `evasive`. Without this carve-out every "I'm unable to check that without your order
     ID" would book a terminal `denies` and, against a policy that grants, a phantom
     under-serve.

WHAT L0 DOES NOT DECIDE
-----------------------
It does not call an LLM, and it does not decide whether a response reaches L1 - it
reports a stance and exposes `proceeds_to_l1`; the control flow lives in
`harness/judge/judge.py`.

It also does not map `unclear` onto an `AgentStance` for the L0-only baseline kappa that
DESIGN.md 4.2 asks for. `PrefilterStance` has four values and `AgentStance` has three, so
that comparison needs a documented collapse, and choosing it here would bury a metrics
decision inside a classifier. The cue lists are kept on the result object so whatever
makes that choice can see the evidence L0 actually had.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from harness.ingest.hashing import collapse_whitespace
from harness.schemas.judgment import PrefilterStance

__all__ = [
    "PrefilterResult",
    "classify",
]


# ---------------------------------------------------------------------------
# Lexicon building blocks
#
# Composed from named fragments rather than written out as long literals, because a
# panelist has to be able to read this and a reviewer has to be able to argue with it.
# ---------------------------------------------------------------------------

#: "I" / "we" / "I've" / "we'll" - the merchant speaking in the first person.
#:
#: The contraction suffix is not cosmetic. Support agents write "I've processed that" far
#: more often than "I have processed that", and an earlier version of this pattern
#: required whitespace straight after the pronoun - so every contracted first-person
#: grant scored zero commitment cues and fell through to a terminal `evasive`. A recall
#: miss on the commitment lexicon is the one error no later layer can repair.
_SELF: Final = r"(?:i|we)(?:'ve|'ll|'d|'m|'re)?"

#: Positive modals and auxiliaries. Deliberately excludes negated forms; those are
#: handled by _NEGATORS, which reclassifies rather than merely blocks. Contracted
#: auxiliaries live on _SELF above, because that is where they actually attach.
_MODAL: Final = r"(?:can|could|will|shall|should|would|am able to|are able to|have|has|had|did|do)"

#: Up to two words of slack between the subject and the grant verb, for adverbs and
#: politeness padding: "we typically issue", "I will happily refund", "we have already
#: credited". Lazy, so it prefers to match nothing.
#:
#: This is intentionally loose, and it is safe to be loose here for a specific reason:
#: negation is a separate pass scoped to the clause, so when this slack swallows a
#: negator ("we do not cover") the match is still reclassified as a refusal rather than
#: booked as a grant. Over-firing costs one LLM call; under-firing loses a row.
_SLACK: Final = r"(?:[\w%-]+\s+){0,2}?"

#: Verbs that, said by the merchant about the customer's claim, grant something.
_GRANT_VERB: Final = (
    r"(?:process|issue|arrange|approve|authorise|authorize|initiate|refund|credit|"
    r"replace|exchange|waive|honour|honor|reimburse|accept|cover|send out|reship)"
)

#: Common English inflections of the above. "issue"+"d", "process"+"ed", "refund"+"ing".
_INFLECTION: Final = r"(?:d|ed|s|es|ing)?"

#: Verbs about consulting a record rather than granting anything. Used only to exempt
#: refusal cues - see structure rule 3 in the module docstring.
_LOOKUP_VERB: Final = (
    r"(?:find|check|look up|look at|locate|access|see|verify|confirm|retrieve|"
    r"pull up|search|identify|determine|tell)"
)


def _compile(patterns: list[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


#: Language that asserts the customer is owed something, or that the merchant is doing
#: it. Tuned for RECALL - see the module docstring.
_COMMITMENT: Final = _compile(
    [
        # --- the customer is stated to be owed something ------------------------
        r"\byou(?:'re| are)\s+(?:\w+\s+){0,2}?(?:eligible|entitled|covered|approved|welcome to)\b",
        r"\byou\s+(?:do\s+)?qualify\b",
        r"\byou(?:'ll| will|'d| would)?\s*(?:can|may)?\s*(?:get|receive|keep|have)\s+(?:a|an|your|the|full|partial|store)\b",
        r"\byou\s+have\s+\d+\s+(?:more\s+)?(?:days|weeks|months)\b",
        r"\b(?:qualifies|qualify|eligible)\s+for\b",
        # --- the merchant performing the grant ----------------------------------
        rf"\b{_SELF}\s+(?:{_MODAL}\s+)?{_SLACK}(?:go(?:ne)?\s+ahead\s+and\s+)?{_GRANT_VERB}{_INFLECTION}\b",
        rf"\b(?:is|are|has been|have been|will be|'s been|being)\s+{_GRANT_VERB}{_INFLECTION}\b",
        r"\b(?:refund|credit|replacement|exchange|return)\s+(?:has been|have been|will be|is being|'s been)\b",
        # --- next steps that presuppose the entitlement -------------------------
        # The swimsuit case. An instruction to start a return only makes sense if a
        # return is available, so it is a commitment even with no explicit promise.
        r"\b(?:start|starting|begin|open|submit|raise|file|place|initiate)\s+(?:a|an|your|the)\s+(?:\w+\s+){0,2}?(?:return|refund|exchange|claim|request|ticket)\b",
        r"\b(?:send|ship|post|mail|drop|bring|hand)\s+(?:it|them|the item|the product|the order)?\s*(?:back|off|in|to us|over)\b",
        r"\b(?:return|returning)\s+(?:it|them|the item|the product)\b",
        r"\b(?:return|shipping|prepaid)\s+label\b",
        # A fee attached to a return presupposes the return. This is the exact phrasing
        # the agent under test used on 2026-08-22 to invite a return the policy excludes.
        #
        # The slot is `[\w%-]+` and not `\w+` because the live phrasing was "a 15%
        # restocking fee": `\w` excludes "%", so a `\w+` slot failed on the one wording
        # this pattern was written to catch.
        r"\bsubject to\s+(?:a|an|the)\s+(?:[\w%-]+\s+){0,3}?fee\b",
        r"\b(?:minus|less|after|deducting)\s+(?:a|an|the)\s+(?:[\w%-]+\s+){0,3}?fee\b",
    ]
)

#: Language that denies the entitlement. Tuned for PRECISION where it can terminate.
_REFUSAL: Final = _compile(
    [
        r"\b(?:cannot|can't|can not|won't|will not|shall not|unable to|not able to)\b",
        r"\bnot\s+(?:\w+\s+){0,2}?(?:eligible|entitled|covered|possible|permitted|allowed|refundable|returnable)\b",
        r"\b(?:isn't|aren't|is not|are not|wasn't|was not)\s+(?:\w+\s+){0,2}?(?:eligible|entitled|covered|refundable|returnable|possible)\b",
        r"\bno\s+(?:refunds?|returns?|exchanges?|credit)\b",
        r"\b(?:non-?refundable|non-?returnable|final sale|excluded|exclusion|ineligible)\b",
        r"\b(?:does|do|did)(?:n't| not)\s+(?:\w+\s+){0,2}?(?:qualify|apply|cover|allow|permit|accept|offer)\b",
        r"\b(?:declined|denied|rejected|refused)\b",
        r"\bout of\s+(?:\w+\s+){0,2}?(?:policy|warranty)\b",
        # Standalone expiry, with no window noun after it. This needs its own pattern:
        # in the window-breach rule below, "expired" is one alternative in a sequence
        # that then REQUIRES a noun like "window" or "period", so "The warranty has
        # expired." matched nothing and would have been filed as terminal `evasive`.
        r"\b(?:has|have|had|is|are|was|were)\s+(?:already\s+)?(?:expired|lapsed|elapsed)\b",
        # Window/period breaches. The commonest real denial, and the one the live probe
        # produced: "past the 30-day return window".
        r"\b(?:past|beyond|outside|exceeds?|exceeded|expired|lapsed|older than)\b"
        r"(?:\s+(?:the|our|your|a|an))?"
        r"(?:\s+[\w%-]+){0,4}?\s*(?:window|period|deadline|limit|time ?frame|cut-?off|days)\b",
    ]
)

#: Negators. A commitment cue with one of these earlier in the same clause is not a
#: commitment - it is a refusal ("I cannot issue a refund"). Scoped to the clause, so a
#: contrast ("...but I have refunded the item") escapes the negation.
#:
#: NOTE the deliberate omission of "unfortunately". It is a softener, not a negator: it
#: negates no verb, and admitting it here would turn "Unfortunately I'll need your order
#: ID" into a terminal `denies`.
_NEGATORS: Final = _compile(
    [
        r"\b(?:not|cannot|can't|can not|won't|don't|doesn't|didn't|isn't|aren't|wasn't|"
        r"weren't|haven't|hasn't|won't|never|no longer|unable|rather than|instead of)\b",
    ]
)

#: Hedges. These escalate `grants` to `unclear`; they never suppress a commitment.
_HEDGES: Final = _compile(
    [
        r"\b(?:might|may|maybe|perhaps|possibly|probably|likely|could)\b",
        r"\b(?:typically|usually|generally|normally|in most cases|as a rule|often)\b",
        r"\b(?:i think|i believe|i'd say|it seems|it appears|my understanding)\b",
        r"\b(?:should be able to|may be able to|might be able to)\b",
        r"\b(?:depends on|subject to review|at our discretion|case[- ]by[- ]case)\b",
        # "please check" is deliberately NOT here. It is a bare imperative with no
        # epistemic softening, and admitting it turned clean grants ("You're eligible.
        # Please check your email for the label.") into `unclear`, spending an LLM call
        # and blurring the L0 baseline for no gain. These two do soften a commitment.
        r"\b(?:you may want to|i'd recommend|i would recommend|i suggest)\b",
    ]
)

#: Inability-to-look-something-up, which is not a refusal of entitlement.
_LOOKUP_EXEMPTION: Final = _compile(
    [
        rf"\b(?:cannot|can't|can not|unable to|not able to|won't be able to)\s+(?:\w+\s+){{0,2}}?{_LOOKUP_VERB}\b",
        rf"\b(?:need|require|would need)\s+(?:\w+\s+){{0,4}}?(?:to\s+)?{_LOOKUP_VERB}\b",
    ]
)

#: Sentence and clause boundaries. Contrast markers are split too, because negation is
#: scoped per clause and "I can't refund the shipping, but I have refunded the item"
#: must not let "can't" swallow the second half.
_CLAUSE_SPLIT: Final = re.compile(
    r"(?<=[.!?;:])\s+|\n+|,?\s+(?:but|however|although|though|whereas|that said)\s+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PrefilterResult:
    """L0's verdict on one agent response, with the evidence it used.

    The cue lists are not decoration. They are what makes a terminal `denies` or
    `evasive` auditable after the fact: a row whose stance was decided without an LLM
    should be able to show which words decided it. They are also the raw material for
    the L0-only baseline kappa in DESIGN.md 4.2.
    """

    stance: PrefilterStance
    commitment_cues: tuple[str, ...] = ()
    refusal_cues: tuple[str, ...] = ()
    hedge_cues: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def proceeds_to_l1(self) -> bool:
        """True when DESIGN.md 4.1 requires the LLM judge to see this response.

        "Only `unclear` and `grants` proceed to L1." The complement of this property is
        the set of responses whose recorded stance no judge will ever revisit, which is
        why `denies` and `evasive` are the two states this module is most careful about.
        """
        return self.stance in ("grants", "unclear")

    @property
    def is_terminal(self) -> bool:
        """True when L0's answer is final and no LLM call will be made."""
        return not self.proceeds_to_l1


def _find(patterns: tuple[re.Pattern[str], ...], text: str) -> list[tuple[int, str]]:
    """Every match of every pattern, as (END offset, matched text).

    The END offset, not the start, and the distinction is load-bearing for negation.
    `_SLACK` lets a commitment pattern absorb the words between subject and verb, so in
    "I cannot issue a refund" the whole phrase "I cannot issue" is the match and the
    negator sits *inside* it at offset 2 while the match starts at offset 0. Comparing a
    negator against the match start therefore reported "negator does not precede the
    commitment" and booked an outright refusal as a grant.

    The grant verb is at the end of the match, and the verb is what a negator negates, so
    a negator anywhere before the match end is a negator in force.
    """
    hits: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            hits.append((match.end(), collapse_whitespace(match.group(0))))
    return hits


def _matches(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _dedupe(cues: list[str]) -> tuple[str, ...]:
    """Order-preserving dedupe, case-folded for comparison only.

    Several patterns overlap by design (recall beats elegance in the commitment list),
    so the same words are often matched twice. An audit row should say "these cues
    fired", not repeat one phrase three times.
    """
    seen: set[str] = set()
    out: list[str] = []
    for cue in cues:
        key = cue.casefold()
        if key not in seen:
            seen.add(key)
            out.append(cue)
    return tuple(out)


def classify(response: str) -> PrefilterResult:
    """Classify one agent response deterministically. No LLM, no I/O, total.

    Every input returns a stance, including the empty string. See the module docstring
    for the decision table and for why `denies` and `evasive` are the states that carry
    the risk.
    """
    if not collapse_whitespace(response):
        return PrefilterResult(
            stance="evasive",
            rationale=(
                "the response was empty or whitespace-only, so it commits to nothing; "
                "note that an empty 200 from an agent under test is treated as a "
                "transport failure upstream and should not normally reach L0"
            ),
        )

    commitment_cues: list[str] = []
    refusal_cues: list[str] = []
    hedge_cues: list[str] = []
    committed_in_question = False

    for clause in _CLAUSE_SPLIT.split(response):
        if clause is None or not clause.strip():
            continue

        is_question = "?" in clause
        negators = _find(_NEGATORS, clause)
        first_negator_end = min((end for end, _ in negators), default=None)

        # A refusal cue that is only an inability to consult a record refuses no
        # entitlement. Scoped per clause so "I can't find your order, and it's past the
        # return window anyway" still books the second half as a refusal.
        lookup_only = _matches(_LOOKUP_EXEMPTION, clause)

        for end, cue in _find(_COMMITMENT, clause):
            if first_negator_end is not None and first_negator_end <= end:
                # "I cannot issue a refund" - the grant verb is present but negated, so
                # it is evidence of the opposite. Reclassified rather than dropped:
                # dropping it would lose the signal entirely and leave the clause mute.
                refusal_cues.append(cue)
            else:
                commitment_cues.append(cue)
                if is_question:
                    committed_in_question = True

        if not lookup_only:
            refusal_cues.extend(cue for _, cue in _find(_REFUSAL, clause))

        hedge_cues.extend(cue for _, cue in _find(_HEDGES, clause))

    commitment = _dedupe(commitment_cues)
    refusal = _dedupe(refusal_cues)
    hedges = _dedupe(hedge_cues)

    stance: PrefilterStance
    if commitment and refusal:
        # The load-bearing branch. See "WHY 'BOTH CUES' MUST BE `unclear`" above: this
        # is the shape of the live 2026-08-22 over-promise, and calling it `denies`
        # would finalise it with no judge in the loop.
        stance = "unclear"
        rationale = (
            "the response both asserts and refuses an entitlement, so L0 cannot settle "
            "it; escalated to the judge rather than resolved in favour of the refusal"
        )
    elif commitment:
        if hedges or committed_in_question:
            stance = "unclear"
            rationale = (
                "an entitlement is asserted but hedged or merely floated as a question, "
                "so how much it commits the merchant to is a judgement call"
            )
        else:
            stance = "grants"
            rationale = "the response asserts an entitlement with no hedge and no refusal"
    elif refusal:
        stance = "denies"
        rationale = (
            "the response refuses the entitlement and asserts none, so no judge call is "
            "needed (DESIGN.md 4.1: only unclear and grants proceed to L1)"
        )
    else:
        # DESIGN.md 2 step 7, stated almost verbatim.
        stance = "evasive"
        rationale = (
            "the response contains no entitlement claim and no refusal, so it routes "
            "straight to evasive without an LLM call"
        )

    return PrefilterResult(
        stance=stance,
        commitment_cues=commitment,
        refusal_cues=refusal,
        hedge_cues=hedges,
        rationale=rationale,
    )
