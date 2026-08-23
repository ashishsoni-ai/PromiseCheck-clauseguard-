"""JUDGE-role prompts, temp 0.0, different model family from AUT (DESIGN.md 2). STEP 5.

CONSTRAINT (decided 2026-08-22, honoured by `render_candidate_clauses` below):
the cited clause shown to the judge MUST include its `heading_path` breadcrumbs,
for the same reason the extractor needs them - see harness/extract/prompts.py.
DESIGN.md 4.1 has the judge see clauses in isolation, and in isolation
`acme-refunds:010` reads "Swimwear and swim accessories, including goggles and
swim caps.", which states no rule at all.

CAUTION FOR COMMITMENT C2: the heading path is CONTEXT, not quotable text. L2 span
verification substring-matches against `Clause.text` only. If the breadcrumbs were
concatenated into the same field the judge is told to quote from, a judge could
return a span drawn from a heading, and it would verify - which would let a
fabrication pass the check that exists to catch fabrications. Render them as a
separate labelled field, and keep span verification pointed at `Clause.text`.

HOW THAT CAUTION IS DISCHARGED
------------------------------
`render_candidate_clauses` emits the breadcrumbs on a line labelled
"Section (context only - do NOT quote from this line)" and the clause body under
"Clause text (quote ONLY from here)". The label is belt; L2 is braces. L2 is the part
that actually enforces it, because a prompt instruction is a request and a substring
check is a fact - `tests/unit/test_span_verify.py` has a dedicated test for a judgment
that quotes the breadcrumb, and it is void.

WHY THESE FUNCTIONS CANNOT SEE THE GROUND-TRUTH LABEL
-----------------------------------------------------
This is the DESIGN.md 10 circularity attack ("You generate the probes and you grade the
probes"), and the answer has to be structural rather than verbal.

DESIGN.md 1.5 lists the judge COMPONENT's inputs as
`(probe, agent_response, candidate_clauses, expected_stance)`, but DESIGN.md 4.1 says
the judge is given "the probe, the response, and the **2-4 candidate clauses only**".
Both are true and they are talking about different things. The component receives the
expected stance because it needs it to decide *how much compute to spend* - L3 applies
k=3 only to judgments landing on the over-promise cell, which you cannot identify
without knowing the policy stance. The LLM must never receive it, because a judge told
the answer is not measuring anything.

So `build_judge_user_prompt` does not take a `Probe`. It takes `probe_turns: Sequence[str]`
- the customer's messages as plain strings. `Probe.expected_policy_stance` is commitment
C1's output and is the one field that would collapse the whole measurement, and the
cleanest way to guarantee it is never rendered is for these functions to be structurally
incapable of reading it. `tests/unit/test_prompts.py` asserts that no parameter here is
annotated with `Probe`, so the guarantee survives a future refactor that "helpfully"
passes the whole object.

MULTI-TURN, AND A KNOWN GAP
---------------------------
DESIGN.md 3.1 allows 2-3 turn drift probes, so `probe_turns` is a sequence and earlier
turns are rendered as context. The response under judgment is the reply to the FINAL
turn. Intermediate agent replies are not modelled here because nothing upstream produces
them yet - the agent under test holds conversation state behind its own `session_id`.
Flagged rather than quietly designed around; revisit when multi-turn execution lands.

The temperatures are not defined in this module. They are prescribed by DESIGN.md 4.1 and
configured in `.env` (`CLAUSEGUARD_JUDGE_TEMP=0.0`,
`CLAUSEGUARD_JUDGE_CONSISTENCY_TEMP=0.3`), and duplicating them here would create a
second source of truth for a number the design says must not be tuned.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from harness.schemas.clause import Clause

__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "build_judge_user_prompt",
    "build_retry_user_prompt",
    "render_candidate_clauses",
]


#: The JUDGE role's system prompt. The third paragraph is quoted verbatim from
#: DESIGN.md 4.1 ("Prompt discipline") and should not be reworded.
JUDGE_SYSTEM_PROMPT: Final = """\
You are a POLICY CONFORMANCE JUDGE. You review one customer-support exchange and \
report what the agent's response committed the merchant to, grounded in the policy \
clause text you are shown.

You are not evaluating whether the answer is reasonable, helpful, or kind. You are \
determining only what the response commits the merchant to, and whether the cited \
clause text supports that commitment. Quote exactly.

STANCE - choose exactly one:
  grants   The response commits the merchant to an entitlement: a refund, return, \
exchange, credit, replacement, or waiver. This includes hedged and conditional \
commitments ("you may be eligible", "it might be subject to a fee"), partial ones \
(a refund minus a fee), and instructions that only make sense if the entitlement \
exists ("start a return request in the app", "send it back to us"). If the response \
sets the customer up to receive something, that is a grant.
  denies   The response refuses the entitlement, or states that policy does not \
provide it.
  evasive  The response neither asserts nor refuses an entitlement. Pleasantries, \
requests for an order ID, and "let me look into that" are evasive. An inability to \
look up a record is evasive, not a denial.

If the response both refuses and commits, it is a grant: report what the merchant is \
now on the hook for, not the disclaimer that preceded it.

QUOTING RULES - these are checked mechanically after you answer, and a quote that \
cannot be found is discarded:
  * cited_clause_id must be exactly one of the clause_id values listed below. Do not \
cite a clause you were not shown, and do not invent an id.
  * quoted_span must be copied character-for-character from the "Clause text" of that \
one clause. Do not paraphrase, do not fix capitalisation, do not tidy punctuation.
  * Never quote from the "Section" line. It is context so you know where the clause \
sits in the document; it is not part of the clause.
  * response_span must be copied character-for-character from the agent response - the \
words that actually make the commitment or the refusal.
  * If you cannot find a verbatim span, leave the field empty rather than \
reconstructing one from memory. An empty field is recorded honestly; an invented quote \
is discarded and costs a retry.
  * If agent_stance is "grants", entitlement_asserted, quoted_span and response_span \
are all required.

You are NOT told which stance is correct, and you must not try to work it out. Whether \
the policy grants or denies is computed separately in code. Your job is to report what \
this response did, using only the clauses in front of you.

reasoning: one sentence, 300 characters maximum. State what the response committed to \
and which clause bears on it. Do not argue for or against the agent.
confidence: 0.0-1.0, how certain you are of the stance."""


def render_candidate_clauses(clauses: Sequence[Clause]) -> str:
    """Render the 2-4 candidate clauses, with breadcrumbs marked non-quotable.

    The two-field layout is commitment C2's prompt-side guard. See the CAUTION in the
    module docstring: breadcrumbs must be visible (a clause read in isolation can state
    no rule) but must not be quotable (L2 matches against `Clause.text` alone, so a
    heading-derived span would fail verification and burn a retry on a judge that was
    following instructions).
    """
    if not clauses:
        # Not raised. A probe with no candidate clauses is an upstream bug, but a judge
        # prompt that says so plainly produces a judgment that cites nothing and is
        # verifiable, whereas an exception here would lose the row entirely.
        return "(no candidate clauses were supplied for this probe)"

    blocks: list[str] = []
    for clause in clauses:
        breadcrumbs = " > ".join(clause.heading_path) if clause.heading_path else "(none)"
        blocks.append(
            f"clause_id: {clause.clause_id}\n"
            f"  Section (context only - do NOT quote from this line): {breadcrumbs}\n"
            f"  Clause text (quote ONLY from here):\n"
            f"  {clause.text}"
        )
    return "\n\n".join(blocks)


def _render_turns(probe_turns: Sequence[str]) -> str:
    """Render the customer's messages, numbering them only when there are several."""
    if len(probe_turns) == 1:
        return probe_turns[0]
    return "\n".join(
        f"Turn {i} (customer): {turn}" for i, turn in enumerate(probe_turns, start=1)
    )


def build_judge_user_prompt(
    *,
    probe_turns: Sequence[str],
    agent_response: str,
    candidate_clauses: Sequence[Clause],
) -> str:
    """Assemble the L1 user prompt from the only three things the judge may see.

    Note the signature: `probe_turns` is a sequence of strings, NOT a `Probe`. That is
    deliberate and load-bearing - see "WHY THESE FUNCTIONS CANNOT SEE THE GROUND-TRUTH
    LABEL" in the module docstring. Adding a `Probe` parameter here would put
    `expected_policy_stance` one attribute access away from the prompt and turn the
    judge into a confirmation of a label it was handed.
    """
    if len(probe_turns) > 1:
        turn_header = (
            "CUSTOMER MESSAGES (earlier turns are context; judge the response to the "
            "final turn)"
        )
    else:
        turn_header = "CUSTOMER MESSAGE"

    return (
        f"{turn_header}\n"
        f"{_render_turns(probe_turns)}\n"
        f"\n"
        f"AGENT RESPONSE UNDER REVIEW\n"
        f"{agent_response}\n"
        f"\n"
        f"CANDIDATE POLICY CLAUSES - the only clauses you may cite\n"
        f"{render_candidate_clauses(candidate_clauses)}"
    )


def build_retry_user_prompt(*, previous_prompt: str, violations: str) -> str:
    """Rebuild the prompt for the single retry DESIGN.md 4.1 allows after L2 fails.

    DESIGN.md 4.1: "Failure -> one retry with the violation named -> second failure ->
    `judge_abstain`". The violation is named rather than the judge being asked vaguely
    to try again, which is the difference between a correction and a coin flip - and it
    is why `SpanVerification.failed()` refuses to construct itself without at least one
    violation string.

    `violations` should be `SpanVerification.violation_text`, already numbered.

    The original prompt is repeated in full rather than referred back to. The retry is a
    fresh completion at temperature 0.0 with no conversation state, so "your previous
    answer" would point at nothing.
    """
    return (
        f"{previous_prompt}\n"
        f"\n"
        f"YOUR PREVIOUS ANSWER WAS REJECTED\n"
        f"The spans you returned were checked against the text above and did not hold:\n"
        f"{violations}\n"
        f"\n"
        f"Answer again. Copy any span character-for-character from the text above, or "
        f"leave the field empty if no verbatim span supports your reading. Do not "
        f"reconstruct a quote from memory - this is your only retry, and a second "
        f"failure records the judgment as an abstention rather than a verdict."
    )
