"""L2 span verification - exact substring after normalization, retry-once-then-abstain. Commitment C2. STEP 5.

This module is the mechanical teeth of commitment C2 (DESIGN.md 0):

    "The judge must quote a span that literally exists in the cited clause.
     Exact-substring verification after normalisation. If the quote isn't in the
     clause, the judgment is void and retried, then abstained. This converts
     'trust the LLM judge' into a falsifiable mechanical check on every single
     row."

DESIGN.md 4.1 calls L2 "deterministic, non-negotiable". There is no LLM in this file
and there must never be one: the whole value of the layer is that it cannot be talked
out of a verdict. Everything here is pure and total - same inputs, same answer, no I/O.

WHY `collapse_whitespace` AND NOT `normalize`
---------------------------------------------
`harness.ingest.hashing` exposes both, and picking the wrong one hollows out C2 without
failing a single test. `normalize()` says so itself: "Lossy by design; never for span
checks." It casefolds and replaces every Unicode punctuation character with a space,
because for *hashing* the question is "did this clause's meaning change?" and editorial
punctuation edits should not count.

For span verification the question is the opposite one - "did the judge invent this?" -
and under `normalize()` these would all verify against a clause reading
"Refunds are not available after 30 days":

    "refunds are not available after 30 days"   <- casefolded, so a quote that was
                                                   never in the document passes
    "refunds are  not  available"               <- fine, but only by accident
    "REFUNDS-ARE-NOT-AVAILABLE-AFTER-30-DAYS"   <- punctuation to spaces, then collapsed

A judge that fabricates a confident-sounding paraphrase and mangles the casing would
sail through the check that exists to catch fabrication. So L2 uses
`collapse_whitespace`, which DESIGN.md 4.1 authorises with the words "after whitespace
normalisation" and which `hashing.py` documents as "the ONLY normalisation permitted
before commitment C2's exact substring check". It is information-preserving apart from
layout, so a quote that survives it is still verbatim.

The one thing it must fold is line wrapping. Clause text arrives from `ingest` with
newlines wherever the source document wrapped, and an LLM asked to quote a sentence
spanning a wrap will return it with a single space. Refusing that would abstain on
correct quotes, which raises the abstain rate for a reason that has nothing to do with
honesty.

THE EMPTY-SPAN HOLE
-------------------
`Judgment.quoted_span` is typed `str | None` with no minimum length, and in Python
`"" in anything` is True. An empty or whitespace-only span would therefore *pass* a
naive substring check - not because it was verified, but because nothing was checked.
That is C2 silently switched off, and it would look identical to a healthy row in every
metric. Empty spans are rejected explicitly below.

A degenerate-but-nonempty span ("a", or a single comma) is a weaker version of the same
problem: it verifies, and it evidences nothing. DESIGN.md sets no minimum quote length,
so this module does not invent a threshold - inventing one would silently change the
reported span-verification failure rate, which DESIGN.md 4.2 asks us to publish. It is
flagged here as a known judgement call rather than quietly legislated.

WHAT L2 DELIBERATELY DOES NOT DECIDE
------------------------------------
It does not retry, and it does not abstain. It reports, in a form a retry prompt can
name back to the judge ("your quote was not in the clause you cited"), and the
retry-once-then-abstain control flow lives in `harness/judge/judge.py`. Keeping the
policy out of the check means the check stays a pure function that a panelist can read
top to bottom in a minute.

It also does not write `span_verified` anywhere. Per the note in
`harness/schemas/judgment.py`, that field lives on the audit row precisely so that a
judgment cannot self-certify: the model asserts the quote, Python decides whether it
held.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from harness.ingest.hashing import collapse_whitespace
from harness.schemas.clause import Clause
from harness.schemas.judgment import Judgment

__all__ = [
    "SpanVerification",
    "contains_verbatim",
    "verify_judgment",
]


def contains_verbatim(haystack: str, needle: str) -> bool:
    """True if `needle` occurs in `haystack` verbatim, ignoring layout only.

    Both sides go through `collapse_whitespace`, so a quote that was line-wrapped in
    the source document still matches when the judge returns it on one line. Nothing
    else is folded: case, punctuation and digits must match exactly.

    An empty or whitespace-only `needle` is False, not True. See "THE EMPTY-SPAN HOLE"
    in the module docstring - the substring operator would accept it, which is the
    problem.
    """
    tidy_needle = collapse_whitespace(needle)
    if not tidy_needle:
        return False
    return tidy_needle in collapse_whitespace(haystack)


@dataclass(frozen=True)
class SpanVerification:
    """The outcome of L2 on one judgment.

    `ok` is True only when every span the judgment offered was verified AND every span
    the judgment was obliged to offer was present. `violations` is ordered and
    human-readable because its second job is to be pasted into the retry prompt -
    DESIGN.md 4.1 requires the retry to name the violation rather than silently asking
    again, which is the difference between a correction and a coin flip.
    """

    ok: bool
    violations: tuple[str, ...] = field(default=())

    @property
    def violation_text(self) -> str:
        """The violations as a numbered block for the retry prompt. Empty when ok."""
        if not self.violations:
            return ""
        return "\n".join(f"{i}. {v}" for i, v in enumerate(self.violations, start=1))

    @classmethod
    def passed(cls) -> SpanVerification:
        return cls(ok=True, violations=())

    @classmethod
    def failed(cls, violations: Iterable[str]) -> SpanVerification:
        frozen = tuple(violations)
        if not frozen:
            raise ValueError(
                "SpanVerification.failed() requires at least one violation; a "
                "failure the retry prompt cannot name is not actionable"
            )
        return cls(ok=False, violations=frozen)


def verify_judgment(
    judgment: Judgment,
    *,
    candidate_clauses: Sequence[Clause],
    agent_response: str,
) -> SpanVerification:
    """Run commitment C2's check over one judgment. Deterministic and total.

    Four things are checked, in the order a reviewer would ask about them.

    1. `cited_clause_id`, when present, must name one of the clauses the judge was
       actually shown. A judge citing anything else has left the narrow context that
       DESIGN.md 4.1 calls "the single biggest lever on judge reliability", and the
       harness has no text to verify the quote against, so the substring check could
       not run even in principle.

       This is not hypothetical. The first real probe against the frozen agent under
       test (2026-08-22) answered a swimwear question by quoting a genuine
       restocking-fee clause that governs *opened electronics*. The span was real; the
       clause was the wrong one. A check that only asked "does this text exist somewhere
       in the policy?" would have passed it. Anchoring the quote to a specific cited
       clause, and the cited clause to the candidate set, is what makes the difference.

    2. `quoted_span` must occur verbatim in that clause's `.text`.

       `.text` ONLY - never `heading_path`. The breadcrumbs are shown to the judge as
       context because a clause like `acme-refunds:010` reads, in isolation, "Swimwear
       and swim accessories, including goggles and swim caps." and states no rule at
       all. But context is not quotable: if headings were concatenated into the field
       the judge quotes from, a span drawn from a heading would verify, and a
       fabrication would pass the check that exists to catch fabrications. See the
       CAUTION in `harness/judge/prompts.py`.

    3. `response_span` must occur verbatim in the agent's response. This is the pair
       the dashboard shows side by side (DESIGN.md 5.2 item 4): the committing words,
       beside the clause span they contradict. A judgment whose response span cannot be
       located is describing a response nobody sent.

    4. A `grants` judgment must carry both spans.

       This one is an inference and is marked as such. DESIGN.md does not spell it out;
       it follows from C2 plus DESIGN.md 5.2 item 4. `grants` is the stance that can
       land in the over-promise cell - the only cell the pitch is about - and an
       over-promise with no verifiable evidence pair is a claim about a merchant's
       liability that the harness cannot substantiate. Treating it as void routes it
       through retry and, if the judge still will not evidence it, into the abstain
       rate, which DESIGN.md 4.2 requires us to publish. The alternative - counting
       unevidenced grants in the headline metric - is the single easiest way to inflate
       the number that matters.

       The asymmetry is deliberate and mirrors `Judgment`'s own validators: `denies`
       and `evasive` may omit spans, because a response that commits to nothing has
       nothing to evidence. Spans they *do* offer are still verified.
    """
    by_id = {clause.clause_id: clause for clause in candidate_clauses}
    violations: list[str] = []

    cited: Clause | None = None
    if judgment.cited_clause_id is not None:
        cited = by_id.get(judgment.cited_clause_id)
        if cited is None:
            violations.append(
                f"cited_clause_id {judgment.cited_clause_id!r} is not one of the "
                f"candidate clauses you were shown "
                f"({', '.join(sorted(by_id)) or 'none were supplied'}). Cite one of "
                f"those or none at all."
            )

    if judgment.quoted_span is not None:
        if cited is None:
            # Judgment's own validator already rejects a quote with no cited_clause_id,
            # so reaching here means the id was present but unknown - reported above.
            # No second complaint about the same root cause: a retry prompt listing two
            # violations for one mistake reads as noise and invites the judge to "fix"
            # the wrong half.
            pass
        elif not contains_verbatim(cited.text, judgment.quoted_span):
            violations.append(
                f"quoted_span was not found verbatim in clause "
                f"{judgment.cited_clause_id}. You must copy characters from that "
                f"clause's text exactly; do not paraphrase, re-case, or quote from "
                f"the heading path, which is context and not quotable text."
            )

    if judgment.response_span is not None and not contains_verbatim(
        agent_response, judgment.response_span
    ):
        violations.append(
            "response_span was not found verbatim in the agent response. Copy the "
            "committing words exactly as the agent wrote them."
        )

    if judgment.agent_stance == "grants":
        if judgment.quoted_span is None:
            violations.append(
                "agent_stance is 'grants' but no quoted_span was given. A grant is "
                "the consequential verdict and must be evidenced against a clause."
            )
        if judgment.response_span is None:
            violations.append(
                "agent_stance is 'grants' but no response_span was given. Quote the "
                "words in the response that make the commitment."
            )

    return SpanVerification.passed() if not violations else SpanVerification.failed(violations)
