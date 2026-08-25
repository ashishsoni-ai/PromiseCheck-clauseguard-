"""`source_span` grounding - every condition must quote the clause it came from.

DESIGN.md states this three times, in three different registers, which is a fair
signal that it is load-bearing:

    line  77  `source_span: str        # must appear verbatim in the clause`
    line  94  "**`source_span` on every condition** must be an exact substring of
               the clause. **Same mechanical check as C2.** Extraction that can't
               ground itself gets `needs_human_review=True` rather than being
               silently accepted."
    line 194  "Post-check: every `source_span` must be a substring of the clause,
               else retry once, then flag `needs_human_review`."

"Same mechanical check as C2" is an explicit instruction to reuse the machinery in
`harness/judge/span_verify.py` rather than to write a second, subtly different
substring check. So `contains_verbatim` is imported, not reimplemented - which also
means `collapse_whitespace` and never `normalize()`, for the reason that module
documents at length: `normalize()` casefolds and turns punctuation into spaces, so a
re-cased paraphrase would pass the check whose entire job is to catch paraphrase.

WHAT AN UNGROUNDED SPAN ACTUALLY BREAKS
---------------------------------------
It is tempting to say "C1 has a hole", and that is imprecise enough to be worth
correcting, because the imprecise version invites the wrong fix.

C1 is: ground-truth labels are derived deterministically from rules in Python, never
from an LLM. That property is untouched here. `evaluate_rules()` reads `attribute`,
`op` and `value`; it never reads `source_span`. A rule with a fabricated span still
produces the same label on every run, on every machine.

What breaks is the only mechanical link between a rule and the policy text it claims
to encode. `source_span` is the entire provenance story: it is what lets a reviewer
ask "where does this rule come from?" and get an answer that can be checked rather
than trusted. Without verification, ground truth can be deterministically derived
from a rule that no clause in the policy supports - and it will look exactly as
authoritative as a correct one, because determinism was never the thing at risk.

That distinction matters for the writeup: the failure mode is not "the numbers move",
it is "the numbers are stable, confident, and about a policy nobody wrote".

WHY THIS CHECK IS WEAKER THAN THE AUTHORING CHECK, ON PURPOSE
-------------------------------------------------------------
`scripts/author_rules.py` checks each span against ONE specific clause, because at
authoring time the human says which: `a.cond(..., clause=12, span=...)`.

The harness cannot ask that question. `Condition` has no clause pointer - it carries
`attribute`, `op`, `value` and `source_span`, and nothing else (DESIGN.md 3.1's schema
block). The clause link lives one level up, on `EntitlementRule.clause_ids`, which is
a list. So the strongest question available here is:

    is this span verbatim in AT LEAST ONE of the clauses its rule cites?

For a single-clause rule those questions are identical. For the five multi-clause
rules in the current lockfile they are not: a span could be grounded in the rule's
clause 003 while the human meant clause 007, and this check would pass it. That is a
real gap and it is recorded in `docs/limitations.md` rather than papered over.

It is still worth having. The threat model at this stage is not a subtly mis-attributed
span; it is a hand-edited `rules.lock.json`, or a future extractor, inventing text that
appears in the policy nowhere at all. This catches that, and it catches it before any
probe runs.

WHY `needs_human_review` IS THE ESCAPE HATCH AND NOT A SEVERITY DIAL
--------------------------------------------------------------------
DESIGN.md's sentence is precise: extraction that cannot ground itself "gets
`needs_human_review=True` rather than being silently accepted". The contrast drawn is
against *silence*, not against *stopping*. So the three-way rule is:

    grounded                              -> runs
    ungrounded, needs_human_review=True   -> runs, and is reported every time
    ungrounded, needs_human_review=False  -> refuses

The middle case is the spec's own declared outcome for the extractor's retry-then-flag
path (line 194), so refusing it outright would make that path unusable: an extractor
would flag a rule exactly as instructed and then find the harness would not run it.

The bottom case is the one that matters. `EntitlementRule.needs_human_review` is
required and deliberately not defaulted - the schema says "a default here would fail
open" - and this is the check that finally makes that field load-bearing. A rule
asserting both "my provenance does not check out" and "nobody needs to look at this"
is internally inconsistent, and the harness resolves the inconsistency by refusing
rather than by picking whichever half is more convenient.

Note the flag is read per *node*: an exception carries its own, and a nested exception
flagged for review does not license an ungrounded span in its parent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from harness.judge.span_verify import contains_verbatim
from harness.schemas.clause import Clause, PolicyDocument
from harness.schemas.rule import EntitlementRule

__all__ = [
    "GroundingReport",
    "UngroundedSpan",
    "UngroundedSpanError",
    "assert_spans_grounded",
    "check_spans_grounded",
]


class UngroundedSpanError(RuntimeError):
    """A condition's `source_span` is not verbatim in any clause its rule cites."""


@dataclass(frozen=True)
class UngroundedSpan:
    """One condition that failed the check, with enough context to fix it.

    Carries `needs_human_review` from the owning rule node so the caller can sort
    refusals from flagged-but-allowed without re-walking the tree.
    """

    rule_id: str
    attribute: str
    op: str
    source_span: str
    clause_ids: tuple[str, ...]
    unknown_clause_ids: tuple[str, ...]
    needs_human_review: bool

    def describe(self) -> str:
        """A block a human can act on: which rule, which condition, which clauses."""
        lines = [
            f"  rule      : {self.rule_id}"
            f"{'  [needs_human_review]' if self.needs_human_review else ''}",
            f"    condition : {self.attribute} {self.op}",
            f"    span      : {self.source_span!r}",
        ]
        if self.clause_ids:
            lines.append(f"    cites     : {', '.join(self.clause_ids)}")
        else:
            lines.append("    cites     : nothing - the rule lists no clause_ids")
        if self.unknown_clause_ids:
            lines.append(
                f"    NOT IN POLICY: {', '.join(self.unknown_clause_ids)} - the "
                f"lockfile cites clauses this document does not contain"
            )
        lines.append("    not a verbatim substring of any clause it cites")
        return "\n".join(lines)


@dataclass(frozen=True)
class GroundingReport:
    """The outcome of the grounding check over a whole rule set.

    `flagged` is not an error but must not be invisible either - `render_small_print`
    prints it, so an ungrounded-but-flagged rule announces itself on every single run
    rather than only in the run where somebody thought to look.
    """

    checked: int
    refused: tuple[UngroundedSpan, ...] = ()
    flagged: tuple[UngroundedSpan, ...] = ()

    @property
    def ok(self) -> bool:
        """True when nothing refuses. Flagged spans are allowed and still `ok`."""
        return not self.refused

    @property
    def grounded(self) -> int:
        return self.checked - len(self.refused) - len(self.flagged)


def check_spans_grounded(
    rules: Sequence[EntitlementRule], policy: PolicyDocument
) -> GroundingReport:
    """Walk every rule node and test every condition's span. Pure and total.

    Reports rather than raises, so the authoring script can print all failures at
    once instead of surfacing them one exception at a time - the same reason
    `SpanVerification` collects violations into a tuple.
    """
    # Built here rather than imported from `runner.clause_index`: `runner` imports
    # `lockfiles`, `lockfiles` imports this module, so reaching back into `runner`
    # would close an import cycle. One dict comprehension is the cheaper price.
    by_id: Mapping[str, Clause] = {c.clause_id: c for c in policy.clauses}

    checked = 0
    refused: list[UngroundedSpan] = []
    flagged: list[UngroundedSpan] = []

    for root in rules:
        for node in root.walk():
            known = [cid for cid in node.clause_ids if cid in by_id]
            unknown = tuple(cid for cid in node.clause_ids if cid not in by_id)
            for condition in node.conditions:
                checked += 1
                if any(
                    contains_verbatim(by_id[cid].text, condition.source_span)
                    for cid in known
                ):
                    continue
                failure = UngroundedSpan(
                    rule_id=node.rule_id,
                    attribute=condition.attribute,
                    op=condition.op,
                    source_span=condition.source_span,
                    clause_ids=tuple(node.clause_ids),
                    unknown_clause_ids=unknown,
                    needs_human_review=node.needs_human_review,
                )
                if node.needs_human_review:
                    flagged.append(failure)
                else:
                    refused.append(failure)

    return GroundingReport(
        checked=checked, refused=tuple(refused), flagged=tuple(flagged)
    )


def assert_spans_grounded(
    rules: Sequence[EntitlementRule],
    policy: PolicyDocument,
    *,
    source: str = "the rules",
) -> GroundingReport:
    """`check_spans_grounded`, but refusing when any span is ungrounded and unflagged.

    Returns the report on success so the caller can still surface `flagged`.
    `source` names the artefact in the error - a path when called from a lockfile
    load, a description when called from an authoring script - because "a span is
    ungrounded" is only actionable once you know which file to open.
    """
    report = check_spans_grounded(rules, policy)
    if report.refused:
        detail = "\n".join(failure.describe() for failure in report.refused)
        raise UngroundedSpanError(
            f"{len(report.refused)} of {report.checked} condition span(s) in "
            f"{source} are not verbatim in any clause they cite, against policy "
            f"{policy.doc_slug!r} at {policy.policy_version}.\n"
            f"`source_span` is the only mechanical link between a rule and the "
            f"policy text it claims to encode (DESIGN.md 3.1); an unverifiable one "
            f"lets ground truth be derived from a rule no clause supports.\n"
            f"Fix the span, or set needs_human_review=True on the rule to declare "
            f"it as awaiting review rather than as settled.\n\n{detail}"
        )
    return report
