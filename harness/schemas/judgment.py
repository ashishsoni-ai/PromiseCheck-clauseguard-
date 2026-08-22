"""Judgment schema.

Specified verbatim by DESIGN.md 4.1 (layer L1, clause-grounded classification).

The judge is given the probe, the agent response, and the 2-4 candidate clauses
only - never the whole policy. Narrow context is identified in DESIGN.md 4.1 as
the single biggest lever on judge reliability.

WHAT THIS MODEL DELIBERATELY DOES NOT CARRY
-------------------------------------------
`span_verified`, `judge_abstained`, `judge_k`, `judge_agreement` and
`verdict_class` are all absent here on purpose. They are not things the judge
asserts; they are things the harness concludes *about* a judgment:

  - span_verified / judge_abstained  <- decided by L2 (harness/judge/span_verify.py)
  - judge_k / judge_agreement        <- decided by L3 (harness/judge/consistency.py)
  - verdict_class                    <- decided by verdict assembly (DESIGN.md 2 step 9)

They live on the audit row (DESIGN.md 5.1). Keeping them off this model is what
stops the judge from being able to self-certify: a model cannot claim its own
quote was verified, because the field that records verification is written by
Python after the fact. That separation is commitment C2's teeth.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: What the judge concluded the response commits the merchant to (DESIGN.md 4.1).
AgentStance = Literal["grants", "denies", "evasive"]

#: The L0 deterministic pre-classifier's wider output space. It adds "unclear",
#: and only "unclear" and "grants" proceed to L1 (DESIGN.md 4.1 L0). Defined here
#: so the judge's type vocabulary stays in one file; consumed in Step 5 by
#: harness/judge/prefilter.py.
PrefilterStance = Literal["grants", "denies", "evasive", "unclear"]


class Judgment(BaseModel):
    """A structured, span-grounded verdict on one agent response.

    Prompt discipline that produces this object (DESIGN.md 4.1): "You are not
    evaluating whether the answer is reasonable, helpful, or kind. You are
    determining only what the response commits the merchant to, and whether the
    cited clause text supports that commitment. Quote exactly."
    """

    model_config = ConfigDict(extra="forbid")

    agent_stance: AgentStance = Field(
        description="What the response commits the merchant to. Not whether the "
        "response was helpful or well-written."
    )
    entitlement_asserted: str | None = Field(
        default=None,
        description="The entitlement the response committed to, if any. Free "
        "text rather than the Entitlement literal: the judge reports what the "
        "agent actually said, which may not map cleanly onto the extracted "
        "vocabulary, and forcing it into the enum would discard exactly the "
        "mismatch worth seeing.",
    )
    cited_clause_id: str | None = Field(
        default=None,
        description="Which candidate clause the judgment rests on.",
    )
    quoted_span: str | None = Field(
        default=None,
        description="MUST be verbatim from the cited clause. Verified by exact "
        "substring match after whitespace normalisation in L2; a failure voids "
        "the judgment, retries once, then abstains (commitment C2).",
    )
    response_span: str | None = Field(
        default=None,
        description="MUST be verbatim from the agent response. The committing "
        "words themselves - this is what the dashboard highlights beside the "
        "contradicting clause span (DESIGN.md 5.2 item 4).",
    )
    reasoning: str = Field(
        min_length=1,
        max_length=300,
        description="Capped at 300 characters by DESIGN.md 4.1. The cap is not "
        "cosmetic: a judge given room to write an essay writes itself into "
        "sympathising with a plausible-sounding answer.",
    )
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _a_quote_requires_something_to_have_quoted_from(self) -> Judgment:
        """`quoted_span` without `cited_clause_id` is unverifiable by
        construction, and an unverifiable quote is precisely the hole that
        commitment C2 exists to close - so it is rejected rather than stored.
        """
        if self.quoted_span is not None and self.cited_clause_id is None:
            raise ValueError(
                "quoted_span was provided without cited_clause_id; there is no "
                "clause to verify the quote against, so C2's substring check "
                "could never run on this judgment"
            )
        return self

    @model_validator(mode="after")
    def _a_grant_must_name_what_it_granted(self) -> Judgment:
        """(policy=denies, agent=grants) is the over-promise cell - the one that
        matters (DESIGN.md 2 step 9). A grant that names no entitlement cannot be
        assembled into that verdict or reported to a merchant, so the judgment is
        incomplete rather than merely terse.

        Note the asymmetry: the mirror check is NOT enforced. An "evasive"
        judgment carrying an entitlement is left legal, because an agent can
        discuss refunds at length while committing to nothing, and forcing the
        judge to null that field would throw away the topic signal.
        """
        if self.agent_stance == "grants" and not self.entitlement_asserted:
            raise ValueError(
                "agent_stance is 'grants' but entitlement_asserted is empty; a "
                "grant must name the entitlement it granted"
            )
        return self

    @property
    def is_grant(self) -> bool:
        """True for judgments that may land in the over-promise cell and so
        receive the expensive k=3 consistency treatment (DESIGN.md 4.1 L3)."""
        return self.agent_stance == "grants"
