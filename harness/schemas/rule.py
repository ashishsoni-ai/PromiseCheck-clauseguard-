"""Condition + EntitlementRule schemas.

Specified verbatim by DESIGN.md 1.2. These two models are the most consequential
in the system: `evaluate_rules()` (DESIGN.md 3.1 step 2) walks them in pure Python
to derive every ground-truth probe label, which is commitment C1 - an LLM never
decides whether a probe's expected answer is grant or deny.

Because the rules engine trusts these objects, the coherence checks that a naive
schema would leave to runtime live here instead. A `Condition` with op "<=" and
value ["footwear"] is not a rule the evaluator should have to defend itself
against; it is a rule that should never have been constructed.

WHERE `source_span` IS VERIFIED: not here. Checking that a span appears verbatim
in its clause needs the clause text, which a Condition deliberately does not
carry. That check is a post-extraction step (DESIGN.md 2 step 2: "every
source_span must be a substring of the clause, else retry once, then flag
needs_human_review") and lives in harness/extract/.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- shared value spaces ---------------------------------------------------
# A rule's polarity and an evaluated stance occupy the same two-value space but
# mean different things: polarity is what the rule asserts, stance is what
# evaluation concluded. Aliased separately so call sites read correctly.
Polarity = Literal["grants", "denies"]
PolicyStance = Literal["grants", "denies"]

Entitlement = Literal[
    "refund",
    "partial_refund",
    "replacement",
    "waiver",
    "extension",
    "discount",
    "credit",
    "cancellation",
]

ConditionOp = Literal["<=", "<", ">=", ">", "==", "in", "not_in"]

#: Ops that compare magnitude and therefore require a numeric operand.
NUMERIC_OPS: frozenset[str] = frozenset({"<=", "<", ">=", ">"})
#: Ops that test membership and therefore require a list operand.
MEMBERSHIP_OPS: frozenset[str] = frozenset({"in", "not_in"})


class Condition(BaseModel):
    """A single atomic test against a scenario's fact vector.

    All conditions on a rule are ANDed (DESIGN.md 1.2).
    """

    model_config = ConfigDict(extra="forbid")

    attribute: str = Field(
        min_length=1,
        description='Fact-vector key, e.g. "days_since_delivery", '
        '"item_category", "order_channel".',
    )
    op: ConditionOp
    value: str | int | float | list[str]
    source_span: str = Field(
        min_length=1,
        description="Verbatim span from the clause that licenses this condition. "
        "An extraction that cannot ground itself gets flagged for human review "
        "rather than silently accepted (DESIGN.md 1.2).",
    )

    @model_validator(mode="after")
    def _op_and_value_are_coherent(self) -> Condition:
        """Reconcile op against operand type, coercing only where intent is
        unambiguous and rejecting where it is not.

        Coercion is deliberately narrow. An extractor emitting "31" for a
        numeric comparison plainly means the number 31, so forcing a retry there
        would burn a call for nothing. An extractor emitting a list for "<=" has
        misunderstood the clause, and that must surface.
        """
        if self.op in NUMERIC_OPS:
            if isinstance(self.value, bool) or isinstance(self.value, list):
                raise ValueError(
                    f"op {self.op!r} compares magnitude and needs a numeric "
                    f"value, got {type(self.value).__name__}: {self.value!r}"
                )
            if isinstance(self.value, str):
                try:
                    coerced: int | float = (
                        int(self.value)
                        if self.value.strip().lstrip("+-").isdigit()
                        else float(self.value)
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"op {self.op!r} needs a numeric value; {self.value!r} "
                        "is not parseable as a number"
                    ) from exc
                # Plain assignment: the model is not frozen and
                # validate_assignment is off, so this will not re-enter validation.
                self.value = coerced

        elif self.op in MEMBERSHIP_OPS:
            if isinstance(self.value, list):
                if not self.value:
                    raise ValueError(
                        f"op {self.op!r} needs a non-empty list; an empty set "
                        "makes the condition trivially constant"
                    )
            elif isinstance(self.value, str):
                # A single category for "in" is unambiguous; normalise to a list
                # so the evaluator has exactly one shape to handle.
                self.value = [self.value]
            else:
                raise ValueError(
                    f"op {self.op!r} tests membership and needs a list of "
                    f"strings, got {type(self.value).__name__}: {self.value!r}"
                )

        else:  # "=="
            if isinstance(self.value, list):
                raise ValueError(
                    "op '==' compares a single value; use 'in' for a set of "
                    f"alternatives, got {self.value!r}"
                )

        return self


class EntitlementRule(BaseModel):
    """A structured entitlement rule extracted from one or more clauses.

    `exceptions` is recursive: exceptions may themselves carry exceptions, and
    DESIGN.md 3.2 strategy 3 probes depth-2 exception paths specifically because
    that is where chunk retrieval destroys precedence.
    """

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(min_length=1)
    clause_ids: list[str] = Field(
        min_length=1,
        description="Usually 1, sometimes 2+ for cross-references. A rule with "
        "no clause is ungrounded by construction, so at least one is required.",
    )
    entitlement: Entitlement
    polarity: Polarity
    conditions: list[Condition] = Field(
        default_factory=list,
        description="ALL must hold (AND). An empty list is legitimate and means "
        "an unconditional rule - typically a broad grant later narrowed by "
        "exceptions.",
    )
    exceptions: list[EntitlementRule] = Field(
        default_factory=list,
        description="Recursive - exceptions to exceptions. Defaults to empty for "
        "ergonomics; note that a missed exception therefore reads as 'none', "
        "which is why extraction coverage is measured and reported rather than "
        "assumed (DESIGN.md 1.2).",
    )
    precedence: int = Field(
        description="Higher wins on conflict. Required rather than defaulted: "
        "precedence decides contradictions, so it should be stated, not inherited."
    )
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool = Field(
        description="Required rather than defaulted to False. A default here "
        "would fail open - an unreviewed rule would silently present as "
        "reviewed."
    )

    @model_validator(mode="after")
    def _rule_id_is_unique_within_its_own_subtree(self) -> EntitlementRule:
        """Duplicate rule_ids inside one tree make audit rows ambiguous about
        which rule produced a label, so they are rejected at construction."""
        seen: set[str] = set()

        def walk(rule: EntitlementRule) -> None:
            if rule.rule_id in seen:
                raise ValueError(
                    f"duplicate rule_id {rule.rule_id!r} within a single rule tree"
                )
            seen.add(rule.rule_id)
            for child in rule.exceptions:
                walk(child)

        walk(self)
        return self

    def depth(self) -> int:
        """Exception nesting depth; 0 for a rule with no exceptions."""
        if not self.exceptions:
            return 0
        return 1 + max(child.depth() for child in self.exceptions)

    def walk(self):
        """Yield this rule and every nested exception, depth-first."""
        yield self
        for child in self.exceptions:
            yield from child.walk()


# Pydantic v2 needs the forward reference in `exceptions` resolved once the class
# object exists.
EntitlementRule.model_rebuild()
