"""Compare a hand-authored rule set against an extractor-produced one.

This is a measurement tool, not a gate. `rules.lock.json` is the reviewed,
hand-authored ground truth that every published number rests on; an extractor's
output is a candidate that gets compared, not swapped in. The comparison answers
the four questions a reviewer would ask:

1. For each hand-authored rule, does the extractor produce something equivalent?
2. ... or something different (same entitlement+polarity, different conditions)?
3. ... or miss it entirely?
4. What rules did the extractor invent that are not in the hand-authored set?

MATCHING, AND WHY IT IS STRUCTURAL
----------------------------------
Two rules are "equivalent" when their *meaning* is the same, and meaning here is
the fact-vector contract: entitlement, polarity, and the ANDed set of
(attribute, op, value) conditions. source_span, precedence, confidence and
needs_human_review are deliberately NOT part of the signature - the first is
provenance the extractor must still get grounded, and the last three are
extractor-specific bookkeeping, not the rule's content. Two rules that agree on
the fact-vector contract produce identical ground-truth labels under
`evaluate_rules()`, which is what actually matters.

A "different" rule shares entitlement and polarity with a hand rule but disagrees
on the conditions. That is the interesting middle: it means the extractor found
the same entitlement in the same direction but encoded it differently, and the
difference could be a paraphrase of the same test or a genuinely different test.
The report prints both rules so a human can judge which.

A "miss" is a hand rule with no extracted rule in the same (entitlement,
polarity) cell at all.

An "invention" is an extracted rule whose (entitlement, polarity) cell contains
no hand rule. It may be a genuine finding the hand author overlooked, a
misreading, or a hallucination - the report says which clause it cites and what
it asserts, and a human decides.

Exceptions are walked into the same comparison: a hand rule with nested
exceptions is compared as its whole tree, node by node, so a missed carve-out
shows up as a miss rather than being hidden inside a matched root.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from harness.schemas.rule import EntitlementRule

__all__ = [
    "RuleSignature",
    "ComparisonRow",
    "CompareReport",
    "compare_rule_sets",
    "signature_of",
]


def _signature_condition(rule: EntitlementRule) -> tuple:
    """A condition reduced to a hashable, order-independent triple."""
    conds = []
    for c in rule.conditions:
        value = c.value if not isinstance(c.value, list) else tuple(c.value)
        conds.append((c.attribute, c.op, value))
    return frozenset(conds)


def signature_of(rule: EntitlementRule) -> tuple[str, str, frozenset]:
    """The structural signature of a rule node: (entitlement, polarity, conds)."""
    return (rule.entitlement, rule.polarity, _signature_condition(rule))


def _walk_all(rules: Sequence[EntitlementRule]):
    """Yield every node (root and exceptions) of every rule."""
    for root in rules:
        yield from root.walk()


@dataclass(frozen=True)
class RuleSignature:
    """A reduced (entitlement, polarity, conditions) triple plus its rule_id."""

    entitlement: str
    polarity: str
    conditions: frozenset
    rule_id: str


@dataclass(frozen=True)
class ComparisonRow:
    """One hand-authored rule's fate in the extractor output."""

    hand_rule_id: str
    entitlement: str
    polarity: str
    status: str  # "equivalent" | "different" | "missed"
    extracted_rule_id: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "hand_rule_id": self.hand_rule_id,
            "entitlement": self.entitlement,
            "polarity": self.polarity,
            "status": self.status,
            "extracted_rule_id": self.extracted_rule_id,
            "note": self.note,
        }


@dataclass(frozen=True)
class CompareReport:
    """The full comparison, ready to render."""

    hand_rules: tuple[EntitlementRule, ...]
    extracted_rules: tuple[EntitlementRule, ...]
    rows: tuple[ComparisonRow, ...]
    inventions: tuple[EntitlementRule, ...]

    @property
    def equivalent(self) -> tuple[ComparisonRow, ...]:
        return tuple(r for r in self.rows if r.status == "equivalent")

    @property
    def different(self) -> tuple[ComparisonRow, ...]:
        return tuple(r for r in self.rows if r.status == "different")

    @property
    def missed(self) -> tuple[ComparisonRow, ...]:
        return tuple(r for r in self.rows if r.status == "missed")

    @property
    def hand_rule_count(self) -> int:
        return len(self.hand_rules)

    @property
    def extracted_rule_count(self) -> int:
        return len(self.extracted_rules)

    def to_dict(self) -> dict:
        return {
            "hand_rule_count": self.hand_rule_count,
            "extracted_rule_count": self.extracted_rule_count,
            "equivalent": len(self.equivalent),
            "different": len(self.different),
            "missed": len(self.missed),
            "invented": len(self.inventions),
            "rows": [r.to_dict() for r in self.rows],
            "invented_rules": [
                rule.model_dump(mode="json") for rule in self.inventions
            ],
        }


def compare_rule_sets(
    hand_rules: Sequence[EntitlementRule],
    extracted_rules: Sequence[EntitlementRule],
) -> CompareReport:
    """Compare hand-authored rules against extracted rules, node by node.

    Pure and deterministic. `hand_rules` are the reviewed rules.lock.json roots;
    `extracted_rules` are the extractor's output roots.
    """
    hand_nodes = list(_walk_all(hand_rules))
    extracted_nodes = list(_walk_all(extracted_rules))

    # Index extracted nodes by (entitlement, polarity) cell.
    by_cell: dict[tuple[str, str], list[EntitlementRule]] = {}
    for node in extracted_nodes:
        by_cell.setdefault((node.entitlement, node.polarity), []).append(node)

    rows: list[ComparisonRow] = []
    used_extracted: set[str] = set()

    for hand in hand_nodes:
        cell = (hand.entitlement, hand.polarity)
        hand_sig = signature_of(hand)
        candidates = by_cell.get(cell, [])

        # Prefer a previously-unclaimed exact match.
        exact = next(
            (e for e in candidates if signature_of(e) == hand_sig and e.rule_id not in used_extracted),
            None,
        )
        if exact is not None:
            used_extracted.add(exact.rule_id)
            rows.append(
                ComparisonRow(
                    hand_rule_id=hand.rule_id,
                    entitlement=hand.entitlement,
                    polarity=hand.polarity,
                    status="equivalent",
                    extracted_rule_id=exact.rule_id,
                )
            )
            continue

        # Any extracted rule in the same cell = "different" (same entitlement,
        # same direction, conditions disagree).
        any_in_cell = next(
            (e for e in candidates if e.rule_id not in used_extracted), None
        )
        if any_in_cell is not None:
            used_extracted.add(any_in_cell.rule_id)
            rows.append(
                ComparisonRow(
                    hand_rule_id=hand.rule_id,
                    entitlement=hand.entitlement,
                    polarity=hand.polarity,
                    status="different",
                    extracted_rule_id=any_in_cell.rule_id,
                    note=(
                        "same entitlement and polarity, conditions differ from "
                        "the hand-authored rule"
                    ),
                )
            )
            continue

        rows.append(
            ComparisonRow(
                hand_rule_id=hand.rule_id,
                entitlement=hand.entitlement,
                polarity=hand.polarity,
                status="missed",
            )
        )

    # Inventions: extracted nodes whose cell has no hand node.
    hand_cells = {(n.entitlement, n.polarity) for n in hand_nodes}
    inventions = [
        node for node in extracted_nodes if (node.entitlement, node.polarity) not in hand_cells
    ]

    return CompareReport(
        hand_rules=tuple(hand_rules),
        extracted_rules=tuple(extracted_rules),
        rows=tuple(rows),
        inventions=tuple(inventions),
    )