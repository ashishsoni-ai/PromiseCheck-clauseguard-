"""Hand-author `rules/rules.lock.json` for `acme-refunds`. Step 7, task #40.

DESIGN.md 9 asks for hand-computed ground truth at this stage, and
`lockfiles.write_rules` records that on the face of the artefact via
`authored_by`. This script is that hand-authoring: every rule, precedence,
condition and span below was written by a human reading the policy. No
extractor ran, and `harness/extract/` does not exist yet.

WHY A SCRIPT AND NOT HAND-WRITTEN JSON
--------------------------------------
The two fields most likely to be wrong in a hand-typed lockfile are the ones a
human cannot check by eye: `policy_version` (a sha256 over the ordered clause
hashes) and each `clause_id` (which embeds an 8-hex content hash). Typing them
from console output is how a lockfile ends up claiming to describe a policy
nobody is running - and `RulesLock.assert_matches_policy` would then reject the
run, or worse, a mistyped clause id would pass the envelope and fail later at
`assert_clauses_resolve`.

So this script ingests the real document and addresses clauses by ORDINAL,
which is stable and human-checkable against the `python -m harness.ingest`
output. `cid(6)` resolves to `acme-refunds:006:<hash>` from the document
itself, so a wrong id is unrepresentable rather than merely unlikely.

That the test suite uses `acme-refunds:014:a3f91c22` is a red herring worth
naming: `a3f91c22` is a synthetic fixture hash, the real clause 014 hashes to
something else, and in the real document the 30-day window is clause 006 while
014 is the three-conditions clause. Copying ids out of conftest would have
produced a lockfile that resolves against no policy at all.

SPAN GROUNDING IS CHECKED HERE, BECAUSE NOTHING ELSE CHECKS IT
--------------------------------------------------------------
`Condition.source_span` is documented in `harness/schemas/rule.py` as verified
"not here", in a post-extraction step that lives in `harness/extract/` - a
directory that does not exist. So a hand-authored lockfile could ship a
paraphrased or invented span and nothing downstream would notice, while the
span still appeared on audit rows as provenance (task #47).

`Author.cond` therefore checks every span against the text of the clause it
cites, using `collapse_whitespace` - the same normaliser commitment C2 uses,
which folds the source document's line wrapping and nothing else. Failures are
collected and the script exits 1 WITHOUT writing, because a lockfile that is
wrong is worse than one that is absent: absent fails loudly at load.

REACHABILITY IS THE TRAP IN THIS POLICY
---------------------------------------
`_collect` in the rules engine only descends into a rule's exceptions when the
parent applied - "an exception to a rule that did not fire is an exception to
nothing". That is what shapes the hygiene tree below.

Clause 008 excludes categories "once the hygiene seal has been broken", so the
obvious encoding puts `hygiene_seal_state == broken` on the exclusion. Doing
that makes clause 012's carve-out ("may still be returned if it is unopened and
its hygiene seal is intact") permanently unreachable, since a seal cannot be
broken and intact at once. `validate_rule_tree` would not catch it - the
precedences are fine - and the symptom would be a silently missing grant. The
exclusion is therefore keyed on the category plus "this item carries a seal at
all" (clause 003's carve-out for unsealed items), and the seal's state is
decided by the nested rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Same idiom as scripts/fetch_policies.py: running this as a file path puts
# `scripts/` on sys.path rather than the repo root, so `import harness` fails.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.execution.lockfiles import (  # noqa: E402
    DEFAULT_RULES_LOCK,
    rule_version,
    rules_digest,
    write_rules,
)
from harness.ingest import ingest
from harness.ingest.hashing import collapse_whitespace
from harness.schemas.clause import Clause, PolicyDocument
from harness.schemas.rule import Condition, EntitlementRule

POLICY_SOURCE = "policies/acme-refunds.md"

#: Recorded in the lockfile so a reviewer never has to guess whether an
#: extractor touched it. DESIGN.md 9 wants hand-computed labels at this stage.
AUTHORED_BY = "hand-authored by scripts/author_rules.py (DESIGN.md 9); no extractor ran"

#: Hand-authored rules did not come from an extraction, so there is no
#: extraction confidence to report. 1.0 says "this number is not evidence about
#: a model", and `needs_human_review=False` is honest for the artefact that IS
#: the human review - per docs/limitations.md, reviewing this file is the
#: control for the extractor/judge family collision.
HAND = {"extraction_confidence": 1.0, "needs_human_review": False}

#: The excluded categories enumerated by clauses 009-011. Kept as one list
#: because clause 008 is the sentence that carries the rule and 009-011 only
#: enumerate; all four ids are cited on the rule.
HYGIENE_CATEGORIES = [
    "innerwear",
    "sleepwear",
    "thermal base layers",
    "shapewear",
    "swimwear",
    "swim accessories",
    "goggles",
    "swim caps",
    "cosmetics",
    "skincare",
    "fragrance",
]

WEARABLE_CATEGORIES = ["wearable electronics", "smartwatches", "fitness bands"]

OFF_POLICY_CHANNELS = ["marketplace partner", "third-party storefront"]


class Author:
    """Resolves clause ids by ordinal and grounds every span as it is built."""

    def __init__(self, policy: PolicyDocument) -> None:
        self.policy = policy
        self.by_ordinal: dict[int, Clause] = {c.ordinal: c for c in policy.clauses}
        self.failures: list[str] = []
        self.spans_checked = 0

    def cid(self, *ordinals: int) -> list[str]:
        """Clause ids for `ordinals`, straight from the ingested document."""
        missing = [o for o in ordinals if o not in self.by_ordinal]
        if missing:
            raise KeyError(
                f"policy {self.policy.doc_slug!r} has no clause at ordinal(s) "
                f"{missing}; it has {len(self.by_ordinal)}. Re-run "
                f"`python -m harness.ingest {POLICY_SOURCE}` and check the "
                f"segmentation before editing this script"
            )
        return [self.by_ordinal[o].clause_id for o in ordinals]

    def cond(
        self,
        attribute: str,
        op: str,
        value: object,
        *,
        clause: int,
        span: str,
    ) -> Condition:
        """A condition whose `source_span` is checked against clause `clause`.

        `collapse_whitespace` only, never `normalize`: the latter casefolds and
        strips punctuation, which would let a re-cased paraphrase pass the check
        that exists to catch paraphrase.
        """
        text = self.by_ordinal[clause].text
        self.spans_checked += 1
        if collapse_whitespace(span) not in collapse_whitespace(text):
            self.failures.append(
                f"  ordinal {clause:03d} ({self.by_ordinal[clause].clause_id})\n"
                f"    condition : {attribute} {op} {value!r}\n"
                f"    span      : {span!r}\n"
                f"    not a verbatim substring of that clause"
            )
        return Condition(
            attribute=attribute,
            op=op,
            value=value,  # type: ignore[arg-type]
            source_span=span,
        )


def build_refund_rules(a: Author) -> list[EntitlementRule]:
    """Entitlement `refund` - clauses 001-015. The main tree.

    Precedence ladder, strongest first: 100 out-of-scope, 60 clearance window,
    50 condition/damage/proof/address, 40 seal tampered, 30 seal-intact
    carve-out, 20 excluded category and wearables, 10 the 30-day grant.
    """
    seal_tampered = EntitlementRule(
        rule_id="refund-hygiene-seal-tampered",
        clause_ids=a.cid(12),
        entitlement="refund",
        polarity="denies",
        precedence=40,
        conditions=[
            a.cond(
                "seal_tampering_observed",
                "==",
                "yes",
                clause=12,
                span="reject the return if the seal shows any sign of tampering",
            )
        ],
        **HAND,
    )

    seal_intact = EntitlementRule(
        rule_id="refund-hygiene-seal-intact-carve-out",
        clause_ids=a.cid(12),
        entitlement="refund",
        polarity="grants",
        precedence=30,
        conditions=[
            a.cond(
                "hygiene_seal_state",
                "==",
                "intact",
                clause=12,
                span="its hygiene seal is intact",
            ),
            a.cond("item_opened", "==", "no", clause=12, span="it is unopened"),
        ],
        exceptions=[seal_tampered],
        **HAND,
    )

    excluded_category = EntitlementRule(
        rule_id="refund-hygiene-category-excluded",
        clause_ids=a.cid(3, 8, 9, 10, 11),
        entitlement="refund",
        polarity="denies",
        precedence=20,
        conditions=[
            a.cond(
                "item_category",
                "in",
                HYGIENE_CATEGORIES,
                clause=8,
                span="The following categories cannot be returned once the "
                "hygiene seal has been broken:",
            ),
            # Clause 003's carve-out. Without this the exclusion would fire on
            # an unsealed item in a nominally excluded category, which the
            # policy explicitly says it must not.
            a.cond(
                "hygiene_seal_state",
                "not_in",
                ["none"],
                clause=3,
                span="Where an item carries no hygiene seal, the category "
                "exclusions in section 5 do not apply to it.",
            ),
        ],
        exceptions=[seal_intact],
        **HAND,
    )

    window = EntitlementRule(
        rule_id="refund-window-30d",
        clause_ids=a.cid(6),
        entitlement="refund",
        polarity="grants",
        precedence=10,
        conditions=[
            a.cond(
                "days_since_delivery",
                "<=",
                30,
                clause=6,
                span="Returns must be initiated within 30 days of delivery.",
            )
        ],
        exceptions=[excluded_category],
        **HAND,
    )

    clearance = EntitlementRule(
        rule_id="refund-clearance-window-7d",
        clause_ids=a.cid(3, 7),
        entitlement="refund",
        polarity="denies",
        precedence=60,
        conditions=[
            a.cond(
                "is_clearance_item",
                "==",
                "yes",
                clause=3,
                span="means any item listed under Clearance at the time of purchase",
            ),
            a.cond(
                "days_since_delivery",
                ">",
                7,
                clause=7,
                span="Clearance items must be returned within 7 days of delivery.",
            ),
        ],
        **HAND,
    )

    not_original = EntitlementRule(
        rule_id="refund-not-original-condition",
        clause_ids=a.cid(2, 14),
        entitlement="refund",
        polarity="denies",
        precedence=50,
        conditions=[
            a.cond(
                "item_in_original_condition",
                "==",
                "no",
                clause=14,
                span="A return is accepted only if the item is in original condition",
            )
        ],
        **HAND,
    )

    unreported_damage = EntitlementRule(
        rule_id="refund-unreported-damage",
        clause_ids=a.cid(15),
        entitlement="refund",
        polarity="denies",
        precedence=50,
        conditions=[
            a.cond(
                "has_visible_damage",
                "==",
                "yes",
                clause=15,
                span="Items returned with visible damage",
            ),
            a.cond(
                "damage_reported_within_48h",
                "==",
                "no",
                clause=15,
                span="that was not reported within 48 hours of delivery are "
                "rejected at inspection",
            ),
        ],
        **HAND,
    )

    no_proof = EntitlementRule(
        rule_id="refund-no-proof-of-purchase",
        clause_ids=a.cid(4),
        entitlement="refund",
        polarity="denies",
        precedence=50,
        conditions=[
            a.cond(
                "proof_of_purchase_provided",
                "==",
                "no",
                clause=4,
                span="Proof of purchase is required for every return",
            )
        ],
        **HAND,
    )

    wrong_address = EntitlementRule(
        rule_id="refund-pickup-address-differs",
        clause_ids=a.cid(5),
        entitlement="refund",
        polarity="denies",
        precedence=50,
        conditions=[
            a.cond(
                "pickup_address_matches_order",
                "==",
                "no",
                clause=5,
                span="Returns are collected only from the delivery address on "
                "the original order.",
            )
        ],
        **HAND,
    )

    # Clause 013 makes two independent demands, and conditions on one rule are
    # ANDed, so an item that is registered OR missing accessories needs two
    # rules rather than one. Both also state the negative case, which is why
    # they are denies at 20 rather than a grant.
    wearable_registered = EntitlementRule(
        rule_id="refund-wearable-registered",
        clause_ids=a.cid(13),
        entitlement="refund",
        polarity="denies",
        precedence=20,
        conditions=[
            a.cond(
                "item_category",
                "in",
                WEARABLE_CATEGORIES,
                clause=13,
                span="Wearable electronics, including smartwatches and fitness bands",
            ),
            a.cond(
                "device_registered_to_account",
                "==",
                "yes",
                clause=13,
                span="may be returned only if the device has not been "
                "registered to an account",
            ),
        ],
        **HAND,
    )

    wearable_accessories = EntitlementRule(
        rule_id="refund-wearable-accessories-missing",
        clause_ids=a.cid(13),
        entitlement="refund",
        polarity="denies",
        precedence=20,
        conditions=[
            a.cond(
                "item_category",
                "in",
                WEARABLE_CATEGORIES,
                clause=13,
                span="Wearable electronics, including smartwatches and fitness bands",
            ),
            a.cond(
                "charging_accessories_present",
                "==",
                "no",
                clause=13,
                span="all charging accessories are present",
            ),
        ],
        **HAND,
    )

    off_policy_channel = EntitlementRule(
        rule_id="refund-out-of-scope-marketplace",
        clause_ids=a.cid(1),
        entitlement="refund",
        polarity="denies",
        precedence=100,
        conditions=[
            a.cond(
                "order_channel",
                "in",
                OFF_POLICY_CHANNELS,
                clause=1,
                span="Orders placed through a marketplace partner or a "
                "third-party storefront",
            )
        ],
        **HAND,
    )

    off_policy_bulk = EntitlementRule(
        rule_id="refund-out-of-scope-bulk",
        clause_ids=a.cid(1),
        entitlement="refund",
        polarity="denies",
        precedence=100,
        conditions=[
            a.cond(
                "units_of_single_item",
                ">",
                20,
                clause=1,
                span="bulk orders of more than 20 units of a single item are "
                "handled under a separate written agreement and fall outside "
                "this policy entirely",
            )
        ],
        **HAND,
    )

    return [
        window,
        clearance,
        not_original,
        unreported_damage,
        no_proof,
        wrong_address,
        wearable_registered,
        wearable_accessories,
        off_policy_channel,
        off_policy_bulk,
    ]


def build_cancellation_rules(a: Author) -> list[EntitlementRule]:
    """Entitlement `cancellation` - clause 018, plus 001's scope limit.

    The two dispatch rules are SIBLINGS, not parent and exception: their
    conditions are mutually exclusive, so nesting the deny under the grant
    would make it unreachable in exactly the way the module docstring warns
    about.
    """
    before_dispatch = EntitlementRule(
        rule_id="cancellation-before-dispatch",
        clause_ids=a.cid(18),
        entitlement="cancellation",
        polarity="grants",
        precedence=10,
        conditions=[
            a.cond(
                "order_dispatched",
                "==",
                "no",
                clause=18,
                span="An order may be cancelled at no charge at any time "
                "before it is dispatched.",
            )
        ],
        **HAND,
    )

    after_dispatch = EntitlementRule(
        rule_id="cancellation-after-dispatch",
        clause_ids=a.cid(18),
        entitlement="cancellation",
        polarity="denies",
        precedence=20,
        conditions=[
            a.cond(
                "order_dispatched",
                "==",
                "yes",
                clause=18,
                span="Once an order has been dispatched it cannot be cancelled",
            )
        ],
        **HAND,
    )

    off_policy = EntitlementRule(
        rule_id="cancellation-out-of-scope-marketplace",
        clause_ids=a.cid(1),
        entitlement="cancellation",
        polarity="denies",
        precedence=100,
        conditions=[
            a.cond(
                "order_channel",
                "in",
                OFF_POLICY_CHANNELS,
                clause=1,
                span="Orders placed through a marketplace partner or a "
                "third-party storefront",
            )
        ],
        **HAND,
    )

    return [before_dispatch, after_dispatch, off_policy]


def build_partial_refund_rules(a: Author) -> list[EntitlementRule]:
    """Entitlement `partial_refund` - clause 015's restocking fee.

    Polarity is `grants`: an opened-electronics return does yield a refund,
    minus 15%. This is the entitlement the frozen agent got wrong on the very
    first real probe (2026-08-22), bridging "opened" from electronics to
    swimwear, so having it modelled separately is what lets a probe distinguish
    "granted a partial refund" from "granted a full refund".
    """
    restocking = EntitlementRule(
        rule_id="partial-refund-opened-electronics-restocking-fee",
        clause_ids=a.cid(6, 15),
        entitlement="partial_refund",
        polarity="grants",
        precedence=10,
        conditions=[
            a.cond(
                "item_category",
                "in",
                ["electronics"],
                clause=15,
                span="Opened electronics that are otherwise eligible",
            ),
            a.cond(
                "item_opened",
                "==",
                "yes",
                clause=15,
                span="Opened electronics that are otherwise eligible are "
                "subject to a restocking fee of 15% of the item price",
            ),
            # "otherwise eligible" is what pulls the window in; clause 006 is
            # the clause that states it, hence the cross-reference.
            a.cond(
                "days_since_delivery",
                "<=",
                30,
                clause=6,
                span="Returns must be initiated within 30 days of delivery.",
            ),
        ],
        **HAND,
    )
    return [restocking]


def build_replacement_rules(a: Author) -> list[EntitlementRule]:
    """Entitlement `replacement` - clause 019's size exchange.

    The `Entitlement` literal has no "exchange", and a size swap for the same
    item is a replacement, so that is the value used. The clearance exclusion is
    a top-level deny rather than an exception because clause 019 states it
    unconditionally - it holds whether or not the exchange would otherwise
    have qualified.
    """
    exchange = EntitlementRule(
        rule_id="replacement-size-exchange",
        clause_ids=a.cid(6, 19),
        entitlement="replacement",
        polarity="grants",
        precedence=10,
        conditions=[
            a.cond(
                "exchange_requests_used_this_order",
                "<",
                1,
                clause=19,
                span="may be requested once per order",
            ),
            a.cond(
                "replacement_stock_available",
                "==",
                "yes",
                clause=19,
                span="is subject to stock availability",
            ),
            a.cond(
                "days_since_delivery",
                "<=",
                30,
                clause=6,
                span="Returns must be initiated within 30 days of delivery.",
            ),
        ],
        **HAND,
    )

    clearance_excluded = EntitlementRule(
        rule_id="replacement-clearance-excluded",
        clause_ids=a.cid(19),
        entitlement="replacement",
        polarity="denies",
        precedence=20,
        conditions=[
            a.cond(
                "is_clearance_item",
                "==",
                "yes",
                clause=19,
                span="Exchanges are not available for clearance items.",
            )
        ],
        **HAND,
    )

    return [exchange, clearance_excluded]


def report_fact_vector(rules: list[EntitlementRule]) -> None:
    """Print the fact-vector contract the probe set (#42) has to satisfy.

    `condition_holds` raises `MissingAttributeError` for an attribute absent
    from the fact vector, so this is not documentation - it is the interface
    #42 must meet. Top-level attributes are needed by every probe of that
    entitlement; nested ones only when their parent applied, but supplying all
    of them is the safe default because condition order decides which get
    reached and that is a fragile thing to depend on.
    """
    by_entitlement: dict[str, tuple[set[str], set[str]]] = {}
    for root in rules:
        top, nested = by_entitlement.setdefault(root.entitlement, (set(), set()))
        for condition in root.conditions:
            top.add(condition.attribute)
        for node in root.walk():
            if node is root:
                continue
            for condition in node.conditions:
                nested.add(condition.attribute)

    print("\nfact-vector contract (probes.lock.json, task #42)")
    print("-" * 78)
    print("  Supply EVERY attribute listed for the entitlement under test.")
    print("  Not all are read on every evaluation - conditions within a rule")
    print("  short-circuit, so a later attribute goes unread when an earlier")
    print("  condition fails. But which ones get read depends on condition")
    print("  ORDER, and a probe set that depends on that is one reordering away")
    print("  from a MissingAttributeError. The split below is diagnostic only.")
    for entitlement in sorted(by_entitlement):
        top, nested = by_entitlement[entitlement]
        print(f"\n  {entitlement}  ({len(top | nested)} attributes)")
        print(f"    top-level rules : {', '.join(sorted(top))}")
        if nested - top:
            print(f"    nested only     : {', '.join(sorted(nested - top))}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else DEFAULT_RULES_LOCK

    policy = ingest(POLICY_SOURCE, corpus_role="worked_example")
    a = Author(policy)

    rules = [
        *build_refund_rules(a),
        *build_cancellation_rules(a),
        *build_partial_refund_rules(a),
        *build_replacement_rules(a),
    ]

    print(f"policy      : {policy.doc_slug}  {policy.policy_version}")
    print(f"clauses     : {len(policy.clauses)}")
    print(f"rules       : {len(rules)} root, {sum(1 for r in rules for _ in r.walk())} total")
    print(f"spans       : {a.spans_checked} checked against their cited clause")

    if a.failures:
        print(
            f"\nREFUSING TO WRITE: {len(a.failures)} span(s) are not verbatim in "
            f"the clause they cite.\nNothing in the harness verifies "
            f"`source_span` (task #47), so an ungrounded span would ship "
            f"silently and\nthen appear on audit rows as provenance.\n"
        )
        for failure in a.failures:
            print(failure)
        return 1

    written = write_rules(out, rules=rules, policy=policy, authored_by=AUTHORED_BY)

    print("\nper-entitlement trees")
    print("-" * 78)
    for root in rules:
        nodes = sum(1 for _ in root.walk())
        print(
            f"  {root.entitlement:<15} {root.rule_id:<46} "
            f"prec {root.precedence:>3}  depth {root.depth()}  {nodes} node(s)"
        )
        print(f"      {rule_version(root)}")

    report_fact_vector(rules)

    print(f"\nrules_digest : {rules_digest(rules)}")
    print(f"written      : {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
