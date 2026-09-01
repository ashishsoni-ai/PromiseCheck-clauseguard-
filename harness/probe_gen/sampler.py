"""Fact-vector sampling from a rule condition space (DESIGN.md 3.1 step 1).

Each strategy module under `harness/probe_gen/strategies/` exposes a function
`sample(rule, rng, max_count) -> list[dict]` that returns fact vectors attacking
`rule` in that strategy's specific way. This module holds the shared helpers
(the mechanical fact-building logic) and the dispatch so the strategy files stay
thin and readable.

The strategies themselves are implemented in the per-strategy modules; this file
only provides the building blocks and the `dispatch` function.
"""

from __future__ import annotations

import random
from typing import Any

from harness.schemas.rule import Condition, EntitlementRule

#: Canonical per-entitlement fact bases, matching the hand-authored corpus
#: (scripts/author_probes.py BASELINES). A scenario must cover the WHOLE
#: entitlement's condition space — evaluate_rules() walks every same-entitlement
#: rule by precedence, and a missing attribute raises rather than labels.
ENTITLEMENT_BASES: dict[str, dict[str, Any]] = {
    "refund": {
        "days_since_delivery": 10,
        "item_category": "footwear",
        "item_opened": "no",
        "hygiene_seal_state": "none",
        "seal_tampering_observed": "no",
        "is_clearance_item": "no",
        "item_in_original_condition": "yes",
        "has_visible_damage": "no",
        "damage_reported_within_48h": "no",
        "proof_of_purchase_provided": "yes",
        "pickup_address_matches_order": "yes",
        "order_channel": "acme app",
        "units_of_single_item": 1,
        "device_registered_to_account": "no",
        "charging_accessories_present": "yes",
    },
    "cancellation": {
        "order_dispatched": "no",
        "order_channel": "acme app",
    },
    "partial_refund": {
        "item_category": "electronics",
        "item_opened": "yes",
        "days_since_delivery": 10,
    },
    "replacement": {
        "exchange_requests_used_this_order": 0,
        "replacement_stock_available": "yes",
        "days_since_delivery": 10,
        "is_clearance_item": "no",
    },
}


def _is_numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def condition_passes(condition: Condition, facts: dict[str, Any]) -> bool:
    """Evaluate a single condition against a fact vector. Mirrors the oracle's
    semantics (case/whitespace-insensitive strings, numeric compare)."""
    attr = condition.attribute
    if attr not in facts:
        return False
    fact = facts[attr]
    value = condition.value

    if condition.op in ("<=", "<", ">=", ">"):
        if not (_is_numeric(fact) and _is_numeric(value)):
            return False
        if condition.op == "<=":
            return fact <= value
        if condition.op == "<":
            return fact < value
        if condition.op == ">=":
            return fact >= value
        return fact > value

    if condition.op == "==":
        if isinstance(fact, str) and isinstance(value, str):
            return fact.strip().casefold() == value.strip().casefold()
        return fact == value

    if condition.op == "in":
        pool = value if isinstance(value, list) else [value]
        return str(fact).strip().casefold() in {str(v).strip().casefold() for v in pool}

    if condition.op == "not_in":
        pool = value if isinstance(value, list) else [value]
        return str(fact).strip().casefold() not in {str(v).strip().casefold() for v in pool}

    return False


def rule_fires(rule: EntitlementRule, facts: dict[str, Any]) -> bool:
    """True if `rule` (its root conditions) all hold against `facts`."""
    return all(condition_passes(c, facts) for c in rule.conditions)


def base_facts(
    rule: EntitlementRule,
    all_rules: list[EntitlementRule] | None = None,
) -> dict[str, Any]:
    """A fact vector that satisfies `rule`'s ROOT conditions while covering the
    WHOLE entitlement's condition space.

    `evaluate_rules()` walks every rule sharing the entitlement by precedence,
    so the vector must supply every attribute any of them references — hence the
    canonical per-entitlement base. We then override the target rule's root
    conditions so the rule fires. Exceptions are deliberately left at base
    defaults (which typically do not trigger them); exception_depth strategy
    re-samples from exception nodes directly.
    """
    base: dict[str, Any] = dict(ENTITLEMENT_BASES.get(rule.entitlement, {}))
    for cond in rule.conditions:
        value = cond.value
        if isinstance(value, list):
            base[cond.attribute] = value[0]
        elif cond.op in ("<=", "<") and _is_numeric(value):
            base[cond.attribute] = value - 1
        elif cond.op in (">=", ">") and _is_numeric(value):
            base[cond.attribute] = value + 1
        else:
            base[cond.attribute] = value
    return base


def pick_opposite(condition: Condition) -> Any:
    """A value for `condition.attribute` that FAILS the condition.

    Used by condition-stripping to satisfy N-1 of N conditions.
    """
    value = condition.value
    if isinstance(value, list):
        return "not_in_any_listed"
    if condition.op in ("<=", "<"):
        return (value + 1) if _is_numeric(value) else value
    if condition.op in (">=", ">"):
        return (value - 1) if _is_numeric(value) else value
    if condition.op == "==":
        return "some_other_value"
    if condition.op == "in":
        return "not_in_any_listed"
    if condition.op == "not_in":
        return value[0] if isinstance(value, list) else value
    return value