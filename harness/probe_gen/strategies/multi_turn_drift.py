"""Probe strategy: multi_turn_drift (DESIGN.md 3.2 #7).

Turn 1 innocuous and in-policy, turn 2 shifts one fact out of policy. Context
carryover; the model reuses turn 1's conclusion.
"""

from __future__ import annotations

import random
from typing import Any

from harness.probe_gen.sampler import base_facts
from harness.schemas.rule import EntitlementRule

__all__ = ["sample"]


def sample(rule: EntitlementRule, rng: random.Random, max_count: int = 3) -> list[dict[str, Any]]:
    """Return two fact vectors per drift: turn 1 in-policy, turn 2 shifted out.

    Encoded as a list where every two entries are (in_policy, drifted). The
    driver pairs them up when rendering two-turn text.
    """
    base = base_facts(rule)
    results: list[dict[str, Any]] = []
    for node in rule.walk():
        for cond in node.conditions:
            value = cond.value
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            attr = cond.attribute
            turn1 = dict(base)
            turn2 = dict(base)
            if cond.op in ("<=", "<"):
                turn2[attr] = value + 1  # drift just over the boundary
            elif cond.op in (">=", ">"):
                turn2[attr] = value - 1
            else:
                continue
            results.append(turn1)
            results.append(turn2)
            break
        if results:
            break
    if not results:
        results = [base, dict(base)]
    return results[: max_count * 2]