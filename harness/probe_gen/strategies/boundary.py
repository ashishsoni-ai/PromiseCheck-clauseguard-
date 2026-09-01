"""Probe strategy: boundary (DESIGN.md 3.2 #1).

For every numeric/temporal condition, sample at v-1, v, v+1. Retrieval returns
the right clause; the model rounds. This strategy probes the model's
quantitative fidelity at the exact edge of a rule's numeric conditions.
"""

from __future__ import annotations

import random
from typing import Any

from harness.probe_gen.sampler import base_facts
from harness.schemas.rule import EntitlementRule

__all__ = ["sample"]


def sample(rule: EntitlementRule, rng: random.Random, max_count: int = 3) -> list[dict[str, Any]]:
    """Facts at v-1, v, v+1 for each numeric condition of the rule tree."""
    base = base_facts(rule)
    results: list[dict[str, Any]] = []

    for node in rule.walk():
        for cond in node.conditions:
            value = cond.value
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            attr = cond.attribute
            for offset in (-1, 0, 1):
                fb = dict(base)
                fb[attr] = value + offset
                results.append(fb)
            break  # one numeric condition per rule is enough for a boundary probe

    if not results:
        results = [base]
    return results[:max_count]