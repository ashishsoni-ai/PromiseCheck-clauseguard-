"""Probe strategy: condition_stripping (DESIGN.md 3.2 #2).

Satisfy N-1 of N ANDed conditions, assert the rest confidently. The model sees
a mostly-matching clause and completes the pattern.
"""

from __future__ import annotations

import random
from typing import Any

from harness.probe_gen.sampler import base_facts, pick_opposite
from harness.schemas.rule import EntitlementRule

__all__ = ["sample"]


def sample(rule: EntitlementRule, rng: random.Random, max_count: int = 3) -> list[dict[str, Any]]:
    """Facts satisfying all but one condition of the root rule."""
    conditions = list(rule.conditions)
    if not conditions:
        return [base_facts(rule)]
    base = base_facts(rule)
    results: list[dict[str, Any]] = []
    for skip in conditions:
        fb = dict(base)
        fb[skip.attribute] = pick_opposite(skip)
        results.append(fb)
    return results[:max_count]