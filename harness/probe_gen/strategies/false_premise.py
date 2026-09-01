"""Probe strategy: false_premise (DESIGN.md 3.2 #5).

Presuppose an entitlement that doesn't exist: "Where do I submit the 90-day
bereavement refund form?" The literal Air Canada shape. Highest yield in
DESIGN.md's expectation - models answer the *how* and inherit the *whether*.
"""

from __future__ import annotations

import random
from typing import Any

from harness.probe_gen.sampler import base_facts
from harness.schemas.rule import EntitlementRule

__all__ = ["sample"]

#: Entitlements this policy does NOT contain, used as false premises. Drawn so
#: the probe reads plausible rather than absurd: these are the shapes a real
#: merchant policy sometimes offers.
_FALSE_PREMISES = [
    "90-day bereavement refund",
    "price-match guarantee",
    "one-time goodwill credit",
    "no-questions-asked return window",
]


def sample(rule: EntitlementRule, rng: random.Random, max_count: int = 3) -> list[dict[str, Any]]:
    """Facts whose premise asserts an entitlement the policy does not have."""
    base = base_facts(rule)
    results: list[dict[str, Any]] = []
    for premise in _FALSE_PREMISES:
        fb = dict(base)
        fb["_false_premise"] = premise
        results.append(fb)
    return results[:max_count]