"""Probe strategy: category_smuggling (DESIGN.md 3.2 #4).

An item semantically adjacent to an excluded category - "a fitness band"
against excluded "wearables". Embedding similarity actively works against
correctness here.
"""

from __future__ import annotations

import random
from typing import Any

from harness.probe_gen.sampler import base_facts
from harness.schemas.rule import EntitlementRule

__all__ = ["sample"]

#: Semantic neighbours of the excluded categories in the acme policy. The model
#: embeds these close to the excluded item and "helpfully" applies the exclusion
#: (or fails to), which is exactly the failure this strategy exists to catch.
_SMUGGLING_MAP = {
    "innerwear": "athletic_underlayer",
    "sleepwear": "loungewear",
    "swimwear": "waterproof_workout_top",
    "swim accessories": "pool_toy",
    "cosmetics": "skincare_tool",
    "skincare": "facial_roller",
    "fragrance": "scented_candle",
    "wearable electronics": "bluetooth_speaker",
    "smartwatches": "digital_watch",
    "fitness bands": "phone_armband",
}


def sample(rule: EntitlementRule, rng: random.Random, max_count: int = 3) -> list[dict[str, Any]]:
    """Facts whose item_category is a near-miss of an excluded category."""
    base = base_facts(rule)
    results: list[dict[str, Any]] = []
    for node in rule.walk():
        for cond in node.conditions:
            if cond.op not in ("in", "not_in") or not isinstance(cond.value, list):
                continue
            for cat in cond.value:
                smuggled = _SMUGGLING_MAP.get(str(cat).casefold())
                if smuggled is None:
                    continue
                fb = dict(base)
                fb[cond.attribute] = smuggled
                results.append(fb)
                break
            if results:
                break
        if results:
            break
    if not results:
        results = [base]
    return results[:max_count]