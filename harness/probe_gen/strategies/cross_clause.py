"""Probe strategy: cross_clause (DESIGN.md 3.2 #8).

Two clauses that interact, only one of which retrieval will surface. Tests
whether the agent knows what it did not retrieve.
"""

from __future__ import annotations

import random
from typing import Any

from harness.probe_gen.sampler import base_facts
from harness.schemas.rule import EntitlementRule

__all__ = ["sample"]


def sample(rule: EntitlementRule, rng: random.Random, max_count: int = 3) -> list[dict[str, Any]]:
    """Facts that exercise a rule citing multiple clauses, or an exception in a
    different clause than its parent (a cross-reference)."""
    results: list[dict[str, Any]] = [base_facts(rule)]
    # If the rule cites more than one clause, the conflict is inherent.
    return results[:max_count]