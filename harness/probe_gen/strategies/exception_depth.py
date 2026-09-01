"""Probe strategy: exception_depth (DESIGN.md 3.2 #3).

Traverse to depth-2 exception paths. Requires correct precedence, which
chunk-retrieval destroys. Facts target a rule's nested exception.
"""

from __future__ import annotations

import random
from typing import Any

from harness.probe_gen.sampler import base_facts
from harness.schemas.rule import EntitlementRule

__all__ = ["sample"]


def _exceptions_at_depth(rule: EntitlementRule, depth: int) -> list[EntitlementRule]:
    nodes: list[EntitlementRule] = []

    def walk(node: EntitlementRule, d: int) -> None:
        if d == depth:
            nodes.append(node)
            return
        for exc in node.exceptions:
            walk(exc, d + 1)

    walk(rule, 0)
    return nodes


def sample(rule: EntitlementRule, rng: random.Random, max_count: int = 3) -> list[dict[str, Any]]:
    """Facts that trigger depth-2 (or the deepest available) exception paths."""
    results: list[dict[str, Any]] = []
    for depth in (2, 1):
        targets = _exceptions_at_depth(rule, depth)
        if not targets:
            continue
        for exc in targets[:max_count]:
            fb = base_facts(exc)
            results.append(fb)
        break
    if not results:
        results = [base_facts(rule)]
    return results[:max_count]