"""The eight probe strategies (DESIGN.md 3.2).

Each module exposes a single `sample(rule, rng, max_count) -> list[dict]`
function that returns fact vectors attacking `rule` in that strategy's way.
`STRATEGY_MODULES` maps the wire-format strategy name to its module, so the
driver can dispatch without a big if/elif chain.
"""

from __future__ import annotations

from importlib import import_module

STRATEGY_MODULES = {
    name: import_module(f"harness.probe_gen.strategies.{name}")
    for name in (
        "authority_pressure",
        "boundary",
        "category_smuggling",
        "condition_stripping",
        "cross_clause",
        "exception_depth",
        "false_premise",
        "multi_turn_drift",
    )
}

__all__ = ["STRATEGY_MODULES"]