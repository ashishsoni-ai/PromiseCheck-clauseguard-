"""Probe-generation driver: the shared pipeline behind the script AND the CLI.

DESIGN.md 3.1 order of operations, enforced here:
  1. sample a fact vector (strategy module, Python)
  2. label it with `evaluate_rules()` (Python, no LLM — C1)
  3. render the surface (adversary LLM, temp 0.9)
  4. self-critique rewrite (adversary LLM)
  5. oracle-check that every fact the probe turns on still appears in the text

This module is importable by both `scripts/generate_probes.py` and
`harness/cli.py` (`clauseguard generate`), so there is exactly one pipeline.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from harness.probe_gen.adversary import render_surface, self_critique
from harness.probe_gen.oracle import oracle_check
from harness.probe_gen.strategies import STRATEGY_MODULES
from harness.rules_engine.evaluate import RuleEvaluationError, evaluate_rules
from harness.schemas.probe import Probe, ProbeScenario, ProbeStrategy

#: The 8 strategies, in DESIGN.md 3.2's order.
STRATEGY_ORDER = [
    ProbeStrategy.BOUNDARY,
    ProbeStrategy.CONDITION_STRIPPING,
    ProbeStrategy.EXCEPTION_DEPTH,
    ProbeStrategy.CATEGORY_SMUGGLING,
    ProbeStrategy.FALSE_PREMISE,
    ProbeStrategy.AUTHORITY_PRESSURE,
    ProbeStrategy.MULTI_TURN_DRIFT,
    ProbeStrategy.CROSS_CLAUSE,
]


def _rendered_surfaces(facts: Any, client: Any, strategy: ProbeStrategy):
    """Render + self-critique the surface(s) for a fact vector.

    Multi-turn probes get two turns: turn 1 from the in-policy vector, turn 2
    from the drifted one. The fact vectors come paired from the strategy module.
    Single-turn strategies get one render + critique.
    """
    turns: list[str] = []
    if (
        strategy is ProbeStrategy.MULTI_TURN_DRIFT
        and isinstance(facts, list)
        and len(facts) >= 2
    ):
        turn1_facts, turn2_facts = facts[0], facts[1]
        t1 = render_surface(turn1_facts, client=client, turn_hint="first message")
        t2 = render_surface(
            turn2_facts,
            client=client,
            turn_hint="a few minutes later, the customer writes back",
        )
        turns = [t1, self_critique(t2, client=client)]
        final_facts = turn2_facts
    else:
        surface = render_surface(facts, client=client)
        turns = [self_critique(surface, client=client)]
        final_facts = facts
    return turns, final_facts


def generate_probes(
    rules,
    policy,
    *,
    client,
    limit_per_rule: int,
) -> dict:
    """Run the full pipeline. Returns a report dict with probes + oracle stats.

    The report carries per-strategy `attempted` and `passed` counts (the raw
    numbers a reviewer needs to trust an oracle pass rate), plus the overall
    stat counters.
    """
    generated: list[Probe] = []
    stats = Counter()
    attempted = Counter()
    passed = Counter()

    for root in rules:
        for strategy in STRATEGY_ORDER:
            module = STRATEGY_MODULES[strategy.value]
            samples = module.sample(
                root, random.Random(str(root.rule_id)), max_count=limit_per_rule
            )
            items: list = list(samples)
            if strategy is ProbeStrategy.MULTI_TURN_DRIFT:
                items = [items[i:i + 2] for i in range(0, len(items) - 1, 2)]

            for sample_facts in items:
                stats["sampled"] += 1
                attempted[strategy.value] += 1
                if strategy is ProbeStrategy.MULTI_TURN_DRIFT:
                    facts_to_render = sample_facts
                    scenario_facts = sample_facts[1]
                else:
                    facts_to_render = sample_facts
                    scenario_facts = sample_facts

                scenario = ProbeScenario(
                    facts=scenario_facts,
                    target_rule_id=root.rule_id,
                    strategy=strategy,
                    difficulty_tier=(
                        3
                        if strategy
                        in (ProbeStrategy.FALSE_PREMISE, ProbeStrategy.MULTI_TURN_DRIFT)
                        else 2
                    ),
                )

                # C1: label deterministically, never by the model.
                try:
                    label = evaluate_rules(scenario, rules)
                except RuleEvaluationError as exc:
                    stats["unlabellable"] += 1
                    continue

                try:
                    turns, final_facts = _rendered_surfaces(
                        facts_to_render, client, strategy
                    )
                except Exception as exc:
                    stats["render_error"] += 1
                    continue

                # Oracle: the surface must match the facts vector. Scoped to the
                # facts the TARGET RULE's conditions reference — the full
                # entitlement vector (15 fields) can never all appear in a short
                # customer message, so checking all of it would report every
                # probe invalid. Same scoping as the hand-authored corpus.
                rule_attrs = {c.attribute for n in root.walk() for c in n.conditions}
                checkable = {k: v for k, v in final_facts.items() if k in rule_attrs}
                joined = " ".join(turns)
                oracle = oracle_check(checkable, joined)
                if not oracle.passed:
                    stats["oracle_failed"] += 1
                    continue
                stats["oracle_passed"] += 1

                probe_id = (
                    f"P-acme-{root.clause_ids[0].split(':')[1]}-"
                    f"{strategy.value}-{passed[strategy.value] + 1:03d}"
                )
                probe = Probe(
                    probe_id=probe_id,
                    scenario=scenario,
                    turns=turns,
                    expected_policy_stance=label.stance,
                    clause_ids=root.clause_ids,
                    style_seed_id=None,
                )
                generated.append(probe)
                passed[strategy.value] += 1

    return {
        "generated": generated,
        "stats": dict(stats),
        "attempted": dict(attempted),
        "passed": dict(passed),
    }