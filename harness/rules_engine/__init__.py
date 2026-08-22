"""The deterministic rules engine - commitment C1's implementation.

`evaluate_rules` derives every ground-truth probe label in pure Python
(DESIGN.md 3.1 step 2). Nothing in this package may import an LLM client, and
that is the point: C1 is "ground-truth labels are derived from rules in Python,
never from a model", and the cheapest way to keep a promise like that is to make
breaking it visible in an import line.

Re-exported so call sites read `from harness.rules_engine import evaluate_rules`.
"""

from harness.rules_engine.evaluate import (
    DEFAULT_STANCE,
    AmbiguousPolicyError,
    AppliedRule,
    EntitlementScopeError,
    FactTypeError,
    MalformedRuleTreeError,
    MissingAttributeError,
    PolicyLabel,
    RuleEvaluationError,
    condition_holds,
    evaluate_rules,
    policy_stance,
    resolve_entitlement,
    validate_rule_tree,
)

__all__ = [
    "evaluate_rules",
    "policy_stance",
    "condition_holds",
    "validate_rule_tree",
    "resolve_entitlement",
    "AppliedRule",
    "PolicyLabel",
    "DEFAULT_STANCE",
    "RuleEvaluationError",
    "MissingAttributeError",
    "FactTypeError",
    "AmbiguousPolicyError",
    "MalformedRuleTreeError",
    "EntitlementScopeError",
]
