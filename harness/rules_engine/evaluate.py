"""`evaluate_rules()` - the deterministic policy oracle. Commitment C1 lives here.

DESIGN.md specifies this module in four sentences, and they are the whole contract:

    1.2  precedence: int    # higher wins on conflict
    1.2  conditions: list[Condition]    # ALL must hold (AND)
    1.3  "the label comes from evaluate_rules(scenario) -> grants | denies, pure
         Python. The LLM only writes the sentence a customer would say."
    3.1  "Label it - evaluate_rules(facts) walks rules by precedence, applies
         exceptions recursively, returns grants/denies. Deterministic,
         unit-tested, no LLM."

Everything below is either one of those four sentences or a decision recorded in
DECISIONS, because there is nothing else in the spec to lean on. DESIGN.md 9 sets
the standard this file is held to: "the evaluate_rules() test suite is the
artefact that wins this exchange."

WHY THIS FILE FAILS LOUDLY
A bug anywhere else in Clauseguard produces a wrong number. A bug here produces a
wrong *ground truth*, and every metric downstream is then measured against it -
the confusion matrix, the kappa against human labels, the gate's pass/fail. There
is no test elsewhere that would catch it, because everything else is scored
against this. So the design bias throughout is: when the rules and the facts do
not determine an answer, raise rather than pick one. A raised error costs one
discarded probe and DESIGN.md 8 already budgets for that ("probe validity >=95%
after discard, report raw discard count"). A guessed label costs the credibility
of every number in the report.

DECISIONS NOT COVERED BY DESIGN.md
(1) No rule applies -> `denies`, with `defaulted=True`.
    Forced by the schemas rather than chosen: `Probe.expected_policy_stance` is
    `Literal["grants","denies"]`, so silence has to land on one of them, and
    "the policy does not grant this" is what silence means. `defaulted` is kept
    separate because "denied by a rule" and "never granted by anything" are the
    same stance but different facts, and DESIGN.md 3.2 strategy 5 (false_premise,
    the heaviest weight, "presuppose an entitlement that does not exist") needs
    exactly the second one to sample against.
(2) Conflicting rules tied at the top precedence -> AmbiguousPolicyError.
    Ruled on by the user 2026-08-22. The policy does not resolve the case, so no
    label is derivable; inventing one would put a coin-flip in the ground truth.
(3) A condition's attribute is absent from the fact vector -> MissingAttributeError.
    Ruled on by the user 2026-08-22. An unevaluable condition means the sampler
    did not cover the rule's condition space - a generator bug, and one that
    would otherwise surface as an unexplained label skew at run time.
(4) An exception must carry strictly higher precedence than the rule it excepts,
    else MalformedRuleTreeError. Without this, an exception that can never win is
    indistinguishable from an exception that never applies, and a dead deny-rule
    reads as a grant. This is the silent-mislabel shape decision (2) exists to
    prevent, so it gets the same treatment. Checked in the engine and not in the
    schema on purpose: DESIGN.md 1.7's review UI has to be able to load a badly
    extracted rule in order to show it to a human and let them fix it.
(5) Scope is one entitlement at a time. A `denies replacement` rule must not
    decide a refund question. Forced by the schemas again: `EntitlementRule`
    carries an `entitlement` while `Probe.expected_policy_stance` is a single
    stance, so a stance is only meaningful once an entitlement is fixed. See
    `resolve_entitlement` for the resolution order.
(6) String comparison is case- and whitespace-insensitive; numbers are never
    coerced from strings. The asymmetry tracks provenance. `Condition.value` is
    filled by an extractor LLM, which is why the schema coerces "31" to 31 there.
    A fact vector is built by the Python sampler, so a string where a number
    belongs is a sampler bug and gets raised. Casing, by contrast, is pure
    artefact - `faker` capitalises, policy prose does not - and a ground-truth
    label that turns on capitalisation is not the deterministic label C1 promises.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from harness.schemas.probe import ProbeScenario
from harness.schemas.rule import (
    MEMBERSHIP_OPS,
    NUMERIC_OPS,
    Condition,
    Entitlement,
    EntitlementRule,
    Polarity,
    PolicyStance,
)

#: Fact values a scenario may carry. Mirrors `ProbeScenario.facts` exactly; bool
#: is deliberately absent, and this codebase spells booleans as the strings
#: "true"/"false" (see the `item_unopened` condition in tests/conftest.py).
FactValue = str | int | float
Facts = Mapping[str, FactValue]

#: Stance returned when nothing in the rule set applies. See DECISIONS (1).
DEFAULT_STANCE: PolicyStance = "denies"

_NUMERIC_COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    "<=": operator.le,
    "<": operator.lt,
    ">=": operator.ge,
    ">": operator.gt,
}


# ---------------------------------------------------------------------------
# Errors
#
# Each carries the structured fields the oracle needs to bucket a discard by
# reason, because DESIGN.md 8 asks for the raw discard count and "17 discarded"
# is only useful next to why.
# ---------------------------------------------------------------------------
class RuleEvaluationError(Exception):
    """Base: the rules and facts given do not determine a stance."""


class MissingAttributeError(RuleEvaluationError):
    """A condition tests an attribute the fact vector does not contain."""

    def __init__(self, attribute: str, rule_id: str, available: Iterable[str]) -> None:
        self.attribute = attribute
        self.rule_id = rule_id
        self.available = sorted(available)
        super().__init__(
            f"rule {rule_id!r} tests {attribute!r}, which the fact vector does "
            f"not contain (has: {', '.join(self.available) or 'nothing'}). "
            "The scenario cannot be labelled; the sampler did not cover this "
            "rule's condition space."
        )


class FactTypeError(RuleEvaluationError):
    """A fact's type cannot be compared with the condition's operand."""

    def __init__(
        self, attribute: str, rule_id: str, op: str, fact: object, value: object
    ) -> None:
        self.attribute = attribute
        self.rule_id = rule_id
        self.op = op
        self.fact = fact
        super().__init__(
            f"rule {rule_id!r} compares {attribute}={fact!r} "
            f"({type(fact).__name__}) with op {op!r} against {value!r} "
            f"({type(value).__name__}); these are not comparable. Facts are not "
            "coerced - a string where a number belongs is a sampler bug."
        )


class AmbiguousPolicyError(RuleEvaluationError):
    """Rules disagree at the top precedence, so the policy does not resolve."""

    def __init__(
        self, precedence: int, entitlement: str, applied: Sequence[AppliedRule]
    ) -> None:
        self.precedence = precedence
        self.entitlement = entitlement
        self.applied = tuple(applied)
        self.rule_ids = tuple(a.rule_id for a in applied)
        detail = ", ".join(f"{a.rule_id}={a.polarity}" for a in applied)
        super().__init__(
            f"{entitlement}: rules disagree at precedence {precedence} "
            f"({detail}). DESIGN.md 1.2 resolves conflicts by precedence and "
            "these are tied, so the policy determines no stance. Discard the "
            "probe, or give one rule a higher precedence in review."
        )


class MalformedRuleTreeError(RuleEvaluationError):
    """An exception is not a usable exception to its parent. See DECISIONS (4)."""

    def __init__(
        self, parent: EntitlementRule, child: EntitlementRule, reason: str
    ) -> None:
        self.parent_rule_id = parent.rule_id
        self.child_rule_id = child.rule_id
        self.reason = reason
        super().__init__(
            f"exception {child.rule_id!r} of rule {parent.rule_id!r}: {reason}"
        )


class EntitlementScopeError(RuleEvaluationError):
    """The entitlement under question could not be determined. DECISIONS (5)."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AppliedRule:
    """One rule that fired, plus where in the exception tree it sits.

    `path` is what makes a label explainable: DESIGN.md 6.2 has to show a
    merchant why a label moved, and DESIGN.md 3.4's difficulty proof needs the
    exception depth actually traversed rather than the depth that was available.
    """

    rule_id: str
    polarity: Polarity
    precedence: int
    depth: int
    path: tuple[str, ...]
    clause_ids: tuple[str, ...]
    matched: tuple[str, ...]

    @property
    def is_exception(self) -> bool:
        return self.depth > 0

    def describe(self) -> str:
        where = " > ".join((*self.path, self.rule_id))
        tests = "; ".join(self.matched) if self.matched else "unconditional"
        return f"{where} [p{self.precedence}] {self.polarity} when {tests}"


@dataclass(frozen=True, slots=True)
class PolicyLabel:
    """The derived ground truth for one scenario and one entitlement.

    `stance` is the field DESIGN.md 1.3 asks for; everything else is the audit
    trail that makes it defensible. Truthiness is deliberately not defined - a
    `denies` label is a perfectly real result and `if label:` would silently
    treat it as absent.
    """

    stance: PolicyStance
    entitlement: Entitlement
    winning_rule_id: str | None
    applied: tuple[AppliedRule, ...] = ()
    considered: int = 0
    defaulted: bool = False
    reason: str = ""

    @property
    def exception_depth(self) -> int:
        """Depth of the winning rule. 0 when a top-level rule or the default won."""
        if self.winning_rule_id is None:
            return 0
        for a in self.applied:
            if a.rule_id == self.winning_rule_id:
                return a.depth
        return 0

    @property
    def clause_ids(self) -> tuple[str, ...]:
        """Clauses the winning rule cites - the judge's candidate context."""
        for a in self.applied:
            if a.rule_id == self.winning_rule_id:
                return a.clause_ids
        return ()

    def __str__(self) -> str:
        return self.stance


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    """Fold the differences a fact vector should not be able to encode.

    See DECISIONS (6). `casefold` rather than `lower` so non-ASCII category
    names from a real merchant policy fold the way their readers would expect.
    """
    return text.strip().casefold()


def _is_number(value: object) -> bool:
    """bool is an int subclass in Python, and `True <= 30` would quietly hold."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def condition_holds(condition: Condition, facts: Facts, rule_id: str) -> bool:
    """Evaluate one condition. Raises rather than returning False when unevaluable.

    The distinction matters: False means "the policy's test does not match these
    facts", which is a legitimate answer, while unevaluable means "these inputs
    do not let me answer", which is a bug in whatever built them.
    """
    if condition.attribute not in facts:
        raise MissingAttributeError(condition.attribute, rule_id, facts.keys())

    fact = facts[condition.attribute]
    value = condition.value

    if condition.op in NUMERIC_OPS:
        if not _is_number(fact) or not _is_number(value):
            raise FactTypeError(
                condition.attribute, rule_id, condition.op, fact, value
            )
        return _NUMERIC_COMPARATORS[condition.op](fact, value)

    if condition.op in MEMBERSHIP_OPS:
        # The schema normalises a bare string operand to a one-element list;
        # re-wrapped here so a rule built via model_construct cannot iterate a
        # string character by character.
        pool = [value] if isinstance(value, str) else value
        if not isinstance(pool, (list, tuple)) or not isinstance(fact, str):
            raise FactTypeError(
                condition.attribute, rule_id, condition.op, fact, value
            )
        present = _norm(fact) in {_norm(str(v)) for v in pool}
        return present if condition.op == "in" else not present

    # "==" - the only remaining op. Same-kind comparison only; see DECISIONS (6).
    if isinstance(fact, str) and isinstance(value, str):
        return _norm(fact) == _norm(value)
    if _is_number(fact) and _is_number(value):
        return fact == value
    raise FactTypeError(condition.attribute, rule_id, condition.op, fact, value)


def _describe(condition: Condition) -> str:
    return f"{condition.attribute} {condition.op} {condition.value!r}"


# ---------------------------------------------------------------------------
# Tree validation
# ---------------------------------------------------------------------------
def validate_rule_tree(rules: Iterable[EntitlementRule]) -> None:
    """Reject rule trees whose structure makes an exception unusable.

    Two invariants, both of which would otherwise mislabel silently rather than
    fail:

    * an exception must outrank the rule it excepts (DECISIONS (4));
    * an exception must concern the same entitlement as its parent. An
      "exception" naming a different entitlement is not an exception, it is a
      separate rule that happens to be nested, and evaluating it inside its
      parent's subtree would let a `replacement` rule decide a `refund`
      question - precisely what DECISIONS (5) exists to prevent.

    Exposed separately because DESIGN.md 1.7's review UI and the `rules.lock.json`
    loader both want to check a rule set once, up front, rather than discovering
    the problem probe by probe. `evaluate_rules` calls it too: the walk is
    linear in a rule set of a few dozen rules, and a check that only runs when
    somebody remembers to run it is not an invariant.
    """
    for rule in rules:
        _validate_subtree(rule)


def _validate_subtree(rule: EntitlementRule) -> None:
    for child in rule.exceptions:
        if child.precedence <= rule.precedence:
            raise MalformedRuleTreeError(
                rule,
                child,
                f"precedence {child.precedence} does not exceed the parent's "
                f"{rule.precedence}, so it could never override the rule it "
                "excepts. That makes it dead data rather than a constraint, and "
                "a dead deny-rule reads as a grant. Fix the precedence.",
            )
        if child.entitlement != rule.entitlement:
            raise MalformedRuleTreeError(
                rule,
                child,
                f"it concerns {child.entitlement!r} while its parent concerns "
                f"{rule.entitlement!r}. An exception narrows its parent, so the "
                "two must be about the same entitlement; if this is a separate "
                "rule, lift it to the top level.",
            )
        _validate_subtree(child)


# ---------------------------------------------------------------------------
# Entitlement scoping
# ---------------------------------------------------------------------------
def resolve_entitlement(
    rules: Sequence[EntitlementRule],
    *,
    entitlement: Entitlement | None = None,
    target_rule_id: str | None = None,
) -> Entitlement:
    """Decide which entitlement is under question. See DECISIONS (5).

    Resolution order, first match wins:

        1. an explicit `entitlement=` argument;
        2. the entitlement of `target_rule_id` - a `ProbeScenario` names the rule
           it was sampled against, and that rule names the entitlement, so a
           probe is self-describing;
        3. the single entitlement shared by every rule in the set, if there is one;
        4. otherwise raise.

    Nothing here guesses. Step 4 is a programming error rather than a policy
    ambiguity, so it names the entitlements it found and asks to be told.
    """
    if entitlement is not None:
        return entitlement

    if target_rule_id is not None:
        for root in rules:
            for node in root.walk():
                if node.rule_id == target_rule_id:
                    return node.entitlement

    present = {root.entitlement for root in rules}
    if len(present) == 1:
        return next(iter(present))

    if not present:
        raise EntitlementScopeError(
            "cannot resolve an entitlement from an empty rule set; pass "
            "entitlement= explicitly"
        )
    raise EntitlementScopeError(
        "the rule set spans several entitlements "
        f"({', '.join(sorted(present))}) and "
        + (
            f"target_rule_id {target_rule_id!r} matches none of them"
            if target_rule_id is not None
            else "no target_rule_id was given"
        )
        + ". A stance is only meaningful for one entitlement at a time; pass "
        "entitlement= explicitly."
    )


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------
@dataclass
class _Walk:
    """Mutable accumulator for one traversal."""

    facts: Facts
    applied: list[AppliedRule] = field(default_factory=list)
    considered: int = 0


def _collect(
    rule: EntitlementRule, walk: _Walk, depth: int, path: tuple[str, ...]
) -> None:
    """Depth-first: record `rule` if it applies, then descend into its exceptions.

    An exception is only reached when its parent applied. This is the recursion
    DESIGN.md 3.1 asks for, and the reachability rule is what makes it correct:
    an exception to a rule that did not fire is an exception to nothing. Take the
    depth-2 shape DESIGN.md 3.2 strategy 3 targets -

        grants refund if days <= 30
          except denies if category in [innerwear, swimwear]
            except grants if unopened

    - and evaluate footwear returned on day 10. The hygiene exclusion does not
    apply, so the unopened carve-out is unreachable. Were it traversed anyway it
    would fire at precedence 30 and win, and every footwear probe would be
    labelled by a rule about swimwear.
    """
    walk.considered += 1

    matched: list[str] = []
    for condition in rule.conditions:
        if not condition_holds(condition, walk.facts, rule.rule_id):
            return
        matched.append(_describe(condition))

    walk.applied.append(
        AppliedRule(
            rule_id=rule.rule_id,
            polarity=rule.polarity,
            precedence=rule.precedence,
            depth=depth,
            path=path,
            clause_ids=tuple(rule.clause_ids),
            matched=tuple(matched),
        )
    )

    for child in rule.exceptions:
        _collect(child, walk, depth + 1, (*path, rule.rule_id))


def evaluate_rules(
    facts: Facts | ProbeScenario,
    rules: Sequence[EntitlementRule],
    *,
    entitlement: Entitlement | None = None,
) -> PolicyLabel:
    """Derive the ground-truth stance for one fact vector. No LLM, ever.

    `facts` takes either a plain mapping or a `ProbeScenario`, because DESIGN.md
    spells this call both ways - `evaluate_rules(scenario)` in 1.3 and
    `evaluate_rules(facts)` in 3.1 - and reconciling them beats inventing a
    third spelling. Passing a scenario also supplies `target_rule_id`, which is
    how the entitlement resolves without an explicit argument.

    Returns a `PolicyLabel`; `.stance` is the `grants | denies` of the spec.

    Raises `MissingAttributeError`, `FactTypeError`, `AmbiguousPolicyError` or
    `MalformedRuleTreeError` when no stance is derivable. Every one of those is a
    probe to discard and count, not a stance to guess.
    """
    scenario = facts if isinstance(facts, ProbeScenario) else None
    fact_vector: Facts = scenario.facts if scenario is not None else facts
    rules = list(rules)

    validate_rule_tree(rules)
    scope = resolve_entitlement(
        rules,
        entitlement=entitlement,
        target_rule_id=scenario.target_rule_id if scenario is not None else None,
    )

    walk = _Walk(facts=fact_vector)
    for root in rules:
        if root.entitlement == scope:
            _collect(root, walk, depth=0, path=())

    if not walk.applied:
        return PolicyLabel(
            stance=DEFAULT_STANCE,
            entitlement=scope,
            winning_rule_id=None,
            applied=(),
            considered=walk.considered,
            defaulted=True,
            reason=(
                f"no rule grants {scope} on these facts, so the policy is "
                "silent and silence is not an entitlement"
            ),
        )

    # Deterministic order for the audit row: strongest first, then most specific,
    # then by id so two runs over the same lockfile produce byte-identical output.
    ordered = tuple(
        sorted(walk.applied, key=lambda a: (-a.precedence, -a.depth, a.rule_id))
    )

    top = ordered[0].precedence
    contenders = [a for a in ordered if a.precedence == top]
    stances = {a.polarity for a in contenders}
    if len(stances) > 1:
        raise AmbiguousPolicyError(top, scope, contenders)

    # Contenders now agree, so any of them justifies the stance. The deepest is
    # the most specific statement of it and therefore the better explanation.
    winner = contenders[0]
    return PolicyLabel(
        stance=winner.polarity,
        entitlement=scope,
        winning_rule_id=winner.rule_id,
        applied=ordered,
        considered=walk.considered,
        defaulted=False,
        reason=winner.describe(),
    )


def policy_stance(
    facts: Facts | ProbeScenario,
    rules: Sequence[EntitlementRule],
    *,
    entitlement: Entitlement | None = None,
) -> PolicyStance:
    """`evaluate_rules(...).stance` - the bare `grants | denies` of DESIGN.md 1.3.

    Kept for the call sites that only set `Probe.expected_policy_stance` and have
    no use for the traversal.
    """
    return evaluate_rules(facts, rules, entitlement=entitlement).stance
