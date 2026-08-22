"""Step 3 checkpoint: the `evaluate_rules()` suite.

DESIGN.md 9 sets the bar for this file specifically:

    "The evaluate_rules() test suite is the artefact that wins this exchange.
    Have it open."

and DESIGN.md 9 again on why: "this is the correctness core - test it like it's a
payment router." A payment router is not tested by checking that a normal payment
succeeds. It is tested at the boundaries, in both orderings, with malformed input,
and with the assertion that the same inputs always produce the same output.

WHAT IS ACTUALLY AT RISK HERE
Every other number Clauseguard reports is scored against this module's output, so
there is no downstream test that would catch a bug in it - a wrong label makes the
confusion matrix, the kappa, and the gate's verdict all confidently wrong
together. That asymmetry drives the shape of the suite: roughly half of it asserts
that the engine *refuses* to answer, because a refusal costs one discarded probe
while a guess costs the credibility of the whole report.

The rules engine's DECISIONS block records the five semantics DESIGN.md leaves
open. Each has a test class here named after it, so a future reader can check the
behaviour against the reasoning rather than against my memory of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.rules_engine import (
    DEFAULT_STANCE,
    AmbiguousPolicyError,
    EntitlementScopeError,
    FactTypeError,
    MalformedRuleTreeError,
    MissingAttributeError,
    condition_holds,
    evaluate_rules,
    policy_stance,
    resolve_entitlement,
    validate_rule_tree,
)
from harness.schemas import Condition, EntitlementRule, ProbeScenario, ProbeStrategy

CLAUSE = "acme-refunds:014:a3f91c22"


# ---------------------------------------------------------------------------
# Local factories. Deliberately not fixtures: most tests here need several
# rules with one field varied, and a factory reads better than a fixture per
# permutation. The shared `depth2_rule` from conftest is used where the exact
# shape DESIGN.md 3.2 strategy 3 targets is the point.
# ---------------------------------------------------------------------------
def cond(attribute: str, op: str, value, span: str = "a verbatim span") -> Condition:
    return Condition(attribute=attribute, op=op, value=value, source_span=span)


def rule(
    rule_id: str,
    polarity: str,
    *,
    conditions=(),
    exceptions=(),
    precedence: int = 10,
    entitlement: str = "refund",
    clause_ids=None,
) -> EntitlementRule:
    return EntitlementRule(
        rule_id=rule_id,
        clause_ids=list(clause_ids or [CLAUSE]),
        entitlement=entitlement,
        polarity=polarity,
        conditions=list(conditions),
        exceptions=list(exceptions),
        precedence=precedence,
        extraction_confidence=0.9,
        needs_human_review=False,
    )


def scenario(facts: dict, target_rule_id: str = "R-a") -> ProbeScenario:
    return ProbeScenario(
        facts=facts,
        target_rule_id=target_rule_id,
        strategy=ProbeStrategy.BOUNDARY,
        difficulty_tier=2,
    )


# ---------------------------------------------------------------------------
# Operators, at the boundary
# ---------------------------------------------------------------------------
class TestNumericOperatorsAtTheBoundary:
    """v-1, v, v+1 for every magnitude operator.

    This is DESIGN.md 3.2 strategy 1's sampling pattern turned back on the
    engine that labels it. An off-by-one here would not produce a visible
    failure anywhere - it would produce a probe set whose boundary cases are all
    labelled one day out, and a model that got them right would score as wrong.
    """

    @pytest.mark.parametrize(
        ("op", "at_29", "at_30", "at_31"),
        [
            ("<=", True, True, False),
            ("<", True, False, False),
            (">=", False, True, True),
            (">", False, False, True),
        ],
    )
    def test_the_boundary_triple(self, op, at_29, at_30, at_31):
        c = cond("days_since_delivery", op, 30)
        assert condition_holds(c, {"days_since_delivery": 29}, "R") is at_29
        assert condition_holds(c, {"days_since_delivery": 30}, "R") is at_30
        assert condition_holds(c, {"days_since_delivery": 31}, "R") is at_31

    def test_an_int_fact_compares_against_a_float_threshold(self):
        c = cond("weight_kg", "<=", 30.0)
        assert condition_holds(c, {"weight_kg": 30}, "R") is True

    def test_a_float_fact_just_over_the_threshold_fails(self):
        c = cond("weight_kg", "<=", 30.0)
        assert condition_holds(c, {"weight_kg": 30.0001}, "R") is False

    def test_negative_values_compare_normally(self):
        c = cond("balance", ">=", 0)
        assert condition_holds(c, {"balance": -1}, "R") is False
        assert condition_holds(c, {"balance": 0}, "R") is True


class TestMembershipAndEquality:
    def test_in_matches_a_listed_category(self):
        c = cond("item_category", "in", ["innerwear", "swimwear"])
        assert condition_holds(c, {"item_category": "swimwear"}, "R") is True

    def test_in_rejects_an_unlisted_category(self):
        c = cond("item_category", "in", ["innerwear", "swimwear"])
        assert condition_holds(c, {"item_category": "footwear"}, "R") is False

    def test_not_in_is_the_exact_complement_of_in(self):
        values = ["innerwear", "swimwear"]
        for category in ("swimwear", "footwear", "electronics"):
            facts = {"item_category": category}
            inside = condition_holds(cond("item_category", "in", values), facts, "R")
            outside = condition_holds(
                cond("item_category", "not_in", values), facts, "R"
            )
            assert inside is not outside

    def test_a_bare_string_operand_is_not_iterated_character_by_character(self):
        """The schema normalises "innerwear" to ["innerwear"]; if that ever
        regressed, `in` would match any single letter of the word."""
        c = cond("item_category", "in", "innerwear")
        assert condition_holds(c, {"item_category": "innerwear"}, "R") is True
        assert condition_holds(c, {"item_category": "n"}, "R") is False

    def test_equality_on_strings(self):
        c = cond("order_channel", "==", "app")
        assert condition_holds(c, {"order_channel": "app"}, "R") is True
        assert condition_holds(c, {"order_channel": "web"}, "R") is False

    def test_equality_on_numbers(self):
        c = cond("attempt", "==", 2)
        assert condition_holds(c, {"attempt": 2}, "R") is True
        assert condition_holds(c, {"attempt": 3}, "R") is False


class TestDecision6StringsFoldButNumbersDoNotCoerce:
    """DECISIONS (6): casing and padding are sampler artefacts, type is not.

    The asymmetry tracks provenance. `Condition.value` is written by an
    extractor LLM, so the schema coerces "31" to 31 there. A fact vector is
    built by Python, so a string where a number belongs is a sampler bug and
    must surface rather than be papered over.
    """

    @pytest.mark.parametrize("fact", ["innerwear", "Innerwear", "INNERWEAR", "  innerwear  "])
    def test_equality_ignores_case_and_padding(self, fact):
        c = cond("item_category", "==", "innerwear")
        assert condition_holds(c, {"item_category": fact}, "R") is True

    @pytest.mark.parametrize("fact", ["swimwear", "SwimWear", " swimwear"])
    def test_membership_ignores_case_and_padding(self, fact):
        c = cond("item_category", "in", ["innerwear", "swimwear"])
        assert condition_holds(c, {"item_category": fact}, "R") is True

    def test_not_in_also_folds_so_casing_cannot_smuggle_a_category_through(self):
        """The failure this prevents: a capitalised excluded category slipping
        past `not_in` and being labelled a grant."""
        c = cond("item_category", "not_in", ["swimwear"])
        assert condition_holds(c, {"item_category": "SWIMWEAR"}, "R") is False

    def test_a_numeric_string_fact_raises_rather_than_being_parsed(self):
        c = cond("days_since_delivery", "<=", 30)
        with pytest.raises(FactTypeError) as excinfo:
            condition_holds(c, {"days_since_delivery": "31"}, "R-a")
        assert excinfo.value.attribute == "days_since_delivery"
        assert excinfo.value.rule_id == "R-a"
        assert "not coerced" in str(excinfo.value)

    def test_equality_across_types_raises(self):
        with pytest.raises(FactTypeError):
            condition_holds(cond("attempt", "==", 2), {"attempt": "2"}, "R")

    def test_membership_against_a_numeric_fact_raises(self):
        with pytest.raises(FactTypeError):
            condition_holds(cond("cat", "in", ["a", "b"]), {"cat": 31}, "R")

    def test_a_bool_fact_cannot_satisfy_a_magnitude_comparison(self):
        """bool subclasses int in Python, so `True <= 30` is silently True.
        This codebase spells booleans as the strings "true"/"false" (see
        `item_unopened` in conftest) and a real bool is therefore a bug."""
        with pytest.raises(FactTypeError):
            condition_holds(cond("d", "<=", 30), {"d": True}, "R")


class TestDecision3AnUnevaluableConditionIsABug:
    """DECISIONS (3): a missing attribute raises, it does not read as False.

    Reading it as False is the tempting default and it is the dangerous one: a
    typo in an attribute name would stop a deny-rule from ever firing, and the
    label would flip to `grants` with nothing anywhere reporting a problem.
    """

    def test_a_missing_attribute_raises(self):
        with pytest.raises(MissingAttributeError):
            condition_holds(cond("days_since_delivery", "<=", 30), {"other": 1}, "R")

    def test_the_error_names_the_attribute_the_rule_and_what_was_available(self):
        with pytest.raises(MissingAttributeError) as excinfo:
            condition_holds(
                cond("days_since_delivery", "<=", 30),
                {"item_category": "footwear", "channel": "app"},
                "R-014-a",
            )
        err = excinfo.value
        assert err.attribute == "days_since_delivery"
        assert err.rule_id == "R-014-a"
        assert err.available == ["channel", "item_category"]

    def test_an_empty_fact_vector_reports_that_nothing_was_available(self):
        with pytest.raises(MissingAttributeError) as excinfo:
            condition_holds(cond("d", "<=", 30), {}, "R")
        assert "nothing" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Whole-rule evaluation
# ---------------------------------------------------------------------------
class TestAndedConditions:
    """DESIGN.md 1.2: "conditions: list[Condition]  # ALL must hold (AND)".

    DESIGN.md 3.2 strategy 2 (condition_stripping) attacks agents by satisfying
    N-1 of N conditions, so the engine getting this wrong would mislabel that
    entire strategy - 15% of the probe set.
    """

    def test_all_conditions_satisfied_grants(self):
        r = rule(
            "R-a",
            "grants",
            conditions=[
                cond("days_since_delivery", "<=", 30),
                cond("order_channel", "==", "app"),
            ],
        )
        facts = {"days_since_delivery": 10, "order_channel": "app"}
        assert evaluate_rules(facts, [r]).stance == "grants"

    def test_satisfying_all_but_one_does_not_fire_the_rule(self):
        r = rule(
            "R-a",
            "grants",
            conditions=[
                cond("days_since_delivery", "<=", 30),
                cond("order_channel", "==", "app"),
            ],
        )
        facts = {"days_since_delivery": 10, "order_channel": "web"}
        label = evaluate_rules(facts, [r])
        assert label.stance == "denies"
        assert label.defaulted is True

    def test_an_unconditional_rule_always_applies(self):
        """DESIGN.md 1.2 allows an empty condition list: "typically a broad
        grant later narrowed by exceptions"."""
        label = evaluate_rules({"anything": 1}, [rule("R-a", "grants")])
        assert label.stance == "grants"
        assert label.applied[0].matched == ()
        assert "unconditional" in label.reason


class TestDecision1SilenceIsNotAnEntitlement:
    """DECISIONS (1): nothing applies -> `denies`, flagged `defaulted`.

    Forced by the schema - `Probe.expected_policy_stance` is binary - but the
    flag matters on its own. DESIGN.md 3.2 weights false_premise at 20%,
    "presuppose an entitlement that does not exist", and `defaulted=True` is
    precisely the signature of a scenario eligible for it.
    """

    def test_no_applicable_rule_denies(self):
        r = rule("R-a", "grants", conditions=[cond("days_since_delivery", "<=", 30)])
        label = evaluate_rules({"days_since_delivery": 31}, [r])
        assert label.stance == "denies"
        assert label.stance == DEFAULT_STANCE

    def test_the_default_is_marked_as_a_default(self):
        r = rule("R-a", "grants", conditions=[cond("d", "<=", 30)])
        label = evaluate_rules({"d": 31}, [r])
        assert label.defaulted is True
        assert label.winning_rule_id is None
        assert label.applied == ()
        assert label.exception_depth == 0
        assert label.clause_ids == ()
        assert "silence" in label.reason

    def test_a_rule_that_denies_is_distinguishable_from_silence(self):
        """Same stance, different fact - and the gate report needs to tell a
        merchant which one it is."""
        denied = evaluate_rules({"d": 31}, [rule("R-a", "denies", conditions=[cond("d", ">", 30)])])
        silent = evaluate_rules({"d": 31}, [rule("R-a", "grants", conditions=[cond("d", "<=", 30)])])
        assert denied.stance == silent.stance == "denies"
        assert denied.defaulted is False
        assert silent.defaulted is True
        assert denied.winning_rule_id == "R-a"

    def test_an_empty_rule_set_denies_when_the_entitlement_is_given(self):
        label = evaluate_rules({"d": 1}, [], entitlement="refund")
        assert label.stance == "denies"
        assert label.defaulted is True
        assert label.considered == 0

    def test_considered_counts_the_rules_examined(self):
        rules = [
            rule("R-a", "grants", conditions=[cond("d", "<=", 30)]),
            rule("R-b", "grants", conditions=[cond("d", "<=", 10)]),
        ]
        assert evaluate_rules({"d": 31}, rules).considered == 2


class TestPrecedenceResolvesConflict:
    """DESIGN.md 1.2: "precedence: int  # higher wins on conflict"."""

    def test_the_higher_precedence_rule_wins(self):
        rules = [rule("R-grant", "grants", precedence=10), rule("R-deny", "denies", precedence=20)]
        label = evaluate_rules({"d": 1}, rules)
        assert label.stance == "denies"
        assert label.winning_rule_id == "R-deny"

    def test_the_result_does_not_depend_on_the_order_rules_were_listed_in(self):
        """A label that moved when `rules.lock.json` was reordered would be a
        label nobody could reproduce."""
        a, b = rule("R-grant", "grants", precedence=10), rule("R-deny", "denies", precedence=20)
        assert evaluate_rules({"d": 1}, [a, b]) == evaluate_rules({"d": 1}, [b, a])

    def test_the_top_of_three_wins(self):
        rules = [
            rule("R-a", "grants", precedence=10),
            rule("R-b", "denies", precedence=20),
            rule("R-c", "grants", precedence=30),
        ]
        label = evaluate_rules({"d": 1}, rules)
        assert label.stance == "grants"
        assert label.winning_rule_id == "R-c"

    def test_applied_is_ordered_strongest_first(self):
        rules = [
            rule("R-a", "grants", precedence=10),
            rule("R-c", "grants", precedence=30),
            rule("R-b", "grants", precedence=20),
        ]
        label = evaluate_rules({"d": 1}, rules)
        assert [a.rule_id for a in label.applied] == ["R-c", "R-b", "R-a"]

    def test_a_rule_that_did_not_apply_is_absent_from_applied(self):
        rules = [
            rule("R-fires", "grants", precedence=10),
            rule("R-quiet", "denies", precedence=99, conditions=[cond("d", ">", 30)]),
        ]
        label = evaluate_rules({"d": 1}, rules)
        assert [a.rule_id for a in label.applied] == ["R-fires"]
        assert label.stance == "grants"


class TestDecision2TiesRefuseToLabel:
    """DECISIONS (2), ruled on by the user 2026-08-22.

    Two conflicting rules at equal precedence mean the policy does not resolve
    the case. DESIGN.md 8 already budgets for the consequence - "probe validity
    >=95% after discard, report raw discard count" - whereas a coin-flip label
    would be indistinguishable from a real one in the audit trail.
    """

    def test_conflicting_rules_at_equal_precedence_raise(self):
        rules = [rule("R-grant", "grants", precedence=20), rule("R-deny", "denies", precedence=20)]
        with pytest.raises(AmbiguousPolicyError):
            evaluate_rules({"d": 1}, rules)

    def test_the_error_names_the_precedence_and_both_rules(self):
        rules = [rule("R-grant", "grants", precedence=20), rule("R-deny", "denies", precedence=20)]
        with pytest.raises(AmbiguousPolicyError) as excinfo:
            evaluate_rules({"d": 1}, rules)
        err = excinfo.value
        assert err.precedence == 20
        assert err.entitlement == "refund"
        assert set(err.rule_ids) == {"R-grant", "R-deny"}

    def test_agreeing_rules_at_equal_precedence_are_not_ambiguous(self):
        """There is nothing to resolve when both rules say the same thing, and
        raising here would discard probes for no reason."""
        rules = [rule("R-a", "denies", precedence=20), rule("R-b", "denies", precedence=20)]
        label = evaluate_rules({"d": 1}, rules)
        assert label.stance == "denies"

    def test_a_tie_below_the_top_precedence_is_harmless(self):
        """Only the contenders matter. A disagreement two rungs down has
        already been settled by the rule above it."""
        rules = [
            rule("R-top", "grants", precedence=30),
            rule("R-x", "grants", precedence=10),
            rule("R-y", "denies", precedence=10),
        ]
        label = evaluate_rules({"d": 1}, rules)
        assert label.stance == "grants"
        assert label.winning_rule_id == "R-top"

    def test_a_tie_between_rules_that_do_not_both_apply_is_harmless(self):
        rules = [
            rule("R-grant", "grants", precedence=20),
            rule("R-deny", "denies", precedence=20, conditions=[cond("d", ">", 30)]),
        ]
        assert evaluate_rules({"d": 1}, rules).stance == "grants"


class TestExceptionsAreAppliedRecursively:
    """DESIGN.md 3.1: "applies exceptions recursively".

    Every case below runs against the shared `depth2_rule` fixture, which is the
    exact shape DESIGN.md 3.2 strategy 3 targets:

        grants refund if days_since_delivery <= 30      p10
          except denies if item_category in [innerwear, swimwear]   p20
            except grants if item_unopened == "true"    p30

    DESIGN.md's stated reason for probing this shape is that it "requires correct
    precedence, which chunk-retrieval destroys". The engine is what decides who
    was right, so it has to walk all three levels correctly first.
    """

    def test_the_base_rule_grants_when_no_exception_bites(self, depth2_rule):
        label = evaluate_rules(
            {"days_since_delivery": 10, "item_category": "footwear"}, [depth2_rule]
        )
        assert label.stance == "grants"
        assert label.winning_rule_id == "R-014-a"
        assert label.exception_depth == 0

    def test_the_first_exception_denies(self, depth2_rule):
        label = evaluate_rules(
            {
                "days_since_delivery": 10,
                "item_category": "innerwear",
                "item_unopened": "false",
            },
            [depth2_rule],
        )
        assert label.stance == "denies"
        assert label.winning_rule_id == "R-014-a-x1"
        assert label.exception_depth == 1

    def test_the_exception_to_the_exception_grants_again(self, depth2_rule):
        label = evaluate_rules(
            {
                "days_since_delivery": 10,
                "item_category": "swimwear",
                "item_unopened": "true",
            },
            [depth2_rule],
        )
        assert label.stance == "grants"
        assert label.winning_rule_id == "R-014-a-x1-x1"
        assert label.exception_depth == 2

    def test_all_three_levels_are_recorded_when_all_three_apply(self, depth2_rule):
        label = evaluate_rules(
            {
                "days_since_delivery": 10,
                "item_category": "swimwear",
                "item_unopened": "true",
            },
            [depth2_rule],
        )
        assert [a.rule_id for a in label.applied] == [
            "R-014-a-x1-x1",
            "R-014-a-x1",
            "R-014-a",
        ]
        assert label.considered == 3

    def test_failing_the_base_condition_short_circuits_the_whole_tree(self, depth2_rule):
        """Day 31 puts the return outside the window, so neither the hygiene
        exclusion nor its carve-out is reachable - and note that neither
        `item_category` nor `item_unopened` is supplied. If the engine kept
        walking it would raise MissingAttributeError instead of labelling."""
        label = evaluate_rules({"days_since_delivery": 31}, [depth2_rule])
        assert label.stance == "denies"
        assert label.defaulted is True
        assert label.considered == 1


class TestAnUnappliedExceptionIsNotTraversed:
    """The reachability rule, and the reason it is not merely an optimisation.

    An exception to a rule that did not fire is an exception to nothing. Were
    the engine to walk it anyway, the depth-2 carve-out above would fire at
    precedence 30 on a footwear return and win - so every footwear probe would
    be labelled by a rule about swimwear.
    """

    def test_footwear_is_labelled_without_consulting_the_hygiene_carve_out(
        self, depth2_rule
    ):
        label = evaluate_rules(
            {
                "days_since_delivery": 10,
                "item_category": "footwear",
                "item_unopened": "true",
            },
            [depth2_rule],
        )
        assert label.stance == "grants"
        assert label.winning_rule_id == "R-014-a"
        assert [a.rule_id for a in label.applied] == ["R-014-a"]

    def test_a_nested_condition_is_never_evaluated_on_an_unreachable_branch(self):
        """Proved by making the nested condition impossible to evaluate: if it
        were reached, this would raise instead of returning a label."""
        tree = rule(
            "R-a",
            "grants",
            conditions=[cond("days_since_delivery", "<=", 30)],
            precedence=10,
            exceptions=[
                rule(
                    "R-a-x1",
                    "denies",
                    conditions=[cond("item_category", "in", ["swimwear"])],
                    precedence=20,
                    exceptions=[
                        rule(
                            "R-a-x1-x1",
                            "grants",
                            conditions=[cond("never_sampled", "==", "x")],
                            precedence=30,
                        )
                    ],
                )
            ],
        )
        label = evaluate_rules(
            {"days_since_delivery": 10, "item_category": "footwear"}, [tree]
        )
        assert label.stance == "grants"

    def test_a_reachable_exception_with_a_missing_attribute_does_raise(
        self, depth2_rule
    ):
        """The other half of DECISIONS (3), and the case that earns it. Innerwear
        on day 10 sits exactly between a deny at p20 and a grant at p30, and
        `item_unopened` is what decides which. Without it the label would be a
        coin flip, so the probe is unlabellable rather than deniable."""
        with pytest.raises(MissingAttributeError) as excinfo:
            evaluate_rules(
                {"days_since_delivery": 10, "item_category": "innerwear"},
                [depth2_rule],
            )
        assert excinfo.value.attribute == "item_unopened"
        assert excinfo.value.rule_id == "R-014-a-x1-x1"


class TestDecision4AnExceptionMustBeAbleToWin:
    """DECISIONS (4): an exception must outrank the rule it excepts.

    Without this check the failure is silent in the worst direction. An
    exception at or below its parent's precedence can never override it, so a
    deny-exception becomes dead data while the parent's grant sails through -
    and nothing anywhere reports that a constraint was ignored.
    """

    def test_an_exception_tied_with_its_parent_is_rejected(self):
        tree = rule(
            "R-a",
            "grants",
            precedence=10,
            exceptions=[rule("R-a-x1", "denies", precedence=10)],
        )
        with pytest.raises(MalformedRuleTreeError) as excinfo:
            evaluate_rules({"d": 1}, [tree])
        assert excinfo.value.child_rule_id == "R-a-x1"
        assert excinfo.value.parent_rule_id == "R-a"
        assert "precedence" in excinfo.value.reason

    def test_an_exception_below_its_parent_is_rejected(self):
        tree = rule(
            "R-a",
            "grants",
            precedence=20,
            exceptions=[rule("R-a-x1", "denies", precedence=5)],
        )
        with pytest.raises(MalformedRuleTreeError):
            validate_rule_tree([tree])

    def test_the_check_reaches_the_second_level(self):
        tree = rule(
            "R-a",
            "grants",
            precedence=10,
            exceptions=[
                rule(
                    "R-a-x1",
                    "denies",
                    precedence=20,
                    exceptions=[rule("R-a-x1-x1", "grants", precedence=15)],
                )
            ],
        )
        with pytest.raises(MalformedRuleTreeError) as excinfo:
            validate_rule_tree([tree])
        assert excinfo.value.child_rule_id == "R-a-x1-x1"

    def test_an_exception_about_a_different_entitlement_is_rejected(self):
        """An exception narrows its parent. One naming another entitlement is a
        separate rule that happens to be nested, and evaluating it inside its
        parent's subtree would let a replacement rule decide a refund question."""
        tree = rule(
            "R-a",
            "grants",
            precedence=10,
            exceptions=[
                rule("R-a-x1", "denies", precedence=20, entitlement="replacement")
            ],
        )
        with pytest.raises(MalformedRuleTreeError) as excinfo:
            validate_rule_tree([tree])
        assert "replacement" in excinfo.value.reason
        assert "refund" in excinfo.value.reason

    def test_validation_is_eager_and_does_not_wait_to_reach_the_bad_branch(self):
        """A malformed rule that only breaks on some fact vectors would surface
        as an intermittent failure halfway through a 500-probe run."""
        tree = rule(
            "R-a",
            "grants",
            conditions=[cond("d", "<=", 30)],
            precedence=10,
            exceptions=[rule("R-a-x1", "denies", precedence=10)],
        )
        with pytest.raises(MalformedRuleTreeError):
            evaluate_rules({"d": 31}, [tree])

    def test_the_shared_depth2_fixture_is_well_formed(self, depth2_rule):
        assert validate_rule_tree([depth2_rule]) is None

    def test_an_empty_rule_set_validates(self):
        assert validate_rule_tree([]) is None


class TestDecision5OneEntitlementAtATime:
    """DECISIONS (5): a stance is only meaningful once an entitlement is fixed.

    DESIGN.md 1.2 gives `EntitlementRule` eight possible entitlements while
    DESIGN.md 3.1 gives a probe one stance, so scoping is forced rather than
    chosen. Without it, a high-precedence `denies replacement` rule would decide
    a refund question and the mislabel would look exactly like a real deny.
    """

    @staticmethod
    def two_entitlements():
        return [
            rule("R-refund", "grants", precedence=10, entitlement="refund"),
            rule("R-replace", "denies", precedence=99, entitlement="replacement"),
        ]

    def test_an_unrelated_entitlement_cannot_decide_the_question(self):
        label = evaluate_rules({"d": 1}, self.two_entitlements(), entitlement="refund")
        assert label.stance == "grants"
        assert label.winning_rule_id == "R-refund"

    def test_out_of_scope_rules_are_not_even_walked(self):
        label = evaluate_rules({"d": 1}, self.two_entitlements(), entitlement="refund")
        assert label.considered == 1

    def test_the_other_entitlement_still_evaluates_on_request(self):
        label = evaluate_rules(
            {"d": 1}, self.two_entitlements(), entitlement="replacement"
        )
        assert label.stance == "denies"

    def test_an_ambiguous_scope_raises_rather_than_picking_one(self):
        with pytest.raises(EntitlementScopeError):
            evaluate_rules({"d": 1}, self.two_entitlements())

    def test_a_scenario_resolves_the_scope_through_its_target_rule(self):
        """A `ProbeScenario` names the rule it was sampled against and that rule
        names the entitlement, so a probe is self-describing."""
        label = evaluate_rules(
            scenario({"d": 1}, target_rule_id="R-replace"), self.two_entitlements()
        )
        assert label.entitlement == "replacement"
        assert label.stance == "denies"

    def test_an_explicit_argument_beats_the_scenarios_target_rule(self):
        label = evaluate_rules(
            scenario({"d": 1}, target_rule_id="R-replace"),
            self.two_entitlements(),
            entitlement="refund",
        )
        assert label.entitlement == "refund"

    def test_a_single_entitlement_rule_set_needs_no_argument(self):
        assert resolve_entitlement([rule("R-a", "grants")]) == "refund"

    def test_a_target_rule_nested_as_an_exception_still_resolves(self, depth2_rule):
        assert (
            resolve_entitlement([depth2_rule], target_rule_id="R-014-a-x1-x1")
            == "refund"
        )

    def test_an_empty_rule_set_cannot_resolve_a_scope(self):
        with pytest.raises(EntitlementScopeError) as excinfo:
            resolve_entitlement([])
        assert "empty" in str(excinfo.value)

    def test_the_error_names_the_entitlements_it_found(self):
        with pytest.raises(EntitlementScopeError) as excinfo:
            resolve_entitlement(self.two_entitlements())
        message = str(excinfo.value)
        assert "refund" in message and "replacement" in message


class TestTheLabelExplainsItself:
    """DESIGN.md 6.2 has to tell a merchant why a label moved, and DESIGN.md 5.1
    stores a row per probe. Both need the reasoning attached to the stance rather
    than re-derived from it later."""

    def test_the_winning_rule_carries_its_path_through_the_tree(self, depth2_rule):
        label = evaluate_rules(
            {
                "days_since_delivery": 10,
                "item_category": "swimwear",
                "item_unopened": "true",
            },
            [depth2_rule],
        )
        winner = label.applied[0]
        assert winner.path == ("R-014-a", "R-014-a-x1")
        assert winner.depth == 2
        assert winner.is_exception is True

    def test_describe_renders_the_route_the_precedence_and_the_test(self, depth2_rule):
        label = evaluate_rules(
            {
                "days_since_delivery": 10,
                "item_category": "swimwear",
                "item_unopened": "true",
            },
            [depth2_rule],
        )
        described = label.applied[0].describe()
        assert "R-014-a > R-014-a-x1 > R-014-a-x1-x1" in described
        assert "[p30]" in described
        assert "grants" in described
        assert "item_unopened" in described

    def test_the_reason_is_the_winners_description(self, depth2_rule):
        label = evaluate_rules(
            {"days_since_delivery": 10, "item_category": "footwear"}, [depth2_rule]
        )
        assert label.reason == label.applied[0].describe()

    def test_clause_ids_come_from_the_winning_rule(self, depth2_rule):
        label = evaluate_rules(
            {"days_since_delivery": 10, "item_category": "footwear"}, [depth2_rule]
        )
        assert label.clause_ids == (CLAUSE,)

    def test_a_top_level_rule_has_an_empty_path(self):
        label = evaluate_rules({"d": 1}, [rule("R-a", "grants")])
        assert label.applied[0].path == ()
        assert label.applied[0].is_exception is False

    def test_the_label_stringifies_to_its_stance(self):
        assert str(evaluate_rules({"d": 1}, [rule("R-a", "grants")])) == "grants"


class TestDeterminism:
    """C1 is a promise about reproducibility as much as about provenance: the
    same lockfile and the same fact vector must give the same label on any
    machine, on any run, or the audit trail proves nothing."""

    def test_repeated_evaluation_is_identical(self, depth2_rule):
        facts = {
            "days_since_delivery": 10,
            "item_category": "innerwear",
            "item_unopened": "true",
        }
        results = [evaluate_rules(facts, [depth2_rule]) for _ in range(5)]
        assert all(r == results[0] for r in results)

    def test_permuting_the_rule_set_changes_nothing(self):
        rules = [
            rule("R-a", "grants", precedence=10),
            rule("R-b", "grants", precedence=30),
            rule("R-c", "grants", precedence=20),
        ]
        forward = evaluate_rules({"d": 1}, rules)
        backward = evaluate_rules({"d": 1}, list(reversed(rules)))
        assert forward == backward
        assert forward.applied == backward.applied

    def test_a_scenario_and_a_plain_mapping_agree(self):
        r = rule("R-a", "grants", conditions=[cond("d", "<=", 30)])
        assert evaluate_rules(scenario({"d": 10}), [r]) == evaluate_rules({"d": 10}, [r])

    def test_evaluation_does_not_mutate_the_facts_it_was_given(self, depth2_rule):
        facts = {
            "days_since_delivery": 10,
            "item_category": "innerwear",
            "item_unopened": "true",
        }
        evaluate_rules(facts, [depth2_rule])
        assert facts == {
            "days_since_delivery": 10,
            "item_category": "innerwear",
            "item_unopened": "true",
        }


class TestRelabelWithoutRegenerate:
    """DESIGN.md 2 step 5, which DESIGN.md 6.2 calls the dominant speed win:

        "only the correct answer moved from `grants` to `denies`.
        `evaluate_rules()` recomputes it. *No LLM call.* This is the single
        biggest speed win in the whole system."

    The property that makes it work is that the label is a function of the fact
    vector and the rules alone. Nothing about the probe's rendered text
    participates, so moving a threshold re-labels without regenerating.
    """

    def test_moving_a_threshold_flips_the_label_for_an_unchanged_scenario(self):
        scen = scenario({"days_since_delivery": 31}, target_rule_id="R-a")
        before = [rule("R-a", "grants", conditions=[cond("days_since_delivery", "<=", 30)])]
        after = [rule("R-a", "grants", conditions=[cond("days_since_delivery", "<=", 45)])]

        assert evaluate_rules(scen, before).stance == "denies"
        assert evaluate_rules(scen, after).stance == "grants"

    def test_the_scenario_is_untouched_by_relabelling(self):
        """The fact vector is the probe's identity. If re-labelling mutated it,
        the cached surface text would no longer describe the scenario it was
        rendered from."""
        scen = scenario({"days_since_delivery": 31}, target_rule_id="R-a")
        snapshot = dict(scen.facts)
        for threshold in (30, 45, 15, 31):
            evaluate_rules(
                scen,
                [rule("R-a", "grants", conditions=[cond("days_since_delivery", "<=", threshold)])],
            )
        assert dict(scen.facts) == snapshot

    def test_the_boundary_moves_with_the_threshold(self):
        """Day 31 is the deny side of a 30-day window and the grant side of a
        31-day one. DESIGN.md 3.2 strategy 1 lives entirely on this edge."""
        scen = scenario({"days_since_delivery": 31}, target_rule_id="R-a")
        at_30 = [rule("R-a", "grants", conditions=[cond("days_since_delivery", "<=", 30)])]
        at_31 = [rule("R-a", "grants", conditions=[cond("days_since_delivery", "<=", 31)])]
        assert evaluate_rules(scen, at_30).stance == "denies"
        assert evaluate_rules(scen, at_31).stance == "grants"


class TestPolicyStanceHelper:
    def test_it_returns_the_bare_literal_design_1_3_specifies(self):
        r = rule("R-a", "grants", conditions=[cond("d", "<=", 30)])
        stance = policy_stance({"d": 10}, [r])
        assert stance == "grants"
        assert isinstance(stance, str)

    def test_it_agrees_with_the_full_result(self, depth2_rule):
        facts = {
            "days_since_delivery": 10,
            "item_category": "innerwear",
            "item_unopened": "false",
        }
        assert policy_stance(facts, [depth2_rule]) == evaluate_rules(facts, [depth2_rule]).stance

    def test_it_propagates_a_refusal_rather_than_defaulting(self):
        rules = [rule("R-a", "grants", precedence=20), rule("R-b", "denies", precedence=20)]
        with pytest.raises(AmbiguousPolicyError):
            policy_stance({"d": 1}, rules)


class TestCommitment1IsStructurallyEnforced:
    """A tripwire, not a behaviour test.

    C1 is "ground-truth labels are derived deterministically from rules in
    Python, never from an LLM". The cheapest durable enforcement of a promise
    like that is to assert the module cannot reach a model at all, so a future
    edit that adds "just one" model call to resolve an ambiguity fails here
    rather than in review.
    """

    BANNED = ("litellm", "instructor", "openai", "anthropic", "ollama", "httpx", "requests")

    def test_the_rules_engine_imports_no_model_client(self):
        from harness import rules_engine

        package = Path(rules_engine.__file__).parent
        offenders: list[str] = []
        for module in sorted(package.glob("*.py")):
            for lineno, line in enumerate(
                module.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if not (stripped.startswith("import ") or stripped.startswith("from ")):
                    continue
                for banned in self.BANNED:
                    if banned in stripped:
                        offenders.append(f"{module.name}:{lineno}: {stripped}")
        assert offenders == []

