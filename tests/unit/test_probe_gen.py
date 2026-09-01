"""Tests for the probe generator pipeline: strategy modules, oracle, adversary, driver.

Offline tests use a fake adversary client; the 1 live test uses the real model
and is gated behind `@pytest.mark.live` + `require_groq_credentials` (or runs
against the local Ollama model pinned by CLAUSEGUARD_ADVERSARY_MODEL).
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from harness.execution.lockfiles import load_rules
from harness.probe_gen.adversary import (
    LitellmAdversaryClient,
    render_surface,
    self_critique,
    resolve_adversary_model,
)
from harness.probe_gen.oracle import OracleResult, oracle_check
from harness.probe_gen.sampler import (
    ENTITLEMENT_BASES,
    base_facts,
    pick_opposite,
    condition_passes,
)
from harness.probe_gen.strategies import STRATEGY_MODULES
from harness.schemas.probe import ProbeStrategy
from harness.schemas.rule import Condition, EntitlementRule
from tests.conftest import basic_grant_rule, make_clause, sample_policy_document

# ---------------------------------------------------------------------------
# Fake adversary client for offline tests
# ---------------------------------------------------------------------------
class FakeAdversary:
    """Returns a pre-set message, no network."""

    def __init__(self, text: str = "I need to return my order.", model: str = "test/fake") -> None:
        self._text = text
        self._model = model
        self.call_count = 0

    @property
    def model(self) -> str:
        return self._model

    def complete(self, *, system: str, user: str, temperature: float) -> str:
        self.call_count += 1
        return self._text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rules():
    return load_rules().rules


# ---------------------------------------------------------------------------
# Tests: sampler
# ---------------------------------------------------------------------------
class TestEntitlementBases:
    """Every entitlement has a canonical base covering its condition space."""

    def test_refund_base_has_all_key_fields(self):
        base = ENTITLEMENT_BASES["refund"]
        assert "days_since_delivery" in base
        assert "item_category" in base
        assert "item_opened" in base
        assert "order_channel" in base
        assert len(base) >= 14

    def test_cancellation_base(self):
        base = ENTITLEMENT_BASES["cancellation"]
        assert "order_dispatched" in base
        assert "order_channel" in base

    def test_partial_refund_base(self):
        base = ENTITLEMENT_BASES["partial_refund"]
        assert "item_category" in base
        assert "item_opened" in base
        assert "days_since_delivery" in base


class TestBaseFacts:
    """base_facts produces a vector that satisfies the target rule's conditions."""

    def test_returns_rule_condition_attributes(self, rules):
        root = rules[0]  # refund-window-30d
        fb = base_facts(root)
        for cond in root.conditions:
            assert cond.attribute in fb
            assert condition_passes(cond, fb)

    def test_includes_entitlement_wide_attributes(self, rules):
        root = rules[0]
        fb = base_facts(root)
        # Every refund rule attribute should be present
        for k in ENTITLEMENT_BASES["refund"]:
            assert k in fb, f"refund base missing {k}"


class TestPickOpposite:
    def test_returns_non_trivial_opposite(self):
        cond = Condition(attribute="x", op="<=", value=30, source_span="30 days")
        opp = pick_opposite(cond)
        assert opp != 30
        assert opp > 30

    def test_membership_opposite(self):
        cond = Condition(attribute="x", op="in", value=["a", "b"], source_span="a or b")
        opp = pick_opposite(cond)
        assert opp not in ("a", "b")


# ---------------------------------------------------------------------------
# Tests: strategy modules
# ---------------------------------------------------------------------------
class TestStrategyModules:
    def test_all_strategies_registered(self):
        assert set(STRATEGY_MODULES) == {
            "boundary", "condition_stripping", "exception_depth",
            "category_smuggling", "false_premise", "authority_pressure",
            "multi_turn_drift", "cross_clause",
        }

    def test_each_returns_list_of_dicts(self, rules):
        rng = random.Random("test")
        for name, mod in STRATEGY_MODULES.items():
            samples = mod.sample(rules[0], rng, max_count=2)
            assert isinstance(samples, list)
            for s in samples:
                assert isinstance(s, dict), f"{name} sample is not a dict: {s!r}"
                assert len(s) > 0

    def test_multi_turn_returns_pairs(self, rules):
        mod = STRATEGY_MODULES["multi_turn_drift"]
        samples = mod.sample(rules[0], random.Random("test"), max_count=2)
        assert len(samples) >= 2
        # Check the first two entries are a pair with different values
        assert samples[0] != samples[1], "multi-turn pairs should differ"

    def test_false_premise_includes_marker(self, rules):
        mod = STRATEGY_MODULES["false_premise"]
        samples = mod.sample(rules[0], random.Random("test"), max_count=1)
        for s in samples:
            assert any(k.startswith("_") for k in s), "false_premise needs _ marker"

    def test_authority_pressure_includes_pressure_marker(self, rules):
        mod = STRATEGY_MODULES["authority_pressure"]
        samples = mod.sample(rules[0], random.Random("test"), max_count=1)
        for s in samples:
            assert any(k.startswith("_") for k in s), "authority_pressure needs _ marker"


# ---------------------------------------------------------------------------
# Tests: oracle
# ---------------------------------------------------------------------------
class TestOracleCheck:
    def test_passes_when_all_facts_stated(self):
        text = "I got it 30 days ago, some footwear, opened it"
        facts = {"days_since_delivery": 30, "item_category": "footwear", "item_opened": "yes"}
        result = oracle_check(facts, text)
        assert result.passed, result.reason

    def test_fails_on_missing_numeric_fact(self):
        text = "my footwear arrived"
        facts = {"days_since_delivery": 30, "item_category": "footwear"}
        result = oracle_check(facts, text)
        assert not result.passed
        assert "days_since_delivery" in result.missing

    def test_skips_boolean_facts(self):
        text = "no mention of yes/no values"
        facts = {"item_opened": "yes", "has_visible_damage": "no"}
        result = oracle_check(facts, text)
        assert result.passed  # booleans are exempt
        assert result.checked == 0

    def test_skips_underscore_keys(self):
        text = "some message"
        facts = {"_false_premise": "90-day refund", "item_category": "footwear"}
        result = oracle_check(facts, text)
        assert not result.passed
        assert "item_category" in result.missing

    def test_numeric_fact_tolerant(self):
        text = "30 days ago I received my order"
        result = oracle_check({"days_since_delivery": 30}, text)
        assert result.passed

    def test_case_insensitive_string_match(self):
        text = "I said footwear and cosmetics"
        result = oracle_check({"item_category": "Footwear"}, text)
        assert result.passed


# ---------------------------------------------------------------------------
# Tests: adversary (offline, fake client)
# ---------------------------------------------------------------------------
class TestRenderSurface:
    def test_returns_text_from_fake_client(self):
        expected = "I need a refund."
        client = FakeAdversary(expected)
        result = render_surface({"item": "shoes"}, client=client, temperature=0.9)
        assert result == expected
        assert client.call_count == 1


class TestSelfCritique:
    def test_rewrites_text(self):
        expected = "I urgently need a refund."
        client = FakeAdversary(expected)
        result = self_critique("I need a refund.", client=client, temperature=0.9)
        assert result == expected
        assert client.call_count == 1


# ---------------------------------------------------------------------------
# Live test
# ---------------------------------------------------------------------------
@pytest.mark.live
class TestLiveAdversary:
    """Deselected by default. Run with `pytest -m live`.

    Verifies the adversary model (local Ollama by default) can produce a
    natural-language message from a fact vector. A failure here means the
    model is unreachable or the prompt is broken.
    """

    def test_live_render_returns_text(self, require_groq_credentials):
        model = resolve_adversary_model()
        client = LitellmAdversaryClient(model=model)
        text = render_surface(
            {"days_since_delivery": 10, "item_category": "footwear"},
            client=client,
        )
        assert len(text) > 20, "adversary returned a very short or empty message"
        # Should reference the category or a natural paraphrase
        assert any(w in text.casefold() for w in ("shoe", "footwear"))

    def test_live_critique_rewrites(self, require_groq_credentials):
        model = resolve_adversary_model()
        client = LitellmAdversaryClient(model=model)
        original = "I need to return my order."
        rewritten = self_critique(original, client=client)
        assert len(rewritten) > 0
        assert "return" in rewritten.casefold()