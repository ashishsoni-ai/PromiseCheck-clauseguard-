"""Tests for `tests/model_families.py` - the DESIGN.md 1.5 family comparison itself.

The helper is test-support code, but it is the thing that decides whether the judge and
the agent under test are separated, so it gets its own tests rather than being trusted.
The controls below encode the bug it was written to fix, so the fix cannot silently
regress into the naive substring scan it replaced.
"""

from __future__ import annotations

import pytest

from tests.model_families import (
    FAMILY_TOKENS,
    PROVIDER_PREFIXES,
    family_of,
    strip_provider_prefix,
)


def naive_family_of(model: str) -> str | None:
    """The buggy implementation, kept as a control. Do not use outside this file.

    This is what `family_of` looked like before 2026-08-23: a raw substring scan with no
    provider-prefix handling. It is reproduced here so the tests can assert that the real
    implementation *disagrees* with it, which is the only way to keep
    `strip_provider_prefix` from decaying into a no-op without a test noticing.
    """
    lowered = model.casefold()
    for token in FAMILY_TOKENS:
        if token in lowered:
            return token
    return None


class TestTheTrapThisModuleExistsToAvoid:
    def test_llama_really_is_a_substring_of_ollama(self):
        """The premise of the whole module, asserted so it reads as fact not folklore."""
        assert "llama" in "ollama"
        assert "ollama"[1:6] == "llama"

    @pytest.mark.parametrize(
        ("model", "correct_family"),
        [
            ("ollama_chat/mistral:7b", "mistral"),
            ("ollama_chat/gemma2:9b", "gemma"),
            ("ollama_chat/phi3:mini", "phi"),
            ("ollama_chat/deepseek-r1:8b", "deepseek"),
            ("ollama/mistral:7b", "mistral"),
        ],
    )
    def test_the_naive_scan_would_call_every_local_model_llama(
        self, model: str, correct_family: str
    ):
        """Positive control on the bug: each of these really would have been wrong."""
        assert naive_family_of(model) == "llama"
        assert family_of(model) == correct_family
        assert family_of(model) != naive_family_of(model)

    def test_the_separation_assertion_would_have_passed_for_the_wrong_reason(self):
        """Why the bug was quiet rather than loud.

        With the agent on Qwen, `family_of(judge) != family_of(agent)` held under the
        naive scan too - it returned "llama" instead of "mistral", and "llama" is still
        not "qwen". The guard would have reported success while measuring nothing.
        """
        agent = "qwen2.5:7b-instruct"
        judge = "ollama_chat/mistral:7b"
        assert naive_family_of(judge) != naive_family_of(agent)  # passes, wrongly
        assert family_of(judge) != family_of(agent)  # passes, correctly
        assert naive_family_of(judge) != family_of(judge)  # for different reasons


class TestStrippingTheProviderPrefix:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("ollama_chat/llama3.1:8b", "llama3.1:8b"),
            ("ollama/mistral:7b", "mistral:7b"),
            ("groq/llama-3.1-8b-instant", "llama-3.1-8b-instant"),
            ("groq/openai/gpt-oss-120b", "openai/gpt-oss-120b"),
        ],
    )
    def test_known_providers_are_dropped(self, model: str, expected: str):
        assert strip_provider_prefix(model) == expected

    @pytest.mark.parametrize(
        "model",
        [
            "qwen2.5:7b-instruct",
            "openai/gpt-oss-120b",
            "meta-llama/llama-prompt-guard-2-86m",
        ],
    )
    def test_a_bare_id_or_unknown_vendor_segment_is_left_alone(self, model: str):
        """`openai/` and `meta-llama/` are parts of the model id, not litellm providers.

        Stripping them would be harmless today but the set is kept minimal on purpose -
        see the module docstring. This pins that decision so it is not "tidied" later.
        """
        assert strip_provider_prefix(model) == model

    def test_every_prefix_in_the_set_is_actually_stripped(self):
        """Keeps PROVIDER_PREFIXES honest: an entry that does nothing is a lie."""
        for prefix in PROVIDER_PREFIXES:
            assert strip_provider_prefix(f"{prefix}/some-model") == "some-model"


class TestFamilyResolution:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # the local pins chosen 2026-08-23
            ("ollama_chat/llama3.1:8b", "llama"),
            ("ollama_chat/mistral:7b", "mistral"),
            # the agent under test, which carries no provider prefix
            ("qwen2.5:7b-instruct", "qwen"),
            # the extractor, as Groq lists it
            ("groq/openai/gpt-oss-120b", "gpt"),
            # the only Qwen left on the account
            ("qwen/qwen3.6-27b", "qwen"),
            # decommissioned, but must still classify for historical audit rows
            ("groq/llama-3.3-70b-versatile", "llama"),
        ],
    )
    def test_the_models_this_project_pins(self, model: str, expected: str):
        assert family_of(model) == expected

    def test_a_locally_served_qwen_is_still_caught(self):
        """The case the guard exists for: pointing a harness role at the agent's family
        via Ollama must not be laundered by the provider prefix."""
        assert family_of("ollama_chat/qwen2.5:7b-instruct") == "qwen"
        assert family_of("ollama_chat/qwen2.5:7b-instruct") == family_of(
            "qwen2.5:7b-instruct"
        )

    def test_token_order_resolves_a_real_ollama_tag(self):
        """`dolphin-mistral` contains "phi" inside "dolphin". FAMILY_TOKENS puts
        "mistral" first for exactly this reason, so the ordering is load-bearing."""
        assert family_of("ollama_chat/dolphin-mistral:7b") == "mistral"

    def test_an_unknown_family_raises_rather_than_returning_a_sentinel(self):
        """An "unknown" bucket would compare unequal to everything and so satisfy every
        separation assertion by default - the failure mode this module is about."""
        with pytest.raises(AssertionError, match="unrecognised model family"):
            family_of("allam-2-7b")

    def test_the_error_names_the_stripped_form_it_actually_scanned(self):
        """So the reader is not left wondering whether the prefix was the problem."""
        with pytest.raises(AssertionError, match="after stripping the provider prefix"):
            family_of("ollama_chat/allam-2-7b")

    def test_comparison_is_case_insensitive(self):
        assert family_of("OLLAMA_CHAT/Llama3.1:8B") == "llama"
