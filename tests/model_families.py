"""One source of truth for "are these two models from the same family?".

DESIGN.md 1.5 requires the judge to come from "a **different family** than the AUT", and
the 2 step 8 role table repeats it as "Different family from AUT; span-verified". That
constraint spans several files - the frozen agent's own constants, `.env.example`, and
`harness/judge/judge.py`'s default - so it cannot be enforced by a comment, and it must
not be enforced by two copies of the rule that can drift apart.

This module is imported by `tests/unit/test_aut_contract.py` (which pins the separation
across the agent and `.env.example`) and `tests/unit/test_judge.py` (which pins the
judge's own default). `tests/unit/test_model_families.py` tests this module itself.

WHY THE PROVIDER PREFIX MUST BE STRIPPED FIRST - THE `ollama` / `llama` TRAP
---------------------------------------------------------------------------
litellm addresses a model as `<provider>/<id>`. The provider for a local model is
`ollama` or `ollama_chat`, and:

    >>> "llama" in "ollama"
    True

because `"ollama"[1:6] == "llama"`. So a naive substring scan reports EVERY locally
served model as the Llama family - `ollama_chat/mistral:7b`, `ollama_chat/gemma2:9b` and
`ollama_chat/deepseek-r1:8b` all come back "llama".

The damage is quiet, which is what makes it worth this much prose. The assertion that
matters most reads

    assert family_of(judge_model) != family_of(agent_model)

and with the agent on Qwen it would still PASS - "llama" != "qwen" - while having stopped
measuring anything at all. A guard that reports success after its subject changed shape
is worse than no guard, because it also removes the pressure to look. This is the same
failure the prose-versus-code tripwire in `test_aut_chunker.py` ran into, and the fix has
the same shape: narrow what is scanned, then pin the narrowing in both directions with a
positive and a negative control.

WHY THE PREFIX SET IS DELIBERATELY SMALL
----------------------------------------
Only `ollama` and `ollama_chat` are load-bearing; they are the ones that collide. `groq`
is stripped too because it is the other provider this project uses and leaving it in
would invite the reader to think prefixes are handled inconsistently. Nothing else is
stripped: `openai/gpt-oss-120b` needs no help (the "gpt" token is in the remainder), and
a prefix set that grows to cover providers nobody uses is a place for a future collision
to hide. Add an entry when a real model id needs it, not in advance.
"""

from __future__ import annotations

from typing import Final

__all__ = ["FAMILY_TOKENS", "PROVIDER_PREFIXES", "family_of", "strip_provider_prefix"]

#: Ordered. The scan returns the first token found, so a token that is a substring of
#: another model's name must come first. "mistral" precedes "phi" because `phi` is a
#: substring of `dolphin`, and `dolphin-mistral` is a real Ollama tag.
FAMILY_TOKENS: Final = (
    "qwen",
    "llama",
    "mistral",
    "mixtral",
    "gemma",
    "phi",
    "deepseek",
    "gpt",
    "claude",
)

#: litellm provider segments this project actually addresses models through.
PROVIDER_PREFIXES: Final = frozenset({"groq", "ollama", "ollama_chat"})


def strip_provider_prefix(model: str) -> str:
    """Drop leading `<provider>/` segments, leaving the model id.

    Loops, so `groq/openai/gpt-oss-120b` loses only `groq/` (openai is not a provider
    prefix here) and a hypothetical doubled prefix would still resolve.
    """
    remainder = model
    while "/" in remainder:
        head, _, tail = remainder.partition("/")
        if head.strip().casefold() not in PROVIDER_PREFIXES:
            break
        remainder = tail
    return remainder


def family_of(model: str) -> str:
    """The model's family token, e.g. `ollama_chat/llama3.1:8b` -> "llama".

    Raises rather than returning a sentinel for an unknown family: DESIGN.md 1.5 depends
    on families being comparable, and an "unknown" bucket would compare unequal to
    everything and so satisfy every separation assertion by default.
    """
    lowered = strip_provider_prefix(model).casefold()
    for token in FAMILY_TOKENS:
        if token in lowered:
            return token
    raise AssertionError(
        f"unrecognised model family in {model!r} (scanned {lowered!r} after stripping "
        "the provider prefix). Add the family to FAMILY_TOKENS rather than loosening "
        "this check - DESIGN.md 1.5 depends on families being comparable."
    )
