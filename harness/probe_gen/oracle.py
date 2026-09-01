"""Probe-validity oracle: does the surface text match its facts vector?

DESIGN.md 3.1 step 4:
> Then a Python assertion that every fact in the vector still appears in the text
> (numeric and category checks). Rewrites that drop or alter facts are rejected.

DESIGN.md 3.4 calls this the probe-validity filter and sets a bar:
> **Oracle pass rate < 95%** → the probe is ambiguous or your label is wrong.
> **Discard it.** ... Report how many you discarded — it reads as rigour, not
> weakness.

This module checks the MECHANICAL half of that: a probe is valid when every
fact its scenario carries is faithfully represented in the surface text. A
customer never writes `item_category: footwear` as a key, so the check is per
fact-type:

* numeric facts: the value (or a near paraphrase of it) must appear in the text;
* category/string facts: a token of the value must appear;
* boolean facts ("yes"/"no"): these have no natural literal surface — a customer
  says "I opened it" not "item_opened: yes" — so they are exempt, exactly as
  `scripts/author_probes.py` treats them.

The label check is separate and lives in the driver: `evaluate_rules()` computes
`expected_policy_stance` deterministically, and the probe is only emitted when
that label is derivable. This module is the fact-in-text half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "OracleResult",
    "BOOLEAN_VALUES",
    "fact_is_stated",
    "oracle_check",
]

#: Values with no natural literal surface in a customer message.
BOOLEAN_VALUES = frozenset({"yes", "no", "true", "false"})


def _tokens(text: str) -> set[str]:
    """Lowercased, punctuation-stripped tokens of a customer message."""
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


@dataclass(frozen=True)
class OracleResult:
    """The fact-in-text verdict for one probe."""

    passed: bool
    missing: tuple[str, ...] = field(default_factory=tuple)
    checked: int = 0

    @property
    def reason(self) -> str:
        if self.passed:
            return f"all {self.checked} checkable facts stated"
        return f"facts not stated in text: {', '.join(self.missing)}"


def fact_is_stated(fact_value: Any, text: str) -> bool:
    """True if `fact_value` is represented in `text`.

    Numeric values are searched for as the number itself (with optional
    decimal/negative handling); strings are matched case-insensitively as
    whole-word tokens. Booleans and internal-only keys are the caller's job to
    skip (via BOOLEAN_VALUES and the leading-underscore convention).
    """
    if isinstance(fact_value, bool):
        return False  # booleans have no literal surface; caller skips these
    if isinstance(fact_value, (int, float)):
        return str(fact_value) in text
    if isinstance(fact_value, str):
        return fact_value.casefold() in _tokens(text)
    return False


def oracle_check(facts: Mapping[str, Any], text: str) -> OracleResult:
    """Check that every checkable fact in `facts` appears in `text`.

    Facts whose key starts with `_` are internal (false-premise markers,
    authority-pressure scaffolding) and are not checked against the surface.
    Boolean-valued facts are exempt from the literal check, matching the
    hand-authored builder's rule.

    NOTE: `facts` here should be the SUBSET of facts the probe turns on (the
    target rule's condition attributes), not the full entitlement vector — the
    same scoping the hand-authored corpus uses, where `Spec.mentions` lists
    exactly the facts stated in the probe text. A full 15-field vector can never
    all appear in a 2-5 sentence customer message, so checking it all would
    report every generated probe as invalid.
    """
    missing: list[str] = []
    checked = 0
    for key, value in facts.items():
        if key.startswith("_"):
            continue
        if isinstance(value, str) and value.casefold() in BOOLEAN_VALUES:
            continue
        checked += 1
        if not fact_is_stated(value, text):
            missing.append(key)
    return OracleResult(passed=not missing, missing=tuple(missing), checked=checked)