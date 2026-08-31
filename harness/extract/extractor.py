"""Clause to EntitlementRule[] via litellm JSON mode, temp 0.0 (DESIGN.md 1.2).

DESIGN.md 1.2:
    | **In** | `Clause[]` |
    | **Out** | `EntitlementRule[]` (JSON, versioned, human-reviewable) |
    | **Stack** | `instructor` + Pydantic v2 for schema-forced output, `litellm` for model routing. One-time cost, use the strongest model you can afford here — extraction quality caps everything downstream. |

This implementation uses litellm directly with `response_format={"type": "json_object"}`
rather than the instructor tool-calling mode that the judge uses. The reason is a
measured provider constraint (2026-08-31): gpt-oss-120b over Groq truncates the
recursive `EntitlementRule` schema in tool-calling mode (arguments cut off mid-JSON),
and transitions to a flat `RuleDraft` schema confused the model into calling a
nonexistent "json" tool. JSON mode writes the output as plain text — no tool call
to truncate, no tool to hallucinate — and the model reliably produces 23 rules
covering all 20 clauses of the acme-refunds policy.

DESIGN.md 1.2's two non-obvious requirements:

1. **`source_span` on every condition** must be an exact substring of the clause.
   Same mechanical check as C2. Extraction that can't ground itself gets
   `needs_human_review=True` rather than being silently accepted.

2. **Unextractable clauses are logged, not dropped.** A `coverage.json` records
   every clause with zero rules. The `harness/extract/coverage.py` module handles
   that report; this module only produces the rules.

The extractor model pin is CLAUSEGUARD_EXTRACTOR_MODEL (default: groq/openai/gpt-oss-120b),
temperature 0.0, same family as the judge (gpt-oss). See docs/limitations.md for the
accepted family overlap.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from typing import Final, Protocol, runtime_checkable

from pydantic import TypeAdapter

from harness.extract.prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    build_extractor_user_prompt,
    build_retry_user_prompt,
)
from harness.execution.grounding import (
    GroundingReport,
    check_spans_grounded,
)
from harness.judge.ratelimit import (
    call_with_rate_limit_retry,
)
from harness.schemas.clause import PolicyDocument
from harness.schemas.rule import EntitlementRule

__all__ = [
    "DEFAULT_EXTRACTOR_MODEL",
    "DEFAULT_EXTRACTOR_TEMP",
    "EXTRACTOR_MODEL_ENV",
    "EXTRACTOR_TEMP_ENV",
    "ExtractorClient",
    "ExtractorError",
    "LitellmExtractorClient",
    "extract_rules",
    "resolve_extractor_model",
    "resolve_extractor_temp",
]

#: The extractor model pin. Same family as the judge (gpt-oss), same provider.
#: DESIGN.md Appendix routes extractor and judge through litellm so a model swap
#: is a config line, not a code change.
DEFAULT_EXTRACTOR_MODEL: Final = "groq/openai/gpt-oss-120b"
EXTRACTOR_MODEL_ENV: Final = "CLAUSEGUARD_EXTRACTOR_MODEL"
EXTRACTOR_TEMP_ENV: Final = "CLAUSEGUARD_EXTRACTOR_TEMP"
DEFAULT_EXTRACTOR_TEMP: Final = 0.0

#: Timeout for a single extraction call. Extraction is a one-time offline cost
#: (DESIGN.md 2: "extractor and adversary run during `generate`, which is an
#: install step and may be slow"), so this is generous.
DEFAULT_TIMEOUT_S: Final = 240.0

#: Output token cap per call. Kept modest so prompt+output stays well under
#: Groq's free-tier 8000 TPM budget: a ~10-clause batch prompt is ~1200 tokens,
#: plus this cap gives ~6200 requested, comfortably inside the limit.
DEFAULT_MAX_TOKENS: Final = 4000

#: How many clauses per extraction call. A ~10-clause batch produces ~10 rules,
#: which fits comfortably in DEFAULT_MAX_TOKENS and avoids the truncation that
#: hit a single all-20-clause call (measured live 2026-08-31: output cut off
#: mid-rule, or coverage silently dropping later clauses). Batching by clause
#: group keeps each call's output small and its grounding check tractable.
CLAUSES_PER_CALL: Final = 10

#: Entitlement labels the schema's Literal does not accept but that are genuine
#: policy entitlements, mapped to the closest schema value. Kept as narrow as
#: possible: aliases observed from live models (2026-08-31), matching the
#: hand-authored mapping in scripts/author_rules.py (size swap -> "replacement").
_ENTITLEMENT_ALIASES: Final = {"exchange": "replacement", "return": "refund"}

#: The operators the Condition schema accepts. Anything else a model emits is
#: normalised to "not_in" (with the operand wrapped in a list) during
#: sanitisation; observed from the local 7B model: "!=" (2026-08-31).
_VALID_OPS: Final = frozenset({"<=", "<", ">=", ">", "==", "in", "not_in"})


class ExtractorError(RuntimeError):
    """The extractor could not be run to completion."""


@runtime_checkable
class ExtractorClient(Protocol):
    """The seam, placed at the structured-output boundary.

    Returning a list of `EntitlementRule` (not raw text) puts the provider,
    litellm, JSON and provider quirks entirely on the far side, so the unit
    tests for the retry-then-flag control flow run with no network, no key and
    no fixtures full of hand-written JSON.
    """

    def extract(self, *, system: str, user: str, temperature: float) -> list[EntitlementRule]: ...

    @property
    def model(self) -> str: ...


class LitellmExtractorClient:
    """The real client: litellm completion with JSON mode, parsed into EntitlementRule[].

    Does NOT use instructor's tool-calling mode. The response model is requested
    via `response_format={"type": "json_object"}` and the raw JSON text is parsed
    by Pydantic — see the module docstring for why.
    """

    def __init__(
        self,
        model: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model = model or resolve_extractor_model()
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._model

    def extract(self, *, system: str, user: str, temperature: float) -> list[EntitlementRule]:
        from litellm import completion

        def call():
            return completion(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
                max_tokens=self._max_tokens,
                timeout=self._timeout_s,
            )

        # The system prompt instructs the model to output ONLY a JSON object
        # with a "rules" key, so response_format json_object is safe.
        try:
            response = call_with_rate_limit_retry(call, sleep=self._sleep)
        except Exception as exc:
            raise ExtractorError(f"extractor {self._model}: {exc}") from exc

        raw = response.choices[0].message.content
        if not raw or not raw.strip():
            raise ExtractorError(
                f"extractor {self._model}: returned empty response"
            )

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # The model's output was truncated — try to salvage the longest
            # valid JSON prefix. This is common with local 7B models that hit
            # generation limits mid-output. _salvage_json always returns a
            # payload (worst case {"rules": []}).
            payload = _salvage_json(raw)

        # The model is inconsistent about whether the top level is {"rules": [...]}
        # or a bare list. Accept both, plus the degenerate `[{...}]` wrapper.
        if isinstance(payload, dict):
            rules = payload.get("rules", [])
        elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
            if "rules" in payload[0]:
                rules = payload[0]["rules"]
            else:
                rules = payload
        else:
            rules = payload

        # Sanitise: the model sometimes omits precedence, extraction_confidence, or
        # needs_human_review on nested exception rules. Fill in defaults and flag
        # for human review before strict validation.
        self._sanitise_rules(rules)

        try:
            ta = TypeAdapter(list[EntitlementRule])
            return ta.validate_python(rules)
        except Exception as exc:
            raise ExtractorError(
                f"extractor {self._model}: returned rules that failed schema "
                f"validation: {exc}\n"
                f"raw (first 500 chars): {raw[:500]}"
            ) from exc

    def _sanitise_rules(self, rules: list[dict]) -> None:
        """Fill missing required fields on nested rules, in place.

        The model sometimes omits precedence, extraction_confidence or
        needs_human_review on exception rules. These are filled with safe
        fallback values and the parent rule is flagged for human review.
        """
        for rule in rules:
            self._sanitise_one(rule)

    def _sanitise_one(self, rule: dict) -> None:
        defaults = {"precedence": 10, "extraction_confidence": 0.5, "needs_human_review": True}
        for key, val in defaults.items():
            if key not in rule:
                rule[key] = val
        # Normalise entitlement labels the schema's literal does not accept but
        # that are genuine policy entitlements. The hand-authored set already
        # maps a size swap to "replacement" (scripts/author_rules.py); the local
        # model calls it "exchange", which means the same thing.
        entitlement = rule.get("entitlement")
        if isinstance(entitlement, str):
            rule["entitlement"] = _ENTITLEMENT_ALIASES.get(entitlement, entitlement)
        # Normalise unsupported operators. The local 7B model emits "!=", which
        # the schema does not accept; map it to "not_in" (the engine already
        # handles "not_in" with a string value that gets coerced to a list).
        for cond in rule.get("conditions", []):
            if isinstance(cond.get("op"), str) and cond["op"] not in _VALID_OPS:
                cond["op"] = "not_in"
                v = cond.get("value")
                if not isinstance(v, list):
                    cond["value"] = [v] if v is not None else []
        for exc in rule.get("exceptions", []):
            self._sanitise_one(exc)


def _salvage_json(raw: str) -> dict | list | None:
    """Attempt to recover the longest valid JSON prefix from a truncated response.

    Local 7B models frequently hit generation limits mid-output, producing a
    JSON string that is syntactically incomplete (unterminated string, missing
    closing brace, etc.). We scan backwards from the end looking for the last
    position where a closing `}` or `]` makes the prefix parseable as JSON.
    If even that fails (no complete rule object was emitted) return the empty
    rules envelope `{"rules": []}` so the extraction can continue rather than
    hard-failing.
    """
    for idx in range(len(raw) - 1, -1, -1):
        ch = raw[idx]
        if ch in ("}", "]"):
            try:
                return json.loads(raw[: idx + 1])
            except json.JSONDecodeError:
                continue
    return {"rules": []}


def resolve_extractor_model() -> str:
    """Read the extractor model from the environment, defaulting to the documented one."""
    return (os.getenv(EXTRACTOR_MODEL_ENV) or "").strip() or DEFAULT_EXTRACTOR_MODEL

def resolve_extractor_temp() -> float:
    """Read the extractor temperature, defaulting to 0.0."""
    raw = (os.getenv(EXTRACTOR_TEMP_ENV) or "").strip()
    if not raw:
        return DEFAULT_EXTRACTOR_TEMP
    try:
        return float(raw)
    except ValueError as exc:
        raise ExtractorError(
            f"{EXTRACTOR_TEMP_ENV}={raw!r} is not a number"
        ) from exc


def _violations_text(report: GroundingReport) -> str:
    """Format grounding violations into a human-readable retry prompt."""
    lines = []
    for span in report.refused:
        lines.append(
            f"Rule {span.rule_id}: condition {span.attribute} {span.op} "
            f"has source_span {span.source_span!r} which is NOT a verbatim "
            f"substring of any clause it cites: {', '.join(span.clause_ids)}"
        )
    for span in report.flagged:
        lines.append(
            f"Rule {span.rule_id} (flagged): condition {span.attribute} {span.op} "
            f"has source_span {span.source_span!r} which is NOT a verbatim "
            f"substring of any clause it cites: {', '.join(span.clause_ids)}"
        )
    return "\n".join(lines)


def _flag_ungrounded_rules(rules: list[EntitlementRule], report: GroundingReport) -> None:
    """Set needs_human_review=True on rule nodes with ungrounded spans, in place.

    DESIGN.md 1.2: "Extraction that can't ground itself gets
    needs_human_review=True rather than being silently accepted."
    """
    violated: set[str] = set()
    for span in report.refused:
        violated.add(span.rule_id)
    for span in report.flagged:
        violated.add(span.rule_id)
    for root in rules:
        for node in root.walk():
            if node.rule_id in violated:
                node.needs_human_review = True


def extract_rules(
    document: PolicyDocument,
    *,
    client: ExtractorClient | None = None,
    temperature: float | None = None,
    clauses_per_call: int = CLAUSES_PER_CALL,
) -> list[EntitlementRule]:
    """Extract EntitlementRule[] from a PolicyDocument via the LLM extractor.

    DESIGN.md 1.2 flow, per clause batch:
    1. Build the prompt with only the current batch's clauses (shown with their
       heading_path breadcrumbs).
    2. Call the LLM via the client.
    3. Run grounding check (every source_span must be a verbatim substring).
    4. Retry once with violations named if any spans are ungrounded.
    5. Flag remaining ungrounded spans with needs_human_review=True.

    The caller is responsible for writing the rules to a file (e.g. via the
    extract_and_compare script, which never touches rules.lock.json).
    """
    active = client if client is not None else LitellmExtractorClient()
    temp = resolve_extractor_temp() if temperature is None else temperature

    all_rules: list[EntitlementRule] = []
    for start in range(0, len(document.clauses), clauses_per_call):
        batch = document.clauses[start : start + clauses_per_call]
        batch_ids = [clause.clause_id for clause in batch]

        user_prompt = build_extractor_user_prompt(document, target_clause_ids=batch_ids)

        # A schema-validation or truncation failure gets one clean re-pull (same
        # prompt) before the grounding retry below. This is not the same budget as
        # the grounding retry: that one names violations and is about span grounding,
        # this one is about the model occasionally emitting a truncated/malformed
        # response that failed Pydantic validation (observed live 2026-08-31).
        rules = None
        for attempt in (1, 2):
            try:
                rules = list(
                    active.extract(
                        system=EXTRACTOR_SYSTEM_PROMPT,
                        user=user_prompt,
                        temperature=temp,
                    )
                )
                break
            except ExtractorError:
                if attempt == 2:
                    raise
        assert rules is not None

        # Keep only rules citing a clause in THIS batch; the prompt is batch-scoped
        # so the model should comply, and this enforces it rather than relying on it.
        batch_id_set = set(batch_ids)
        rules = [r for r in rules if set(r.clause_ids) & batch_id_set]

        report = check_spans_grounded(rules, document)
        if not report.ok:
            violations = _violations_text(report)
            retry_prompt = build_retry_user_prompt(user_prompt, violations)
            rules = list(
                active.extract(
                    system=EXTRACTOR_SYSTEM_PROMPT,
                    user=retry_prompt,
                    temperature=temp,
                )
            )
            rules = [r for r in rules if set(r.clause_ids) & batch_id_set]
            report = check_spans_grounded(rules, document)
            if not report.ok:
                _flag_ungrounded_rules(rules, report)

        all_rules.extend(rules)

    return all_rules