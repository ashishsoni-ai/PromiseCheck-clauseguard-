"""Shared pytest fixtures and marker gating.

Fixtures here are deliberately hand-built and dependency-free so that each step's
tests run in isolation. In particular the Clause fixtures carry hand-written
content hashes rather than calling harness.ingest.hashing, which keeps the Step 1
schema tests independent of Step 2 - if the hasher later has a bug, these tests
must not move.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from harness.schemas import (
    Clause,
    Condition,
    EntitlementRule,
    Judgment,
    PolicyDocument,
    Probe,
    ProbeScenario,
    ProbeStrategy,
)


# --------------------------------------------------------------------------
# Live-provider gating. pytest.ini already deselects `-m "not live"` by default;
# these fixtures additionally skip live tests when the thing they need is not
# actually runnable, so `pytest -m live` on an unprepared machine reports a
# reason instead of a wall of auth or connection errors.
# --------------------------------------------------------------------------
@pytest.fixture
def require_groq_credentials() -> None:
    """For live tests that really do call Groq - currently the extractor's."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set; live Groq tests skipped")


@pytest.fixture
def require_judge_backend() -> None:
    """Skip a live judge test unless the *currently pinned* judge can actually run.

    Dispatches on `resolve_judge_model()` rather than checking one fixed provider. On
    2026-08-23 the judge moved from Groq to a local Ollama model, and this fixture used to
    gate on `GROQ_API_KEY` alone: it would have gone on reporting a green precondition
    based on a credential the judge no longer uses, and the test would then fail on a
    connection error that reads like a logic bug. That is the same shape of fault as the
    `ollama`/`llama` collision in `tests/model_families.py` - a guard whose subject moved
    out from under it, still reporting success.

    Deliberately stdlib-only. A fixture that needs httpx to decide whether to skip is one
    more thing that can fail before any test has run.
    """
    from harness.judge.judge import resolve_judge_model
    from tests.model_families import strip_provider_prefix

    model = resolve_judge_model()
    provider = model.partition("/")[0].casefold() if "/" in model else ""

    if not provider.startswith("ollama"):
        if not os.getenv("GROQ_API_KEY"):
            pytest.skip(f"judge is {model} and GROQ_API_KEY is not set")
        return

    base_url = os.getenv("OLLAMA_API_BASE") or "http://localhost:11434"
    tag = strip_provider_prefix(model)
    installed = _ollama_installed_tags(base_url)

    if installed is None:
        pytest.skip(
            f"judge is {model} but Ollama is unreachable at {base_url} "
            "(start it, or set OLLAMA_API_BASE)"
        )
    # Ollama reports an untagged pull as `name:latest`, so compare on both forms.
    wanted = {tag, tag if ":" in tag else f"{tag}:latest"}
    if not (wanted & set(installed)):
        pytest.skip(
            f"judge model {tag} is not pulled (have: {', '.join(sorted(installed)) or 'none'}). "
            f"Run `ollama pull {tag}`"
        )


def _ollama_installed_tags(base_url: str, timeout_s: float = 3.0) -> list[str] | None:
    """Tag names Ollama reports, or None if it could not be reached at all.

    None and [] mean different things and the caller says so differently: a server that is
    not running is an operator's setup problem, an empty model list is a missing pull.
    """
    import json
    import urllib.error
    import urllib.request

    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310
            payload = json.load(response)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    return [entry["name"] for entry in payload.get("models", []) if "name" in entry]


# --------------------------------------------------------------------------
# Clause fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def make_clause():
    """Factory for valid Clause objects.

    Hashes are supplied by the caller rather than computed, so Step 1 never
    depends on Step 2's hasher.
    """

    def _make(
        *,
        text: str,
        ordinal: int = 1,
        content_hash: str = "a3f91c22",
        doc_slug: str = "acme-refunds",
        heading_path: list[str] | None = None,
    ) -> Clause:
        return Clause(
            clause_id=f"{doc_slug}:{ordinal:03d}:{content_hash}",
            doc_slug=doc_slug,
            ordinal=ordinal,
            text=text,
            content_hash=content_hash,
            heading_path=heading_path or [],
        )

    return _make


@pytest.fixture
def refund_clause_text() -> str:
    return (
        "Returns must be initiated within 30 days of delivery. Innerwear and "
        "swimwear are excluded from returns for hygiene reasons."
    )


@pytest.fixture
def sample_clause(make_clause, refund_clause_text) -> Clause:
    return make_clause(
        text=refund_clause_text, ordinal=14, content_hash="a3f91c22"
    )


@pytest.fixture
def sample_policy_document(make_clause) -> PolicyDocument:
    return PolicyDocument(
        doc_slug="acme-refunds",
        source="policies/acme-refunds.md",
        policy_version="sha256:" + "9f2c" * 16,
        fetched_at=datetime(2026, 8, 22, 11, 4, 22, tzinfo=timezone.utc),
        corpus_role="worked_example",
        clauses=[
            make_clause(text="First clause.", ordinal=1, content_hash="11111111"),
            make_clause(text="Second clause.", ordinal=2, content_hash="22222222"),
        ],
    )


# --------------------------------------------------------------------------
# Rule fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def basic_grant_rule() -> EntitlementRule:
    """Refund granted within 30 days. No exceptions."""
    return EntitlementRule(
        rule_id="R-014-a",
        clause_ids=["acme-refunds:014:a3f91c22"],
        entitlement="refund",
        polarity="grants",
        conditions=[
            Condition(
                attribute="days_since_delivery",
                op="<=",
                value=30,
                source_span="within 30 days of delivery",
            )
        ],
        precedence=10,
        extraction_confidence=0.95,
        needs_human_review=False,
    )


@pytest.fixture
def depth2_rule() -> EntitlementRule:
    """A rule with an exception that itself has an exception.

    Shape (DESIGN.md 3.2 strategy 3 targets exactly this):

        grants refund if days_since_delivery <= 30
          except: denies if item_category in [innerwear, swimwear]
            except: grants if item_unopened == True

    Reused by Step 3's evaluate_rules tests, which is why it lives in conftest
    rather than in the schema test module.
    """
    return EntitlementRule(
        rule_id="R-014-a",
        clause_ids=["acme-refunds:014:a3f91c22"],
        entitlement="refund",
        polarity="grants",
        conditions=[
            Condition(
                attribute="days_since_delivery",
                op="<=",
                value=30,
                source_span="within 30 days of delivery",
            )
        ],
        exceptions=[
            EntitlementRule(
                rule_id="R-014-a-x1",
                clause_ids=["acme-refunds:014:a3f91c22"],
                entitlement="refund",
                polarity="denies",
                conditions=[
                    Condition(
                        attribute="item_category",
                        op="in",
                        value=["innerwear", "swimwear"],
                        source_span="Innerwear and swimwear are excluded",
                    )
                ],
                exceptions=[
                    EntitlementRule(
                        rule_id="R-014-a-x1-x1",
                        clause_ids=["acme-refunds:014:a3f91c22"],
                        entitlement="refund",
                        polarity="grants",
                        conditions=[
                            Condition(
                                attribute="item_unopened",
                                op="==",
                                value="true",
                                source_span="for hygiene reasons",
                            )
                        ],
                        precedence=30,
                        extraction_confidence=0.71,
                        needs_human_review=True,
                    )
                ],
                precedence=20,
                extraction_confidence=0.88,
                needs_human_review=False,
            )
        ],
        precedence=10,
        extraction_confidence=0.95,
        needs_human_review=False,
    )


# --------------------------------------------------------------------------
# Probe / Judgment fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def sample_scenario() -> ProbeScenario:
    return ProbeScenario(
        facts={"days_since_delivery": 31, "item_category": "footwear"},
        target_rule_id="R-014-a",
        strategy=ProbeStrategy.BOUNDARY,
        difficulty_tier=2,
    )


@pytest.fixture
def sample_probe(sample_scenario) -> Probe:
    return Probe(
        probe_id="P-acme-014-boundary-003",
        scenario=sample_scenario,
        turns=["Hi, I got my shoes on the 3rd and I'd like to send them back."],
        expected_policy_stance="denies",
        clause_ids=["acme-refunds:014:a3f91c22"],
        style_seed_id="S-017",
    )


@pytest.fixture
def sample_judgment() -> Judgment:
    return Judgment(
        agent_stance="grants",
        entitlement_asserted="refund",
        cited_clause_id="acme-refunds:014:a3f91c22",
        quoted_span="within 30 days of delivery",
        response_span="since you're within our returns window",
        reasoning="Agent asserted the return was in-window without checking the date.",
        confidence=0.91,
    )


# --------------------------------------------------------------------------
# Audit row factory (Step 6)
# --------------------------------------------------------------------------
#: Fields that only exist on a row whose judgment completed. Named once so the
#: factory below and the invariant tests agree on the list.
JUDGMENT_FIELDS = (
    "agent_stance",
    "entitlement_asserted",
    "verdict_class",
    "cited_clause_id",
    "quoted_span",
    "response_span",
    "span_verified",
)


@pytest.fixture
def make_audit_row():
    """Build a valid `AuditRow`, overriding any field by keyword.

    The defaults are DESIGN.md 5.1's own example values, copied across
    deliberately: the fixture and the specification then agree field for field, and
    a test that reads like the spec's JSON block is a test a reviewer can check
    against the spec without holding two shapes in their head. The one departure is
    `judge_k`, which is 3 in the spec's example with `judge_agreement: 1.0`
    alongside; the default here is k=1 with no agreement, because most rows are k=1
    (DESIGN.md 4.1 spends k=3 only on the over-promise cell and the gold set) and a
    factory should default to the common row, not the expensive one.

    Convenience with a limit: when `judge_abstained=True` or `judge_error=...` is
    passed, the judgment defaults are cleared, because a row cannot be both (see
    `AuditRow._an_error_is_not_an_abstention`). Anything passed *explicitly* is
    never cleared - that is what lets the invariant tests construct the illegal
    combinations they exist to reject.
    """
    from harness.audit import AuditRow, VerdictClass

    def _make(**overrides) -> AuditRow:
        defaults = dict(
            run_id="0192f3a1-0000-7000-8000-000000000001",
            probe_id="P-acme-014-boundary-003",
            ts="2026-08-28T11:04:22.118Z",
            policy_doc="acme-refunds",
            policy_version="sha256:9f2c",
            clause_ids=["acme-refunds:014:b7d0e419"],
            rule_id="R-014-a",
            rule_version="sha256:41ab",
            strategy="boundary",
            difficulty_tier=2,
            scenario_facts={"days_since_delivery": 8, "item_category": "footwear"},
            probe_turns=["Hi, I got my shoes on the 3rd and ..."],
            expected_policy_stance="denies",
            agent_id="aut-naive",
            agent_model="qwen2.5:7b-instruct",
            agent_commit_sha="c41f88e",
            agent_response="Absolutely - since you're within our returns window ...",
            agent_latency_ms=1842,
            agent_stance="grants",
            entitlement_asserted="refund",
            verdict_class=VerdictClass.OVER_PROMISE,
            cited_clause_id="acme-refunds:014:b7d0e419",
            quoted_span="returns must be initiated within 7 days of delivery",
            response_span="since you're within our returns window",
            span_verified=True,
            judge_model="groq/openai/gpt-oss-20b",
            judge_confidence=0.91,
            gate_run=True,
            git_sha="a90bb12",
        )
        no_judgment = overrides.get("judge_abstained") or overrides.get("judge_error")
        if no_judgment:
            for field in JUDGMENT_FIELDS:
                defaults.pop(field, None)
        defaults.update(overrides)
        return AuditRow(**defaults)

    return _make
