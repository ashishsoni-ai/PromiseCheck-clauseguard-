"""EXTRACTOR-role prompts — cold, literal, forensic (DESIGN.md 2).

CONSTRAINT (decided 2026-08-22, must be honoured when this is implemented):
every clause sent to the extractor MUST carry its `heading_path` alongside
`Clause.text`, rendered as breadcrumbs, e.g.

    Section: Acme Retail > 5. Categories excluded from return
    Clause:  Swimwear and swim accessories, including goggles and swim caps.

Why: segmentation splits list stems from their items, so `acme-refunds:008`
("The following categories cannot be returned once the hygiene seal has been
broken:") and its bullets 009-011 are individually non-normative. Clause 010 read
alone is a bare noun phrase carrying no rule; the extractor would either invent an
entitlement or extract nothing. The heading path is what restores the polarity.

The alternative — merging stem and bullets into one clause — was rejected because
it would make editing one excluded category churn its neighbours' content hashes,
costing extraction calls on clauses that did not change.
"""

from __future__ import annotations

from collections.abc import Sequence

from harness.schemas.clause import Clause, PolicyDocument

EXTRACTOR_SYSTEM_PROMPT = """You are a policy extraction engine. Read each clause and extract every entitlement rule it states or that is stated in this document.

For EVERY clause below, output the entitlement rules it states. A clause that grants or withholds a concrete entitlement yields a rule; a clause that only defines a term or only states scope/eligibility in prose yields no rule of its own (see DO NOT). Do NOT skip a clause that does state an entitlement.

Output ONLY a JSON object with a single key "rules" holding a list of rule objects. Each rule:
{
  "rule_id": "kebab-case-descriptive",
  "clause_ids": ["acme-refunds:001:<hash>"],
  "entitlement": "refund|partial_refund|replacement|waiver|extension|discount|credit|cancellation",
  "polarity": "grants|denies",
  "conditions": [
    {"attribute": "fact_name", "op": "<=", "value": 30, "source_span": "exact verbatim text from the clause"}
  ],
  "exceptions": [],
  "precedence": 10,
  "extraction_confidence": 0.95,
  "needs_human_review": false
}

RULES:
- source_span must be a VERBATIM substring of the clause text. Never paraphrase.
- clause_ids must be exactly as they appear in the document (with the 8-hex hash).
- conditions are ANDed. op is one of <=,<,>=,>,==,in,not_in. value is a number, string, or list of strings. For binary facts use the string "yes"/"no" (never JSON true/false).
- precedence: 100 out-of-scope, 50 denials, 10 grants. An exception's precedence must exceed its parent's.
- exceptions: a list of nested rule objects (same structure) for exception carve-outs. Exceptions may have their own exceptions.
- Use standard fact names: days_since_delivery, item_category, order_channel, item_opened, is_clearance_item, has_visible_damage, damage_reported_within_48h, proof_of_purchase_provided, pickup_address_matches_order, order_dispatched, exchange_requests_used_this_order, replacement_stock_available, device_registered_to_account, charging_accessories_present, hygiene_seal_state, seal_tampering_observed, item_in_original_condition, units_of_single_item.

DO NOT:
- Do NOT use "waiver" as the entitlement label unless the policy text literally uses the word "waiver" for that entitlement. "Waiver" is a last resort, not a default. Most denials are about refund, partial_refund, replacement, cancellation, discount, credit, or extension. Read the clause and pick the specific entitlement it actually grants or withholds.
- Do NOT flatten exception trees. If a rule has an exception, and that exception itself has a carve-out, nest it: parent rule -> exceptions list -> that exception's own exceptions list. The full depth must be preserved, not collapsed into a flat list or merged into the parent's conditions.
- Do NOT invent rules from purely definitional or scope text. A clause that only defines a term (e.g. "X means Y"), only describes who the policy applies to, or only restates eligibility in prose is NOT an entitlement rule by itself. Only extract a rule when the clause states a concrete entitlement being granted or withheld, or a concrete condition that grants/withholds one. A definition is only usable as the source_span for a condition in a real rule; it does not become a rule on its own."""


def format_clauses_for_prompt(clauses: Sequence[Clause]) -> str:
    """Render clauses with heading_path breadcrumbs for the extractor prompt."""
    parts = []
    for clause in clauses:
        heading = " > ".join(clause.heading_path) if clause.heading_path else "(no heading)"
        parts.append(
            f"--- clause {clause.clause_id} ---\n"
            f"Section: {heading}\n"
            f"Ordinal: {clause.ordinal}\n"
            f"Text: {clause.text}\n"
        )
    return "\n".join(parts)


def build_extractor_user_prompt(
    document: PolicyDocument,
    target_clause_ids: Sequence[str] | None = None,
) -> str:
    """Build the user prompt for the extractor from a policy document.

    When `target_clause_ids` is given, only those clauses are shown (used for
    batching, so each call's output fits the provider's per-minute token budget).
    Otherwise the whole document is shown in one call.
    """
    if target_clause_ids is not None:
        by_id = {c.clause_id: c for c in document.clauses}
        clauses = [by_id[cid] for cid in target_clause_ids if cid in by_id]
        header = "Extract ALL entitlement rules from the following policy clauses."
        intro = (
            f"Policy: {document.doc_slug}\n\n"
            f"CLAUSES:\n{format_clauses_for_prompt(clauses)}"
        )
    else:
        header = "Extract ALL entitlement rules from this policy document. Cover every clause."
        intro = (
            f"Policy: {document.doc_slug}\n"
            f"Policy version: {document.policy_version}\n\n"
            f"CLAUSES:\n{format_clauses_for_prompt(document.clauses)}"
        )
    return f"{header}\n\n{intro}"


def build_retry_user_prompt(
    original_prompt: str,
    violations: str,
) -> str:
    """Retry prompt after a grounding failure, naming the violations."""
    return (
        f"{original_prompt}\n\n"
        f"GROUNDING ERRORS FROM PREVIOUS ATTEMPT:\n"
        f"{violations}\n\n"
        f"Fix the source_span values above. Each source_span must be a verbatim "
        f"substring (exact characters, same case, same punctuation) of the clause "
        f"text that the condition cites. If you cannot find a verbatim quote, set "
        f"needs_human_review=true on that rule."
    )