"""Run the LLM extractor over acme-refunds, compare against the hand-authored rules.

This is a MEASUREMENT exercise, not a replacement. It writes the extractor's raw
output to `rules/rules.extracted.json` (never to `rules.lock.json`), computes
DESIGN.md 8's extraction-coverage metric, and prints a per-rule comparison of the
16 hand-authored rules against what the extractor produced.

Usage:
    python scripts/extract_and_compare.py

Requires GROQ_API_KEY (the extractor is a hosted model). The output file is
side-by-side with the lockfile so a reviewer can diff the two.

Why `rules.extracted.json` is written by hand rather than via `write_rules`:
`write_rules` is a correctness gate for a rule set the harness will actually run
on - it refuses any tree that fails `validate_rule_tree` or carries an ungrounded,
unflagged span. The extractor's output is a *candidate* whose whole point is to be
compared, not trusted, so this script records exactly what the model returned even
where it would not pass the gate, and reports the gate's verdict alongside. The
comparison then tells you which differences are content and which are shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Same idiom as author_rules.py: running this as a file path puts `scripts/` on
# sys.path rather than the repo root, so `import harness` fails.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.execution.lockfiles import (  # noqa: E402
    DEFAULT_RULES_LOCK,
    RULES_LOCK_SCHEMA,
    load_rules,
)
from harness.extract.compare import compare_rule_sets  # noqa: E402
from harness.extract.coverage import compute_coverage  # noqa: E402
from harness.extract.extractor import (  # noqa: E402
    DEFAULT_EXTRACTOR_MODEL,
    LitellmExtractorClient,
    extract_rules,
    resolve_extractor_temp,
)
from harness.ingest import ingest  # noqa: E402
from harness.rules_engine import validate_rule_tree  # noqa: E402
from harness.rules_engine.evaluate import MalformedRuleTreeError  # noqa: E402

POLICY_SOURCE = "policies/acme-refunds.md"
OUT_PATH = Path("rules/rules.extracted.json")

#: Written into the envelope so a reviewer never has to guess whether an
#: extractor touched this file. DESIGN.md 9 wants hand-computed labels in the
#: lockfile, and this file is explicitly NOT that.
EXTRACTED_BY = "extracted by harness/extract/extractor.py; ungrounded spans flagged for human review"


def _validate_report(rules) -> str:
    """The gate verdict: does the extractor output pass the runnable-rule check?"""
    try:
        validate_rule_tree(rules)
        return "passes"
    except MalformedRuleTreeError as exc:
        return f"FAILS: {exc}"


def _write_extracted(path: Path, *, rules, policy, model: str) -> Path:
    """Write the extractor output with the lockfile envelope, to a NEW path."""
    payload = {
        "schema": RULES_LOCK_SCHEMA,
        "policy_doc": policy.doc_slug,
        "policy_version": policy.policy_version,
        "authored_by": f"{EXTRACTED_BY} (model {model})",
        "rules": [rule.model_dump(mode="json") for rule in rules],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _format_conditions(rule) -> str:
    parts = []
    for c in rule.conditions:
        value = c.value if not isinstance(c.value, list) else ", ".join(c.value)
        parts.append(f"{c.attribute} {c.op} {value}")
    return " AND ".join(parts) if parts else "(unconditional)"


def _format_rule(rule) -> str:
    return (
        f"    {rule.rule_id}  [{rule.entitlement} / {rule.polarity} / "
        f"prec {rule.precedence}]\n"
        f"      {_format_conditions(rule)}"
    )


def render_report(report) -> str:
    """Render the comparison as a readable console report."""
    # Lookup over the WHOLE tree (roots + nested exceptions): a comparison row
    # may name an exception node, which is not in the top-level list.
    hand_by_id = {n.rule_id: n for r in report.hand_rules for n in r.walk()}
    ext_by_id = {n.rule_id: n for r in report.extracted_rules for n in r.walk()}

    lines = []
    lines.append("=" * 78)
    lines.append("EXTRACTOR vs HAND-AUTHORED RULES")
    lines.append("=" * 78)
    lines.append(
        f"  hand rules    : {report.hand_rule_count} (incl. nested exceptions)"
    )
    lines.append(f"  extracted     : {report.extracted_rule_count}")
    lines.append(
        f"  equivalent    : {len(report.equivalent)}"
        f"   different: {len(report.different)}   missed: {len(report.missed)}"
    )
    lines.append(f"  invented      : {len(report.inventions)}")

    if report.different:
        lines.append("\n-- DIFFERENT (same entitlement+polarity, conditions differ) --")
        for row in report.different:
            lines.append(f"  hand {row.hand_rule_id}  ->  {row.extracted_rule_id}")
            hand = hand_by_id.get(row.hand_rule_id)
            ext = ext_by_id.get(row.extracted_rule_id or "")
            if hand:
                lines.append("    hand:")
                lines.append(_format_rule(hand).replace("\n", "\n    "))
            if ext:
                lines.append("    extracted:")
                lines.append(_format_rule(ext).replace("\n", "\n    "))

    if report.missed:
        lines.append("\n-- MISSED (no extracted rule in the same cell) --")
        for row in report.missed:
            hand = hand_by_id.get(row.hand_rule_id)
            if hand:
                lines.append(_format_rule(hand))

    if report.inventions:
        lines.append("\n-- INVENTED (no hand rule in the same cell) --")
        for rule in report.inventions:
            lines.append(_format_rule(rule))

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract rules from acme-refunds policy and compare against hand-authored set."
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Timeout in seconds per extraction call (default: 240s for hosted, bump for local Ollama)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Output token cap per call (default 4000; raise for local models with no TPM limit)"
    )
    args = parser.parse_args(argv)

    policy = ingest(POLICY_SOURCE, corpus_role="worked_example")
    print(f"policy      : {policy.doc_slug}  {policy.policy_version}")
    print(f"clauses     : {len(policy.clauses)}")

    if args.timeout and args.max_tokens:
        client = LitellmExtractorClient(timeout_s=args.timeout, max_tokens=args.max_tokens)
    elif args.timeout:
        client = LitellmExtractorClient(timeout_s=args.timeout)
    elif args.max_tokens:
        client = LitellmExtractorClient(max_tokens=args.max_tokens)
    else:
        client = LitellmExtractorClient()
    print(f"extractor   : {client.model}  (temperature {resolve_extractor_temp()})")
    if client.model != DEFAULT_EXTRACTOR_MODEL:
        print(
            f"  NOTE      : this run used {client.model}, NOT the pinned "
            f"{DEFAULT_EXTRACTOR_MODEL}. One-off comparison run, not the canonical "
            f"extraction config (see docs/limitations.md for the local-model argument)."
        )
    print(f"extracting  : this is a live LLM call, may take a minute...")
    extracted = extract_rules(policy, client=client)

    path = _write_extracted(OUT_PATH, rules=extracted, policy=policy, model=client.model)
    print(f"written     : {path}  ({len(extracted)} root rule(s))")

    # Gate verdict (does not stop the write - this is a measurement).
    verdict = _validate_report(extracted)
    print(f"tree check  : {verdict}")

    # DESIGN.md 8 extraction-coverage metric.
    coverage = compute_coverage(policy, extracted)
    print(coverage.summary())
    print(f"  DESIGN.md 8 band    : {coverage.band}")

    # Hand-authored comparison.
    hand = load_rules(DEFAULT_RULES_LOCK)
    report = compare_rule_sets(hand.rules, extracted)
    print(render_report(report))

    print(f"\nhand lockfile: {DEFAULT_RULES_LOCK} (UNTOUCHED by this script)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else None))