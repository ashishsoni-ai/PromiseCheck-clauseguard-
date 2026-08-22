"""`python -m harness.ingest <source> [...]` - the Step 2 standalone checkpoint.

Prints the `Clause[]` a source segments into, with IDs, so clause segmentation and
hashing can be inspected by a human before anything downstream trusts them. Also
prints the 40-400 token audit from DESIGN.md 1.1 and, with `--manifest`, the
step ① diff against the committed baseline.

Deliberately not part of `harness/cli.py`: that CLI is Step 7's `clauseguard run`,
and this stays runnable on its own so ingest can be debugged without the rest of
the harness existing or working.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harness.ingest import (
    MANIFEST_PATH,
    MAX_CLAUSE_TOKENS,
    MIN_CLAUSE_TOKENS,
    diff_against_manifest,
    ingest,
    update_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m harness.ingest",
        description="Segment and hash a policy source into addressable clauses.",
    )
    p.add_argument("sources", nargs="+", help="Markdown/PDF path, or http(s) URL.")
    p.add_argument(
        "--corpus-role",
        choices=["real", "synthetic_stress", "worked_example"],
        default="worked_example",
        help="Provenance (DESIGN.md 7.1). Defaults to worked_example, which is "
        "never counted as evidence - so an unlabelled run cannot inflate results.",
    )
    p.add_argument("--holdout", action="store_true", help="Mark as held out (7.3).")
    p.add_argument(
        "--full-text", action="store_true", help="Print whole clauses, not excerpts."
    )
    p.add_argument(
        "--manifest",
        nargs="?",
        const=str(MANIFEST_PATH),
        default=None,
        metavar="PATH",
        help=f"Diff against a manifest (default {MANIFEST_PATH}).",
    )
    p.add_argument(
        "--write-manifest",
        action="store_true",
        help="Update the baseline after diffing. Implies --manifest.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest_path = Path(args.manifest) if args.manifest else MANIFEST_PATH
    exit_code = 0

    for source in args.sources:
        doc = ingest(
            source, corpus_role=args.corpus_role, is_holdout=args.holdout
        )

        print("=" * 78)
        print(f"{doc.doc_slug}  <- {doc.source}")
        print(f"policy_version : {doc.policy_version}")
        print(f"fetched_at     : {doc.fetched_at.isoformat()}")
        print(
            f"corpus_role    : {doc.corpus_role}"
            f"   evidence={doc.counts_as_evidence}  holdout={doc.is_holdout}"
        )
        print(f"clauses        : {len(doc.clauses)}")
        print("=" * 78)

        for clause in doc.clauses:
            path = " > ".join(clause.heading_path) or "(root)"
            print(f"\n{clause.clause_id}   [{clause.token_estimate}t]  {path}")
            body = clause.text if args.full_text else clause.text[:150]
            suffix = "" if args.full_text or len(clause.text) <= 150 else " ..."
            print(f"    {body}{suffix}")

        lengths = [c.token_estimate or 0 for c in doc.clauses]
        under = sum(1 for n in lengths if n < MIN_CLAUSE_TOKENS)
        over = sum(1 for n in lengths if n > MAX_CLAUSE_TOKENS)
        print(
            f"\nlength audit (target {MIN_CLAUSE_TOKENS}-{MAX_CLAUSE_TOKENS} tokens): "
            f"{len(lengths) - under - over}/{len(lengths)} in band, "
            f"{under} under, {over} over, "
            f"min={min(lengths, default=0)}, max={max(lengths, default=0)}"
        )

        if args.manifest or args.write_manifest:
            diff = diff_against_manifest(doc, manifest_path)
            print(f"\nmanifest diff ({manifest_path}): {diff.summary()}")
            if diff.is_new_document:
                print("  document is not in the baseline yet (first sighting)")
            for change in diff.changed:
                print(
                    f"  ordinal {change.ordinal:03d} changed: "
                    f"{change.old_hash} -> {change.new_hash}"
                )
            for move in diff.moved:
                print(
                    f"  content {move.content_hash} moved: "
                    f"{move.old_ordinal:03d} -> {move.new_ordinal:03d}"
                )
            for clause_id in diff.removed:
                print(f"  removed: {clause_id}")
            print(f"  needs extraction: {len(diff.needs_extraction)} clause(s)")

            if args.write_manifest:
                written = update_manifest(doc, diff, manifest_path)
                print(f"  baseline written: {written}")
            elif not diff.is_clean:
                # Non-zero so the diff is usable in a shell conditional before the
                # real gate (Step 8) exists.
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
