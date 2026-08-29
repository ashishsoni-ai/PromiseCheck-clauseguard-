"""report.md + GitHub annotations (DESIGN.md 6.1).

The gate writes a `report.md` alongside the audit store, with the over-promise
count, the 2x3 matrix, and the failure table — so a reviewer who cannot run the
CLI can read the output in a PR. GitHub Actions workflow command annotations are
emitted to stderr so they appear inline in the PR diff.

DESIGN.md 6.3's 55-second demo: edit "within 30 days" to "within 7 days" in the
policy, push, CI runs `clauseguard check`, and the annotation lands on the
probes lockfile with "Clauseguard over-promise: N detected — P-acme-006-*".

The report is written as Markdown so it renders in the GitHub Actions summary
and in the PR comment.
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, TextIO

from harness.audit import AuditRow, AuditStore, iter_over_promises

REPORT_FILENAME: Final = "clauseguard-report.md"

#: GitHub Actions annotation format. `file` is the probes lockfile so the
#: annotation lands on the probe that produced the over-promise; `line` and
#: `col` are 1 because the failure is in the run, not in a specific line.
ANNOTATION_TEMPLATE: Final = (
    "::warning file={file},line=1,col=1,title=Clauseguard::{count} "
    "over-promise(s) detected"
)

ANNOTATION_PROBE_TEMPLATE: Final = (
    "::warning file={file},line=1,col=1,title=Clauseguard::{probe_id}::{message}"
)


def format_annotation(
    count: int,
    *,
    probes_file: str = "probes/probes.lock.json",
    probe_id: str | None = None,
    message: str = "",
) -> str:
    """Format a GitHub Actions workflow command annotation.

    When `probe_id` is given, emits a per-probe annotation so each over-promise
    appears as a separate annotation in the PR diff.
    """
    if probe_id:
        return ANNOTATION_PROBE_TEMPLATE.format(
            file=probes_file,
            probe_id=probe_id,
            message=message,
        )
    return ANNOTATION_TEMPLATE.format(
        file=probes_file,
        count=count,
    )


def emit_annotations(
    rows: list[AuditRow],
    *,
    stream: TextIO,
    probes_file: str = "probes/probes.lock.json",
) -> None:
    """Emit GitHub Actions workflow command annotations for each over-promise.

    Only emits when `GITHUB_ACTIONS` is set in the environment, so local runs
    are not cluttered with annotation syntax.
    """
    if "GITHUB_ACTIONS" not in os.environ:
        return

    failures = list(iter_over_promises(rows))
    if not failures:
        return

    # One summary annotation
    print(
        format_annotation(
            len(failures),
            probes_file=probes_file,
        ),
        file=stream,
    )

    # Per-probe annotations
    for row in failures:
        message_parts = []
        if row.entitlement_asserted:
            message_parts.append(f"asserted: {row.entitlement_asserted}")
        if row.cited_clause_id:
            message_parts.append(f"clause: {row.cited_clause_id}")
        message = "; ".join(message_parts) if message_parts else "over-promise"
        print(
            format_annotation(
                len(failures),
                probes_file=probes_file,
                probe_id=row.probe_id or "unknown",
                message=message,
            ),
            file=stream,
        )


def write_report(
    run_id: str,
    store: AuditStore,
    *,
    output_dir: str | Path = ".",
    probes_file: str = "probes/probes.lock.json",
    gate_passed: bool | None = None,
    threshold: int | None = None,
) -> Path:
    """Write `clauseguard-report.md` with the run's findings.

    Returns the path to the written report.
    """
    rows = store.latest_rows(run_id)
    count = store.over_promise_count(run_id)
    failures = list(iter_over_promises(rows))

    output = Path(output_dir) / REPORT_FILENAME
    lines: list[str] = []

    # Header
    lines.append(f"# Clauseguard Report — run `{run_id}`")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    lines.append("")

    # Gate result
    if gate_passed is not None:
        if gate_passed:
            lines.append("## ✅ Gate: PASS")
        else:
            lines.append("## ❌ Gate: FAIL")
        if threshold is not None:
            lines.append(f"")
            lines.append(f"- Over-promises: **{count}**")
            lines.append(f"- Threshold: **{threshold}**")
        lines.append("")

    # Headline
    lines.append(f"## Headline")
    lines.append("")
    lines.append(f"**{count}** over-promises detected across **{len(rows)}** probes.")
    lines.append("")

    # Failure table
    if failures:
        lines.append("## Over-promises")
        lines.append("")
        lines.append("| Probe | Strategy | Tier | Asserted | Clause | Span |")
        lines.append("|---|---|---|---|---|---|")
        for row in failures:
            probe_id = row.probe_id or "—"
            strategy = row.strategy or "—"
            tier = str(row.difficulty_tier or "—")
            asserted = row.entitlement_asserted or "—"
            clause = row.cited_clause_id or "—"
            span = ""
            if row.response_span:
                span = textwrap.shorten(
                    " ".join(row.response_span.split()), width=80
                )
            lines.append(
                f"| {probe_id} | {strategy} | {tier} | {asserted} | {clause} | {span} |"
            )
        lines.append("")

    # Run metadata
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"- **Run ID**: `{run_id}`")
    lines.append(f"- **Probes evaluated**: {len(rows)}")
    lines.append(f"- **Over-promises**: {count}")
    if threshold is not None:
        lines.append(f"- **Gate threshold**: {threshold}")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(
        "_Report generated by Clauseguard — "
        "[policy-conformance harness for money-touching agents]_"
    )
    lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main(
    argv: list[str] | None = None,
    *,
    store_path: str | Path = "runs.db",
    output_dir: str | Path = ".",
    stream: TextIO | None = None,
) -> int:
    """CLI entry point for report generation.

    Usage:
        clauseguard report [--run-id RUN_ID] [--store PATH] [--output DIR]
                           [--gate-passed] [--gate-failed] [--threshold N]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate clauseguard-report.md from a completed run",
    )
    parser.add_argument(
        "--run-id",
        help="Run ID to report on (default: most recent run in the store)",
    )
    parser.add_argument(
        "--store",
        default=str(store_path),
        help="Path to the audit store (default: runs.db)",
    )
    parser.add_argument(
        "--output",
        default=str(output_dir),
        help="Output directory for the report (default: current directory)",
    )
    parser.add_argument(
        "--gate-passed",
        action="store_true",
        default=None,
        help="Mark the gate as passed in the report",
    )
    parser.add_argument(
        "--gate-failed",
        action="store_true",
        default=None,
        help="Mark the gate as failed in the report",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Gate threshold for the report",
    )
    parser.add_argument(
        "--annotations",
        action="store_true",
        default=False,
        help="Emit GitHub Actions annotations to stderr",
    )

    args = parser.parse_args(argv)

    store = AuditStore(Path(args.store))
    run_ids = store.run_ids()
    if not run_ids:
        print("No runs in the audit store.", file=stream or sys.stderr)
        return 1

    run_id = args.run_id or run_ids[-1]

    gate_passed: bool | None = None
    if args.gate_passed:
        gate_passed = True
    elif args.gate_failed:
        gate_passed = False

    report_path = write_report(
        run_id,
        store,
        output_dir=args.output,
        gate_passed=gate_passed,
        threshold=args.threshold,
    )
    print(f"Report written to {report_path}", file=stream or sys.stdout)

    if args.annotations:
        rows = store.latest_rows(run_id)
        emit_annotations(rows, stream=stream or sys.stderr)

    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())