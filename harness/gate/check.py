"""clauseguard check — the CI gate (DESIGN.md 6). Exit 0/1.

DESIGN.md 2 step 11: the gate compares the over-promise count against a threshold
and exits 0 (pass) or 1 (fail). The threshold can be an absolute number, or
"--baseline" which reads the previous run's count from the audit store and fails
if the current run is worse.

The harness half was already done before this module existed: `gate_run` is a
column in every audit row, and `clauseguard run` already produces the count a
threshold would compare against. What was missing was the exit-code contract,
the baseline comparison, and the GitHub Actions annotation that lands the
reviewer on the offending line.

DESIGN.md 6.3's 55-second demo — edit "within 30 days" to "within 7 days",
push, watch CI go red — is now recordable because this gate exists.

`check` is deliberately separate from `run`. DESIGN.md 2 step 11 says "the
number has to be readable by someone who disagrees with the threshold", and
putting the comparison inside `run` would make the number and the verdict about
the number the same code path. Exit 1 is reserved for the gate and never
returned by `run`, so `check` can claim it without redefining what `run` means.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Final, TextIO

from harness.audit import AuditStore, iter_over_promises
from harness.execution import (
    DEFAULT_PROBES_LOCK,
    DEFAULT_RULES_LOCK,
    ProbesLock,
    RulesLock,
    load_probes,
    load_rules,
)

EXIT_PASS: Final = 0
EXIT_FAIL: Final = 1
EXIT_OPERATIONAL: Final = 2

#: Default threshold: zero over-promises tolerated. A merchant can raise this
#: during a known-bad rollout, but the default says "the gate exists to catch
#: things, not to approve them".
DEFAULT_MAX_OVER_PROMISE = 0

#: The GitHub Actions annotation format. `file` is the probes lockfile so the
#: annotation lands on the probe that produced the over-promise; `line` and
#: `col` are 1 because the failure is in the run, not in a specific line of the
#: lockfile.
ANNOTATION_TEMPLATE = (
    "::warning file={file},line=1,col=1,title=Clauseguard over-promise::{count} "
    "over-promise(s) detected — {probes}"
)


class GateError(RuntimeError):
    """The gate could not determine pass or fail."""


def resolve_threshold(
    max_over_promise: int | None,
    baseline: str | None,
    store: AuditStore,
    *,
    default: int = DEFAULT_MAX_OVER_PROMISE,
) -> int:
    """Resolve the over-promise threshold for this gate run.

    Three modes, in precedence order:

    1. `--max-overpromise N` — absolute threshold. Explicit, reviewable,
       and the one a CI config file should pin.
    2. `--baseline` — compare against the previous run's count. The threshold
       becomes "no worse than the last run". This is the right default for a
       regression gate: it catches regressions without requiring a human to
       decide what number is acceptable.
    3. Neither — use `default` (0). Strictest possible gate. Useful for
       critical-path deployments where any over-promise blocks the merge.

    `--baseline` with no previous run is an operational error: there is nothing
    to compare against, and defaulting to 0 would silently switch modes.
    """
    if max_over_promise is not None:
        if max_over_promise < 0:
            raise GateError(
                f"--max-overpromise must be >= 0, got {max_over_promise}"
            )
        return max_over_promise

    if baseline is not None:
        run_ids = store.run_ids()
        if len(run_ids) < 2:
            raise GateError(
                "--baseline requires at least two runs in the audit store "
                f"({len(run_ids)} found). Run `clauseguard run` twice, or "
                "use --max-overpromise N for the first gate."
            )
        previous_id = run_ids[-2]
        previous_count = store.over_promise_count(previous_id)
        return previous_count

    return default


def check_run(
    run_id: str,
    store: AuditStore,
    *,
    max_over_promise: int | None = None,
    baseline: str | None = None,
    probes: ProbesLock | None = None,
    rules: RulesLock | None = None,
    annotations: bool = False,
    stream: TextIO = sys.stdout,
) -> int:
    """Evaluate a run against the gate threshold.

    Returns EXIT_PASS (0) or EXIT_FAIL (1). Never raises for a threshold
    exceedance — that is the normal condition this module exists to detect.
    Raises `GateError` for operational failures that prevent evaluation.

    The run must already exist in `store`. `clauseguard check` is always
    post-hoc: it reads a completed run and decides pass/fail. It does not
    re-run probes or re-judge anything.
    """
    run_ids = store.run_ids()
    if run_id not in run_ids:
        raise GateError(f"run {run_id} not found in the audit store")

    threshold = resolve_threshold(max_over_promise, baseline, store)
    count = store.over_promise_count(run_id)

    rows = store.latest_rows(run_id)
    failures = list(iter_over_promises(rows))

    print(f"  Gate evaluation for run {run_id}", file=stream)
    print(f"    Over-promises: {count}", file=stream)
    print(f"    Threshold:     {threshold}", file=stream)
    print(file=stream)

    if count <= threshold:
        print(f"  ✓ PASS — {count} ≤ {threshold}", file=stream)
        return EXIT_PASS

    print(f"  ✗ FAIL — {count} > {threshold}", file=stream)
    print(file=stream)

    # Print the failure table, same format as `clauseguard run`'s summary
    for row in failures:
        probe_id = row.probe_id or "(unknown)"
        strategy = row.strategy or "?"
        tier = row.difficulty_tier or "?"
        print(f"    {probe_id}  [{strategy}, tier {tier}]", file=stream)
        if row.entitlement_asserted:
            print(f"      asserted: {row.entitlement_asserted}", file=stream)
        if row.cited_clause_id:
            print(f"      clause:   {row.cited_clause_id}", file=stream)
        if row.response_span:
            print(f"      span:     {textwrap.shorten(row.response_span, width=120)}",
                  file=stream)
        print(file=stream)

    # GitHub Actions annotation
    if annotations and "GITHUB_ACTIONS" in os.environ:
        probe_names = ", ".join(
            row.probe_id for row in failures if row.probe_id
        )
        annotation = ANNOTATION_TEMPLATE.format(
            file=str(probes._path if probes else DEFAULT_PROBES_LOCK),
            count=count,
            probes=probe_names or "(unknown)",
        )
        print(annotation, file=stream)

    return EXIT_FAIL


def main(
    argv: list[str] | None = None,
    *,
    store_path: str | Path = "runs.db",
    probes_path: str | Path = DEFAULT_PROBES_LOCK,
    rules_path: str | Path = DEFAULT_RULES_LOCK,
    stream: TextIO = sys.stdout,
) -> int:
    """CLI entry point for `clauseguard check`.

    Usage:
        clauseguard check [--run-id RUN_ID] [--max-overpromise N] [--baseline]
                          [--annotations] [--store PATH] [--probes PATH]

    Without --run-id, checks the most recent run in the store.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Clauseguard CI gate — evaluate a run against a threshold",
    )
    parser.add_argument(
        "--run-id",
        help="Run ID to evaluate (default: most recent run in the store)",
    )
    parser.add_argument(
        "--max-overpromise",
        type=int,
        default=None,
        help=f"Absolute over-promise threshold (default: {DEFAULT_MAX_OVER_PROMISE})",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Compare against the previous run's over-promise count",
    )
    parser.add_argument(
        "--annotations",
        action="store_true",
        default=False,
        help="Emit GitHub Actions workflow command annotations",
    )
    parser.add_argument(
        "--store",
        default=str(store_path),
        help="Path to the audit store (default: runs.db)",
    )
    parser.add_argument(
        "--probes",
        default=str(probes_path),
        help="Path to probes.lock.json (default: probes/probes.lock.json)",
    )
    parser.add_argument(
        "--rules",
        default=str(rules_path),
        help="Path to rules.lock.json (default: rules/rules.lock.json)",
    )

    args = parser.parse_args(argv)

    store = AuditStore(Path(args.store))
    run_ids = store.run_ids()
    if not run_ids:
        print("No runs in the audit store — nothing to gate on.", file=stream)
        return EXIT_PASS

    run_id = args.run_id or run_ids[-1]

    probes = None
    rules = None
    if args.probes:
        try:
            probes = load_probes(Path(args.probes))
        except Exception:
            pass  # Optional — annotations just won't name the probe file
    if args.rules:
        try:
            rules = load_rules(Path(args.rules))
        except Exception:
            pass

    try:
        return check_run(
            run_id,
            store,
            max_over_promise=args.max_overpromise,
            baseline="--baseline" if args.baseline else None,
            probes=probes,
            rules=rules,
            annotations=args.annotations,
            stream=stream,
        )
    except GateError as exc:
        print(f"Gate error: {exc}", file=stream)
        return EXIT_OPERATIONAL


if __name__ == "__main__":
    raise SystemExit(main())