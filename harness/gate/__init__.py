"""Gate: pass/fail decision for `clauseguard check` (DESIGN.md 6).

Three modules:

    check       the exit-code contract: compare over-promise count against
                a threshold (absolute or baseline), return 0 or 1
    report     `report.md` writer and GitHub Actions annotation emitter

Re-exported so call sites read `from harness.gate import check_run` rather
than reaching into module paths.
"""

from __future__ import annotations

from harness.gate.check import (
    ANNOTATION_TEMPLATE,
    DEFAULT_MAX_OVER_PROMISE,
    EXIT_FAIL,
    EXIT_OPERATIONAL,
    EXIT_PASS,
    GateError,
    check_run,
    main as check_main,
    resolve_threshold,
)
from harness.gate.report import (
    REPORT_FILENAME,
    emit_annotations,
    format_annotation,
    main as report_main,
    write_report,
)

__all__ = [
    "ANNOTATION_TEMPLATE",
    "DEFAULT_MAX_OVER_PROMISE",
    "EXIT_FAIL",
    "EXIT_OPERATIONAL",
    "EXIT_PASS",
    "REPORT_FILENAME",
    "GateError",
    "check_main",
    "check_run",
    "emit_annotations",
    "format_annotation",
    "report_main",
    "resolve_threshold",
    "write_report",
]