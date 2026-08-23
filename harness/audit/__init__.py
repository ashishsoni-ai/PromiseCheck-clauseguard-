"""Audit: one immutable row per probe attempted, in a single `runs.db`.

DESIGN.md 1.6 and 5.1. Two modules, split along the line where validation stops:

    models   the row schema and its invariants  (AuditRow / AuditRowRecord)
    store    append-only access to the file      (AuditStore)

Re-exported so call sites read `from harness.audit import AuditStore` rather than
reaching into module paths, matching `harness.schemas` and `harness.ingest`.
"""

from __future__ import annotations

from harness.audit.models import (
    AGENT_STANCES,
    JSON_COLUMNS,
    POLICY_STANCES,
    AuditRow,
    AuditRowRecord,
    VerdictClass,
    classify_verdict,
    new_row_id,
    new_run_id,
    utc_now_iso,
)
from harness.audit.store import (
    DEFAULT_DB_PATH,
    AuditError,
    AuditIntegrityError,
    AuditStore,
    Reconciliation,
    SupersedeError,
    iter_over_promises,
    new_run,
)

__all__ = [
    "AGENT_STANCES",
    "DEFAULT_DB_PATH",
    "JSON_COLUMNS",
    "POLICY_STANCES",
    "AuditError",
    "AuditIntegrityError",
    "AuditRow",
    "AuditRowRecord",
    "AuditStore",
    "Reconciliation",
    "SupersedeError",
    "VerdictClass",
    "classify_verdict",
    "iter_over_promises",
    "new_row_id",
    "new_run",
    "new_run_id",
    "utc_now_iso",
]
