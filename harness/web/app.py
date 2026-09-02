"""harness/web - minimal run-summary dashboard (DESIGN.md 1.7, minimal slice).

This is the smallest honest build of DESIGN.md 1.7's dashboard: a single FastAPI
route that renders what `clauseguard run` already prints - the over-promise
headline, the 2x3 matrix, and the small print - as a plain HTML page. No HTMX,
no diff view, no review queue; the CLI serves all of that content today, so the
web slice only needs to be *real*, not comprehensive. The deferred screens are
documented as such in web/routes/diff.py and web/routes/review.py.

It reads the same audit store `clauseguard run` writes, and reuses the same
tally/matrix arithmetic (VerdictClass cells), so the page cannot drift from the
numbers the CLI prints - both are derived from the same rows.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from harness.audit import AuditStore, VerdictClass

#: The audit store the dashboard reads. Same default as `clauseguard run`.
DEFAULT_DB_PATH = "runs.db"

#: Template directory, sibling to this file.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="Clauseguard run summary")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _tally(rows) -> dict[VerdictClass, int]:
    """Count every verdict cell, the same tally the CLI summary uses."""
    counts = {cell: 0 for cell in VerdictClass}
    for row in rows:
        if row.verdict_class is not None:
            counts[row.verdict_class] += 1
    return counts


def _summary_context(db_path: str) -> dict:
    """Build the template context from the latest run in the audit store."""
    store = AuditStore(Path(db_path)).initialise()
    run_ids = store.run_ids()
    if not run_ids:
        return {
            "has_run": False,
            "message": "No runs in the audit store yet.",
        }
    latest = run_ids[-1]
    rows = store.latest_rows(latest)
    counts = _tally(rows)

    over_promises = counts[VerdictClass.OVER_PROMISE]
    under_serves = counts[VerdictClass.UNDER_SERVE]
    evasive = (
        counts[VerdictClass.EVASIVE_ON_GRANT] + counts[VerdictClass.EVASIVE_ON_DENIAL]
    )
    abstained = sum(1 for row in rows if row.judge_abstained)
    attempted = len(rows)

    # The 2x3 matrix, cell by cell, in the CLI's order.
    grid = [
        # (policy_stance, agent_stance, cell)
        ("grants", "grants", VerdictClass.CORRECT_GRANT),
        ("grants", "denies", VerdictClass.UNDER_SERVE),
        ("grants", "evasive", VerdictClass.EVASIVE_ON_GRANT),
        ("denies", "grants", VerdictClass.OVER_PROMISE),
        ("denies", "denies", VerdictClass.CORRECT_DENIAL),
        ("denies", "evasive", VerdictClass.EVASIVE_ON_DENIAL),
    ]
    matrix = []
    for policy_stance, agent_stance, cell in grid:
        matrix.append(
            {
                "policy_stance": policy_stance,
                "agent_stance": agent_stance,
                "count": counts[cell],
                "is_over_promise": cell is VerdictClass.OVER_PROMISE,
            }
        )

    scored = sum(counts.values())
    unscored = attempted - scored
    errored = sum(1 for row in rows if row.judge_error is not None)

    return {
        "has_run": True,
        "run_id": latest,
        "attempted": attempted,
        "over_promises": over_promises,
        "under_serves": under_serves,
        "evasive": evasive,
        "abstained": abstained,
        "unscored": unscored,
        "errored": errored,
        "matrix": matrix,
    }


@app.get("/", response_class=HTMLResponse)
def summary(request: Request, db: str = DEFAULT_DB_PATH) -> HTMLResponse:
    """The one route: render the latest run's summary as HTML."""
    context = _summary_context(db)
    return templates.TemplateResponse(
        request=request, name="dashboard.html", context=context
    )