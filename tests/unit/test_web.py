"""Smoke test for harness/web/app.py — the minimal run-summary dashboard.

Verifies the single route renders from a real audit store, and that the empty
store case is handled. Uses the same `make_audit_row` fixture and `AuditStore`
append path as the rest of the suite, so it is hermetic (no real runs.db).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from harness.audit import AuditStore
from harness.web.app import _summary_context, app


@pytest.fixture
def seeded_store(tmp_path, make_audit_row):
    """A temp audit store with one over-promise row and one clean denial."""
    store = AuditStore(tmp_path / "runs.db").initialise()
    over = make_audit_row(
        run_id="0192f3a1-0000-7000-8000-000000000001",
        expected_policy_stance="denies",
        agent_stance="grants",
        verdict_class="OVER_PROMISE",
    )
    clean = make_audit_row(
        probe_id="P-acme-014-boundary-004",
        run_id="0192f3a1-0000-7000-8000-000000000001",
        expected_policy_stance="denies",
        agent_stance="denies",
        verdict_class="CORRECT_DENIAL",
    )
    store.append_many([over, clean])
    return tmp_path / "runs.db"


class TestWebApp:
    def test_summary_context_from_real_store(self, seeded_store):
        ctx = _summary_context(str(seeded_store))
        assert ctx["has_run"] is True
        assert ctx["attempted"] == 2
        assert ctx["over_promises"] == 1
        # The over-promise cell carries the count.
        op = next(c for c in ctx["matrix"] if c["is_over_promise"])
        assert op["count"] == 1
        assert op["policy_stance"] == "denies"
        assert op["agent_stance"] == "grants"

    def test_empty_store_reports_no_run(self, tmp_path):
        empty = tmp_path / "empty.db"
        AuditStore(empty).initialise()
        ctx = _summary_context(str(empty))
        assert ctx["has_run"] is False
        assert "No runs" in ctx["message"]

    def test_route_renders_html_with_seeded_rows(self, seeded_store):
        client = TestClient(app)
        resp = client.get("/", params={"db": str(seeded_store)})
        assert resp.status_code == 200
        assert "OVER-PROMISES: 1 / 2" in resp.text
        assert "run" in resp.text

    def test_route_renders_html_for_empty_store(self, tmp_path):
        empty = tmp_path / "empty.db"
        AuditStore(empty).initialise()
        client = TestClient(app)
        resp = client.get("/", params={"db": str(empty)})
        assert resp.status_code == 200
        assert "No runs" in resp.text