"""STEP 6 checkpoint tests - the append-only store per DESIGN.md 1.6.

    Append-only by convention: no `UPDATE`, no `DELETE`. Each run is immutable and
    identified by `run_id = uuid7()`; corrections are new rows with `supersedes_id`.

Three claims are under test, and one non-claim:

  - **Appends work and read back in order**, which is the whole of what a run needs.
  - **UPDATE and DELETE are refused by SQLite itself.** `TestAppendOnlyEnforcement`
    asserts the refusal did not come from a Python guard, because a Python guard is
    bypassed by the next script someone writes against the file.
  - **Corrections are additive.** A superseded row stays readable and stops
    counting, and `TestSupersede` checks the second half by watching the
    over-promise headline move.
  - The non-claim: **append-only does not mean the contents are trustworthy.**
    `TestIntegrityOnRead` appends a malformed row through raw SQL - which the
    triggers permit, because appending is the one thing they allow - and shows the
    read path catching it. That is the honest boundary of this design, and it is
    better as a test than as a paragraph.
"""

from __future__ import annotations

import re

import pytest
from sqlmodel import text

from harness.audit.models import AuditRowRecord, VerdictClass
from harness.audit.store import (
    AuditError,
    AuditIntegrityError,
    AuditStore,
    SupersedeError,
    iter_over_promises,
    new_run,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path):
    """A fresh, initialised, trigger-protected store per test."""
    with AuditStore(tmp_path / "runs.db") as opened:
        yield opened


@pytest.fixture
def unprotected_store(tmp_path):
    """A store whose append-only triggers were never installed."""
    with AuditStore(tmp_path / "unprotected.db", enforce_append_only=False) as opened:
        yield opened


def raw_execute(store: AuditStore, sql: str, params: dict | None = None):
    """Run SQL against the file directly, bypassing every model in the harness.

    Rows are materialised inside the connection's scope and returned as a list: a
    `CursorResult` is not usable once the connection that produced it has closed, so
    returning the result object itself would fail at the point of use rather than
    here.
    """
    with store.engine.begin() as connection:
        result = connection.execute(text(sql), params or {})
        return list(result) if result.returns_rows else None


# ==========================================================================
# Lifecycle
# ==========================================================================
class TestInitialisation:
    def test_it_creates_the_file(self, tmp_path):
        path = tmp_path / "runs.db"
        with AuditStore(path):
            assert path.exists()

    def test_it_creates_missing_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "runs.db"
        with AuditStore(path):
            assert path.exists()

    def test_initialise_is_idempotent(self, tmp_path, make_audit_row):
        """Opening an existing database is the same call as creating one, so the
        demo path and the test path cannot diverge."""
        path = tmp_path / "runs.db"
        first = AuditStore(path).initialise()
        row = first.append(make_audit_row())
        first.close()

        second = AuditStore(path).initialise()
        try:
            assert second.get(row.row_id) is not None
            assert len(second.rows()) == 1
        finally:
            second.close()

    def test_using_the_store_before_initialise_is_an_error(self, tmp_path, make_audit_row):
        """Creating the table lazily on first write would mean a read against a
        fresh database returned zero rows instead of saying it was not set up."""
        store = AuditStore(tmp_path / "runs.db")
        with pytest.raises(AuditError, match="before initialise"):
            store.append(make_audit_row())
        with pytest.raises(AuditError, match="before initialise"):
            store.rows()

    def test_the_triggers_are_installed(self, store):
        installed = store.installed_triggers()
        assert installed == [
            "audit_rows_is_append_only_no_delete",
            "audit_rows_is_append_only_no_update",
        ]

    def test_enforcement_can_be_switched_off(self, unprotected_store):
        assert unprotected_store.installed_triggers() == []

    def test_the_flag_does_not_remove_protection_from_an_existing_file(
        self, tmp_path
    ):
        """`enforce_append_only=False` controls whether triggers are *installed*, not
        whether existing ones are dropped. Worth pinning: someone reading the flag
        name could reasonably expect it to disable protection on an already-protected
        file, and it does not - the triggers live in the database, not in this class.
        """
        path = tmp_path / "runs.db"
        AuditStore(path).initialise().close()
        reopened = AuditStore(path, enforce_append_only=False).initialise()
        try:
            assert len(reopened.installed_triggers()) == 2
        finally:
            reopened.close()


# ==========================================================================
# Appending
# ==========================================================================
class TestAppend:
    def test_a_row_round_trips_through_sqlite(self, store, make_audit_row):
        row = make_audit_row()
        store.append(row)
        assert store.get(row.row_id).model_dump() == row.model_dump()

    def test_append_returns_the_row_unchanged(self, store, make_audit_row):
        row = make_audit_row()
        assert store.append(row) is row

    def test_get_of_an_unknown_row_is_none_not_an_error(self, store):
        assert store.get("no-such-row") is None

    def test_append_many_returns_a_count(self, store, make_audit_row):
        assert store.append_many([make_audit_row() for _ in range(5)]) == 5
        assert len(store.rows()) == 5

    def test_an_empty_batch_is_a_no_op(self, store):
        assert store.append_many([]) == 0
        assert store.rows() == []

    def test_a_batch_with_a_repeated_row_id_is_refused_before_it_is_written(
        self, store, make_audit_row
    ):
        """Caught in Python rather than left to the primary key.

        Two records sharing a primary key inside one session is a case where
        SQLAlchemy's behaviour depends on its identity map rather than on SQLite, so
        rather than depend on which of those wins, the batch is rejected before any
        of it reaches the database. What is asserted here is only what this module
        promises: the call raises, and nothing was stored.
        """
        row = make_audit_row()
        with pytest.raises(AuditError, match="appears twice"):
            store.append_many([row, row])
        assert store.rows() == []

    def test_reusing_a_row_id_in_a_later_batch_is_refused_by_the_primary_key(
        self, store, make_audit_row
    ):
        row = make_audit_row()
        store.append(row)
        with pytest.raises(Exception) as excinfo:
            store.append(make_audit_row(row_id=row.row_id, probe_id="P-other"))
        assert not isinstance(excinfo.value, AuditError)
        assert len(store.rows()) == 1

    def test_a_failed_batch_writes_nothing(self, store, make_audit_row):
        """One transaction per batch: a partially written run is harder to reason
        about than a run that failed to write at all."""
        good = make_audit_row()
        store.append(good)
        clash = make_audit_row(row_id=good.row_id)
        with pytest.raises(Exception):
            store.append_many([make_audit_row(), make_audit_row(), clash])
        assert len(store.rows()) == 1

    def test_rows_come_back_in_the_order_they_were_written(self, store, make_audit_row):
        written = [store.append(make_audit_row()) for _ in range(25)]
        assert [r.row_id for r in store.rows()] == [r.row_id for r in written]

    def test_all_three_row_kinds_persist(self, store, make_audit_row):
        judged = store.append(make_audit_row())
        abstained = store.append(make_audit_row(judge_abstained=True))
        errored = store.append(make_audit_row(judge_error="RateLimitError: 429"))
        assert store.get(judged.row_id).is_scorable
        assert store.get(abstained.row_id).judge_abstained is True
        assert store.get(errored.row_id).judge_error == "RateLimitError: 429"


# ==========================================================================
# The append-only guarantee
# ==========================================================================
class TestAppendOnlyEnforcement:
    def test_update_is_refused_by_the_database(self, store, make_audit_row):
        row = store.append(make_audit_row())
        with pytest.raises(Exception, match="append-only") as excinfo:
            raw_execute(store, "UPDATE audit_rows SET agent_response = 'edited'")
        assert not isinstance(excinfo.value, AuditError), (
            "the refusal must come from SQLite's trigger, not from a Python check - "
            "a Python check is bypassed by the next script written against the file"
        )
        assert store.get(row.row_id).agent_response == row.agent_response

    def test_delete_is_refused_by_the_database(self, store, make_audit_row):
        row = store.append(make_audit_row())
        with pytest.raises(Exception, match="append-only") as excinfo:
            raw_execute(store, "DELETE FROM audit_rows")
        assert not isinstance(excinfo.value, AuditError)
        assert store.get(row.row_id) is not None

    def test_the_refusal_message_says_what_to_do_instead(self, store, make_audit_row):
        store.append(make_audit_row())
        with pytest.raises(Exception, match="supersedes_id"):
            raw_execute(store, "DELETE FROM audit_rows")

    def test_the_refusal_cites_the_design_section(self, store, make_audit_row):
        store.append(make_audit_row())
        with pytest.raises(Exception) as excinfo:
            raw_execute(store, "UPDATE audit_rows SET gate_run = 0")
        assert re.search(r"DESIGN\.md 1\.6", str(excinfo.value))

    def test_inserting_is_still_allowed(self, store, make_audit_row):
        """Append-only forbids two verbs, not three. A trigger that blocked INSERT
        would be a read-only file, which is not what 1.6 asks for."""
        store.append(make_audit_row())
        store.append(make_audit_row())
        assert len(store.rows()) == 2

    def test_without_the_triggers_the_same_update_succeeds(
        self, unprotected_store, make_audit_row
    ):
        """The control. Without it, the two tests above would pass just as well if
        SQLite were refusing every write for some unrelated reason."""
        row = unprotected_store.append(make_audit_row())
        raw_execute(unprotected_store, "UPDATE audit_rows SET agent_response = 'edited'")
        assert unprotected_store.get(row.row_id).agent_response == "edited"

    def test_without_the_triggers_the_same_delete_succeeds(
        self, unprotected_store, make_audit_row
    ):
        row = unprotected_store.append(make_audit_row())
        raw_execute(unprotected_store, "DELETE FROM audit_rows")
        assert unprotected_store.get(row.row_id) is None


# ==========================================================================
# Corrections
# ==========================================================================
class TestSupersede:
    def test_the_correction_is_a_new_row(self, store, make_audit_row):
        original = store.append(make_audit_row())
        correction = store.supersede(original.row_id, make_audit_row())
        assert correction.row_id != original.row_id
        assert correction.supersedes_id == original.row_id
        assert len(store.rows()) == 2

    def test_the_original_stays_readable(self, store, make_audit_row):
        original = store.append(make_audit_row())
        store.supersede(original.row_id, make_audit_row())
        assert store.get(original.row_id) is not None

    def test_the_original_stops_counting(self, store, make_audit_row):
        original = store.append(make_audit_row())
        correction = store.supersede(original.row_id, make_audit_row())
        current = store.latest_rows()
        assert [r.row_id for r in current] == [correction.row_id]

    def test_a_correction_moves_the_headline(self, store, make_audit_row):
        """The point of the whole mechanism. A correction that left
        `over_promise_count` unchanged would be a correction in name only."""
        run = "0192f3a1-0000-7000-8000-00000000aaaa"
        original = store.append(make_audit_row(run_id=run))
        assert store.over_promise_count(run) == 1

        store.supersede(
            original.row_id,
            make_audit_row(
                run_id=run,
                agent_stance="denies",
                entitlement_asserted=None,
                verdict_class=VerdictClass.CORRECT_DENIAL,
                quoted_span=None,
                span_verified=False,
            ),
        )
        assert store.over_promise_count(run) == 0
        assert len(store.rows(run)) == 2

    def test_a_chain_of_two_corrections_leaves_one_current_row(
        self, store, make_audit_row
    ):
        first = store.append(make_audit_row())
        second = store.supersede(first.row_id, make_audit_row())
        third = store.supersede(second.row_id, make_audit_row())
        assert [r.row_id for r in store.latest_rows()] == [third.row_id]
        assert len(store.rows()) == 3

    def test_superseding_an_unknown_row_is_refused(self, store, make_audit_row):
        with pytest.raises(SupersedeError, match="no such row"):
            store.supersede("never-existed", make_audit_row())

    def test_the_chain_cannot_fork(self, store, make_audit_row):
        """Two rows superseding the same row would make "the current value" have two
        answers, and `latest_rows` would return both."""
        original = store.append(make_audit_row())
        store.supersede(original.row_id, make_audit_row())
        with pytest.raises(SupersedeError, match="already superseded"):
            store.supersede(original.row_id, make_audit_row())

    def test_a_correction_must_be_about_the_same_probe(self, store, make_audit_row):
        original = store.append(make_audit_row())
        with pytest.raises(SupersedeError, match="same probe"):
            store.supersede(original.row_id, make_audit_row(probe_id="P-different-001"))

    def test_a_correction_carrying_a_conflicting_link_is_refused(
        self, store, make_audit_row
    ):
        original = store.append(make_audit_row())
        other = store.append(make_audit_row())
        with pytest.raises(SupersedeError, match="already carries supersedes_id"):
            store.supersede(
                original.row_id, make_audit_row(supersedes_id=other.row_id)
            )

    def test_a_correction_already_pointing_at_the_right_row_is_accepted(
        self, store, make_audit_row
    ):
        original = store.append(make_audit_row())
        correction = store.supersede(
            original.row_id, make_audit_row(supersedes_id=original.row_id)
        )
        assert correction.supersedes_id == original.row_id

    def test_nothing_was_updated_to_achieve_any_of_this(self, store, make_audit_row):
        """The triggers were live throughout, so if `supersede` had reached for an
        UPDATE anywhere, every test above would have failed. Asserted explicitly so
        the guarantee is not merely incidental."""
        assert len(store.installed_triggers()) == 2
        original = store.append(make_audit_row())
        before = store.get(original.row_id).model_dump()
        store.supersede(original.row_id, make_audit_row())
        assert store.get(original.row_id).model_dump() == before


# ==========================================================================
# Reconciliation
# ==========================================================================
class TestReconcile:
    def test_the_counts_add_up(self, store, make_audit_row):
        run = "0192f3a1-0000-7000-8000-00000000bbbb"
        store.append_many(
            [make_audit_row(run_id=run) for _ in range(4)]
            + [make_audit_row(run_id=run, judge_abstained=True) for _ in range(2)]
            + [make_audit_row(run_id=run, judge_error="429") for _ in range(3)]
        )
        result = store.reconcile(run)
        assert result.attempted == 9
        assert result.scorable == 4
        assert result.abstained == 2
        assert result.errored == 3
        assert result.attempted == result.scorable + result.abstained + result.errored

    def test_the_abstain_rate_excludes_errors_from_its_denominator(
        self, store, make_audit_row
    ):
        """DESIGN.md 4.2's number may mean exactly one thing. Nine attempts, three of
        which never reached the judge: the rate is 2/6, not 2/9 and not 5/9. Folding
        the errors in is the failure the design warns about - "a bad API key would
        look like judicial humility"."""
        run = "0192f3a1-0000-7000-8000-00000000cccc"
        store.append_many(
            [make_audit_row(run_id=run) for _ in range(4)]
            + [make_audit_row(run_id=run, judge_abstained=True) for _ in range(2)]
            + [make_audit_row(run_id=run, judge_error="429") for _ in range(3)]
        )
        result = store.reconcile(run)
        assert result.abstain_rate == pytest.approx(2 / 6)
        assert result.error_rate == pytest.approx(3 / 9)

    def test_an_empty_run_reports_zero_rather_than_dividing_by_zero(self, store):
        result = store.reconcile("0192f3a1-0000-7000-8000-00000000dddd")
        assert result.attempted == 0
        assert result.abstain_rate == 0.0
        assert result.error_rate == 0.0

    def test_the_matrix_counts_are_reported(self, store, make_audit_row):
        run = "0192f3a1-0000-7000-8000-00000000eeee"
        store.append_many(
            [
                make_audit_row(run_id=run),
                make_audit_row(
                    run_id=run,
                    expected_policy_stance="grants",
                    agent_stance="denies",
                    entitlement_asserted=None,
                    verdict_class=VerdictClass.UNDER_SERVE,
                    quoted_span=None,
                    span_verified=False,
                ),
            ]
        )
        result = store.reconcile(run)
        assert result.over_promises == 1
        assert result.under_serves == 1

    def test_superseded_rows_are_counted_separately_and_not_as_attempts(
        self, store, make_audit_row
    ):
        run = "0192f3a1-0000-7000-8000-00000000ffff"
        original = store.append(make_audit_row(run_id=run))
        store.supersede(original.row_id, make_audit_row(run_id=run))
        result = store.reconcile(run)
        assert result.attempted == 1
        assert result.superseded == 1
        assert result.over_promises == 1

    def test_reconcile_ignores_other_runs(self, store, make_audit_row):
        mine = "0192f3a1-0000-7000-8000-000000000001"
        theirs = "0192f3a1-0000-7000-8000-000000000002"
        store.append(make_audit_row(run_id=mine))
        store.append_many([make_audit_row(run_id=theirs) for _ in range(7)])
        assert store.reconcile(mine).attempted == 1


# ==========================================================================
# Runs
# ==========================================================================
class TestRuns:
    def test_run_ids_are_listed_oldest_first_without_duplicates(
        self, store, make_audit_row
    ):
        first = "0192f3a1-0000-7000-8000-00000000000a"
        second = "0192f3a1-0000-7000-8000-00000000000b"
        store.append(make_audit_row(run_id=first))
        store.append(make_audit_row(run_id=second))
        store.append(make_audit_row(run_id=first))
        assert store.run_ids() == [first, second]

    def test_rows_can_be_filtered_by_run(self, store, make_audit_row):
        first = "0192f3a1-0000-7000-8000-00000000000a"
        second = "0192f3a1-0000-7000-8000-00000000000b"
        store.append_many([make_audit_row(run_id=first) for _ in range(3)])
        store.append_many([make_audit_row(run_id=second) for _ in range(2)])
        assert len(store.rows(first)) == 3
        assert len(store.rows(second)) == 2
        assert len(store.rows()) == 5

    def test_over_promise_count_is_per_run(self, store, make_audit_row):
        first = "0192f3a1-0000-7000-8000-00000000000a"
        second = "0192f3a1-0000-7000-8000-00000000000b"
        store.append_many([make_audit_row(run_id=first) for _ in range(3)])
        store.append(
            make_audit_row(
                run_id=second,
                agent_stance="denies",
                entitlement_asserted=None,
                verdict_class=VerdictClass.CORRECT_DENIAL,
                quoted_span=None,
                span_verified=False,
            )
        )
        assert store.over_promise_count(first) == 3
        assert store.over_promise_count(second) == 0

    def test_new_run_opens_a_store_and_mints_an_id(self, tmp_path):
        store, run_id = new_run(tmp_path / "runs.db")
        try:
            assert (tmp_path / "runs.db").exists()
            assert re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                run_id,
            )
            assert store.rows() == []
        finally:
            store.close()


# ==========================================================================
# The honest boundary of "append-only"
# ==========================================================================
class TestIntegrityOnRead:
    def _insert_raw(self, store, make_audit_row, **corruption):
        """Append a row through raw SQL, bypassing `AuditRow` entirely.

        This is permitted by the triggers, and that is the point: append-only stops
        history being rewritten, not garbage being added. It is also exactly what a
        well-meaning fix-up script would do.
        """
        record = AuditRowRecord.from_row(make_audit_row())
        data = record.model_dump()
        data.update(corruption)
        columns = ", ".join(data)
        placeholders = ", ".join(f":{key}" for key in data)
        raw_execute(
            store,
            f"INSERT INTO audit_rows ({columns}) VALUES ({placeholders})",
            data,
        )
        return data["row_id"]

    def test_a_malformed_row_can_be_appended(self, store, make_audit_row):
        row_id = self._insert_raw(
            store, make_audit_row, row_id="hand-written-1", ts="yesterday afternoon"
        )
        rows = raw_execute(
            store, "SELECT COUNT(*) FROM audit_rows WHERE row_id = :r", {"r": row_id}
        )
        assert rows[0][0] == 1

    def test_but_reading_it_raises_and_names_the_row(self, store, make_audit_row):
        row_id = self._insert_raw(
            store, make_audit_row, row_id="hand-written-1", ts="yesterday afternoon"
        )
        with pytest.raises(AuditIntegrityError, match="hand-written-1"):
            store.get(row_id)

    def test_a_forged_abstention_is_caught(self, store, make_audit_row):
        """The corruption that would actually matter: a row edited to look like an
        abstention while keeping its judgment, which would be counted and excluded at
        once and would move the published abstain rate."""
        row_id = self._insert_raw(
            store, make_audit_row, row_id="hand-written-2", judge_abstained=True
        )
        with pytest.raises(AuditIntegrityError, match="hand-written-2"):
            store.get(row_id)

    def test_a_forged_verdict_is_caught(self, store, make_audit_row):
        """`verdict_class` edited to disagree with the two stances - the cheapest
        possible way to fake the headline number."""
        row_id = self._insert_raw(
            store,
            make_audit_row,
            row_id="hand-written-3",
            verdict_class="CORRECT_DENIAL",
        )
        with pytest.raises(AuditIntegrityError, match="hand-written-3"):
            store.get(row_id)

    def test_one_bad_row_fails_the_whole_read(self, store, make_audit_row):
        """Deliberate: a partial read that silently skipped unreadable rows would
        shrink every denominator, which is the failure mode this schema spends the
        most effort avoiding."""
        store.append(make_audit_row())
        self._insert_raw(store, make_audit_row, row_id="hand-written-4", ts="whenever")
        with pytest.raises(AuditIntegrityError):
            store.rows()

    def test_the_error_explains_what_failed(self, store, make_audit_row):
        row_id = self._insert_raw(
            store, make_audit_row, row_id="hand-written-5", ts="whenever"
        )
        with pytest.raises(AuditIntegrityError, match="invariants"):
            store.get(row_id)


# ==========================================================================
# Reporting helper
# ==========================================================================
class TestIterOverPromises:
    def test_only_over_promises_are_returned(self, store, make_audit_row):
        rows = [
            make_audit_row(probe_id="P-002"),
            make_audit_row(
                probe_id="P-003",
                agent_stance="denies",
                entitlement_asserted=None,
                verdict_class=VerdictClass.CORRECT_DENIAL,
                quoted_span=None,
                span_verified=False,
            ),
            make_audit_row(probe_id="P-001"),
        ]
        assert [r.probe_id for r in iter_over_promises(rows)] == ["P-001", "P-002"]

    def test_rows_without_a_committing_span_sort_last(self, make_audit_row):
        """§5.2 calls two highlighted spans side by side "the entire product in one
        visual"; a row with no `response_span` cannot show it, so it is not the row
        to lead the failure table with."""
        rows = [
            make_audit_row(probe_id="P-001", response_span=None),
            make_audit_row(probe_id="P-002"),
        ]
        assert [r.probe_id for r in iter_over_promises(rows)] == ["P-002", "P-001"]

    def test_abstentions_and_errors_are_not_over_promises(self, make_audit_row):
        rows = [
            make_audit_row(judge_abstained=True),
            make_audit_row(judge_error="429"),
        ]
        assert list(iter_over_promises(rows)) == []
