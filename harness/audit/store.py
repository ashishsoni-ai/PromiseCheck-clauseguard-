"""Append-only `runs.db` store (DESIGN.md 1.6).

    | Stack | SQLite via SQLModel. Not Postgres. A single `runs.db` file that a
    | panelist can download and query is worth more than a database service.

    Append-only by convention: no UPDATE, no DELETE. Each run is immutable and
    identified by `run_id = uuid7()`; corrections are new rows with `supersedes_id`.

WHAT "APPEND-ONLY" IS WORTH, AND WHAT IT IS NOT
-----------------------------------------------
The design says "by convention". This module goes one step further and installs
two SQLite triggers that abort any UPDATE or DELETE on `audit_rows`. That is an
addition to the specification, so it is flagged rather than blended in, and it is
one keyword argument to switch off (`AuditStore(..., enforce_append_only=False)`).

The honest limit of that enforcement: **a trigger stops accidents, not tampering.**
Anyone holding the file can drop the trigger and rewrite history, and no scheme
that keeps the evidence and the guard in the same file can prevent that. What the
triggers buy is that a bug in our own code - a stray `session.merge`, a helpful
"fix this row" script written at 2am on demo day - fails loudly at the moment it
runs instead of silently changing a published number.

The second, weaker guard is on the read path: `AuditRowRecord.to_row()` re-runs
every invariant in `models.py`, so a row edited by hand has to survive all of them
to be read back without complaint. Between the two, an inconsistent database is
detectable. Neither makes it impossible.

CORRECTIONS
-----------
`supersede()` is the only sanctioned way to change what a run reports. It writes a
new row carrying `supersedes_id`, leaves the original exactly where it was, and
`latest_rows()` then hides the original from metrics while keeping it readable.
This is why the gate counts through `latest_rows()` and never through a raw
`SELECT`: a correction that did not move the headline would be a correction in
name only.

RECONCILIATION IS THE POINT OF PERSISTING FAILURES
--------------------------------------------------
`reconcile()` exists because of a specific hole found during Step 5. DESIGN.md 4.2
publishes the abstain rate, and a `JudgeError` used to produce no row at all - so a
flaky backend moved the denominator of a published number with nothing anywhere
recording that it had. `reconcile()` returns attempted, scorable, abstained and
errored counts that must sum, which makes "we lost 3 rows to a backend crash" a
fact on the report rather than a gap nobody measured.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select, text

from harness.audit.models import (
    AuditRow,
    AuditRowRecord,
    VerdictClass,
    new_run_id,
)

#: DESIGN.md 1.6's single downloadable file, and the default `CLAUSEGUARD_DB_PATH`
#: in .env.example.
DEFAULT_DB_PATH = "runs.db"

#: Installed by `AuditStore.initialise()` unless enforcement is switched off. Named
#: so they can be checked for by the tests rather than matched on message text.
_NO_UPDATE_TRIGGER = "audit_rows_is_append_only_no_update"
_NO_DELETE_TRIGGER = "audit_rows_is_append_only_no_delete"

_TRIGGER_SQL = {
    _NO_UPDATE_TRIGGER: (
        f"CREATE TRIGGER IF NOT EXISTS {_NO_UPDATE_TRIGGER} "
        "BEFORE UPDATE ON audit_rows BEGIN "
        "SELECT RAISE(ABORT, 'audit_rows is append-only (DESIGN.md 1.6): "
        "no UPDATE. Corrections are new rows with supersedes_id.'); END"
    ),
    _NO_DELETE_TRIGGER: (
        f"CREATE TRIGGER IF NOT EXISTS {_NO_DELETE_TRIGGER} "
        "BEFORE DELETE ON audit_rows BEGIN "
        "SELECT RAISE(ABORT, 'audit_rows is append-only (DESIGN.md 1.6): "
        "no DELETE. Corrections are new rows with supersedes_id.'); END"
    ),
}


class AuditError(RuntimeError):
    """Base class for audit-store failures.

    Subclasses `RuntimeError` for the same reason `JudgeError` does: call sites
    that write `except Exception` must catch it, because the alternative is a
    harness that swallows a storage failure and reports a clean run.
    """


class AuditIntegrityError(AuditError):
    """A stored row does not satisfy the invariants it was written under.

    Raised on the read path. The row is named, because the useful next action is
    to look at it - and because "the database is corrupt" without a row id is an
    unactionable thing to print at a judging panel.
    """


class SupersedeError(AuditError):
    """A correction does not describe a legal change to the record."""


@dataclass(frozen=True)
class Reconciliation:
    """Counts that must add up, for one run.

    `attempted == scorable + abstained + errored` is checked by `assert_consistent`
    rather than assumed. If that identity ever fails, some row is in a state the
    validators were supposed to make unreachable, and finding out here is much
    cheaper than finding out from a metric that looks plausible.
    """

    run_id: str
    attempted: int
    scorable: int
    abstained: int
    errored: int
    superseded: int
    over_promises: int
    under_serves: int

    @property
    def abstain_rate(self) -> float:
        """Abstentions over rows the judge actually returned an opinion path for.

        Errors are excluded from the denominator, not folded into the numerator.
        An abstention is a judgment the harness rejected; an error is a judgment
        that never happened. DESIGN.md 4.2 publishes the first, and mixing the
        second into it is the exact failure the design warns about - "a bad API
        key would look like judicial humility".
        """
        denominator = self.scorable + self.abstained
        return self.abstained / denominator if denominator else 0.0

    @property
    def error_rate(self) -> float:
        """Errors over everything attempted. Reported beside the abstain rate, not
        inside it."""
        return self.errored / self.attempted if self.attempted else 0.0

    def assert_consistent(self) -> None:
        if self.attempted != self.scorable + self.abstained + self.errored:
            raise AuditIntegrityError(
                f"run {self.run_id}: attempted={self.attempted} but "
                f"scorable={self.scorable} + abstained={self.abstained} + "
                f"errored={self.errored} = "
                f"{self.scorable + self.abstained + self.errored}. Some row is "
                f"simultaneously two of judged/abstained/errored, which the row "
                f"validators are supposed to make impossible"
            )


class AuditStore:
    """Append-only access to one `runs.db`.

    Usable as a context manager. `initialise()` is idempotent, so opening an
    existing database is the same call as creating a new one - which matters
    because the demo path and the test path must not diverge.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        *,
        enforce_append_only: bool = True,
        echo: bool = False,
    ) -> None:
        """Open a store.

        `enforce_append_only` controls whether the triggers are *installed*, not
        whether they are enforced. Triggers live in the database file, so opening an
        already-protected `runs.db` with this set to False leaves the protection in
        place; nothing here drops a trigger, because a flag that quietly disarmed an
        existing audit file would be a worse hazard than the convenience is worth.
        """
        self.db_path = Path(db_path)
        self.enforce_append_only = enforce_append_only
        # `sqlite://` + an absolute path. Resolved rather than passed through so
        # that the URL does not depend on the process's working directory; the CLI
        # and the tests run from different ones.
        url = f"sqlite:///{self.db_path.resolve()}"
        self._engine = create_engine(url, echo=echo)
        self._initialised = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialise(self) -> AuditStore:
        """Create the table and, unless disabled, the append-only triggers."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        SQLModel.metadata.create_all(self._engine)
        if self.enforce_append_only:
            with self._engine.begin() as connection:
                for statement in _TRIGGER_SQL.values():
                    connection.execute(text(statement))
        self._initialised = True
        return self

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self) -> AuditStore:
        return self.initialise()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def engine(self):  # noqa: ANN201 - see docstring
        """The underlying engine, for tests that need raw SQL.

        Deliberately unannotated: importing `sqlalchemy.Engine` purely for a
        return type would make this module depend on SQLAlchemy's public surface
        when everything else here goes through SQLModel. The one place that needs
        the type is a test, and a test can read the docstring.
        """
        return self._engine

    def installed_triggers(self) -> list[str]:
        """Trigger names present on `audit_rows`.

        Exposed so the tests can assert enforcement is really installed instead of
        inferring it from an error message. A guard verified only by the text of
        its own complaint is a guard that can be satisfied by a typo.
        """
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                    "AND tbl_name = 'audit_rows' ORDER BY name"
                )
            ).all()
        return [row[0] for row in rows]

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def append(self, row: AuditRow) -> AuditRow:
        """Write one row. Returns it unchanged, for call-site chaining."""
        self.append_many([row])
        return row

    def append_many(self, rows: Iterable[AuditRow]) -> int:
        """Write rows in one transaction, and return how many were written.

        One transaction per batch rather than per row: a partially written run is
        harder to reason about than a run that failed to write at all, and the
        reconciliation identity above assumes the rows for a run arrived together.
        """
        self._require_initialised()
        batch = list(rows)
        if not batch:
            return 0

        seen: set[str] = set()
        for row in batch:
            if row.row_id in seen:
                raise AuditError(
                    f"row_id {row.row_id!r} appears twice in the same batch; "
                    f"row ids are unique per row, not per probe"
                )
            seen.add(row.row_id)

        with Session(self._engine) as session:
            for row in batch:
                session.add(AuditRowRecord.from_row(row))
            session.commit()
        return len(batch)

    def supersede(self, superseded_row_id: str, correction: AuditRow) -> AuditRow:
        """Record a correction as a new row (DESIGN.md 1.6).

        Refuses three things, each because permitting it would make the chain
        unreadable rather than merely untidy:

          - superseding a row that is not in the store, which would leave a
            correction pointing at nothing;
          - superseding a row that some other row already supersedes, which would
            fork the chain so that "the current value" has two answers;
          - a correction whose `probe_id` differs from the row it corrects, which
            is not a correction but a different result wearing one's clothes.
        """
        self._require_initialised()
        original = self.get(superseded_row_id)
        if original is None:
            raise SupersedeError(
                f"cannot supersede {superseded_row_id!r}: no such row. A "
                f"correction that points at nothing is not append-only, it is lost"
            )
        existing = self._superseding_row_id(superseded_row_id)
        if existing is not None:
            raise SupersedeError(
                f"row {superseded_row_id!r} is already superseded by "
                f"{existing!r}; superseding it again would fork the chain and "
                f"'the current value' would have two answers. Supersede "
                f"{existing!r} instead"
            )
        if correction.probe_id != original.probe_id:
            raise SupersedeError(
                f"correction is for probe {correction.probe_id!r} but "
                f"{superseded_row_id!r} records probe {original.probe_id!r}; a "
                f"correction must be about the same probe"
            )
        if correction.supersedes_id not in (None, superseded_row_id):
            raise SupersedeError(
                f"correction already carries supersedes_id "
                f"{correction.supersedes_id!r}, which is not {superseded_row_id!r}"
            )

        corrected = correction.model_copy(update={"supersedes_id": superseded_row_id})
        self.append(corrected)
        return corrected

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------
    def get(self, row_id: str) -> AuditRow | None:
        self._require_initialised()
        with Session(self._engine) as session:
            record = session.get(AuditRowRecord, row_id)
            return self._decode(record) if record is not None else None

    def rows(self, run_id: str | None = None) -> list[AuditRow]:
        """Every row, superseded ones included, in insertion order.

        Ordered by `row_id` because it is a uuid7 and therefore time-ordered - no
        separate sequence column, and no dependence on SQLite's rowid, which is an
        implementation detail rather than a promise.
        """
        self._require_initialised()
        statement = select(AuditRowRecord)
        if run_id is not None:
            statement = statement.where(AuditRowRecord.run_id == run_id)
        with Session(self._engine) as session:
            records = session.exec(statement.order_by(AuditRowRecord.row_id)).all()
        return [self._decode(record) for record in records]

    def latest_rows(self, run_id: str | None = None) -> list[AuditRow]:
        """Rows that nothing supersedes - the current state of the record.

        Every metric and the gate read through this. A correction that left the
        headline unchanged would be a correction in name only.
        """
        all_rows = self.rows(run_id)
        superseded = {r.supersedes_id for r in all_rows if r.supersedes_id is not None}
        return [row for row in all_rows if row.row_id not in superseded]

    def run_ids(self) -> list[str]:
        """Distinct run ids, oldest first (uuid7 sorts chronologically)."""
        self._require_initialised()
        with Session(self._engine) as session:
            records = session.exec(
                select(AuditRowRecord.run_id).order_by(AuditRowRecord.row_id)
            ).all()
        seen: list[str] = []
        for value in records:
            if value not in seen:
                seen.append(value)
        return seen

    def reconcile(self, run_id: str) -> Reconciliation:
        """Counts for one run, with the identity checked rather than assumed."""
        all_rows = self.rows(run_id)
        current = self.latest_rows(run_id)
        result = Reconciliation(
            run_id=run_id,
            attempted=len(current),
            scorable=sum(1 for r in current if r.is_scorable),
            abstained=sum(1 for r in current if r.judge_abstained),
            errored=sum(1 for r in current if r.judge_error is not None),
            superseded=len(all_rows) - len(current),
            over_promises=sum(1 for r in current if r.is_over_promise),
            under_serves=sum(
                1 for r in current if r.verdict_class is VerdictClass.UNDER_SERVE
            ),
        )
        result.assert_consistent()
        return result

    def over_promise_count(self, run_id: str) -> int:
        """What DESIGN.md 2 step 11 compares against `--max-overpromise`."""
        return sum(1 for row in self.latest_rows(run_id) if row.is_over_promise)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_initialised(self) -> None:
        if not self._initialised:
            raise AuditError(
                "AuditStore was used before initialise(); call initialise() or "
                "use the store as a context manager. Creating the table lazily "
                "on first write would mean a read against a fresh database "
                "silently returned zero rows instead of saying it was not set up"
            )

    def _superseding_row_id(self, row_id: str) -> str | None:
        self._require_initialised()
        with Session(self._engine) as session:
            found = session.exec(
                select(AuditRowRecord.row_id).where(
                    AuditRowRecord.supersedes_id == row_id
                )
            ).first()
        return found

    @staticmethod
    def _decode(record: AuditRowRecord) -> AuditRow:
        """Validate on the way out, and name the row if it fails."""
        try:
            return record.to_row()
        except Exception as exc:  # noqa: BLE001 - re-raised with the row id
            raise AuditIntegrityError(
                f"row {record.row_id!r} in the store does not satisfy the audit "
                f"row invariants: {exc}"
            ) from exc


def new_run(db_path: str | Path = DEFAULT_DB_PATH) -> tuple[AuditStore, str]:
    """Open (or create) a store and mint a `run_id` for a fresh run.

    Convenience for the CLI, so that `run_id = uuid7()` happens in exactly one
    place and no call site invents its own scheme.
    """
    store = AuditStore(db_path).initialise()
    return store, new_run_id()


def iter_over_promises(rows: Sequence[AuditRow]) -> Iterator[AuditRow]:
    """The failure table of DESIGN.md 5.2 item 4, in severity order.

    Sorted so that rows whose committing span is known come first: two highlighted
    spans side by side is described there as "the entire product in one visual",
    and a row missing `response_span` cannot show it.
    """
    over = [row for row in rows if row.is_over_promise]
    over.sort(key=lambda r: (r.response_span is None, r.probe_id))
    return iter(over)
