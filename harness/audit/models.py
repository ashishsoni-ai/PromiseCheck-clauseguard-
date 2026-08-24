"""Append-only audit row schema (DESIGN.md 5.1).

One row per probe *attempted* (DESIGN.md 2 step 10), immutable once written. §1.6
fixes the stack as SQLite via SQLModel, because "a single `runs.db` file that a
panelist can download and query is worth more than a database service".

TWO MODELS, ON PURPOSE
----------------------
`AuditRow` is a validated pydantic model - the thing the harness constructs and
passes around. `AuditRowRecord` is the SQLModel table it is persisted as.

The split is not architecture for its own sake. **SQLModel does not run pydantic
validation on `table=True` models**, so every invariant below would silently not
fire if the table class were also the domain class. The invariants are the whole
point of this file: one of them is the difference between an abstention and a
transport failure, which DESIGN.md 4.2 publishes as a headline number. A schema
whose checks quietly do not run is worse than a schema with no checks, because it
reads as though it is guarding something.

The cost of the split is a duplicated field list, and duplication rots. So it is
pinned by `tests/unit/test_audit_models.py::TestTheTwoModelsCannotDriftApart`,
which asserts the two field sets correspond exactly. Adding a field to one and
not the other is a test failure, not a mystery at write time.

WHY THE TABLE HAS ONLY PRIMITIVE COLUMNS
----------------------------------------
`AuditRowRecord` stores `clause_ids`, `probe_turns` and `scenario_facts` as JSON
*text*, and `ts` as an ISO-8601 string. No `sa_column=Column(JSON)`, no
`sa.Enum`, no `datetime` columns. Three reasons, in descending order of how much
trouble they save:

  - SQLite has no timezone. A `datetime` column takes an aware datetime and hands
    back a naive one, so a round-trip is not an identity and any later comparison
    raises `TypeError`. An ISO-8601 string with a `Z` suffix round-trips exactly,
    and because such strings sort lexicographically in chronological order, range
    queries still work.
  - DESIGN.md 5.1 specifies the row as JSON with `"ts": "2026-08-28T11:04:22.118Z"`.
    Storing that literal string means the database and the specification agree
    character for character.
  - `sqlite3` on any machine, and any panelist's SQL browser, can read it without
    knowing anything about SQLAlchemy type decorators. §1.6's whole argument for
    SQLite is that the artefact is inspectable.

FIELDS BEYOND DESIGN.md 5.1 - FLAGGED FOR REVIEW
------------------------------------------------
§5.1 lists 33 fields. Five more are here, for 38, and each is an addition rather
than an interpretation, so each is named rather than blended in:

  `row_id`             implied rather than added: §5.1's `supersedes_id` has to
                       point at something, and no other field is unique per row.
  `judge_error`        §5.1 has `judge_abstained` but nothing for "the judge never
                       returned". Without this column a `JudgeError` is an absent
                       row, which moves the denominator of the published abstain
                       rate where nobody can see it. See the invariant below.
  `agent_backend`      `LLM_BACKEND` selects ollama or groq and changes the
                       agent's behaviour without changing `agent_commit_sha`, so
                       two rows with identical SHAs can legitimately differ.
                       Recording it is what keeps commitment C3 falsifiable.
  `judge_temperature`  §4.1 mandates 0.0. Measurement on 2026-08-23 showed
  `judge_completions`  byte-identical input at 0.0 still producing different
                       generations, and `grants` judgments needing a second
                       completion 57% of the time against 0% for `denies`. Both
                       facts are only checkable per row if the row says what it
                       actually used. See docs/limitations.md, third entry.

`git_sha` is read as the *harness* commit, not the SHA of the repository being
gated; the agent's is already carried by `agent_commit_sha`.

Those counts are not decoration and not taken on trust:
`test_audit_models.py::TestTheRowMatchesTheSpecification` parses the JSON block out
of DESIGN.md 5.1 and asserts that the difference between it and this model is
exactly those five names. Adding a 39th field without deciding whether it belongs
in the specification is therefore a test failure.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel
from uuid6 import uuid7

from harness.schemas import AgentStance, PolicyStance

#: The literal values, extracted once so the validators and the tests agree with
#: the schema module rather than restating it. A restated vocabulary is a
#: vocabulary that drifts.
AGENT_STANCES: Final[frozenset[str]] = frozenset(get_args(AgentStance))
POLICY_STANCES: Final[frozenset[str]] = frozenset(get_args(PolicyStance))


class VerdictClass(str, Enum):
    """The cells of DESIGN.md 2 step 9's 2x3 matrix.

    §2 names two of the six - OVER_PROMISE and UNDER_SERVE - and leaves the rest
    unnamed. Naming all six rather than collapsing to §5.2's four-way display
    (`UNDER-SERVE / EVASIVE / JUDGE ABSTAINED` beside the headline) means the
    matrix is reconstructible from this column alone; the dashboard aggregates for
    presentation, which is a display concern and reversible. Collapsing at write
    time is not reversible, and it would throw away the difference between evading
    a question the policy grants and evading one it denies - which are different
    merchant-side costs.

    Member names equal their values so that nothing depends on whether a given
    serialiser writes `.name` or `.value`.
    """

    #: (policy=denies, agent=grants). The consequential class: the agent promised
    #: something the policy refuses. §2's "cell that matters".
    OVER_PROMISE = "OVER_PROMISE"

    #: (policy=grants, agent=denies). §2's mirror - "the merchant-CX cost", and
    #: what keeps the tool honest about false-positive framing.
    UNDER_SERVE = "UNDER_SERVE"

    #: (policy=grants, agent=evasive). The customer was entitled and did not
    #: find out.
    EVASIVE_ON_GRANT = "EVASIVE_ON_GRANT"

    #: (policy=denies, agent=evasive). Cheaper than over-promising, and not a
    #: correct answer either.
    EVASIVE_ON_DENIAL = "EVASIVE_ON_DENIAL"

    #: (policy=grants, agent=grants).
    CORRECT_GRANT = "CORRECT_GRANT"

    #: (policy=denies, agent=denies).
    CORRECT_DENIAL = "CORRECT_DENIAL"


#: The matrix itself. Exhaustive over policy x agent by construction, and asserted
#: to be exhaustive in the tests: a missing cell would make `classify_verdict`
#: raise on a legitimate pair, and it would do so only on the rarest combination,
#: which is the worst possible time to find out.
_MATRIX: Final[dict[tuple[str, str], VerdictClass]] = {
    ("denies", "grants"): VerdictClass.OVER_PROMISE,
    ("grants", "denies"): VerdictClass.UNDER_SERVE,
    ("grants", "evasive"): VerdictClass.EVASIVE_ON_GRANT,
    ("denies", "evasive"): VerdictClass.EVASIVE_ON_DENIAL,
    ("grants", "grants"): VerdictClass.CORRECT_GRANT,
    ("denies", "denies"): VerdictClass.CORRECT_DENIAL,
}


def classify_verdict(policy_stance: str, agent_stance: str) -> VerdictClass:
    """Assemble the verdict for one (policy, agent) pair - DESIGN.md 2 step 9.

    Deliberately a pure function of two strings, with no access to the row, the
    judge, or the probe. `verdict_class` is the field a demo is read off, so it
    must not be possible to write a value that the two stances do not imply. The
    row validator below re-derives it and rejects any mismatch, which makes a bug
    in verdict assembly a write-time failure rather than a wrong headline.
    """
    try:
        return _MATRIX[(policy_stance, agent_stance)]
    except KeyError:
        raise ValueError(
            f"no verdict cell for policy_stance={policy_stance!r}, "
            f"agent_stance={agent_stance!r}; expected policy in "
            f"{sorted(POLICY_STANCES)} and agent in {sorted(AGENT_STANCES)}"
        ) from None


def new_row_id() -> str:
    """A fresh row identifier.

    uuid7 rather than uuid4 for the same reason §1.6 chose it for `run_id`: it is
    time-ordered, so rows sort into insertion order without a separate sequence
    column, and an append-only table that cannot be read back in the order it was
    written is missing half of what append-only buys.

    The strength of that ordering, checked against uuid6 2025.0.1 rather than
    assumed: `uuid7()` holds a module-global last-timestamp and increments the
    millisecond field when the clock has not advanced, so ids are *strictly*
    increasing - but only **within one process**, since nothing is shared between
    them. A single clauseguard run writes from one process, so ordering holds for a
    `run_id`; two runs started in the same millisecond by different processes could
    interleave. Nothing here depends on cross-process ordering, and `ts` is the
    field to use if something ever does.
    """
    return str(uuid7())


def new_run_id() -> str:
    """A fresh run identifier - `run_id = uuid7()`, DESIGN.md 1.6 verbatim."""
    return str(uuid7())


def utc_now_iso() -> str:
    """`ts` in the exact shape DESIGN.md 5.1 prints: milliseconds, `Z` suffix.

    `datetime.isoformat()` alone gives `+00:00`, and `timespec="milliseconds"`
    alone still gives the offset form, so both are handled explicitly rather than
    hoping the default matches the spec.
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _parse_iso_z(value: str) -> datetime:
    """Parse the `Z` form. Python's `fromisoformat` only learned `Z` in 3.11, and
    the project requires 3.11+, but being explicit here costs one line and means
    the failure mode is a clear ValueError rather than a version-dependent one."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class AuditRow(BaseModel):
    """One immutable audit row - the validated, in-memory form.

    Field order follows DESIGN.md 5.1's JSON block so the two can be read side by
    side.
    """

    model_config = ConfigDict(extra="forbid")

    # --- identity -----------------------------------------------------------
    row_id: str = Field(default_factory=new_row_id, min_length=1)
    run_id: str = Field(min_length=1)
    probe_id: str = Field(min_length=1)
    ts: str = Field(default_factory=utc_now_iso)

    # --- policy provenance --------------------------------------------------
    policy_doc: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    clause_ids: list[str] = Field(min_length=1)
    rule_id: str = Field(min_length=1)
    rule_version: str = Field(min_length=1)

    # --- probe --------------------------------------------------------------
    strategy: str = Field(min_length=1)
    difficulty_tier: int = Field(ge=1, le=3)
    scenario_facts: dict[str, Any] = Field(min_length=1)
    probe_turns: list[str] = Field(min_length=1, max_length=3)
    expected_policy_stance: PolicyStance

    # --- agent under test ---------------------------------------------------
    agent_id: str = Field(min_length=1)
    agent_model: str = Field(min_length=1)
    agent_commit_sha: str = Field(min_length=1)
    agent_response: str
    agent_latency_ms: int = Field(ge=0)
    agent_backend: str | None = None

    # --- the judgment, absent when the judge abstained or failed ------------
    agent_stance: AgentStance | None = None
    entitlement_asserted: str | None = None
    verdict_class: VerdictClass | None = None
    cited_clause_id: str | None = None
    quoted_span: str | None = None
    response_span: str | None = None
    span_verified: bool | None = None

    # --- judge provenance ---------------------------------------------------
    judge_model: str = Field(min_length=1)
    # `ge=0`, not `ge=1`. Zero means no model was sampled: DESIGN.md 4.1's L0
    # pre-filter terminated the ladder deterministically, or the call failed before
    # a completion came back. Both are real rows, and leaving the default at 1
    # would make each of them claim a sample it never took. 5.1 is silent - its
    # only example is `judge_k: 3` - so this is a decision, recorded here and in
    # `_zero_samples_means_no_model_ran` below.
    judge_k: int = Field(default=1, ge=0)
    judge_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    judge_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    judge_abstained: bool = False
    judge_temperature: float | None = Field(default=None, ge=0.0)
    judge_completions: int | None = Field(default=None, ge=1)
    judge_error: str | None = None

    # --- run context --------------------------------------------------------
    gate_run: bool = False
    git_sha: str = Field(min_length=1)
    supersedes_id: str | None = None

    # ----------------------------------------------------------------------
    # Invariants
    # ----------------------------------------------------------------------
    @model_validator(mode="after")
    def _an_error_is_not_an_abstention(self) -> AuditRow:
        """THE LOAD-BEARING INVARIANT IN THIS FILE.

        DESIGN.md 4.2 publishes the abstain rate, and the rule the judge was built
        to - `harness/judge/judge.py` - is that an abstention may mean exactly one
        thing: a judgment whose span was rejected twice by L2. Transport failure,
        an unparseable reply and missing candidate clauses all raise `JudgeError`
        instead, precisely so they cannot inflate that number. The design's own
        warning is that "a bad API key would look like judicial humility"; it has
        now failed loudly three times for three unrelated causes (a decommissioned
        model returning 404, a local CUDA crash, and a Groq rate limit), and each
        time the separation held.

        This validator is where that separation stops being a convention and
        becomes a constraint on the stored data. A row may be an error or an
        abstention or a judgment - never two of those at once.
        """
        judgment_fields = {
            "agent_stance": self.agent_stance,
            "entitlement_asserted": self.entitlement_asserted,
            "verdict_class": self.verdict_class,
            "cited_clause_id": self.cited_clause_id,
            "quoted_span": self.quoted_span,
            "response_span": self.response_span,
        }
        present = sorted(k for k, v in judgment_fields.items() if v is not None)

        if self.judge_error is not None:
            if self.judge_abstained:
                raise ValueError(
                    "judge_error is set and judge_abstained is True; an error is "
                    "not an abstention, and conflating them corrupts the abstain "
                    "rate DESIGN.md 4.2 publishes"
                )
            if present:
                raise ValueError(
                    f"judge_error is set but the row also carries a judgment "
                    f"({', '.join(present)}); a call that failed produced no "
                    f"judgment to record"
                )
            if self.span_verified:
                raise ValueError(
                    "judge_error is set but span_verified is True; L2 cannot have "
                    "verified a span that was never returned"
                )
            return self

        if self.judge_abstained:
            if present:
                raise ValueError(
                    f"judge_abstained is True but the row also carries a judgment "
                    f"({', '.join(present)}); an abstention is the absence of one. "
                    f"DESIGN.md 4.1 excludes abstentions from headline metrics, so "
                    f"a row that is both would be counted and excluded at once"
                )
            if self.span_verified:
                raise ValueError(
                    "judge_abstained is True but span_verified is True; an "
                    "abstention is what happens when L2 rejected the span twice"
                )
            return self

        # Neither errored nor abstained: this row is a completed judgment and must
        # actually contain one.
        if self.agent_stance is None:
            raise ValueError(
                "row is neither an error nor an abstention, so it must carry "
                "agent_stance; a row with no stance and no reason for having none "
                "is a lost result masquerading as a clean one"
            )
        if self.verdict_class is None:
            raise ValueError(
                "row carries agent_stance but no verdict_class; DESIGN.md 2 step 9 "
                "assembles one for every (policy, agent) pair"
            )
        # Keyed on `quoted_span` and not on the row being a judgment at all. An L0
        # row is a completed judgment - it carries a stance and a verdict_class -
        # that quoted nothing, because no model ran to quote anything, and the same
        # is true of a judged `denies` that cited no clause. There is then no check
        # outcome to record: True would claim a substring match that never ran, and
        # False would claim one that ran and failed. Requiring a boolean here made
        # every L0 row unwritable, which is roughly 30% of a real run (DESIGN.md
        # 4.1), and the runner surfaced it only because `judge.py` had documented
        # None as the honest value for that case since Step 5.
        if self.quoted_span is not None and self.span_verified is None:
            raise ValueError(
                "row quotes a clause but span_verified is None; commitment C2 is "
                "the claim that every quote was mechanically checked, and None "
                "records that the outcome of the check is unknown"
            )
        return self

    @model_validator(mode="after")
    def _verdict_class_must_follow_from_the_two_stances(self) -> AuditRow:
        """`verdict_class` is re-derived and compared, never trusted.

        This is the field a panelist reads the headline off. If verdict assembly
        ever writes the wrong cell - an off-by-one in a lookup, a swapped argument
        pair - the number moves and nothing else in the system notices. Re-deriving
        it here costs one dict lookup per row and converts that class of bug into a
        write-time failure.
        """
        if self.agent_stance is None or self.verdict_class is None:
            return self
        expected = classify_verdict(self.expected_policy_stance, self.agent_stance)
        if self.verdict_class is not expected:
            raise ValueError(
                f"verdict_class is {self.verdict_class.value} but "
                f"(policy={self.expected_policy_stance}, "
                f"agent={self.agent_stance}) implies {expected.value}"
            )
        return self

    @model_validator(mode="after")
    def _a_quote_requires_something_to_have_quoted_from(self) -> AuditRow:
        """Mirrors `Judgment`'s own C2 check, deliberately rather than by import.

        The judge model rejects `quoted_span` without `cited_clause_id` because an
        unverifiable quote is the hole C2 exists to close. The audit row is written
        by a different code path - verdict assembly, not the judge - so it repeats
        the check rather than assuming the value arrived through the validated one.
        """
        if self.quoted_span is not None and self.cited_clause_id is None:
            raise ValueError(
                "quoted_span was provided without cited_clause_id; there is no "
                "clause to verify the quote against, so C2's substring check "
                "could never have run on this row"
            )
        if self.span_verified and self.quoted_span is None:
            raise ValueError(
                "span_verified is True but quoted_span is None; there was no span "
                "to verify. A `denies` judgment legitimately carries no span, and "
                "span_verified=False is the honest record of that"
            )
        return self

    @model_validator(mode="after")
    def _a_grant_must_name_what_it_granted(self) -> AuditRow:
        """Same rule as `Judgment`'s, for the same reason: an over-promise that
        names no entitlement cannot be reported to a merchant."""
        if self.agent_stance == "grants" and not self.entitlement_asserted:
            raise ValueError(
                "agent_stance is 'grants' but entitlement_asserted is empty; a "
                "grant must name the entitlement it granted"
            )
        return self

    @model_validator(mode="after")
    def _ts_must_be_the_shape_the_spec_prints(self) -> AuditRow:
        """A timestamp column that mixes formats cannot be range-queried, and the
        lexicographic ordering this schema relies on quietly stops holding."""
        if not self.ts.endswith("Z"):
            raise ValueError(
                f"ts must be UTC with a 'Z' suffix as in DESIGN.md 5.1 "
                f"(e.g. 2026-08-28T11:04:22.118Z); got {self.ts!r}"
            )
        try:
            parsed = _parse_iso_z(self.ts)
        except ValueError:
            raise ValueError(f"ts is not an ISO-8601 timestamp: {self.ts!r}") from None
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError(f"ts must be UTC; got offset {parsed.utcoffset()!r}")
        return self

    @model_validator(mode="after")
    def _agreement_belongs_to_a_vote(self) -> AuditRow:
        """`judge_agreement` at k=1 is not a measurement of anything.

        DESIGN.md 4.1 applies k=3 only to the over-promise cell and the gold set,
        so most rows are k=1 and an agreement of 1.0 on them would look like
        unanimity where there was only one vote. That is the same failure as
        scoring L0's designed escalations as wrong answers: a number that cannot
        come out any other way, published as though it could.
        """
        if self.judge_k <= 1 and self.judge_agreement is not None:
            votes = "no votes" if self.judge_k == 0 else "a single vote"
            raise ValueError(
                f"judge_agreement is set but judge_k is {self.judge_k}; {votes} has "
                f"no agreement, and storing 1.0 here would read as unanimity"
            )
        return self

    @model_validator(mode="after")
    def _zero_samples_means_no_model_ran(self) -> AuditRow:
        """`judge_k == 0` may not carry anything only a completion could produce.

        Zero samples is how an L0 termination and a failed call are recorded.
        Neither observed a completion, so `judge_confidence` and
        `judge_completions` must be absent on both - and `judge_confidence` is the
        one that would actually mislead, because DESIGN.md 4.2 reports it.

        `judge_temperature` is treated differently on purpose, and the difference
        is not cosmetic. Temperature describes the *request*, not the reply, so it
        is a fact about a call that was made and failed - which is exactly what a
        row with `judge_error` records. An L0 row made no call at all, so a
        temperature on one would be the default masquerading as provenance.
        `judge_error` is what distinguishes the two, since L0 never errors.
        """
        if self.judge_k != 0:
            return self
        forbidden = ["judge_confidence", "judge_completions"]
        if self.judge_error is None:
            forbidden.append("judge_temperature")
        for name in forbidden:
            if getattr(self, name) is not None:
                raise ValueError(
                    f"{name} is set but judge_k is 0, which means no completion "
                    f"came back (L0 terminated, or the call failed). Only a "
                    f"completed judgment could have produced that value"
                )
        return self

    # ----------------------------------------------------------------------
    # Read helpers
    # ----------------------------------------------------------------------
    @property
    def is_over_promise(self) -> bool:
        """The one number DESIGN.md 5.2 puts in huge type."""
        return self.verdict_class is VerdictClass.OVER_PROMISE

    @property
    def is_scorable(self) -> bool:
        """True when this row may enter a headline metric.

        Errors and abstentions are both excluded, for different reasons - §4.1
        excludes abstentions by design, and an error never produced a judgment to
        score - so callers that need a denominator should use this rather than
        writing `judge_error is None` and forgetting the other half.
        """
        return self.judge_error is None and not self.judge_abstained

    @property
    def matrix_cell(self) -> tuple[str, str] | None:
        """(policy_stance, agent_stance), or None for a row with no judgment."""
        if self.agent_stance is None:
            return None
        return (self.expected_policy_stance, self.agent_stance)


#: Columns holding JSON text rather than a native type. Named once so the encoder,
#: the decoder and the drift test all read from the same list.
JSON_COLUMNS: Final[tuple[str, ...]] = ("clause_ids", "probe_turns", "scenario_facts")


class AuditRowRecord(SQLModel, table=True):
    """The SQLite persistence shape of `AuditRow`.

    `__tablename__` is explicit because SQLModel's default is the lowercased class
    name, and a table called `auditrowrecord` is an unkind thing to hand a
    panelist with a SQL browser.

    NOTE: this class performs no validation - SQLModel skips it for table models.
    Construct it only via `from_row`, which takes an already-validated `AuditRow`.
    """

    __tablename__ = "audit_rows"

    row_id: str = SQLField(primary_key=True)
    run_id: str = SQLField(index=True)
    probe_id: str = SQLField(index=True)
    ts: str

    policy_doc: str = SQLField(index=True)
    policy_version: str
    clause_ids: str
    rule_id: str
    rule_version: str

    strategy: str = SQLField(index=True)
    difficulty_tier: int
    scenario_facts: str
    probe_turns: str
    expected_policy_stance: str

    agent_id: str = SQLField(index=True)
    agent_model: str
    agent_commit_sha: str
    agent_response: str
    agent_latency_ms: int
    agent_backend: str | None = None

    agent_stance: str | None = None
    entitlement_asserted: str | None = None
    verdict_class: str | None = SQLField(default=None, index=True)
    cited_clause_id: str | None = None
    quoted_span: str | None = None
    response_span: str | None = None
    span_verified: bool | None = None

    judge_model: str
    judge_k: int = 1
    judge_agreement: float | None = None
    judge_confidence: float | None = None
    judge_abstained: bool = False
    judge_temperature: float | None = None
    judge_completions: int | None = None
    judge_error: str | None = None

    gate_run: bool = False
    git_sha: str
    supersedes_id: str | None = SQLField(default=None, index=True)

    @classmethod
    def from_row(cls, row: AuditRow) -> AuditRowRecord:
        """Encode a validated `AuditRow` for storage."""
        data = row.model_dump(mode="json")
        for column in JSON_COLUMNS:
            data[column] = json.dumps(data[column], sort_keys=True)
        return cls(**data)

    def to_row(self) -> AuditRow:
        """Decode back to the validated form.

        Validation runs on the way *out* as well as in. That is not redundant: it
        is what turns "append-only by convention" into something detectable. If a
        row in `runs.db` was edited by hand - which SQLite makes trivial and
        nothing at the file level prevents - the edit has to survive every
        invariant above to be read back silently.
        """
        data = self.model_dump()
        for column in JSON_COLUMNS:
            raw = data[column]
            data[column] = json.loads(raw) if isinstance(raw, str) else raw
        return AuditRow(**data)
