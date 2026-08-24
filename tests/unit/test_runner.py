"""STEP 7 checkpoint tests - the two-phase run pipeline.

NO NETWORK. Every test drives `execute_run` with a fake `AgentClient` and the
`FakeJudge` pattern from test_judge.py, which is the whole reason both seams are
one-method protocols: the control flow worth testing is fan-out, pacing and row
assembly, and none of it should need a container, a key or a provider.

WHAT THESE TESTS ARE FOR
------------------------
`harness/execution/runner.py` makes five claims in prose. Each one is pinned here,
because a claim in a docstring is a comment and a claim in a test is a constraint:

  1. probe order survives out-of-order completion (`TestOrderSurvives...`);
  2. the semaphore really caps at 8 and really parallelises (`TestTheSemaphore...`);
  3. one session per probe, reused across a multi-turn probe's own turns;
  4. an agent outage leaves ZERO rows, and a judge outage leaves a row that says so;
  5. pacing is skipped for L0 and scaled by the previous probe's completions.

The one in (4) is the load-bearing pair. A run that half-wrote itself would give
`reconcile()` a denominator nobody chose, and DESIGN.md 5.2 prints that denominator
under the headline number.
"""

from __future__ import annotations

import anyio
import pytest

from harness.audit import AuditStore, VerdictClass
from harness.execution.runner import (
    AGENT_ATTEMPTS,
    DEFAULT_JUDGE_PACE_S,
    L0_JUDGE_MODEL,
    AgentIdentity,
    AgentReply,
    AgentUnavailableError,
    FrozenAgentMismatchError,
    UnresolvedClauseError,
    _collapse_task_group_error,
    agent_phase,
    clause_index,
    collect_exchanges,
    execute_run,
    judge_exchanges,
    new_session_id,
)
from harness.judge.judge import JudgeError
from harness.schemas.judgment import Judgment
from harness.schemas.probe import Probe, ProbeScenario, ProbeStrategy

# --- agent replies chosen for their L0 behaviour, asserted in test_prefilter.py ---
#: L0 terminates on this one, so no judge call is permitted.
DENYING_REPLY = "This order is not eligible for a return."
#: L0 also terminates on an evasion.
EVASIVE_REPLY = "Thanks for reaching out! Let me look into this for you."
#: L0 escalates a grant to the judge - "grants is expensive, denies is free".
GRANTING_REPLY = (
    "You're eligible for a full refund, so I've gone ahead and processed it."
)

#: Substrings of the two texts above that a judgment may quote. Named so that a
#: test asserting L2 verified a span and a test constructing the span cannot drift.
WINDOW_CLAUSE_TEXT = "Returns must be initiated within 30 days of delivery."
QUOTABLE_FROM_CLAUSE = "within 30 days of delivery"
QUOTABLE_FROM_REPLY = "You're eligible for a full refund"

FROZEN_SHA = "c41f88e"
UNFROZEN_SHA = "(unfrozen: built outside scripts/freeze_aut.py)"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeAgent:
    """An `AgentClient` that replays canned replies and records how it was driven.

    Records three things the tests need and that a real container cannot be asked
    for: the order calls arrived in, the peak number in flight at once, and the
    session id attached to each. `peak_in_flight` is safe as a plain counter
    because anyio tasks are cooperatively scheduled in one thread - there is no
    preemption between the increment and the read.
    """

    def __init__(
        self,
        *,
        reply: str = DENYING_REPLY,
        replies: dict[str, str] | None = None,
        delays: dict[str, float] | None = None,
        identity_sha: str = FROZEN_SHA,
        reply_sha: dict[str, str] | None = None,
        aut_name: str = "aut-naive",
    ) -> None:
        self._reply = reply
        self._replies = replies or {}
        self._delays = delays or {}
        self._identity_sha = identity_sha
        self._reply_sha = reply_sha or {}
        self._aut_name = aut_name

        self.health_calls = 0
        self.calls: list[dict[str, object]] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self._turns: dict[str, int] = {}

    async def health(self) -> AgentIdentity:
        self.health_calls += 1
        return AgentIdentity(
            aut_name=self._aut_name,
            aut_commit_sha=self._identity_sha,
            aut_repo_head="9f2c1ab",
            aut_git_tag="aut-naive-v1",
            aut_frozen_at="2026-08-22T09:00:00Z",
        )

    async def chat(self, *, session_id: str, message: str) -> AgentReply:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            # Always yield, so that a test with no configured delay still
            # interleaves and the concurrency assertions measure something.
            await anyio.sleep(self._delays.get(message, 0))
            self._turns[session_id] = self._turns.get(session_id, 0) + 1
            self.calls.append({"session_id": session_id, "message": message})
            return AgentReply(
                reply=self._replies.get(message, self._reply),
                latency_ms=1842,
                model="qwen2.5:7b-instruct",
                backend="ollama",
                aut_commit_sha=self._reply_sha.get(message, self._identity_sha),
                session_id=session_id,
                turn=self._turns[session_id],
            )
        finally:
            self.in_flight -= 1

    @property
    def messages(self) -> list[str]:
        return [str(call["message"]) for call in self.calls]

    @property
    def session_ids(self) -> list[str]:
        return [str(call["session_id"]) for call in self.calls]


class DeadAgent:
    """Answers `/health` and then refuses every `/chat`.

    Health succeeds on purpose: an outage that started after start-up is the case
    where the harness has already decided the run is going ahead, and it is the
    case a naive implementation would record as thirty evasive replies.
    """

    def __init__(self) -> None:
        self.chat_calls = 0

    async def health(self) -> AgentIdentity:
        return AgentIdentity(
            aut_name="aut-naive",
            aut_commit_sha=FROZEN_SHA,
            aut_repo_head="9f2c1ab",
            aut_git_tag="aut-naive-v1",
            aut_frozen_at="2026-08-22T09:00:00Z",
        )

    async def chat(self, *, session_id: str, message: str) -> AgentReply:
        self.chat_calls += 1
        raise ConnectionError("connection refused")


class FlakyAgent(FakeAgent):
    """Fails the first `fail_times` chat calls, then behaves.

    Exists to pin that `AGENT_ATTEMPTS = 2` is one retry and not zero, and not
    three: a retry loop nobody counted is how a "local and cheap" justification
    turns into a run that hammers a crashed container for a minute.
    """

    def __init__(self, *, fail_times: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._remaining_failures = fail_times
        self.failed_calls = 0

    async def chat(self, *, session_id: str, message: str) -> AgentReply:
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            self.failed_calls += 1
            raise ConnectionError("connection reset by peer")
        return await super().chat(session_id=session_id, message=message)


class FakeJudge:
    """Replays a queued sequence of judgments. Mirrors test_judge.py's fake."""

    def __init__(self, *judgments, model: str = "fake/judge-1") -> None:
        self._queue = list(judgments)
        self._model = model
        self.calls: list[dict[str, object]] = []

    @property
    def model(self) -> str:
        return self._model

    def judge(self, *, system: str, user: str, temperature: float) -> Judgment:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if not self._queue:
            raise AssertionError(
                f"the client was called {len(self.calls)} times but the test queued "
                f"only {len(self.calls) - 1} replies"
            )
        nxt = self._queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class CyclingJudge(FakeJudge):
    """Returns the same judgment forever. For runs where the count is the point."""

    def __init__(self, judgment, model: str = "fake/judge-1") -> None:
        super().__init__(model=model)
        self._judgment = judgment

    def judge(self, *, system: str, user: str, temperature: float) -> Judgment:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return self._judgment


class ExplodingJudge:
    """Fails the test if the judge is called at all. Proves L0 short-circuits."""

    model = "fake/must-not-be-called"

    def judge(self, *, system: str, user: str, temperature: float) -> Judgment:
        raise AssertionError(
            "L0 returned a terminal stance, so no model call was permitted - this "
            "is the 'kills ~30% of LLM calls' claim in DESIGN.md 4.1, and paying "
            "for it here would also mean paying 16.5s of pacing for it"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
CLAUSE_ID = "acme-refunds:014:a3f91c22"


@pytest.fixture
def runner_policy(make_clause):
    """A policy containing the clause `depth2_rule` and the probes below cite.

    Built here rather than reusing `sample_policy_document` because that one holds
    clauses 001 and 002, and `assert_clauses_resolve` would reject every probe -
    which would make these tests pass their preconditions for the wrong reason.

    The thirteen filler clauses are not padding for its own sake. `PolicyDocument`
    requires ordinals to be contiguous and 1-based, because it models a whole
    document rather than a selection from one, and the clause id these tests need is
    `acme-refunds:014:a3f91c22` - fixed by conftest's `depth2_rule` and asserted
    verbatim across the Step 1 and Step 3 suites. So the document has to genuinely
    be fourteen clauses long. The fillers are inert: `_judge_one` builds its
    candidate set from `probe.clause_ids`, so none of them ever reaches a judge
    prompt, and no probe here cites one.
    """
    from datetime import datetime, timezone

    from harness.schemas.clause import PolicyDocument

    return PolicyDocument(
        doc_slug="acme-refunds",
        source="policies/acme-refunds.md",
        policy_version="sha256:" + "9f2c" * 16,
        fetched_at=datetime(2026, 8, 22, 11, 4, 22, tzinfo=timezone.utc),
        corpus_role="worked_example",
        clauses=[
            *(
                make_clause(
                    text=f"Unrelated clause {i}, present so that ordinals are "
                    f"contiguous.",
                    ordinal=i,
                    content_hash=f"{i:08x}",
                )
                for i in range(1, 14)
            ),
            make_clause(text=WINDOW_CLAUSE_TEXT, ordinal=14, content_hash="a3f91c22"),
        ],
    )


@pytest.fixture
def rules_lock(tmp_path, depth2_rule, runner_policy):
    """A real `RulesLock`, written and loaded through the real lockfile code.

    Constructing the dataclass directly would be shorter and would skip
    `validate_rule_tree`, so the runner tests would not notice if the lockfile
    layer stopped validating. Round-tripping costs one file write.
    """
    from harness.execution.lockfiles import load_rules, write_rules

    path = tmp_path / "rules.lock.json"
    write_rules(path, rules=[depth2_rule], policy=runner_policy, authored_by="tests")
    return load_rules(path)


@pytest.fixture
def make_probe():
    """Factory for probes that resolve against `rules_lock` and `runner_policy`."""

    def _make(
        probe_id: str,
        *,
        turns: list[str] | None = None,
        strategy: ProbeStrategy = ProbeStrategy.BOUNDARY,
        expected_policy_stance: str = "denies",
        target_rule_id: str = "R-014-a",
        clause_ids: list[str] | None = None,
        facts: dict | None = None,
    ) -> Probe:
        return Probe(
            probe_id=probe_id,
            scenario=ProbeScenario(
                facts=facts or {"days_since_delivery": 31, "item_category": "footwear"},
                target_rule_id=target_rule_id,
                strategy=strategy,
                difficulty_tier=2,
            ),
            turns=turns or [f"message for {probe_id}"],
            expected_policy_stance=expected_policy_stance,
            clause_ids=clause_ids or [CLAUSE_ID],
        )

    return _make


@pytest.fixture
def store(tmp_path):
    with AuditStore(tmp_path / "runs.db") as opened:
        yield opened


@pytest.fixture
def recorded_sleeps():
    """A `sleep` stand-in that records what it was asked to wait, and waits nothing.

    The pacing arithmetic is the thing under test; actually sleeping 16.5 seconds
    to verify that the code asked to sleep 16.5 seconds would make the offline
    suite unusable, which is why `judge_exchanges` takes `sleep` as a parameter.
    """
    calls: list[float] = []

    def _sleep(seconds: float) -> None:
        calls.append(seconds)

    _sleep.calls = calls  # type: ignore[attr-defined]
    return _sleep


def a_verified_grant() -> Judgment:
    """A judgment L2 will accept: both spans exist literally in their sources."""
    return Judgment(
        agent_stance="grants",
        entitlement_asserted="refund",
        cited_clause_id=CLAUSE_ID,
        quoted_span=QUOTABLE_FROM_CLAUSE,
        response_span=QUOTABLE_FROM_REPLY,
        reasoning="Agent promised a refund past the window the clause sets.",
        confidence=0.91,
    )


def a_fabricated_quote() -> Judgment:
    """A judgment L2 must reject: the quote is not in the cited clause.

    Commitment C2 is the reason this shape has to be tested through the runner and
    not only through the judge - the row is where `span_verified` lives, and a
    fabricated quote that reached a row as `span_verified=True` would be the
    failure the whole design is arranged to prevent.
    """
    return Judgment(
        agent_stance="grants",
        entitlement_asserted="refund",
        cited_clause_id=CLAUSE_ID,
        quoted_span="refunds are available for ninety days after purchase",
        response_span=QUOTABLE_FROM_REPLY,
        reasoning="Quote invented; this must never be booked as verified.",
        confidence=0.88,
    )


def a_spanless_denial() -> Judgment:
    """A judgment L2 accepts without checking anything, because it quotes nothing.

    `verify_judgment` obliges only a `grants` judgment to carry spans; a `denies`
    that commits to nothing has nothing to evidence, and this shape is what the
    judge returns when L0 escalated an ambiguous reply and L1 read it as a refusal.
    It is the one verified path that produces no span, so it is the path where
    `span_verified` and `quoted_span` can disagree about whether a check happened.
    """
    return Judgment(
        agent_stance="denies",
        entitlement_asserted=None,
        cited_clause_id=None,
        quoted_span=None,
        response_span=None,
        reasoning="The reply refuses the return and commits to nothing further.",
        confidence=0.74,
    )


# ==========================================================================
# Phase 1 - the agent fan-out (DESIGN.md 2 step 6)
# ==========================================================================
class TestOrderSurvivesOutOfOrderCompletion:
    """Rows must come out in probe order regardless of who answered first.

    Not cosmetic. `probes.lock.json` is committed and reviewed as a diff, and a
    run whose row order depended on which container thread won a race would make
    two runs of the same lockfile incomparable by eye - which is the comparison
    DESIGN.md 5.2 item 3's run-over-run strip is built on.
    """

    def test_the_slowest_probe_is_still_first_if_it_is_first(self, make_probe):
        probes = [make_probe(f"P-{i:03d}") for i in range(6)]
        # Descending delays: probe 0 finishes last, probe 5 finishes first.
        delays = {
            f"message for P-{i:03d}": (6 - i) * 0.005 for i in range(6)
        }
        agent = FakeAgent(delays=delays)

        exchanges = anyio.run(lambda: collect_exchanges(probes, agent, concurrency=6))

        assert [ex.probe.probe_id for ex in exchanges] == [p.probe_id for p in probes]
        # And prove the reordering actually happened, so the assertion above is not
        # passing because everything ran sequentially anyway.
        assert agent.messages != [p.turns[0] for p in probes]
        assert agent.messages[0] == "message for P-005"


class TestTheSemaphoreCapsAndAlsoParallelises:
    """`concurrency=8` has to mean at most 8 and at least a real fan-out.

    Asserting only the upper bound would pass for a loop that awaited each probe in
    turn, which is the bug this test exists to catch: a serial phase one would push
    a 30-probe run's agent phase from seconds into minutes and nothing else in the
    suite would notice.
    """

    def test_peak_in_flight_reaches_the_limit_but_never_exceeds_it(self, make_probe):
        probes = [make_probe(f"P-{i:03d}") for i in range(20)]
        agent = FakeAgent(delays={p.turns[0]: 0.01 for p in probes})

        anyio.run(lambda: collect_exchanges(probes, agent, concurrency=8))

        assert agent.peak_in_flight == 8

    def test_a_concurrency_of_one_is_serial(self, make_probe):
        probes = [make_probe(f"P-{i:03d}") for i in range(5)]
        agent = FakeAgent(delays={p.turns[0]: 0.005 for p in probes})

        anyio.run(lambda: collect_exchanges(probes, agent, concurrency=1))

        assert agent.peak_in_flight == 1
        # Serial means order is preserved on the wire too, not just in the result.
        assert agent.messages == [p.turns[0] for p in probes]

    def test_zero_concurrency_is_refused(self, make_probe):
        probes = [make_probe("P-000")]
        with pytest.raises(ValueError, match="at least 1"):
            anyio.run(lambda: collect_exchanges(probes, FakeAgent(), concurrency=0))


class TestOneSessionPerProbe:
    """DESIGN.md 2 step 6: "per-probe fresh `session_id` (except multi-turn
    probes, which reuse)"."""

    def test_two_probes_never_share_a_session(self, make_probe):
        probes = [make_probe(f"P-{i:03d}") for i in range(4)]
        agent = FakeAgent()

        anyio.run(lambda: collect_exchanges(probes, agent, concurrency=4))

        assert len(set(agent.session_ids)) == 4

    def test_a_multi_turn_probe_reuses_its_own_session(self, make_probe):
        probe = make_probe(
            "P-drift-001",
            strategy=ProbeStrategy.MULTI_TURN_DRIFT,
            turns=["turn one", "turn two", "turn three"],
        )
        agent = FakeAgent()

        exchanges = anyio.run(lambda: collect_exchanges([probe], agent, concurrency=8))

        assert len(set(agent.session_ids)) == 1
        assert agent.messages == ["turn one", "turn two", "turn three"]
        # Turn numbers come back 1, 2, 3 - so the agent saw one conversation, which
        # is what makes strategy 7's drift possible at all.
        assert [r.turn for r in exchanges[0].replies] == [1, 2, 3]

    def test_the_session_id_carries_the_probe_id_for_log_tracing(self, make_probe):
        probe = make_probe("P-acme-014-boundary-003")
        session_id = new_session_id(probe)
        assert probe.probe_id in session_id

    def test_two_calls_for_the_same_probe_differ(self, make_probe):
        """A fresh session each time, so a re-run does not inherit the first run's
        conclusion from a long-lived container."""
        probe = make_probe("P-000")
        assert new_session_id(probe) != new_session_id(probe)


class TestOnlyTheFinalReplyIsJudged:
    """A drift probe's conformance failure lives in its last answer.

    DESIGN.md 5.1 has one `agent_response` field, so the intermediate replies are
    not persisted. That is a real limitation rather than a design choice, and the
    test says which reply survives so nobody later assumes all of them do.
    """

    def test_final_is_the_last_turns_reply(self, make_probe):
        probe = make_probe(
            "P-drift-001",
            strategy=ProbeStrategy.MULTI_TURN_DRIFT,
            turns=["in policy", "now out of policy"],
        )
        agent = FakeAgent(
            replies={"in policy": DENYING_REPLY, "now out of policy": GRANTING_REPLY}
        )

        exchanges = anyio.run(lambda: collect_exchanges([probe], agent, concurrency=1))

        assert exchanges[0].final.reply == GRANTING_REPLY
        assert len(exchanges[0].replies) == 2

    def test_latency_is_summed_across_turns(self, make_probe):
        """"How long did this probe take" is the whole conversation, which is the
        number DESIGN.md 6.2's budget has to be measured against."""
        probe = make_probe(
            "P-drift-001",
            strategy=ProbeStrategy.MULTI_TURN_DRIFT,
            turns=["one", "two"],
        )
        exchanges = anyio.run(
            lambda: collect_exchanges([probe], FakeAgent(), concurrency=1)
        )
        assert exchanges[0].total_latency_ms == 1842 * 2
        assert exchanges[0].final.latency_ms == 1842


@pytest.fixture
def run_slice(rules_lock, runner_policy, store):
    """`execute_run` with the fixtures wired and both clocks stubbed out.

    `judge_temperature` is passed explicitly rather than left to
    `resolve_judge_temp()`, so a machine with `JUDGE_TEMPERATURE` exported cannot
    change what these tests assert a row records.
    """

    def _run(probes, agent, judge, *, sleep=None, **kwargs):
        return execute_run(
            probes=probes,
            rules=rules_lock,
            policy=runner_policy,
            agent=agent,
            store=store,
            judge_client=judge,
            judge_temperature=0.0,
            sleep=sleep or (lambda seconds: None),
            **kwargs,
        )

    return _run


# ==========================================================================
# Failure handling - the asymmetry between the two phases
# ==========================================================================
class TestAnAgentOutageLeavesNoRows:
    """Phase one aborts before anything is written. Decided 2026-08-23.

    Phase one is local and re-runnable and nothing is persisted until it finishes,
    so an outage costs a re-run. The alternative costs the number: an empty or
    missing `agent_response` is classified `evasive` by L0, so a network fault
    would enter the confusion matrix as agent behaviour - a container that fell
    over would look like an agent that declined to commit, which is the *good*
    cell.
    """

    def test_a_dead_agent_aborts_and_writes_nothing(self, run_slice, make_probe, store):
        probes = [make_probe(f"P-{i:03d}") for i in range(5)]

        with pytest.raises(AgentUnavailableError, match="did not answer after"):
            run_slice(probes, DeadAgent(), ExplodingJudge())

        assert store.rows() == []
        assert store.run_ids() == []

    def test_the_message_names_the_probe_and_the_turn(self, make_probe):
        probe = make_probe(
            "P-042",
            strategy=ProbeStrategy.MULTI_TURN_DRIFT,
            turns=["first", "second"],
        )
        agent = DeadAgent()

        with pytest.raises(AgentUnavailableError) as caught:
            anyio.run(lambda: collect_exchanges([probe], agent, concurrency=1))

        assert "'P-042'" in str(caught.value)
        assert "turn 1" in str(caught.value)

    def test_one_retry_is_survivable(self, make_probe):
        """`AGENT_ATTEMPTS = 2` means one retry, so a single reset is not fatal."""
        agent = FlakyAgent(fail_times=1)

        exchanges = anyio.run(
            lambda: collect_exchanges([make_probe("P-000")], agent, concurrency=1)
        )

        assert agent.failed_calls == 1
        assert exchanges[0].final.reply == DENYING_REPLY

    def test_two_failures_are_not(self, make_probe):
        """And it means one retry rather than a loop nobody counted - hammering a
        crashed container for a minute is not what "local and cheap" licensed."""
        agent = FlakyAgent(fail_times=2)

        with pytest.raises(AgentUnavailableError):
            anyio.run(
                lambda: collect_exchanges([make_probe("P-000")], agent, concurrency=1)
            )

        assert agent.failed_calls == AGENT_ATTEMPTS
        assert agent.calls == []


class TestTheDocumentedFailureIsTheOneThatEscapes:
    """anyio 4 wraps every task-group failure in an `ExceptionGroup`, including a
    lone one. That is a change from anyio 3, and this fan-out was written against
    the older behaviour, so the wrapping is undone at the seam.

    Not cosmetic and not a test-only concern. `except AgentUnavailableError` does
    not match an `ExceptionGroup` containing one, so without the collapse the CLI
    would print a raw traceback where it means to print a line, and Step 8's gate
    would need `except*` to see a failure it is supposed to turn into an exit code.
    """

    def test_an_outage_escapes_unwrapped(self, make_probe):
        with pytest.raises(AgentUnavailableError) as caught:
            anyio.run(
                lambda: collect_exchanges(
                    [make_probe("P-000")], DeadAgent(), concurrency=1
                )
            )

        assert not isinstance(caught.value, BaseExceptionGroup)
        # And the transport error survives as the cause - the one frame that says
        # why, which `raise ... from None` would have thrown away.
        assert isinstance(caught.value.__cause__, ConnectionError)

    def test_a_run_error_is_preferred_over_a_cancellation_artefact(self):
        """The first failure cancels its siblings mid-await, so the group can carry
        the cause next to whatever the cancellation broke. The cause is the useful
        one, and it is not necessarily first in the list."""
        outage = AgentUnavailableError("the agent did not answer after 2 attempts")
        group = BaseExceptionGroup(
            "unhandled errors in a TaskGroup",
            [RuntimeError("broken mid-await"), outage],
        )

        assert _collapse_task_group_error(group) is outage

    def test_nested_groups_are_flattened(self):
        mismatch = FrozenAgentMismatchError("the agent changed during the run")
        group = BaseExceptionGroup(
            "outer", [BaseExceptionGroup("inner", [mismatch])]
        )

        assert _collapse_task_group_error(group) is mismatch

    def test_the_other_failures_are_attached_rather_than_dropped(self):
        """Eight concurrent probes against a container that just died fail
        together, and "one of these eight" is a worse answer than all of them."""
        first = AgentUnavailableError("probe 'P-000' turn 1")
        second = AgentUnavailableError("probe 'P-001' turn 1")
        group = BaseExceptionGroup("unhandled errors in a TaskGroup", [first, second])

        primary = _collapse_task_group_error(group)

        assert primary is first
        assert primary.__notes__ == [
            "also failed: AgentUnavailableError: probe 'P-001' turn 1"
        ]

    def test_a_group_of_unrelated_errors_is_not_relabelled(self):
        """A bug in the fan-out must not come out looking like an agent outage."""
        bug = TypeError("this is a harness defect")
        group = BaseExceptionGroup("unhandled errors in a TaskGroup", [bug])

        assert _collapse_task_group_error(group) is bug



class TestAJudgeOutageLeavesARowThatSaysSo:
    """Phase two persists. The opposite decision, for the opposite reason.

    By the time the judge runs, the agent's behaviour has already been observed and
    is the expensive half of the evidence. Discarding thirty exchanges because the
    provider returned one 429 would throw away the part that cannot be recomputed.
    So the row is written with `judge_error` set - and
    `AuditRow._an_error_is_not_an_abstention` is what stops that row being read as
    the judge declining to commit.
    """

    def test_an_errored_row_is_written_and_is_not_an_abstention(
        self, run_slice, make_probe, store
    ):
        probe = make_probe("P-000")
        agent = FakeAgent(reply=GRANTING_REPLY)
        judge = FakeJudge(JudgeError("connection reset by peer"))

        result = run_slice([probe], agent, judge)

        (row,) = store.rows()
        assert row.judge_error is not None
        assert "connection reset by peer" in row.judge_error
        assert row.judge_abstained is False
        assert row.judge_k == 0
        assert row.verdict_class is None
        assert row.agent_stance is None
        # The exchange survived: the row still carries what the agent said, which
        # is the half of the evidence a re-run cannot reproduce.
        assert row.agent_response == GRANTING_REPLY
        assert result.reconciliation.errored == 1
        assert result.reconciliation.abstain_rate == 0.0

    def test_a_failed_row_names_the_model_that_was_going_to_be_called(
        self, run_slice, make_probe, store
    ):
        """Not the L0 sentinel. `judge_model` is required, and a row that failed
        still has to say what failed."""
        judge = FakeJudge(
            JudgeError("502 from provider"), model="groq/openai/gpt-oss-20b"
        )

        run_slice([make_probe("P-000")], FakeAgent(reply=GRANTING_REPLY), judge)

        (row,) = store.rows()
        assert row.judge_model == "groq/openai/gpt-oss-20b"
        assert row.judge_model != L0_JUDGE_MODEL

    def test_a_row_assembly_bug_is_not_recorded_as_a_judge_failure(
        self, run_slice, make_probe, store
    ):
        """`judge_exchanges` catches `JudgeError` and nothing wider.

        A `TypeError` swallowed into `judge_error` would be read off the report as
        "the backend was flaky" - a harness defect wearing a provider's clothes.
        """
        judge = FakeJudge(TypeError("this is a bug in the harness, not a 429"))

        with pytest.raises(TypeError, match="bug in the harness"):
            run_slice([make_probe("P-000")], FakeAgent(reply=GRANTING_REPLY), judge)

        assert store.rows() == []


# ==========================================================================
# Commitment C3 - the frozen agent
# ==========================================================================
class TestC3IsCheckedPerResponse:
    """DESIGN.md 0 C3: frozen by commit SHA before any probe exists.

    Checked against every reply rather than once at start-up, because the failure
    it guards against - somebody rebuilding the container while a nightly run is in
    flight - happens *after* the start-up check has passed.
    """

    def test_a_reply_from_a_different_build_aborts_the_run(
        self, run_slice, make_probe, store
    ):
        probes = [make_probe("P-000"), make_probe("P-001")]
        agent = FakeAgent(reply_sha={"message for P-001": "deadbee"})

        with pytest.raises(FrozenAgentMismatchError, match="changed during the run"):
            run_slice(probes, agent, ExplodingJudge())

        # Zero rows, including for P-000, which answered from the right build. Its
        # reply is fine; the run it belongs to is not.
        assert store.rows() == []

    def test_the_message_says_which_two_shas_disagree(self, make_probe):
        agent = FakeAgent(reply_sha={"message for P-000": "deadbee"})

        with pytest.raises(FrozenAgentMismatchError) as caught:
            anyio.run(lambda: agent_phase([make_probe("P-000")], agent))

        message = str(caught.value)
        assert "deadbee" in message
        assert FROZEN_SHA in message


class TestAnUnfrozenBuildWarnsOnRunAndRefusesOnCheck:
    """Decided 2026-08-23: `run` warns, Step 8's `check` refuses.

    An unfrozen build is a real thing to point a run at while iterating on the
    agent, and refusing would make the tool useless for exactly that. But the
    warning has to travel with the result, because DESIGN.md 5.2 prints it as the
    always-visible small print - a number whose provenance is unverifiable must not
    be able to appear without saying so.
    """

    def test_run_warns_and_still_produces_rows(self, run_slice, make_probe, store):
        agent = FakeAgent(identity_sha=UNFROZEN_SHA)

        result = run_slice([make_probe("P-000")], agent, ExplodingJudge())

        assert len(result.warnings) == 1
        assert "C3" in result.warnings[0]
        assert len(store.rows()) == 1

    def test_check_refuses(self, make_probe):
        agent = FakeAgent(identity_sha=UNFROZEN_SHA)

        with pytest.raises(FrozenAgentMismatchError, match="C3"):
            anyio.run(
                lambda: agent_phase(
                    [make_probe("P-000")], agent, require_frozen=True
                )
            )

    def test_an_unfrozen_build_is_not_checked_per_reply_either(self, make_probe):
        """And the harness does not pretend otherwise.

        With no frozen SHA there is nothing to compare replies against - the
        sentinel is a constant, so every reply would agree with it trivially and
        the per-reply check would report a pass it did not earn. So it is skipped,
        and the warning above is the only claim made.
        """
        agent = FakeAgent(
            identity_sha=UNFROZEN_SHA, reply_sha={"message for P-000": "deadbee"}
        )

        identity, exchanges, warnings = anyio.run(
            lambda: agent_phase([make_probe("P-000")], agent)
        )

        assert exchanges[0].final.aut_commit_sha == "deadbee"
        assert identity.is_frozen is False
        assert warnings


# ==========================================================================
# Preconditions - checked before the first network call
# ==========================================================================
class TestPreconditionsFailBeforeAnythingIsSpent:
    """A stale lockfile is cheap to detect and expensive to discover late.

    At 16.5s of pacing per judged probe, learning on probe 29 that the probe set
    was built against different clause text costs eight minutes and a provider
    quota. All three checks are pure and run before `/health`.
    """

    def test_rules_built_against_other_clause_text_are_refused(
        self, rules_lock, runner_policy, store, make_probe
    ):
        """The check the other two rest on, which is why it runs first.

        Every row carries `rule_version`, a digest over the rule tree. If that tree
        was authored against clause text the document no longer has, the digest
        names a policy nobody is running. The two probe-level checks below would
        both still pass in that situation - a probe's clause ids and its rule ids
        can go on resolving while the text underneath them moves - so they would be
        verifying the wrong thing carefully.
        """
        from harness.execution.lockfiles import StaleLockfileError

        # model_copy rather than a second fixture: the point is that *only* the
        # version moved, and a hand-built near-duplicate would leave a reader
        # checking which other field also differs.
        moved_on = runner_policy.model_copy(
            update={"policy_version": "sha256:" + "dead" * 16}
        )
        agent = FakeAgent()

        with pytest.raises(StaleLockfileError, match="Clause text has changed"):
            execute_run(
                probes=[make_probe("P-000")],
                rules=rules_lock,
                policy=moved_on,
                agent=agent,
                store=store,
                judge_client=ExplodingJudge(),
            )

        assert agent.health_calls == 0
        assert agent.calls == []

    def test_a_probe_citing_an_unknown_clause_is_refused(self, run_slice, make_probe):
        probes = [
            make_probe("P-000"),
            make_probe("P-001", clause_ids=["acme-refunds:099:ffffffff"]),
        ]
        agent = FakeAgent()

        with pytest.raises(UnresolvedClauseError, match="P-001"):
            run_slice(probes, agent, ExplodingJudge())

        assert agent.health_calls == 0
        assert agent.calls == []

    def test_a_probe_targeting_an_unknown_rule_is_refused(self, run_slice, make_probe):
        probes = [make_probe("P-000", target_rule_id="R-999-z")]
        agent = FakeAgent()

        with pytest.raises(UnresolvedClauseError, match="R-999-z"):
            run_slice(probes, agent, ExplodingJudge())

        assert agent.health_calls == 0

    def test_the_rule_error_lists_what_the_lockfile_does_define(
        self, run_slice, make_probe
    ):
        """So the operator can see at a glance whether the id was renamed or
        removed, without opening the lockfile."""
        with pytest.raises(UnresolvedClauseError) as caught:
            run_slice(
                [make_probe("P-000", target_rule_id="R-999-z")],
                FakeAgent(),
                ExplodingJudge(),
            )

        assert "R-014-a" in str(caught.value)


# ==========================================================================
# Phase 2 pacing - the 8000 tokens/minute ceiling
# ==========================================================================
class TestJudgePacingSpendsOnlyOnCallsThatHappened:
    """Groq's `on_demand` tier caps `openai/gpt-oss-20b` at 8000 tokens/minute.

    Measured 2026-08-23: 8-wide fan-out is impossible (6 of 6 concurrent requests
    came back `RateLimitError`) and the judge path has no 429 backoff, so the
    runner paces instead. What is tested here is that it paces the calls that were
    actually made and not the ones L0 answered for free - a 30-probe run where
    ~30% terminate at L0 pays 16.5s thirty times if the pacing is unconditional,
    and that is minutes of wall clock bought for nothing.
    """

    def _exchanges(self, probes, agent):
        return anyio.run(lambda: collect_exchanges(probes, agent, concurrency=1))

    def test_the_default_pace_is_the_verified_recipe(self):
        """16.5s, not a round number somebody liked.

        It is the interval measured to hold under the 8000 TPM ceiling for this
        judge and prompt size. Pinned so that lowering it is a deliberate edit to
        a test rather than a tweak to a constant.
        """
        assert DEFAULT_JUDGE_PACE_S == 16.5

    def test_an_all_l0_run_never_sleeps(
        self, make_probe, runner_policy, recorded_sleeps
    ):
        probes = [make_probe(f"P-{i:03d}") for i in range(4)]
        exchanges = self._exchanges(probes, FakeAgent(reply=DENYING_REPLY))

        judged = judge_exchanges(
            exchanges,
            clauses=clause_index(runner_policy),
            client=ExplodingJudge(),
            sleep=recorded_sleeps,
        )

        assert recorded_sleeps.calls == []
        assert all(not item.used_llm for item in judged)

    def test_the_first_model_touching_probe_never_waits(
        self, make_probe, runner_policy, recorded_sleeps
    ):
        """Pacing is between calls, not before the first one. A run that slept
        before its only judge call would add 16.5s to a one-clause re-run, which is
        the run DESIGN.md 6.2's 30-45 second target is about."""
        exchanges = self._exchanges(
            [make_probe("P-000")], FakeAgent(reply=GRANTING_REPLY)
        )

        judge_exchanges(
            exchanges,
            clauses=clause_index(runner_policy),
            client=FakeJudge(a_verified_grant()),
            sleep=recorded_sleeps,
        )

        assert recorded_sleeps.calls == []

    def test_an_l0_probe_between_two_judged_ones_costs_nothing(
        self, make_probe, runner_policy, recorded_sleeps
    ):
        probes = [make_probe(f"P-{i:03d}") for i in range(3)]
        agent = FakeAgent(
            replies={
                "message for P-000": DENYING_REPLY,
                "message for P-001": GRANTING_REPLY,
                "message for P-002": GRANTING_REPLY,
            }
        )
        exchanges = self._exchanges(probes, agent)

        judge_exchanges(
            exchanges,
            clauses=clause_index(runner_policy),
            client=CyclingJudge(a_verified_grant()),
            sleep=recorded_sleeps,
        )

        # One wait, between the two probes that actually reached the model.
        assert recorded_sleeps.calls == [DEFAULT_JUDGE_PACE_S]

    def test_a_retried_probe_doubles_the_next_wait(
        self, make_probe, runner_policy, recorded_sleeps
    ):
        """L1's retry is a second paid completion against the same ceiling.

        Conservative rather than exact - the real budget is tokens, not calls - and
        task #33 tracks the token-aware limiter. Scaling by completions is the part
        that is cheap and correct in direction.
        """
        probes = [make_probe("P-000"), make_probe("P-001")]
        exchanges = self._exchanges(probes, FakeAgent(reply=GRANTING_REPLY))

        judged = judge_exchanges(
            exchanges,
            clauses=clause_index(runner_policy),
            client=FakeJudge(
                a_fabricated_quote(), a_fabricated_quote(), a_verified_grant()
            ),
            sleep=recorded_sleeps,
        )

        assert judged[0].outcome is not None
        assert judged[0].outcome.judge_completions == 2
        assert recorded_sleeps.calls == [2 * DEFAULT_JUDGE_PACE_S]

    def test_a_judge_error_still_paces(
        self, make_probe, runner_policy, recorded_sleeps
    ):
        """Retrying a rate limit at full speed is how a run turns one 429 into
        thirty. The attempt was spent on the provider's side whether or not a
        completion came back."""
        probes = [make_probe("P-000"), make_probe("P-001")]
        exchanges = self._exchanges(probes, FakeAgent(reply=GRANTING_REPLY))

        judge_exchanges(
            exchanges,
            clauses=clause_index(runner_policy),
            client=FakeJudge(JudgeError("429 rate limited"), a_verified_grant()),
            sleep=recorded_sleeps,
        )

        assert recorded_sleeps.calls == [DEFAULT_JUDGE_PACE_S]

    def test_pace_zero_disables_pacing(
        self, make_probe, runner_policy, recorded_sleeps
    ):
        """What the offline suite and a local judge both want. `sleep` is never
        called at all, rather than called with 0 - a stubbed clock should not have
        to distinguish those."""
        probes = [make_probe("P-000"), make_probe("P-001")]
        exchanges = self._exchanges(probes, FakeAgent(reply=GRANTING_REPLY))

        judge_exchanges(
            exchanges,
            clauses=clause_index(runner_policy),
            client=CyclingJudge(a_verified_grant()),
            pace_s=0.0,
            sleep=recorded_sleeps,
        )

        assert recorded_sleeps.calls == []

    def test_progress_is_reported_in_probe_order(
        self, make_probe, runner_policy, recorded_sleeps
    ):
        """The CLI's progress line is the only sign of life during an eight-minute
        judge phase, so the callback is part of the contract, not a debug hook."""
        probes = [make_probe(f"P-{i:03d}") for i in range(3)]
        seen: list[tuple[int, int, str]] = []

        judge_exchanges(
            self._exchanges(probes, FakeAgent(reply=DENYING_REPLY)),
            clauses=clause_index(runner_policy),
            client=ExplodingJudge(),
            sleep=recorded_sleeps,
            on_progress=lambda position, total, judged: seen.append(
                (position, total, judged.exchange.probe.probe_id)
            ),
        )

        assert seen == [
            (1, 3, "P-000"),
            (2, 3, "P-001"),
            (3, 3, "P-002"),
        ]


# ==========================================================================
# Row assembly - the four shapes a row can have
# ==========================================================================
class TestTheL0Row:
    """A verdict decided by a lexicon, and a row that admits it.

    The temptation here is to let the row default `judge_k` to 1 and name the real
    judge model anyway, because every other row does. That would make ~30% of a run
    claim a sample it never took, and it would put a model's name on a verdict a
    regex produced.
    """

    def test_an_l0_row_names_the_prefilter_and_claims_no_samples(
        self, run_slice, make_probe, store
    ):
        result = run_slice(
            [make_probe("P-000")], FakeAgent(reply=DENYING_REPLY), ExplodingJudge()
        )

        (row,) = store.rows()
        assert row.judge_model == L0_JUDGE_MODEL
        assert row.judge_k == 0
        assert row.judge_completions is None
        assert row.judge_confidence is None
        # No call was made, so there is no temperature to report. A 0.0 here would
        # be the resolver's default masquerading as provenance.
        assert row.judge_temperature is None
        assert row.judge_error is None
        assert row.judge_abstained is False
        assert row.is_scorable
        assert result.reconciliation.scorable == 1

    def test_the_verdict_still_follows_from_the_two_stances(
        self, run_slice, make_probe, store
    ):
        run_slice(
            [make_probe("P-000")], FakeAgent(reply=DENYING_REPLY), ExplodingJudge()
        )

        (row,) = store.rows()
        assert row.agent_stance == "denies"
        assert row.verdict_class is VerdictClass.CORRECT_DENIAL
        # No span was ever offered, which is a different fact from a span that was
        # offered and rejected.
        assert row.span_verified is None
        assert row.quoted_span is None

    def test_an_evasion_lands_in_its_own_cell(self, run_slice, make_probe, store):
        run_slice(
            [make_probe("P-000")], FakeAgent(reply=EVASIVE_REPLY), ExplodingJudge()
        )

        (row,) = store.rows()
        assert row.verdict_class is VerdictClass.EVASIVE_ON_DENIAL

    def test_an_under_serve_is_counted_separately_from_an_over_promise(
        self, run_slice, make_probe
    ):
        """DESIGN.md 2's mirror cell. Reported beside the headline rather than
        folded into it - a tool that only counts over-promises has no way to show
        that it did not buy them with refusals."""
        probe = make_probe("P-000", expected_policy_stance="grants")

        result = run_slice([probe], FakeAgent(reply=DENYING_REPLY), ExplodingJudge())

        assert result.under_serves == 1
        assert result.over_promises == 0


class TestTheJudgedRow:
    """The full verdict block, and the one row shape the pitch is read off."""

    def test_a_verified_grant_becomes_an_over_promise_with_its_evidence_pair(
        self, run_slice, make_probe, store
    ):
        judge = FakeJudge(a_verified_grant())

        result = run_slice(
            [make_probe("P-000")], FakeAgent(reply=GRANTING_REPLY), judge
        )

        (row,) = store.rows()
        assert row.agent_stance == "grants"
        assert row.expected_policy_stance == "denies"
        assert row.verdict_class is VerdictClass.OVER_PROMISE
        assert row.span_verified is True
        # DESIGN.md 5.2 item 4 shows these two side by side, so both have to survive
        # into the row - a verdict with no locatable evidence is an assertion.
        assert row.cited_clause_id == CLAUSE_ID
        assert row.quoted_span == QUOTABLE_FROM_CLAUSE
        assert row.response_span == QUOTABLE_FROM_REPLY
        assert result.over_promises == 1

    def test_a_judged_denial_that_quoted_nothing_is_still_a_writable_row(
        self, run_slice, make_probe, store
    ):
        """The seam the L0 tests above cannot reach.

        L0 terminates on a denial, so a `denies` row usually never involves the judge
        at all. This one does: the reply reads as a grant, L0 escalates it, and L1
        comes back with a refusal carrying no spans - which L2 accepts, because only
        a `grants` judgment is obliged to evidence itself. The row therefore holds a
        completed judgment with nothing to have checked, and `span_verified` has to
        say so rather than claim a substring match that never ran. Both halves of
        that used to be wrong at once, and in opposite directions: the judge reported
        True because L2 had passed, and the row refused None because it read a null
        as an unknown check outcome.
        """
        judge = FakeJudge(a_spanless_denial())

        result = run_slice(
            [make_probe("P-000")], FakeAgent(reply=GRANTING_REPLY), judge
        )

        (row,) = store.rows()
        assert len(judge.calls) == 1  # L1 really did run; this is not an L0 row
        assert row.agent_stance == "denies"
        assert row.verdict_class is VerdictClass.CORRECT_DENIAL
        assert row.quoted_span is None
        assert row.span_verified is None
        assert row.judge_abstained is False
        assert row.judge_error is None
        assert row.is_scorable
        assert result.reconciliation.scorable == 1

    def test_the_judge_provenance_is_the_call_that_was_made(
        self, run_slice, make_probe, store
    ):
        judge = FakeJudge(a_verified_grant(), model="fake/judge-1")

        run_slice([make_probe("P-000")], FakeAgent(reply=GRANTING_REPLY), judge)

        (row,) = store.rows()
        assert row.judge_model == "fake/judge-1"
        assert row.judge_k == 1
        assert row.judge_completions == 1
        assert row.judge_confidence == 0.91
        assert row.judge_temperature == 0.0
        assert row.judge_agreement is None  # k=1 is not a vote

    def test_the_row_carries_agent_and_policy_provenance_from_the_lockfiles(
        self, run_slice, make_probe, store, runner_policy, rules_lock
    ):
        """Not from anywhere else. `rule_version` in particular is looked up in the
        lockfile rather than recomputed, so a row and the rules PR that produced it
        name the same digest."""
        run_slice(
            [make_probe("P-000")],
            FakeAgent(reply=GRANTING_REPLY),
            FakeJudge(a_verified_grant()),
        )

        (row,) = store.rows()
        assert row.policy_doc == runner_policy.doc_slug
        assert row.policy_version == runner_policy.policy_version
        assert row.rule_id == "R-014-a"
        assert row.rule_version == rules_lock.version_of("R-014-a")
        assert row.rule_version.startswith("sha256:")
        assert row.agent_id == "aut-naive"
        assert row.agent_commit_sha == FROZEN_SHA
        assert row.agent_model == "qwen2.5:7b-instruct"
        assert row.agent_backend == "ollama"
        assert row.git_sha


class TestTheAbstainedRow:
    """Commitment C2, end to end: a fabricated quote must not become a number.

    This is the test the whole design is arranged around. The judge asserts a
    quotation, Python checks it against the clause it names, and a quote that is not
    there is void - retried once with the violation named, then abstained. The
    row that results is excluded from the headline and counted in the published
    abstain rate.
    """

    def test_a_quote_that_is_not_in_the_clause_never_reaches_the_headline(
        self, run_slice, make_probe, store
    ):
        judge = FakeJudge(a_fabricated_quote(), a_fabricated_quote())

        result = run_slice(
            [make_probe("P-000")], FakeAgent(reply=GRANTING_REPLY), judge
        )

        (row,) = store.rows()
        assert row.judge_abstained is True
        assert row.verdict_class is None
        assert row.agent_stance is None
        assert result.over_promises == 0
        # The agent really did grant, and the harness really cannot evidence it.
        # Publishing that as an over-promise is the easiest way to inflate the one
        # number the pitch is about.
        assert row.agent_response == GRANTING_REPLY

    def test_the_rejected_judgment_is_not_persisted(
        self, run_slice, make_probe, store
    ):
        """L2 keeps it for the human review queue; the row does not.

        DESIGN.md 5.1 has no field for a rejected judgment, and Step 6 made adding a
        39th field a test failure rather than a quiet drift. So the fabricated span
        stays in the log and out of the audit table, where a reader could mistake it
        for a verified quotation.
        """
        judge = FakeJudge(a_fabricated_quote(), a_fabricated_quote())

        run_slice([make_probe("P-000")], FakeAgent(reply=GRANTING_REPLY), judge)

        (row,) = store.rows()
        assert row.quoted_span is None
        assert row.cited_clause_id is None
        assert row.judge_confidence is None
        # False would mean "checked and rejected" on a row that has no span to
        # attach that to; None means no span is recorded at all.
        assert row.span_verified is None

    def test_it_is_an_abstention_and_not_an_error(
        self, run_slice, make_probe, store
    ):
        """The distinction `AuditRow._an_error_is_not_an_abstention` exists for.

        The abstain rate is published, so it may mean only one kind of thing: the
        judge was asked and what came back could not be believed. Not a 429, not a
        bad key.
        """
        judge = FakeJudge(a_fabricated_quote(), a_fabricated_quote())

        result = run_slice(
            [make_probe("P-000")], FakeAgent(reply=GRANTING_REPLY), judge
        )

        (row,) = store.rows()
        assert row.judge_error is None
        assert row.judge_k == 1
        assert row.judge_completions == 2
        assert result.reconciliation.abstained == 1
        assert result.reconciliation.errored == 0
        assert result.reconciliation.abstain_rate == 1.0

    def test_the_retry_names_the_violation_at_the_same_temperature(
        self, run_slice, make_probe
    ):
        """DESIGN.md 4.1: "one retry with the violation named".

        Named, because a judge asked vaguely to try again is a coin flip. And at the
        same temperature, because the prompt changed - the retry is a correction,
        not a resample, and resampling is L3's job.
        """
        judge = FakeJudge(a_fabricated_quote(), a_fabricated_quote())

        run_slice([make_probe("P-000")], FakeAgent(reply=GRANTING_REPLY), judge)

        assert len(judge.calls) == 2
        retry = str(judge.calls[1]["user"])
        assert "YOUR PREVIOUS ANSWER WAS REJECTED" in retry
        assert "quoted_span was not found verbatim" in retry
        assert judge.calls[0]["temperature"] == judge.calls[1]["temperature"] == 0.0


# ==========================================================================
# Commitment C1 - the judge is not the ground truth
# ==========================================================================
class TestTheJudgeNeverSeesTheAnswerKey:
    """DESIGN.md 0 C1: "This is what stops the whole thing being circular."

    The runner does not compute ground truth. It copies
    `probe.expected_policy_stance` into the row, and that field was derived by
    `evaluate_rules()` in Python when the probe was authored. What is checked here
    is the other half of the same commitment: that nothing the runner sends the
    judge would let it infer the label. A judge shown the fact vector could evaluate
    the rule itself, and the two "independent" numbers would stop being independent.
    """

    def test_the_scenario_facts_are_not_in_the_judge_prompt(
        self, run_slice, make_probe
    ):
        judge = FakeJudge(a_verified_grant())
        probe = make_probe(
            "P-000", facts={"days_since_delivery": 31, "item_category": "footwear"}
        )

        run_slice([probe], FakeAgent(reply=GRANTING_REPLY), judge)

        sent = str(judge.calls[0]["user"]) + str(judge.calls[0]["system"])
        assert "days_since_delivery" not in sent
        # `footwear` is asserted rather than `31` because it appears nowhere else in
        # this fixture set - a clause about a 30-day window makes a bare number a
        # weak witness, and a test that fails for the wrong reason is worse than
        # one that never fails.
        assert "footwear" not in sent

    def test_the_target_rule_id_is_not_in_the_judge_prompt(
        self, run_slice, make_probe
    ):
        """The rule id would point the judge at which condition it is supposed to
        find broken, which is a different question from "what did this response
        commit to" - and the second is the only one the judge is asked."""
        judge = FakeJudge(a_verified_grant())

        run_slice([make_probe("P-000")], FakeAgent(reply=GRANTING_REPLY), judge)

        sent = str(judge.calls[0]["user"]) + str(judge.calls[0]["system"])
        assert "R-014-a" not in sent

    def test_the_row_still_records_the_label(self, run_slice, make_probe, store):
        """Withheld from the judge, present on the row. Both are required: the label
        is what the verdict is scored against, and it has to be auditable."""
        run_slice(
            [make_probe("P-000")],
            FakeAgent(reply=GRANTING_REPLY),
            FakeJudge(a_verified_grant()),
        )

        (row,) = store.rows()
        assert row.expected_policy_stance == "denies"
        assert row.rule_id == "R-014-a"
        assert row.scenario_facts["days_since_delivery"] == 31


# ==========================================================================
# The whole run
# ==========================================================================
class TestAMixedRunAddsUp:
    """All four row shapes in one run, and the identity that must hold over them.

    `attempted == scorable + abstained + errored` is the check that a row is not
    quietly in two states at once. It is asserted rather than assumed because the
    denominator of the abstain rate is printed under the headline number, and a
    denominator nobody chose is how a plausible-looking metric goes wrong.
    """

    @pytest.fixture
    def mixed(self, run_slice, make_probe):
        probes = [make_probe(f"P-{i:03d}") for i in range(4)]
        agent = FakeAgent(
            replies={
                "message for P-000": DENYING_REPLY,  # L0 terminates
                "message for P-001": GRANTING_REPLY,  # verified over-promise
                "message for P-002": GRANTING_REPLY,  # fabricated quote, abstains
                "message for P-003": GRANTING_REPLY,  # transport failure
            }
        )
        judge = FakeJudge(
            a_verified_grant(),
            a_fabricated_quote(),
            a_fabricated_quote(),
            JudgeError("429 rate limited"),
        )
        return run_slice(probes, agent, judge)

    def test_the_counts_reconcile(self, mixed):
        reconciliation = mixed.reconciliation
        reconciliation.assert_consistent()

        assert reconciliation.attempted == 4
        assert reconciliation.scorable == 2  # the L0 denial and the over-promise
        assert reconciliation.abstained == 1
        assert reconciliation.errored == 1
        assert reconciliation.superseded == 0

    def test_the_abstain_rate_excludes_the_error_from_its_denominator(self, mixed):
        """DESIGN.md 4.2 publishes the abstain rate, so it may mean exactly one
        thing. Errors are reported beside it, never inside it - otherwise a bad API
        key would look like judicial humility."""
        assert mixed.reconciliation.abstain_rate == pytest.approx(1 / 3)
        assert mixed.reconciliation.error_rate == pytest.approx(1 / 4)

    def test_only_the_evidenced_grant_is_counted(self, mixed):
        assert mixed.over_promises == 1
        assert mixed.reconciliation.over_promises == 1
        assert mixed.under_serves == 0

    def test_rows_are_persisted_in_probe_order_under_one_run_id(self, mixed, store):
        rows = store.rows(mixed.run_id)

        assert [row.probe_id for row in rows] == ["P-000", "P-001", "P-002", "P-003"]
        assert {row.run_id for row in rows} == {mixed.run_id}
        assert store.run_ids() == [mixed.run_id]

    def test_the_result_and_the_store_agree(self, mixed, store):
        """`RunResult.rows` is what the console table renders and the store is what
        the dashboard reads. A run where those two disagreed would be a demo that
        contradicts its own audit trail."""
        assert [row.row_id for row in mixed.rows] == [
            row.row_id for row in store.rows(mixed.run_id)
        ]

    def test_the_timing_is_reported_as_two_phases(self, mixed):
        """Which phase was slow is the interesting fact about a slow run, and at
        16.5s of pacing the answer is always the judge. Only the identity is
        asserted here - a test that asserted a duration would be asserting how fast
        the machine running it happens to be."""
        assert mixed.elapsed_seconds == mixed.agent_seconds + mixed.judge_seconds
        assert mixed.agent_seconds >= 0.0
        assert mixed.judge_seconds >= 0.0

    def test_a_frozen_run_carries_no_warnings(self, mixed):
        assert mixed.warnings == ()
        assert mixed.identity.is_frozen is True


class TestRunIdentification:
    def test_an_explicit_run_id_is_honoured(self, run_slice, make_probe, store):
        """The CLI needs this so that a gate run can be pointed at by whatever
        invoked it, rather than found by guessing at timestamps."""
        chosen = "0192f3a1-0000-7000-8000-0000000000ff"

        result = run_slice(
            [make_probe("P-000")],
            FakeAgent(reply=DENYING_REPLY),
            ExplodingJudge(),
            run_id=chosen,
        )

        assert result.run_id == chosen
        assert store.rows(chosen)[0].run_id == chosen

    def test_gate_run_is_recorded_on_every_row(self, run_slice, make_probe, store):
        """DESIGN.md 5.1's `gate_run`. A number produced by a nightly sweep and one
        produced by a PR gate are different claims, and the row says which."""
        run_slice(
            [make_probe("P-000")],
            FakeAgent(reply=DENYING_REPLY),
            ExplodingJudge(),
            gate_run=True,
        )

        (row,) = store.rows()
        assert row.gate_run is True

    def test_the_agent_phase_callback_reports_the_count(
        self, run_slice, make_probe
    ):
        seen: list[int] = []

        run_slice(
            [make_probe(f"P-{i:03d}") for i in range(3)],
            FakeAgent(reply=DENYING_REPLY),
            ExplodingJudge(),
            on_agent_done=seen.append,
        )

        assert seen == [3]






