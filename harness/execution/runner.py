"""`clauseguard run` - probes in, audit rows out (DESIGN.md 2 steps 6-10).

    | ⑥ Async `httpx` fan-out to the AUT, semaphore of 8, per-probe fresh
    | `session_id` (except multi-turn probes, which reuse)
    | ⑩ One row per probe (§5). Immutable.

TWO PHASES, NOT ONE PIPELINE
----------------------------
Every probe is sent to the agent first, and only then is anything judged. The
alternative - one task per probe doing agent-then-judge - is shorter code and the
wrong shape here, for a measured reason.

The AUT is local and parallel: eight concurrent calls cost nothing but RAM. The
judge is a hosted model on Groq's `on_demand` tier, capped at 8000 tokens per
minute, and eight concurrent judge calls were measured failing 6 for 6 with
`RateLimitError`. So the two stages want opposite concurrency, and interleaving
them would force the slower one's pacing onto the faster one.

Splitting them buys something else worth more than the speed: if the judge dies
half way through - rate limit, expired key, model retired - the agent responses
are already in hand. They are the expensive, non-reproducible part (a frozen
agent at temperature 0 is still not deterministic; see docs/limitations.md), and
losing thirty of them to a 401 would mean re-running the AUT to recover data that
was already collected.

AGENT FAILURES ABORT; JUDGE FAILURES PERSIST
--------------------------------------------
The asymmetry is deliberate and it is not covered by DESIGN.md, so it is stated
here rather than left in the code to be inferred.

A judge failure becomes a row carrying `judge_error` (Step 6's `reconcile()`
exists for exactly this) because the call is remote, expensive, and already spent
- dropping the row would move the denominator of a published number with nothing
recording that it had.

One class of judge failure no longer arrives here at all: a persistent
`tool_use_failed` now returns an abstention instead of raising, so it lands in
the abstain rate rather than the error rate. That was a decision, made after a
live run put two `expected=denies` rows in `judge_error` and out of every cell -
see "WHAT THE ABSTAIN RATE IS ALLOWED TO MEAN" in `harness/judge/judge.py`.

An agent failure aborts the run before anything is written. Nothing is lost by
doing so: phase one is local, fast, and re-runnable, and no row has been
committed yet. The alternative would be a row with an empty `agent_response`,
which the L0 pre-filter scores `evasive` - a transport outage would enter the
confusion matrix as agent behaviour. DESIGN.md 5.1 has no `agent_error` field to
put it in instead, and inventing a 39th field to record an outage is worse than
refusing to report a run that did not happen.

C3 IS CHECKED PER RESPONSE, NOT ONCE
------------------------------------
`aut-naive` repeats its freeze identity on every `/chat` reply specifically so the
harness cannot attribute thirty rows to an identity it read once at start-up. So
`/health` is read first and every reply's `aut_commit_sha` is compared against it;
a mismatch aborts. A container swapped mid-run is the failure this catches, and it
is the only link between the frozen tag and the audit row that the harness can
verify for itself.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol, runtime_checkable
from uuid import uuid4

from harness.audit import (
    AuditRow,
    AuditStore,
    Reconciliation,
    VerdictClass,
    classify_verdict,
    new_row_id,
    new_run_id,
    utc_now_iso,
)
from harness.execution.grounding import assert_spans_grounded
from harness.execution.lockfiles import RulesLock
from harness.judge.consistency import (
    L3_TEMPERATURE,
    ConsistencyError,
    ConsistencyResult,
    apply_consistency,
)
from harness.judge.judge import (
    JudgeError,
    JudgeOutcome,
    judge_response,
    resolve_judge_model,
    resolve_judge_temp,
)
from harness.schemas.clause import Clause, PolicyDocument
from harness.schemas.probe import Probe

#: DESIGN.md 2 step 6, verbatim: "semaphore of 8".
DEFAULT_AGENT_CONCURRENCY = 8

#: Seconds between judge calls. Groq's `on_demand` tier caps
#: `openai/gpt-oss-20b` at 8000 tokens/minute, which measured out at 5-6 calls
#: per minute; 16.5s is the pacing that survived a full sequential sweep without
#: a single 429. Not a guess and not a safety margin pulled from the air - see
#: scripts/time_judge.py and docs/limitations.md entry 2.
DEFAULT_JUDGE_PACE_S = 16.5

#: Generous, because a cold `sentence-transformers` index plus a 7B local
#: generation is slow on first call and a timeout here aborts the whole run.
DEFAULT_AGENT_TIMEOUT_S = 180.0

#: One retry for the agent only. Cheap and local, unlike the judge, where a retry
#: is a paid call against a token ceiling and is already spent inside L1/L2.
AGENT_ATTEMPTS = 2

#: What `judge_model` records when no model ran. DESIGN.md 4.1's L0 is a
#: deterministic lexicon, so naming a hosted model on an L0 row would credit a
#: verdict to something that was never called.
L0_JUDGE_MODEL = "deterministic-prefilter-L0"

#: `git_sha` is `min_length=1`, so a non-checkout needs a value that reads as an
#: absence rather than as a plausible sha.
UNKNOWN_GIT_SHA = "(unknown: not a git checkout)"


class RunError(RuntimeError):
    """Base: the run cannot proceed or cannot be trusted."""


class AgentUnavailableError(RunError):
    """The agent under test did not answer. See the module docstring."""


class FrozenAgentMismatchError(RunError):
    """A reply's freeze identity disagrees with `/health`. Commitment C3."""


class UnresolvedClauseError(RunError):
    """A probe cites a clause the loaded policy does not contain."""


# ---------------------------------------------------------------------------
# The agent under test
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentIdentity:
    """What the container at the other end claims to be (`GET /health`)."""

    aut_name: str
    aut_commit_sha: str
    aut_repo_head: str
    aut_git_tag: str
    aut_frozen_at: str

    @property
    def is_frozen(self) -> bool:
        """False when the image was built outside `scripts/freeze_aut.py`.

        `aut-naive` reports the string "(unfrozen: built outside
        scripts/freeze_aut.py)" rather than a blank in that case, which is what
        makes this checkable at all.
        """
        return not self.aut_commit_sha.startswith("(unfrozen")


@dataclass(frozen=True)
class AgentReply:
    """One `POST /chat` response."""

    reply: str
    latency_ms: int
    model: str
    backend: str
    aut_commit_sha: str
    session_id: str
    turn: int


@runtime_checkable
class AgentClient(Protocol):
    """The seam the offline test suite substitutes at.

    Mirrors `JudgeClient` in harness/judge/judge.py on purpose: the judge already
    proved that a one-method protocol is enough to run the whole pipeline in CI
    with no network, and a second seam shaped differently would be a second thing
    to learn.
    """

    async def health(self) -> AgentIdentity: ...

    async def chat(self, *, session_id: str, message: str) -> AgentReply: ...


class HttpxAgentClient:
    """`AgentClient` over HTTP, per DESIGN.md 1.4's `POST /chat`."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = DEFAULT_AGENT_TIMEOUT_S,
        client: object | None = None,
    ) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout_s
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()  # type: ignore[attr-defined]

    async def health(self) -> AgentIdentity:
        response = await self._client.get("/health")  # type: ignore[attr-defined]
        response.raise_for_status()
        payload = response.json()
        return AgentIdentity(
            aut_name=str(payload.get("aut_name", "(unnamed)")),
            aut_commit_sha=str(payload.get("aut_commit_sha", "(absent)")),
            aut_repo_head=str(payload.get("aut_repo_head", "(absent)")),
            aut_git_tag=str(payload.get("aut_git_tag", "(absent)")),
            aut_frozen_at=str(payload.get("aut_frozen_at", "(absent)")),
        )

    async def chat(self, *, session_id: str, message: str) -> AgentReply:
        response = await self._client.post(  # type: ignore[attr-defined]
            "/chat", json={"session_id": session_id, "message": message}
        )
        response.raise_for_status()
        payload = response.json()
        return AgentReply(
            reply=payload["reply"],
            latency_ms=int(payload["latency_ms"]),
            model=str(payload["model"]),
            backend=str(payload["backend"]),
            aut_commit_sha=str(payload["aut_commit_sha"]),
            session_id=str(payload["session_id"]),
            turn=int(payload["turn"]),
        )


# ---------------------------------------------------------------------------
# Phase 1 - fan out to the agent
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Exchange:
    """What one probe got back. The final reply is what gets judged.

    Multi-turn probes keep every reply in `replies` but only the last is scored.
    DESIGN.md 3.2 strategy 7 is explicit about why: the drift probe works by
    turn 1 being in policy and turn 2 shifting a fact out of it, so the
    conformance failure - if there is one - lives in the last answer. DESIGN.md
    5.1 has one `agent_response` field, so the earlier replies are visible to a
    human reading the run but are not persisted. That is a real limitation and
    it is recorded in docs/limitations.md rather than smoothed over here.
    """

    probe: Probe
    session_id: str
    replies: tuple[AgentReply, ...]

    @property
    def final(self) -> AgentReply:
        return self.replies[-1]

    @property
    def total_latency_ms(self) -> int:
        """Summed across turns.

        For a single-turn probe this is the reply's own latency. For a drift
        probe it is the cost of the whole conversation, which is the honest
        answer to "how long did this probe take" and the number a 45-second
        budget has to be measured against.
        """
        return sum(reply.latency_ms for reply in self.replies)


def new_session_id(probe: Probe) -> str:
    """A fresh session per probe (DESIGN.md 2 step 6).

    The probe id is embedded so that a session visible in the agent's logs can be
    traced back to the row it produced. The random suffix is what makes it fresh:
    re-running the same probe against a long-lived container must not inherit the
    earlier conversation, or a second run of a drift probe would start from the
    first run's conclusion.
    """
    return f"cg-{probe.probe_id}-{uuid4().hex[:12]}"


async def collect_exchanges(
    probes: Sequence[Probe],
    client: AgentClient,
    *,
    concurrency: int = DEFAULT_AGENT_CONCURRENCY,
    expect_commit_sha: str | None = None,
) -> list[Exchange]:
    """Send every probe to the agent, `concurrency` at a time.

    Returns exchanges in the order the probes were given, not the order they
    completed, so that two runs over the same lockfile produce rows in the same
    order and a diff of two runs is readable.

    Raises `AgentUnavailableError` or `FrozenAgentMismatchError` - the two failures
    this module documents - and not the `ExceptionGroup` anyio would otherwise
    deliver them in. See `_collapse_task_group_error`.
    """
    import anyio

    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")

    results: list[Exchange | None] = [None] * len(probes)
    limiter = anyio.Semaphore(concurrency)

    async def run_one(index: int, probe: Probe) -> None:
        async with limiter:
            results[index] = await _exchange_one(
                probe, client, expect_commit_sha=expect_commit_sha
            )

    try:
        async with anyio.create_task_group() as tasks:
            for index, probe in enumerate(probes):
                tasks.start_soon(run_one, index, probe)
    except BaseExceptionGroup as group:
        primary = _collapse_task_group_error(group)
        if primary is not group:
            # The group is the transport, not information: it contains `primary`
            # itself, so leaving it as the context prints the same failure twice.
            # `__cause__` is deliberately untouched - on an
            # `AgentUnavailableError` it holds the ConnectionError that caused it,
            # which is the one frame an operator actually needs.
            primary.__suppress_context__ = True
        raise primary

    # The task group cancelled its siblings on the first failure, so reaching here
    # means every slot was filled. Asserted rather than assumed because a None
    # would become an AttributeError several frames later.
    missing = [i for i, item in enumerate(results) if item is None]
    if missing:
        raise AgentUnavailableError(
            f"no reply recorded for probe index(es) {missing}, which should be "
            f"unreachable once the task group has exited cleanly"
        )
    return [item for item in results if item is not None]


def _collapse_task_group_error(group: BaseExceptionGroup) -> BaseException:
    """The one failure worth re-raising out of a task group's `ExceptionGroup`.

    anyio 4 wraps *every* child failure in a `BaseExceptionGroup`, including a lone
    one - `_backends/_asyncio.py` raises `BaseExceptionGroup("unhandled errors in a
    TaskGroup", self._exceptions)` whenever that list is non-empty. Read out of the
    installed anyio 4.14.2 rather than assumed, because anyio 3 re-raised a single
    exception unwrapped and this file's fan-out was written against that behaviour.

    Left as a group, the two failures in this module's docstring would be
    uncatchable by the code that has to act on them: `except AgentUnavailableError`
    would not match, so the CLI would print a raw traceback where it means to print
    one line, and Step 8's gate would need `except*` to see a verdict it is
    supposed to turn into an exit code. Collapsing here keeps that contract at the
    seam that states it.

    A `RunError` is preferred over any other child, because the first failure
    cancels its siblings mid-await and the group can therefore carry the cause
    alongside whatever the cancellation broke. Extra outright failures are attached
    as notes rather than dropped: eight concurrent probes against a container that
    just died fail together, and "one of these eight" is a worse answer than all of
    them.
    """
    flat: list[BaseException] = []
    queue: list[BaseException] = list(group.exceptions)
    while queue:
        exc = queue.pop(0)
        if isinstance(exc, BaseExceptionGroup):
            queue[:0] = list(exc.exceptions)
        else:
            flat.append(exc)

    if not flat:
        return group

    primary = next((exc for exc in flat if isinstance(exc, RunError)), flat[0])
    for other in flat:
        if other is not primary:
            primary.add_note(f"also failed: {type(other).__name__}: {other}")
    return primary



async def _exchange_one(
    probe: Probe, client: AgentClient, *, expect_commit_sha: str | None
) -> Exchange:
    session_id = new_session_id(probe)
    replies: list[AgentReply] = []

    for turn_index, message in enumerate(probe.turns, start=1):
        reply = await _chat_with_retry(
            client, session_id=session_id, message=message, probe=probe, turn=turn_index
        )
        if expect_commit_sha is not None and reply.aut_commit_sha != expect_commit_sha:
            raise FrozenAgentMismatchError(
                f"probe {probe.probe_id!r} turn {turn_index}: the reply reports "
                f"aut_commit_sha {reply.aut_commit_sha!r} but /health reported "
                f"{expect_commit_sha!r}. The agent under test changed during the "
                f"run, so no row from it can be attributed to a frozen commit "
                f"(commitment C3)"
            )
        replies.append(reply)

    return Exchange(probe=probe, session_id=session_id, replies=tuple(replies))


async def _chat_with_retry(
    client: AgentClient,
    *,
    session_id: str,
    message: str,
    probe: Probe,
    turn: int,
) -> AgentReply:
    last: Exception | None = None
    for attempt in range(1, AGENT_ATTEMPTS + 1):
        try:
            return await client.chat(session_id=session_id, message=message)
        except Exception as exc:  # noqa: BLE001 - re-raised below with context
            last = exc
            if attempt == AGENT_ATTEMPTS:
                break
    raise AgentUnavailableError(
        f"probe {probe.probe_id!r} turn {turn}: the agent did not answer after "
        f"{AGENT_ATTEMPTS} attempts ({type(last).__name__}: {last}). The run is "
        f"abandoned before anything is written - see harness/execution/runner.py"
    ) from last


async def agent_phase(
    probes: Sequence[Probe],
    client: AgentClient,
    *,
    concurrency: int = DEFAULT_AGENT_CONCURRENCY,
    require_frozen: bool = False,
) -> tuple[AgentIdentity, list[Exchange], list[str]]:
    """All of phase one inside one event loop: `/health`, then the fan-out.

    Deliberately one coroutine rather than two calls from sync code. `httpx`'s
    connection pool binds to the loop it is first awaited on, so reading `/health`
    under one `anyio.run` and then fanning out under a second would hand the same
    client to a closed loop - which fails as a hang or an "Event loop is closed",
    neither of which reads like the cause. One loop, one client, one phase.

    Returns the identity as well as the exchanges because the identity is what the
    rows are attributed to, and it must be the identity that was live *for this
    fan-out* rather than one read at some other point in the process.
    """
    identity = await client.health()

    warnings: list[str] = []
    if not identity.is_frozen:
        message = (
            f"the agent under test reports {identity.aut_commit_sha!r}, so it was "
            f"not built by scripts/freeze_aut.py. Commitment C3 - 'frozen by "
            f"commit SHA before any probe exists' - is unverifiable for this run, "
            f"and its rows cannot be attributed to a specific agent"
        )
        if require_frozen:
            raise FrozenAgentMismatchError(message)
        warnings.append(message)

    # Nothing to compare replies against when the build was never frozen: the
    # sentinel is a constant, so every reply would trivially agree with it and the
    # check would report a pass it did not earn.
    expect = identity.aut_commit_sha if identity.is_frozen else None
    exchanges = await collect_exchanges(
        probes, client, concurrency=concurrency, expect_commit_sha=expect
    )
    return identity, exchanges, warnings


# ---------------------------------------------------------------------------
# Phase 2 - judge, paced
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Judged:
    """One exchange plus its verdict, or plus the reason there isn't one."""

    exchange: Exchange
    outcome: JudgeOutcome | None = None
    error: str | None = None
    #: Present when L3 was *considered*, which is every non-errored row. Carries
    #: `applied=False` for the k=1 majority of them. Kept alongside `outcome`
    #: rather than folded into it because `JudgeOutcome` has no room for the votes
    #: and DESIGN.md 5.1's row has no field for them either, so the only place the
    #: tally survives is here, in memory, long enough for the CLI to report it.
    consistency: ConsistencyResult | None = None
    #: The temperature a *failed* judge call was made at, when it is not the run's.
    #: Only L3 sets it, and only via `ConsistencyError`. Separate from `consistency`
    #: because a raised L3 produces no `ConsistencyResult` to hang it on - which is
    #: precisely why the errored row used to record the run's L1 temperature as
    #: though the first pass had been the last thing tried.
    error_temperature: float | None = None

    @property
    def used_llm(self) -> bool:
        return self.outcome is not None and self.outcome.used_llm

    @property
    def l3_applied(self) -> bool:
        return self.consistency is not None and self.consistency.applied


def clause_index(policy: PolicyDocument) -> dict[str, Clause]:
    return {clause.clause_id: clause for clause in policy.clauses}


def assert_clauses_resolve(
    probes: Sequence[Probe], clauses: Mapping[str, Clause]
) -> None:
    """Fail before phase 1 if any probe cites a clause that is not loaded.

    Checked up front because `judge_response` raises `JudgeError` when handed no
    candidate clauses, and discovering that after eight minutes of paced judge
    calls - one probe at a time - is an expensive way to learn that a lockfile is
    stale.
    """
    problems: list[str] = []
    for probe in probes:
        missing = [cid for cid in probe.clause_ids if cid not in clauses]
        if missing:
            problems.append(f"{probe.probe_id}: {', '.join(missing)}")
    if problems:
        raise UnresolvedClauseError(
            "these probes cite clauses the loaded policy does not contain, which "
            "means the probe set was built against different clause text:\n  "
            + "\n  ".join(problems)
        )


class _Pacer:
    """One rate-limit budget shared by every paid judge call in a run.

    Extracted from `judge_exchanges`'s loop when L3 arrived, and for a concrete
    reason rather than tidiness. The old arithmetic paced *between probes*, which
    was complete while one probe meant one or two calls. An L3 row is up to four,
    and three consecutive samples fired with no wait between them would hit the
    8000 tokens/minute ceiling inside a single iteration - the exact thing
    `DEFAULT_JUDGE_PACE_S` exists to prevent. Routing the samples through the same
    debt is what keeps one pace for one provider.

    Debt is booked after a call and paid before the next, so a run never sleeps
    after its final call. `charge(0)` is deliberately free: an L0 termination spent
    no tokens, and pacing after it would add 16.5 seconds per deterministic verdict
    to a run whose whole point is finishing inside DESIGN.md 2's budget.
    """

    def __init__(self, pace_s: float, sleep: Callable[[float], None]) -> None:
        self._pace_s = pace_s
        self._sleep = sleep
        self._pending = 0.0

    def wait(self) -> None:
        if self._pending > 0:
            self._sleep(self._pending)
            self._pending = 0.0

    def charge(self, completions: int) -> None:
        if completions > 0:
            self._pending = self._pace_s * completions


def judge_exchanges(
    exchanges: Sequence[Exchange],
    *,
    clauses: Mapping[str, Clause],
    client: object | None = None,
    temperature: float | None = None,
    pace_s: float = DEFAULT_JUDGE_PACE_S,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[int, int, Judged], None] | None = None,
    consistency: bool = True,
    gold_probe_ids: Sequence[str] = (),
) -> list[Judged]:
    """Judge every exchange in order, pacing only the calls that hit the model.

    `sleep` is injected so the offline suite can assert the pacing arithmetic
    without waiting eight minutes for it.

    Pacing skips L0 terminations, because a deterministic pre-filter verdict
    spends no tokens. It is applied *between* model-touching calls and scaled by
    the previous call's completion count, since L1's retry is a second paid call
    against the same per-minute ceiling. That scaling is conservative rather than
    exact - the true budget is tokens, not calls - and the honest limit is that a
    long response can still exceed 8000 tokens/minute on its own. Task #33 tracks
    the token-aware limiter that would fix it.

    THIS IS WHERE THE GROUND-TRUTH LABEL ENTERS THE JUDGE PIPELINE
    -------------------------------------------------------------
    `judge_response` never sees `expected_policy_stance`; this function does,
    because it holds the `Probe`. DESIGN.md 4.1's L3 needs the label to find the
    over-promise cell, so the escalation decision is taken here and the label is
    handed to `apply_consistency`, which spends compute and cannot write a prompt.
    Keeping those two capabilities apart is what stops the answer leaking into the
    question - see `harness/judge/consistency.py`'s opening section.

    `consistency=False` disables L3 for a whole run. It exists because L3 roughly
    quadruples the paid calls on an over-promise, and at 16.5s of pacing each, a
    probe set with a dozen of them adds around nine minutes - DESIGN.md 4.1
    predicts exactly this ("triple cost and latency"). Turning it off buys a
    *cheaper and weaker* measurement, never a faster equivalent one, and a run that
    does so says so on every row, because `judge_k` stays 1.
    """
    results: list[Judged] = []
    pacer = _Pacer(pace_s, sleep)
    gold = tuple(gold_probe_ids)

    for position, exchange in enumerate(exchanges, start=1):
        candidates = [
            clauses[cid] for cid in exchange.probe.clause_ids if cid in clauses
        ]

        def ask(
            temp: float | None,
            _ex: Exchange = exchange,
            _cands: Sequence[Clause] = candidates,
        ) -> JudgeOutcome:
            """One paid judge call, paced. Also L3's sampler for this exchange.

            The exchange and its candidate clauses are bound as default arguments
            rather than captured, because a closure over the loop variable would
            make every sampler read the *last* exchange - and that bug is invisible
            offline, where a fake judge answers the same way whatever it is asked.
            """
            pacer.wait()
            outcome = judge_response(
                probe_turns=_ex.probe.turns,
                agent_response=_ex.final.reply,
                candidate_clauses=list(_cands),
                client=client,  # type: ignore[arg-type]
                temperature=temp,
            )
            pacer.charge(outcome.judge_completions)
            return outcome

        try:
            first_pass = ask(temperature)
            if consistency:
                resolved = apply_consistency(
                    first_pass,
                    sample=ask,
                    expected_policy_stance=exchange.probe.expected_policy_stance,
                    probe_id=exchange.probe.probe_id,
                    gold_probe_ids=gold,
                )
            else:
                resolved = ConsistencyResult(outcome=first_pass, applied=False)
            judged = Judged(
                exchange=exchange, outcome=resolved.outcome, consistency=resolved
            )
        except JudgeError as exc:
            # Deliberately not caught wider than JudgeError. A bug in row
            # assembly must not be recorded as a judge failure and then read as
            # "the backend was flaky" on the report.
            #
            # An L3 failure lands here too, and it takes the first pass down with
            # it. That is the intended cost of the ruling that a withdrawn verdict
            # may not fall back on the sample that produced it: once resampling has
            # failed to confirm a stance, recording that stance anyway would put a
            # number in the confusion matrix the consistency layer just declined to
            # stand behind.
            judged = Judged(
                exchange=exchange,
                error=f"{type(exc).__name__}: {exc}",
                error_temperature=(
                    exc.temperature if isinstance(exc, ConsistencyError) else None
                ),
            )
            # A transport failure still consumed an attempt on the provider's
            # side, so pace after it too - retrying a rate limit at full speed is
            # how a run turns one 429 into thirty.
            pacer.charge(1)

        results.append(judged)
        if on_progress is not None:
            on_progress(position, len(exchanges), judged)

    return results


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------
def harness_git_sha() -> str:
    """The harness commit, for `git_sha` (DESIGN.md 5.1).

    Distinct from `agent_commit_sha`: one says which harness produced the number,
    the other which agent produced the behaviour, and conflating them would make
    "we changed the judge prompt" indistinguishable from "we changed the agent".
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_GIT_SHA
    sha = completed.stdout.strip()
    return sha if completed.returncode == 0 and sha else UNKNOWN_GIT_SHA


@dataclass(frozen=True)
class RowContext:
    """Everything a row needs that is not specific to one probe."""

    run_id: str
    policy: PolicyDocument
    rule_versions: Mapping[str, str]
    agent_id: str
    judge_model_when_absent: str
    judge_temperature: float | None
    git_sha: str
    gate_run: bool = False


def build_row(judged: Judged, context: RowContext) -> AuditRow:
    """Assemble one DESIGN.md 5.1 row from one judged exchange.

    Three shapes, and which fields are present is what distinguishes them - see
    `AuditRow._an_error_is_not_an_abstention`:

    * judged        - the full verdict block, `verdict_class` re-derived from the
                      two stances by `classify_verdict`;
    * abstained     - no verdict block at all, `judge_abstained=True`;
    * errored       - no verdict block at all, `judge_error` set.

    The abstained case discards the rejected judgment even though L2 kept it for
    the human review queue. There is no field for it in 5.1, and Step 6 made
    adding a 39th field a test failure rather than a quiet drift, so it stays out
    of the row and stays in the log. The same absence covers the second abstention
    cause - a provider that never forwarded a tool call - where there is no
    rejected judgment to discard in the first place, and where the row is
    consequently identical to an L2 abstention. Which of the two applied is a log
    question, not a row question.
    """
    probe = judged.exchange.probe
    reply = judged.exchange.final
    outcome = judged.outcome

    # An L3 row's verdict was decided by samples at `L3_TEMPERATURE`, not by the
    # run-wide L1 temperature, so recording the latter would be a false provenance
    # note on the rows most likely to be argued over. `RowContext` carries one
    # temperature because it is one per run for L1; L3 is the only thing that
    # varies it, and it varies it to a constant this module can read.
    #
    # The third arm is a row L3 *failed* on. It has no `ConsistencyResult`, so
    # `l3_applied` is False and it would otherwise record the run's 0.0 - reading as
    # though the first pass were the last call made, when in fact three more were
    # attempted at 0.3. `ConsistencyError` carries the temperature for exactly this
    # case; `AuditRow` permits it on an errored row because temperature describes a
    # request, and those requests were made.
    row_temperature = context.judge_temperature
    if judged.l3_applied:
        row_temperature = L3_TEMPERATURE
    elif judged.error_temperature is not None:
        row_temperature = judged.error_temperature

    common = dict(
        row_id=new_row_id(),
        run_id=context.run_id,
        probe_id=probe.probe_id,
        ts=utc_now_iso(),
        policy_doc=context.policy.doc_slug,
        policy_version=context.policy.policy_version,
        clause_ids=list(probe.clause_ids),
        rule_id=probe.scenario.target_rule_id,
        rule_version=context.rule_versions[probe.scenario.target_rule_id],
        strategy=probe.scenario.strategy.value,
        difficulty_tier=probe.scenario.difficulty_tier,
        scenario_facts=dict(probe.scenario.facts),
        probe_turns=list(probe.turns),
        expected_policy_stance=probe.expected_policy_stance,
        agent_id=context.agent_id,
        agent_model=reply.model,
        agent_commit_sha=reply.aut_commit_sha,
        agent_response=reply.reply,
        agent_latency_ms=judged.exchange.total_latency_ms,
        agent_backend=reply.backend,
        gate_run=context.gate_run,
        git_sha=context.git_sha,
    )

    if judged.error is not None:
        return AuditRow(
            **common,
            judge_model=context.judge_model_when_absent,
            judge_k=0,
            judge_agreement=None,
            judge_confidence=None,
            judge_temperature=row_temperature,
            judge_completions=None,
            judge_error=judged.error,
        )

    assert outcome is not None  # error is None, so the judge ran to completion
    judge_model = outcome.judge_model or L0_JUDGE_MODEL

    if outcome.abstained:
        return AuditRow(
            **common,
            judge_model=judge_model,
            judge_k=outcome.judge_k,
            judge_agreement=outcome.judge_agreement,
            judge_confidence=None,
            judge_abstained=True,
            judge_temperature=row_temperature,
            judge_completions=outcome.judge_completions or None,
        )

    assert outcome.agent_stance is not None  # not abstained, so a stance exists
    judgment = outcome.judgment
    asserted = None if judgment is None else judgment.entitlement_asserted
    return AuditRow(
        **common,
        agent_stance=outcome.agent_stance,
        entitlement_asserted=asserted,
        verdict_class=classify_verdict(
            probe.expected_policy_stance, outcome.agent_stance
        ),
        cited_clause_id=None if judgment is None else judgment.cited_clause_id,
        quoted_span=None if judgment is None else judgment.quoted_span,
        response_span=None if judgment is None else judgment.response_span,
        span_verified=outcome.span_verified,
        judge_model=judge_model,
        judge_k=outcome.judge_k,
        judge_agreement=outcome.judge_agreement,
        judge_confidence=outcome.judge_confidence,
        judge_temperature=row_temperature if outcome.used_llm else None,
        judge_completions=outcome.judge_completions or None,
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunResult:
    """What one `clauseguard run` produced. Rendering is the CLI's job."""

    run_id: str
    rows: tuple[AuditRow, ...]
    identity: AgentIdentity
    policy: PolicyDocument
    judged: tuple[Judged, ...]
    agent_seconds: float
    judge_seconds: float
    reconciliation: Reconciliation
    warnings: tuple[str, ...] = ()

    @property
    def elapsed_seconds(self) -> float:
        """Agent phase plus judge phase.

        Not wall clock around the whole call: it excludes lockfile loading and the
        `/health` round trip, so it is the number DESIGN.md 6.2's budget is about.
        Reported as two components as well as a total, because the interesting fact
        about a slow run is almost always which phase was slow - and at 16.5s
        pacing the answer is the judge.
        """
        return self.agent_seconds + self.judge_seconds

    @property
    def over_promises(self) -> int:
        """DESIGN.md 5.2 item 1's one huge number."""
        return sum(1 for row in self.rows if row.is_over_promise)

    @property
    def under_serves(self) -> int:
        return sum(
            1 for row in self.rows if row.verdict_class is VerdictClass.UNDER_SERVE
        )

    @property
    def l3_escalations(self) -> int:
        """How many rows DESIGN.md 4.1's k=3 was spent on."""
        return sum(1 for item in self.judged if item.l3_applied)

    @property
    def l3_withdrawals(self) -> int:
        """How many over-promises L3 took off the headline number.

        Worth reporting next to `over_promises` rather than buried, because it is
        the size of the correction the consistency layer applied to the project's
        own flagship metric. A large number here is not a bug: it means k=1 was
        over-counting, which is the finding. Read it with the module docstring in
        `harness/judge/consistency.py` - L3 cannot make this number negative, so it
        bounds the over-count from one side only.
        """
        return sum(
            1
            for item in self.judged
            if item.consistency is not None
            and item.consistency.left_the_over_promise_cell
        )


def execute_run(
    *,
    probes: Sequence[Probe],
    rules: RulesLock,
    policy: PolicyDocument,
    agent: AgentClient,
    store: AuditStore,
    run_id: str | None = None,
    agent_concurrency: int = DEFAULT_AGENT_CONCURRENCY,
    judge_client: object | None = None,
    judge_temperature: float | None = None,
    judge_pace_s: float = DEFAULT_JUDGE_PACE_S,
    sleep: Callable[[float], None] = time.sleep,
    consistency: bool = True,
    gold_probe_ids: Sequence[str] = (),
    require_frozen: bool = False,
    gate_run: bool = False,
    on_agent_done: Callable[[int], None] | None = None,
    on_judge_progress: Callable[[int, int, Judged], None] | None = None,
) -> RunResult:
    """Run every probe against `agent`, judge the replies, persist the rows.

    Preconditions are checked before the first network call, because each one means
    the inputs disagree with each other - the rules with the policy, or the probes
    with either - and finding that out after eight minutes of paced judging is an
    expensive way to learn it.

    `sleep` and the two progress callbacks are injected so the offline suite can
    drive a full thirty-probe run in milliseconds and still assert on the pacing.

    `consistency` and `gold_probe_ids` are DESIGN.md 4.1's L3 knobs, forwarded
    unchanged to `judge_exchanges` - see its docstring for why turning L3 off is a
    weaker measurement rather than a faster one. `gold_probe_ids` has no caller yet
    because the 200-item gold set is still a stub (`scripts/label_gold.py`); it is a
    parameter now so that the escalation rule lives in one place when it arrives.
    """
    import anyio

    run_id = run_id or new_run_id()
    clauses = clause_index(policy)
    # Four preconditions, all pure. The lockfile check comes first because it is
    # the one that invalidates the rest: if the rules were authored against
    # different clause text, then every `rule_version` a row would carry names a
    # digest computed over a policy nobody is running, and the checks below would
    # be verifying the wrong thing carefully.
    #
    # Span grounding sits second, with the other rules-versus-policy check, before
    # the two probe-level ones. It is the most expensive of the four - a substring
    # scan per condition per cited clause - and still microseconds, so ordering is
    # by what invalidates what rather than by cost.
    rules.assert_matches_policy(policy)
    grounding = assert_spans_grounded(rules.rules, policy, source=str(rules.path))
    assert_clauses_resolve(probes, clauses)
    assert_rules_resolve(probes, rules)

    # Ungrounded-but-flagged spans are allowed through (see `grounding`'s module
    # docstring on why `needs_human_review` is the spec's escape hatch), but they
    # do not get to be quiet about it: every run says so.
    span_warnings = [
        f"rule {failure.rule_id} condition {failure.attribute} has an ungrounded "
        f"source_span {failure.source_span!r}, allowed only because the rule is "
        f"flagged needs_human_review"
        for failure in grounding.flagged
    ]

    agent_started = time.perf_counter()
    identity, exchanges, warnings = anyio.run(
        partial(
            agent_phase,
            probes,
            agent,
            concurrency=agent_concurrency,
            require_frozen=require_frozen,
        )
    )
    agent_seconds = time.perf_counter() - agent_started
    if on_agent_done is not None:
        on_agent_done(len(exchanges))

    # Span-grounding warnings lead: they are about the rules the whole run rests
    # on, so a reader should see them before the per-agent ones. `agent_phase`
    # returns a fresh list, so prepending here does not mutate anything shared.
    warnings = span_warnings + list(warnings)

    judge_started = time.perf_counter()
    # Resolved here rather than left as None. `judge_response(temperature=None)`
    # falls back to `resolve_judge_temp()`, so passing None straight through to
    # the row would record "no temperature" for a call that ran at 0.0 - the row
    # has to name the setting the model was actually given.
    temperature = (
        judge_temperature if judge_temperature is not None else resolve_judge_temp()
    )
    judged = judge_exchanges(
        exchanges,
        clauses=clauses,
        client=judge_client,
        temperature=temperature,
        pace_s=judge_pace_s,
        sleep=sleep,
        on_progress=on_judge_progress,
        consistency=consistency,
        gold_probe_ids=gold_probe_ids,
    )
    judge_seconds = time.perf_counter() - judge_started

    context = RowContext(
        run_id=run_id,
        policy=policy,
        rule_versions=rules.versions,
        agent_id=identity.aut_name,
        judge_model_when_absent=_judge_model_for_errors(judge_client),
        judge_temperature=temperature,
        git_sha=harness_git_sha(),
        gate_run=gate_run,
    )
    rows = tuple(build_row(item, context) for item in judged)

    # One transaction for the whole run. A half-written run would give the
    # reconciliation identity a denominator nobody chose.
    store.append_many(rows)

    return RunResult(
        run_id=run_id,
        rows=rows,
        identity=identity,
        policy=policy,
        judged=tuple(judged),
        agent_seconds=agent_seconds,
        judge_seconds=judge_seconds,
        reconciliation=store.reconcile(run_id),
        warnings=tuple(warnings),
    )


def assert_rules_resolve(probes: Sequence[Probe], rules: RulesLock) -> None:
    """Fail before phase 1 if a probe targets a rule the lockfile does not define.

    `rule_version` is required on every audit row, and it is looked up by
    `target_rule_id`. A probe naming a rule that was renamed or removed cannot
    produce a row at all, so it is caught here rather than as a `KeyError` in the
    middle of assembling one.
    """
    known = rules.rule_ids()
    missing = sorted(
        {p.scenario.target_rule_id for p in probes} - set(known)
    )
    if missing:
        raise UnresolvedClauseError(
            f"these probes target rule ids that {str(rules.path)!r} does not "
            f"define: {', '.join(missing)}. Known ids: {', '.join(sorted(known))}"
        )


def _judge_model_for_errors(judge_client: object | None) -> str:
    """What `judge_model` says on a row whose judge call failed.

    A failed row still has to name what failed - `judge_model` is `min_length=1`
    and required - and the answer is the model that was going to be called, not
    the L0 sentinel. Read off the client when it exposes one (`JudgeClient` has a
    `model` property), and otherwise from the same resolver the judge itself uses,
    so the row and the call cannot name different models.
    """
    model = getattr(judge_client, "model", None)
    if isinstance(model, str) and model:
        return model
    return resolve_judge_model()
