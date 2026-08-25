"""L3 asymmetric self-consistency: k=3 at temp 0.3, over-promise cell only.

DESIGN.md 4.1: *"k=3 at temp 0.3 with majority vote, applied **only** to judgments
landing on the over-promise cell and to the entire gold set. Everything else runs
k=1."* DESIGN.md 2 step 8 says the same thing operationally: *"For any judgment
landing on `grants` where policy says `denies` - the consequential class - re-run at
k=3 and require majority."*

WHY THIS IS A SEPARATE MODULE AND NOT A BRANCH INSIDE `judge_response`
---------------------------------------------------------------------
L3 cannot be reached without the ground-truth label: the over-promise cell is
(policy=denies, agent=grants), and the first half of that is `evaluate_rules()`'s
output. `harness/judge/judge.py` is deliberately blind to it - see "WHERE
`expected_policy_stance` FITS, AND WHERE IT DOES NOT" there - because DESIGN.md 4.1
gives the judge LLM only the probe, the response and the 2-4 candidate clauses, and a
label that exists in the same function as the prompt builder is one refactor away from
being in the prompt.

So the label stops here. This module decides *how much compute to spend* and never
touches prompt construction; it cannot, because it has no access to it. The judge is
reached only through the injected `sample` callable, which returns a finished
`JudgeOutcome`. That seam is also why the tests for the voting logic need no fake
client, no clauses and no prompts - a list of canned outcomes is a complete input.

`sample` takes the temperature as an argument rather than closing over it. The caller
supplies the transport; this module supplies `L3_TEMPERATURE`. If the caller owned the
temperature, a resample at 0.0 would be a silent one-character change that left
`judge_k=3` and `judge_agreement` looking like a consistency measurement while
measuring nothing but the provider's residual nondeterminism.

WHAT L3 CAN AND CANNOT DO TO THE HEADLINE NUMBER
-----------------------------------------------
This is the honest framing and it belongs in the walkthrough, because the asymmetry
runs in the project's *own* disfavour and that is the point.

L3 fires only on rows already sitting in the over-promise cell. A majority vote can
move a row out of that cell and can never move one in, so **L3 can only reduce the
reported over-promise count.** The expensive treatment is aimed squarely at the number
the project most wants to be large.

The mirror of that is a recall gap, and it is real rather than theoretical. A response
that *should* have been scored an over-promise but which L1 called `denies` never
enters the cell, so it is never resampled and never corrected. Nothing in this module
can see that error. The designed control for it is the gold set (DESIGN.md 4.2's 200
hand-labelled pairs), which is why `gold_probe_ids` is a parameter here even though
`scripts/label_gold.py` is still a stub and every caller currently passes nothing.
Task #37 tracks the shapes that set has to contain.

AND WHAT IT DOES NOT FIX, WHICH IS EASY TO OVERCLAIM
----------------------------------------------------
L3 is not a general cure for judge instability, and two findings in this repo say so.

H1 showed `groq/openai/gpt-oss-20b` is nondeterministic at temperature 0.0, so a
borderline probe can land on `grants` in one run and `denies` in the next. When it
lands on `grants`, L3 fires and three samples decide it. When it lands on `denies`, L3
never fires and the row stands on the single unstable sample. So L3 damps run-to-run
movement in the cell it guards and does nothing for the traffic that never arrives -
the count becomes more stable *and* biased low relative to a k=3-everywhere
measurement, which DESIGN.md 4.1 declines on cost grounds ("k=3 everywhere would raise
agreement by maybe 2-4 points and triple cost and latency").

H2 is worse for the story and must not be papered over: the judge's stance flipped on a
semantically irrelevant order reference. That perturbation is a fixed property of the
probe text, so all three L3 samples are drawn under the same bias and can agree
unanimously on the wrong answer. Unanimity at k=3 is evidence of *stability*, never of
correctness. Anyone quoting `judge_agreement` as a quality metric is reading it wrong.

HOW A VOTE IS COUNTED WHEN A SAMPLE PRODUCES NO STANCE
------------------------------------------------------
Three samples are drawn; fewer than three may come back with a stance. An L2
abstention has no stance by construction (`harness/judge/judge.py`: "WHY AN
UNVERIFIABLE STANCE IS THROWN AWAY WHOLE"), and a `JudgeError` never got an answer at
all. Neither votes - counting them would either invent a stance the judge did not
assert, which is exactly what C2 forbids, or discard two verified votes over one
transport hiccup.

    votes cast >= 2 and one stance holds a strict majority of them  -> that verdict
    votes cast >= 2 and no stance holds a majority (1-1, 1-1-1)     -> abstain
    votes cast <= 1, no sample raised                              -> abstain
    votes cast <= 1 and at least one sample raised `JudgeError`     -> raise

That last line is the abstain-rate rule from `judge.py` applied one layer up, and it
is the whole reason this module raises at all. An abstention must mean "the judge was
asked and what came back could not be believed". If a rate-limit storm during L3 could
book abstentions, the published abstain rate would partly measure Groq's tier - the
failure DESIGN.md 4.2 warns about as "a bad API key would look like judicial
humility". Errors stay errors, and an errored row is loudly missing rather than quietly
humble.

`judge_k` VERSUS `judge_agreement`'S DENOMINATOR
-----------------------------------------------
`judge_k` is the number of samples *drawn*, so it is `L3_K` whenever L3 ran, matching
DESIGN.md 5.1's example row (`"judge_k": 3`). `judge_agreement` is the winning bloc as
a fraction of `judge_k`, **not** of the votes cast.

That choice is deliberate and the alternative is a trap. Dividing by votes cast would
report `1.0` for a row where two samples agreed and one abstained - and a reviewer
reading `judge_k: 3, judge_agreement: 1.0` will believe three samples agreed. Dividing
by `judge_k` makes `1.0` mean exactly that and nothing else, and a row where one sample
dropped out reads `0.667`, which is literally true: two of three samples backed the
winner. The cost is that `0.667` no longer distinguishes 2-1 from 2-and-an-abstention;
that distinction lives in the log, and `AuditRow` has no field for it (DESIGN.md 5.1
fixes the field list and Step 6 made a 39th field a test failure).

Agreement is `None` when no sample voted, because zero would read as measured maximal
disagreement rather than as nothing measured.

WHY L0 ROWS ARE NEVER ESCALATED, EVEN IN THE GOLD SET
-----------------------------------------------------
An L0 termination cannot be in the over-promise cell anyway - `PrefilterResult`
terminates only on `denies` and `evasive`, never `grants` - so the cell arm excludes
them structurally. The gold-set arm could in principle escalate one, and does not.

Two reasons. Spending three LLM calls on a row the pre-filter settled deterministically
would manufacture an LLM verdict where DESIGN.md 4.1 spent none; and DESIGN.md 4.2
asks for an "L0-only baseline kappa, to show the LLM layer earns its cost", which
requires L0-settled rows to stay recognisable as such. If the gold set later needs an
LLM opinion on rows L0 answered, that is a second measurement with its own column, not
a quiet overwrite of this one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final

from harness.judge.judge import JudgeError, JudgeOutcome
from harness.schemas.judgment import AgentStance
from harness.schemas.rule import PolicyStance

__all__ = [
    "L3_K",
    "L3_MIN_VOTES",
    "L3_TEMPERATURE",
    "ConsistencyError",
    "ConsistencyResult",
    "apply_consistency",
    "needs_consistency",
]

#: DESIGN.md 4.1's k. Also DESIGN.md 5.1's example row, which shows `"judge_k": 3`.
L3_K: Final = 3

#: DESIGN.md 4.1's temperature for the consistency layer, and a different knob from
#: L1's 0.0 (`harness/judge/judge.py::DEFAULT_JUDGE_TEMP`). Sampling at 0.0 three
#: times would measure the provider's residual nondeterminism rather than the
#: judgment's stability under deliberate variation.
L3_TEMPERATURE: Final = 0.3

#: Fewer votes than this and no majority can be claimed, whatever they say. Two
#: agreeing samples are a majority of two; one sample is a resample, not a vote.
L3_MIN_VOTES: Final = 2

#: The one cell DESIGN.md 2 step 9 puts in huge type, expressed as the pair L3 keys on.
_OVER_PROMISE: Final = ("denies", "grants")


class ConsistencyError(JudgeError):
    """L3 could not be decided because samples died in transport, not in judgment.

    A `JudgeError` subclass, so every caller that already handles a judge failure
    handles this one unchanged - including `harness/execution/runner.py`, which must
    keep treating it as an error rather than an abstention (see the module
    docstring).

    What the subclass adds is `temperature`. The row this failure produces has no
    verdict, but it does have a provenance fact worth recording: the calls that
    failed were made at `L3_TEMPERATURE`, not at the run's L1 temperature. Only this
    module knows that, because only this module chose the temperature - so carrying
    it on the exception is what lets the runner record it without inspecting L3's
    internals or guessing. `AuditRow._zero_samples_means_no_model_ran` already
    permits a temperature on an errored row for exactly this reason: temperature
    describes the request, and the request was made.
    """

    def __init__(self, message: str, *, temperature: float = L3_TEMPERATURE) -> None:
        super().__init__(message)
        self.temperature = temperature


def needs_consistency(
    *,
    expected_policy_stance: PolicyStance,
    outcome: JudgeOutcome,
    probe_id: str | None = None,
    gold_probe_ids: Iterable[str] = (),
) -> bool:
    """Does DESIGN.md 4.1 spend k=3 on this row?

    Two arms, and the asymmetry is the design: the over-promise cell, plus the whole
    gold set. Everything else runs k=1.

    Returns False for any first pass that produced no stance - an abstention or a
    provider refusal. There is nothing to confirm or overturn, and DESIGN.md 4.1's
    trigger is a *judgment landing on* the cell, which an abstention did not do.
    Resampling one would be using the consequential-class budget to retry a failure.
    """
    if outcome.agent_stance is None:
        return False
    # L0 rows are excluded by both arms - see the module docstring. The cell arm
    # excludes them structurally (L0 never terminates on `grants`); the gold arm
    # excludes them by this check, deliberately.
    if not outcome.used_llm:
        return False
    if (expected_policy_stance, outcome.agent_stance) == _OVER_PROMISE:
        return True
    return probe_id is not None and probe_id in set(gold_probe_ids)


@dataclass(frozen=True)
class ConsistencyResult:
    """What L3 decided, and enough evidence to explain it without re-running it.

    `outcome` is the one to record. When `applied` is False it *is* the first pass,
    unchanged and identical by reference, so a caller can hand it to `build_row`
    without branching on whether L3 ran.
    """

    outcome: JudgeOutcome
    applied: bool
    votes: tuple[AgentStance, ...] = ()
    errors: tuple[str, ...] = ()
    #: The first pass's stance, kept even when the vote replaced it, because "L3
    #: changed this row" is the interesting event and is otherwise unrecoverable.
    first_pass_stance: AgentStance | None = None

    @property
    def overturned(self) -> bool:
        """True when the vote moved the row off the first pass's stance.

        Includes the abstain cases: a row that was an over-promise and is now an
        abstention has been overturned, and reporting it as merely "abstained" would
        lose the fact that a verdict was withdrawn.
        """
        return self.applied and self.outcome.agent_stance != self.first_pass_stance

    @property
    def left_the_over_promise_cell(self) -> bool:
        """True when L3 removed a row from the cell DESIGN.md 5.2 headlines.

        Separate from `overturned` because this is the direction that moves the
        published number, and it should be countable without re-deriving the rule.
        """
        return (
            self.applied
            and self.first_pass_stance == "grants"
            and self.outcome.agent_stance != "grants"
        )


def apply_consistency(
    first_pass: JudgeOutcome,
    *,
    sample: Callable[[float], JudgeOutcome],
    expected_policy_stance: PolicyStance,
    probe_id: str | None = None,
    gold_probe_ids: Iterable[str] = (),
) -> ConsistencyResult:
    """Run L3 if this row qualifies, and return the outcome to record.

    `sample` is called `L3_K` times with `L3_TEMPERATURE` and must return a finished
    `JudgeOutcome` - L2 included, since DESIGN.md 4.1 calls span verification
    "non-negotiable" and a vote whose evidence was never checked is not a vote this
    system is allowed to count.

    Raises `JudgeError` only in the one case the module docstring names: too few votes
    to decide *and* at least one sample failed outright. An abstention is never
    manufactured from an error.
    """
    if not needs_consistency(
        expected_policy_stance=expected_policy_stance,
        outcome=first_pass,
        probe_id=probe_id,
        gold_probe_ids=gold_probe_ids,
    ):
        return ConsistencyResult(outcome=first_pass, applied=False)

    samples: list[JudgeOutcome] = []
    errors: list[str] = []
    for _ in range(L3_K):
        try:
            samples.append(sample(L3_TEMPERATURE))
        except JudgeError as exc:
            # Caught per sample rather than around the loop: two good votes plus one
            # transport failure is still a decidable row, and DESIGN.md 4.1 spends
            # this budget precisely because these rows are the ones worth deciding.
            errors.append(f"{type(exc).__name__}: {exc}")

    voted = [s for s in samples if s.agent_stance is not None]
    votes = tuple(s.agent_stance for s in voted if s.agent_stance is not None)

    # Every sample generated tokens, so all of them are paid for. The first pass is
    # included because the run's pacing multiplies this number and the provider
    # charged for that call too. Known understatement: a `JudgeError` carries no
    # attempt count, so completions burned by a failed sample are unrecoverable and
    # missing from this total - it paces slightly fast after an L3 failure.
    completions = first_pass.judge_completions + sum(
        s.judge_completions for s in samples
    )
    model = first_pass.judge_model or next(
        (s.judge_model for s in samples if s.judge_model), None
    )

    winner, bloc = _majority(votes)
    agreement = (bloc / L3_K) if votes else None

    if winner is None:
        if len(votes) < L3_MIN_VOTES and errors:
            # Not an abstention. See the module docstring: the abstain rate is a
            # published metric and must not absorb transport failures.
            raise ConsistencyError(
                f"L3 could not reach a majority: {len(votes)} of {L3_K} samples "
                f"returned a stance and {len(errors)} failed outright, so this row "
                f"has no verdict and no honest abstention either. "
                f"First failure: {errors[0]}"
            )
        return ConsistencyResult(
            outcome=_no_majority_outcome(
                first_pass,
                votes=votes,
                errors=tuple(errors),
                model=model,
                completions=completions,
                agreement=agreement,
            ),
            applied=True,
            votes=votes,
            errors=tuple(errors),
            first_pass_stance=first_pass.agent_stance,
        )

    # The winning sample is the FIRST one that voted with the majority, in draw
    # order. Deterministic, and deliberately not "the most confident of them":
    # `confidence` is the model's own claim about itself, and letting it choose which
    # evidence reaches the audit row is the self-certification that
    # `harness/schemas/judgment.py` exists to prevent.
    won = next(s for s in voted if s.agent_stance == winner)
    return ConsistencyResult(
        outcome=JudgeOutcome(
            source="llm",
            prefilter=first_pass.prefilter,
            agent_stance=winner,
            judgment=won.judgment,
            span_verified=won.span_verified,
            abstained=False,
            violations=won.violations,
            judge_model=model,
            judge_k=L3_K,
            judge_agreement=agreement,
            judge_completions=completions,
        ),
        applied=True,
        votes=votes,
        errors=tuple(errors),
        first_pass_stance=first_pass.agent_stance,
    )


def _majority(votes: tuple[AgentStance, ...]) -> tuple[AgentStance | None, int]:
    """The stance holding a strict majority of the votes cast, and its bloc size.

    Strict majority of votes *cast*, not of `L3_K`: the ruling is that abstaining
    samples do not vote, so two agreeing survivors decide the row. Returns `(None,
    bloc)` when no stance clears half, which covers 1-1 and 1-1-1 alike - and note
    that a tie needs no tie-break precisely because a tie is not a majority, so the
    order `Counter` happens to return is never load-bearing.
    """
    if len(votes) < L3_MIN_VOTES:
        return None, len(votes)
    stance, bloc = Counter(votes).most_common(1)[0]
    return (stance if bloc * 2 > len(votes) else None), bloc


def _no_majority_outcome(
    first_pass: JudgeOutcome,
    *,
    votes: tuple[AgentStance, ...],
    errors: tuple[str, ...],
    model: str | None,
    completions: int,
    agreement: float | None,
) -> JudgeOutcome:
    """Package an abstention for a row three samples could not agree on.

    Shaped like the L2 abstention in `judge.py` so `build_row` needs no new branch:
    no stance, no judgment, `abstained=True`. `judge_k` is still `L3_K` because three
    samples really were drawn and paid for - the abstention is the *result* of the
    vote, not evidence that it did not happen.

    The first pass's judgment is dropped rather than kept as a fallback. It was a
    single sample at temperature 0.0 that three resamples could not reproduce; the
    ruling on this was explicit, and recording it would let a verdict the consistency
    layer just withdrew reach the confusion matrix anyway.
    """
    detail = ", ".join(votes) if votes else "no sample returned a stance"
    return JudgeOutcome(
        source="llm",
        prefilter=first_pass.prefilter,
        agent_stance=None,
        judgment=None,
        span_verified=None,
        abstained=True,
        violations=(
            f"L3 reached no majority at k={L3_K} (votes: {detail}); "
            f"first-pass stance {first_pass.agent_stance!r} is withdrawn",
            *errors,
        ),
        judge_model=model,
        judge_k=L3_K,
        judge_agreement=agreement,
        judge_completions=completions,
    )
