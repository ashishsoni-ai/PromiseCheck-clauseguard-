"""L0 -> L1 -> L2 control flow: the judge, assembled. DESIGN.md 4.1. STEP 5.

This module owns the *policy*; the layers own the work. `prefilter.classify` decides
whether an LLM is needed, `prompts.py` decides what it sees, `span_verify.verify_judgment`
decides whether to believe it, and the sequencing - including DESIGN.md 4.1's
"failure -> one retry with the violation named -> second failure -> `judge_abstain`" -
lives here and nowhere else.

WHAT THE ABSTAIN RATE IS ALLOWED TO MEAN
---------------------------------------
`judge_abstained` is a *reported metric* (DESIGN.md 4.2), and the claim it supports is
specific: "a judge that abstains 4% of the time and is verifiable on the other 96% is a
far stronger claim than one that answers everything." That claim only holds if an
abstention means one thing - the judge could not produce evidence that survived
mechanical checking.

So exactly one condition books an abstention: L2 rejected the judgment twice. Everything
else raises `JudgeError`:

* the provider timed out, refused, or returned nothing
* the response could not be coerced into a `Judgment`
* no candidate clauses were supplied

None of those are the judge declining to guess; they are a missing measurement. Booking
them as abstentions would let a bad API key, a rate limit, or an upstream bug in probe
construction look like judicial humility - and would quietly deflate the headline numbers,
because abstentions are excluded from them. A crashed run is obvious and gets fixed; a
run that silently reports 30% abstention because a key expired is a lie with a plausible
story attached.

WHY AN UNVERIFIABLE STANCE IS THROWN AWAY WHOLE
----------------------------------------------
When L2 fails twice, the second judgment usually still has a *plausible* `agent_stance` -
the span check tests the judge's evidence, not its conclusion, and a judge can pick the
right stance while fumbling the quote. This module discards it anyway, and
`JudgeOutcome.agent_stance` is None for an abstention.

That is deliberate, and it is the whole product. An LLM stance with no verifiable quote
behind it is exactly what every other LLM-judge demo ships. Keeping it would mean the
headline number rests partly on unverified assertions, and there would be no way to tell
which part. The rejected judgment is preserved in `JudgeOutcome.judgment` alongside
`violations` so the human review queue (DESIGN.md 4.1 L2) can see what the judge tried to
say and why it was not accepted - but it is not a verdict, and a None cannot be silently
binned into a confusion-matrix cell by downstream code.

L0 AND L1 DO NOT VOTE
---------------------
If the pre-filter escalates, its stance is not consulted again: L1's judgment stands on
its own, and the `PrefilterResult` is carried on the outcome as evidence, not as input to
the verdict. Blending a lexicon guess with a model judgment would produce a number nobody
can explain the provenance of, and would destroy DESIGN.md 4.2's L0-only baseline kappa -
which requires L0's opinion on every row, including the rows L1 also judged, to be
recoverable and uncontaminated.

`judge_k` VERSUS `judge_completions`
-----------------------------------
Two different counts that are easy to conflate, and conflating them corrupts two
different slides.

`judge_k` is DESIGN.md 4.1's consistency parameter - the number of independent samples
majority-voted at temp 0.3. It is 0 when L0 answered (no model ran) and 1 everywhere else
until L3 lands. `judge_completions` is how many times this module actually asked a model
for a judgment, so a C2 retry makes it 2 with `judge_k` still 1.

Summing `judge_completions` is what substantiates DESIGN.md 4.1's claim that L0 "kills
~30% of LLM calls"; summing `judge_k` would understate the retry cost, and reading
`judge_completions` as k would report a consistency measurement that was never taken.

WHERE `expected_policy_stance` FITS, AND WHERE IT DOES NOT
----------------------------------------------------------
It is absent from this module's signatures on purpose. See "WHY THESE FUNCTIONS CANNOT
SEE THE GROUND-TRUTH LABEL" in `harness/judge/prompts.py`: DESIGN.md 1.5 gives the judge
component the expected stance, DESIGN.md 4.1 gives the LLM only the probe, the response
and the clauses, and both are true because the component needs the label to decide *how
much compute to spend*, never as evidence.

That distinction only becomes load-bearing at L3, which applies k=3 exclusively to
judgments landing on the over-promise cell - a cell you cannot identify without the
policy stance. L3 is a later step and `harness/judge/consistency.py` is still a stub, so
the parameter is documented here rather than added unused. When it arrives, the
constraint on it is one sentence long: it may select the sampling policy and must never
reach `build_judge_user_prompt`.

Note the direction of that asymmetry, because it is the honest reading and worth saying
out loud in the walkthrough: spending k=3 on suspected over-promises can only *reduce*
the reported over-promise count, never inflate it. Majority voting three samples discards
grants that only one sample believed. The expensive treatment is aimed at the number the
project most wants to be large, which is the opposite of putting a thumb on the scale.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Final, Literal, Protocol, runtime_checkable

from harness.judge.prefilter import PrefilterResult, classify
from harness.judge.prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_prompt,
    build_retry_user_prompt,
)
from harness.judge.ratelimit import RateLimitExhausted, call_with_rate_limit_retry
from harness.judge.span_verify import verify_judgment
from harness.schemas.clause import Clause
from harness.schemas.judgment import AgentStance, Judgment

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_JUDGE_TEMP",
    "InstructorJudgeClient",
    "JudgeClient",
    "JudgeError",
    "JudgeOutcome",
    "judge_response",
    "resolve_judge_model",
    "resolve_judge_temp",
]

#: DESIGN.md 1.5 requires the judge to come from a different model family than the AUT.
#: The agent under test is Qwen, so this is gpt-oss.
#:
#: WHY THIS PIN MOVED TWICE ON 2026-08-23, AND WHY ONLY THE SECOND MOVE WAS MEASURED
#: Model IDs are provider inventory, not spec. DESIGN.md names no model ID anywhere except
#: the AUT's Qwen; 1.5 and 2 step 8 constrain the model *family*, because a family is a
#: property of the measurement while an ID expires on someone else's schedule. The Appendix
#: anticipates exactly this - "judge and extractor via `litellm` so model swaps are a config
#: line - you will want that when a provider rate-limits you at 11pm on day 12".
#:
#: First move, forced: `groq/llama-3.3-70b-versatile` was decommissioned and began returning
#: 404 `model_not_found`, so the judge went local to `ollama_chat/llama3.1:8b`. That restored
#: availability and took the provider key out of the judge path - worth something, since the
#: key rotation earlier the same day was forced by a pytest traceback printing the
#: `Authorization` header litellm passes as a frame argument. But it was chosen on
#: availability grounds with no latency measurement behind it.
#:
#: Second move, measured: the local judge cost ~11.7s per call warm on this machine - 1294
#: prompt tokens at 786 tok/s, then 121 generated tokens at 12.0 tok/s. At ~30 probes
#: surviving L0 that is close to six minutes serialised, and step 8's k=3 on the
#: consequential class takes it to twelve or more. DESIGN.md 2 step 11 asks for "under 45
#: seconds for an incremental run". The judge is the only harness role on the incremental
#: path - the extractor and adversary run during `generate`, which is an install step and may
#: be slow - so the judge is the one role whose latency is load-bearing, and it was the one
#: sitting on the slowest hardware available.
#:
#: WHAT THE HOSTED JUDGE ACTUALLY BOUGHT, MEASURED AFTERWARDS BY `scripts/time_judge.py`,
#: AND WHY IT IS STILL NOT 45 SECONDS
#: Per call the move was a win, and a larger one than was first claimed here: min 0.88s,
#: median 0.92s, max 1.67s, with imports warmed outside the timed section. Roughly 13x faster
#: than the local pin. An intermediate figure of ~12-16s per call briefly appeared in this
#: repo and is RETRACTED: it came from dividing a two-test pytest wall clock by two, and that
#: wall clock contained a 10.95s `litellm` + `instructor` import paid once per PROCESS rather
#: than once per call - the same mistake as dividing a cold model load by the number of tests
#: that followed it. Do not reintroduce it.
#:
#: But per-call latency is not the binding constraint, and the 45-second target is NOT met on
#: this tier, so nothing above should be read as claiming it. Groq's `on_demand` tier caps
#: this model at 8000 tokens per minute, and a single judge call requests 1152-2178 tokens
#: depending on whether C2's retry fires - which for a `grants` judgment it usually does,
#: since `grants` is the one stance the system prompt requires spans for. That is ~5-6 calls
#: per minute regardless of how fast any one of them returns. Thirty post-L0 probes at ~2200
#: tokens is ~66,000 tokens, so roughly eight minutes of deliberate waiting, and step 8's k=3
#: on the consequential class is worse. Step 6's "semaphore of 8" cannot rescue it: 6 of 6
#: concurrent calls failed inside 0.25s with `RateLimitError`, because a concurrency cap does
#: not model a token budget. What that path needs is a token-budget-aware limiter plus a 429
#: backoff honouring the delay the provider states in the error body. Both now exist - the
#: pacing is `runner.py`'s `--judge-pace`, scaled by `judge_completions`, and the backoff is
#: `harness/judge/ratelimit.py` - so a transient refusal no longer loses the run. Neither
#: makes the run FASTER: pacing is chosen to stay under the budget, so the eight minutes
#: above is the design, not a defect to be optimised away. `SCHEMA_REPAIR_RETRIES` below is
#: still not it; it repairs malformed JSON and never waits.
#:
#: So the decision to go hosted rests on role placement - the argument above, which does not
#: depend on the quota and survives it. The 45-second claim does not rest on anything yet.
#: `docs/limitations.md` carries the arithmetic; do not write "under 45 seconds" into any
#: report until a real end-to-end run has been timed on the tier the demo will use.
#:
#: The local pin was unreliable for a related reason, and the two facts share a cause: 6GB of
#: VRAM does not comfortably hold a 4.9GB Q4 8B alongside a 4096-token KV cache, so the first
#: load attempt died during CUDA init and Ollama retried with layers spilled to CPU - which
#: is what a 12 tok/s generation rate on that GPU means. A judge that dies on cold load
#: raises `JudgeError` and loses the row rather than abstaining (see "WHAT THE ABSTAIN RATE
#: IS ALLOWED TO MEAN"), and step 6's semaphore of 8 would have attempted eight such loads at
#: once. Raising concurrency would have made that worse, not better.
#:
#: ACKNOWLEDGED LIMITATION: THE JUDGE AND THE EXTRACTOR ARE NOW ONE FAMILY
#: Of the 13 models this account can see on 2026-08-23, nine are not judges at all (2 ASR,
#: 2 TTS, 2 prompt-injection classifiers of 22M/86M params, and 2 agentic systems with
#: built-in web search, which 4.1 disqualifies since a judge that can search could pull in
#: text outside the 2-4 candidate clauses). Five chat models remain: three gpt-oss, one Qwen
#: - the agent's own family, closed to the judge by 1.5 - and `allam-2-7b`.
#:
#: So the honest claim is NOT that a hosted judge is *necessarily* gpt-oss; it is that no
#: hosted model in a fourth family is a *suitable* judge. `allam-2-7b` is a real
#: instruction-tuned chat model and would erase this limitation on paper. It was passed over
#: because it is a 7B Arabic-first bilingual model being asked to read English policy prose,
#: its tool-calling support - which instructor's TOOLS mode needs - is unverified here, and
#: the judge is the one role graded mechanically: C2 requires a span that survives exact
#: substring matching. A judge that abstains constantly damages the published numbers more
#: than a judge sharing a family with the extractor does. See `docs/limitations.md`; if
#: `allam-2-7b` is ever spiked and holds up, take it and delete that entry.
#:
#: The four roles therefore hold three families, not four:
#:
#:   agent      qwen2.5:7b-instruct        local, frozen in aut-naive
#:   extractor  groq/openai/gpt-oss-120b   hosted    <-- one family
#:   judge      groq/openai/gpt-oss-20b    hosted    <-- one family
#:   adversary  ollama_chat/mistral:7b     local
#:
#: Recorded here, in `.env.example`, in README's limitations and in
#: `tests/unit/test_aut_contract.py` rather than left to be discovered, because a limitation
#: that lives only in a config diff is one nobody reads.
#:
#: Nothing in DESIGN.md forbids it. 1.5's rule is judge-versus-AUT and is met. 2's
#: circularity warning names one specific mechanism - "a model that generated a probe is
#: measurably more likely to accept a response that pattern-matches its own generation" -
#: which is the adversary/judge relation, and that stays separated (gpt-oss vs mistral) and
#: asserted.
#:
#: The extractor/judge overlap is a weaker concern, for a structural reason worth stating
#: rather than assuming: the extractor produces rules, and under commitment C1 the
#: ground-truth label is computed from those rules by `evaluate_rules()` in Python. The judge
#: never sees that label and never grades the extractor's output - it classifies what the
#: agent's reply committed to and cites a span, which C2 then checks mechanically. For a
#: shared blind spot to reach a headline number it would have to route through a rule the
#: extractor mis-extracted *and* a response the judge mis-classified in the same direction,
#: with the span check passing throughout. That is not zero and it belongs in 8's
#: limitations; it is not the same order of risk as grading a probe with the model that
#: wrote it.
#:
#: (`ollama_chat/` versus `ollama/` still matters for the adversary, which stays local:
#: litellm routes `ollama/` at the legacy `/api/generate` and `ollama_chat/` at `/api/chat`,
#: and instructor's default TOOLS mode needs the chat endpoint.)
DEFAULT_JUDGE_MODEL: Final = "groq/openai/gpt-oss-20b"
JUDGE_MODEL_ENV: Final = "CLAUSEGUARD_JUDGE_MODEL"

#: DESIGN.md 4.1: L1 runs at temp 0.0. L3's 0.3 is a different knob for a different layer
#: and is not read here.
DEFAULT_JUDGE_TEMP: Final = 0.0
JUDGE_TEMP_ENV: Final = "CLAUSEGUARD_JUDGE_TEMP"

#: Handed to instructor for *schema* repair - a reply that is not valid JSON, or omits a
#: required field. Orthogonal to commitment C2's single retry, which concerns a
#: syntactically perfect judgment whose quotes do not exist. Counting one as the other
#: would either hide a fabrication behind a parse error or spend C2's retry on a comma.
SCHEMA_REPAIR_RETRIES: Final = 2

#: A ceiling on a failure, not a target. Back to 120.0 now that the judge is hosted again:
#: the 300.0 it briefly held existed for one reason, a cold Ollama load of an 8B model
#: measured at ~81s on this machine, and cold loads are not on this path any more.
#:
#: The principle it has to satisfy has not changed. A timeout here raises `JudgeError` and
#: loses the row entirely rather than abstaining (see "WHAT THE ABSTAIN RATE IS ALLOWED TO
#: MEAN"), so it must be generous enough never to be the thing that fires under ordinary
#: provider latency - a timeout that trims the slow tail would quietly shrink the denominator
#: of every published metric. It is deliberately far above DESIGN.md 2 step 11's 45-second
#: run target, because that target describes a healthy run and this number describes a broken
#: one; conflating them would buy speed by discarding measurements.
DEFAULT_TIMEOUT_S: Final = 120.0


class JudgeError(RuntimeError):
    """The judge could not be run to completion.

    Deliberately *not* an abstention - see "WHAT THE ABSTAIN RATE IS ALLOWED TO MEAN" in
    the module docstring. Raised for transport failures, unparseable replies, and missing
    candidate clauses.
    """


@runtime_checkable
class JudgeClient(Protocol):
    """The seam, placed at the structured-output boundary rather than at the string one.

    Returning a `Judgment` (not raw text) puts instructor, litellm, JSON and provider
    quirks entirely on the far side, so the unit tests for the retry-then-abstain control
    flow - the part of this module with actual logic in it - run with no network, no key
    and no fixtures full of hand-written JSON.
    """

    def judge(self, *, system: str, user: str, temperature: float) -> Judgment: ...

    @property
    def model(self) -> str: ...


class InstructorJudgeClient:
    """The real client: instructor over litellm, structured into `Judgment`."""

    def __init__(
        self,
        model: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model = model or resolve_judge_model()
        self._timeout_s = timeout_s
        self._client = None  # built lazily; importing litellm is slow and noisy
        # Injected only so the offline suite can assert the 429 backoff arithmetic
        # without elapsing real time. Nothing else should pass it.
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._model

    def _ensure_client(self):
        if self._client is None:
            try:
                import instructor
                from litellm import completion
            except ImportError as exc:  # pragma: no cover - dependency, not logic
                raise JudgeError(f"judge dependencies unavailable: {exc}") from exc
            self._client = instructor.from_litellm(completion)
        return self._client

    def judge(self, *, system: str, user: str, temperature: float) -> Judgment:
        client = self._ensure_client()

        def call() -> Judgment:
            return client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_model=Judgment,
                temperature=temperature,
                max_retries=SCHEMA_REPAIR_RETRIES,
                timeout=self._timeout_s,
            )

        # A 429 is retried HERE rather than by L1's retry-then-abstain control, and
        # the reason is not convenience. L1's budget is about judgment validity, and
        # the abstain rate built on it is a published metric (DESIGN.md 4.2); a quota
        # refusal spent from that budget would make the number partly a measurement
        # of Groq's tier. It also must not reach `judge_completions`, which paces the
        # run - a rejected call burned no tokens and must not buy itself sleep. See
        # harness/judge/ratelimit.py. `SCHEMA_REPAIR_RETRIES` above is unrelated: it
        # repairs malformed JSON, and instructor does not treat a 429 as repairable.
        try:
            return call_with_rate_limit_retry(call, sleep=self._sleep)
        except RateLimitExhausted as exc:
            # Deliberately not flattened into the generic message below: the remedy
            # is to wait or change tier, and an operator who reads "the judge failed"
            # goes looking for a bug that is not there.
            raise JudgeError(f"judge {self._model}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - detail belongs in the message
            raise JudgeError(f"judge {self._model}: {exc}") from exc


def resolve_judge_model() -> str:
    """Read the judge model from the environment, defaulting to the documented one."""
    return (os.getenv(JUDGE_MODEL_ENV) or "").strip() or DEFAULT_JUDGE_MODEL


def resolve_judge_temp() -> float:
    """Read the L1 temperature, failing loudly rather than silently using 0.0.

    A typo'd temperature that quietly fell back to the default would be undetectable in
    the audit trail, and DESIGN.md 4.1 treats 0.0 as part of the judge's definition.
    """
    raw = (os.getenv(JUDGE_TEMP_ENV) or "").strip()
    if not raw:
        return DEFAULT_JUDGE_TEMP
    try:
        return float(raw)
    except ValueError as exc:
        raise JudgeError(f"{JUDGE_TEMP_ENV}={raw!r} is not a number") from exc


@dataclass(frozen=True)
class JudgeOutcome:
    """Everything one probe/response pair produced, shaped for DESIGN.md 5.1's row.

    The fields absent from `Judgment` live here rather than there, per the note in
    `harness/schemas/judgment.py`: a judgment must not be able to certify itself. The
    model asserts `quoted_span`; Python decides `span_verified`.

    Three fields are `None`-able for the same reason, and the None is meaningful rather
    than merely empty:

    * `agent_stance` is None only for an abstention - there is no verdict to record.
    * `span_verified` is None when no span was ever offered - either L0 answered, or the
      judge returned a `denies`/`evasive` judgment that quoted nothing. False means
      checked and rejected, which is a different fact.
    * `judge_confidence` is None when no model was asked. A lexicon has no calibrated
      confidence, and inventing one would corrupt any confidence-versus-accuracy plot.
    """

    source: Literal["prefilter", "llm"]
    prefilter: PrefilterResult
    agent_stance: AgentStance | None = None
    judgment: Judgment | None = None
    span_verified: bool | None = None
    abstained: bool = False
    violations: tuple[str, ...] = field(default=())
    judge_model: str | None = None
    judge_k: int = 0
    judge_agreement: float | None = None
    judge_completions: int = 0

    @property
    def judge_confidence(self) -> float | None:
        """The model's own confidence, or None when no model ran."""
        return None if self.judgment is None else self.judgment.confidence

    @property
    def used_llm(self) -> bool:
        return self.source == "llm"

    @property
    def counts_toward_headline_metrics(self) -> bool:
        """DESIGN.md 4.1: abstentions are "excluded from headline metrics but counted in
        the abstain rate". Stated as a property so the metrics layer asks rather than
        re-deriving the rule and getting it subtly wrong."""
        return not self.abstained


def _prefilter_outcome(result: PrefilterResult) -> JudgeOutcome:
    """Package a terminal L0 verdict. No model ran, so most of the row is None."""
    # `stance` is narrowed by PrefilterResult.is_terminal: only "denies" and "evasive"
    # terminate, and both are valid AgentStance values. "grants" and "unclear" escalate.
    return JudgeOutcome(
        source="prefilter",
        prefilter=result,
        agent_stance=result.stance,  # type: ignore[arg-type]
        judge_k=0,
        judge_completions=0,
    )


def judge_response(
    *,
    probe_turns: Sequence[str],
    agent_response: str,
    candidate_clauses: Sequence[Clause],
    client: JudgeClient | None = None,
    temperature: float | None = None,
) -> JudgeOutcome:
    """Judge one probe/response pair through L0, L1 and L2.

    Note what is not a parameter: the expected policy stance. See the module docstring.

    Raises `JudgeError` on transport or schema failure, or if no candidate clauses were
    supplied. Returns an outcome with `abstained=True` only when L2 rejected the
    judgment's spans twice.
    """
    prefilter_result = classify(agent_response)
    if prefilter_result.is_terminal:
        return _prefilter_outcome(prefilter_result)

    if not candidate_clauses:
        # Refused before spending anything. A probe reaching the judge with no clauses is
        # an upstream bug in probe construction: L2 would reject any citation the judge
        # could possibly make, so the two LLM calls are guaranteed waste and the row
        # would land in the abstain rate carrying a harness defect instead of a judgment.
        raise JudgeError(
            "no candidate clauses supplied; DESIGN.md 4.1 requires 2-4, and a judgment "
            "citing a clause the judge was not shown can never pass L2"
        )

    active = client if client is not None else InstructorJudgeClient()
    temp = resolve_judge_temp() if temperature is None else temperature

    user_prompt = build_judge_user_prompt(
        probe_turns=probe_turns,
        agent_response=agent_response,
        candidate_clauses=candidate_clauses,
    )

    completions = 0
    judgment: Judgment | None = None
    verification = None

    # DESIGN.md 4.1: first attempt, then at most one retry naming the violation. The
    # retry stays at the same temperature because it is not a resample - the prompt
    # itself changed, so a different answer does not require a different temperature.
    for attempt in (1, 2):
        judgment = active.judge(
            system=JUDGE_SYSTEM_PROMPT, user=user_prompt, temperature=temp
        )
        completions += 1

        verification = verify_judgment(
            judgment,
            candidate_clauses=candidate_clauses,
            agent_response=agent_response,
        )
        if verification.ok:
            # L2 passing is not the same as a span having been verified. Point 4 of
            # `verify_judgment`'s docstring keeps the asymmetry deliberately: a
            # `denies` or `evasive` judgment may omit both spans, because a response
            # that commits to nothing has nothing to evidence. Such a judgment
            # passes L2 without any substring check having run, so the outcome is
            # None rather than True - and the audit row refuses True beside a null
            # `quoted_span` for exactly this reason. Reachable whenever L0 escalates
            # an `unclear` reply and L1 comes back with a spanless denial.
            return JudgeOutcome(
                source="llm",
                prefilter=prefilter_result,
                agent_stance=judgment.agent_stance,
                judgment=judgment,
                span_verified=True if judgment.quoted_span is not None else None,
                judge_model=active.model,
                judge_k=1,
                judge_completions=completions,
            )

        if attempt == 1:
            user_prompt = build_retry_user_prompt(
                previous_prompt=user_prompt,
                violations=verification.violation_text,
            )

    # Second failure. The judgment is kept for the human review queue but is not a
    # verdict, so `agent_stance` stays None - see the module docstring.
    assert verification is not None  # loop always runs at least once
    return JudgeOutcome(
        source="llm",
        prefilter=prefilter_result,
        agent_stance=None,
        judgment=judgment,
        span_verified=False,
        abstained=True,
        violations=verification.violations,
        judge_model=active.model,
        judge_k=1,
        judge_completions=completions,
    )
