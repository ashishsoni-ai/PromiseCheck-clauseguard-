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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol, runtime_checkable

from harness.judge.prefilter import PrefilterResult, classify
from harness.judge.prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_prompt,
    build_retry_user_prompt,
)
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
#: The agent under test is Qwen, so this is Llama - and since 2026-08-23 a *local* Llama.
#:
#: WHY LOCAL, AND WHY THIS IS A CONFIG CHANGE RATHER THAN A DESIGN CHANGE
#: The previous pin, `groq/llama-3.3-70b-versatile`, was decommissioned by the provider and
#: began returning 404 `model_not_found`. DESIGN.md names no model ID anywhere except the
#: AUT's Qwen; 1.5 and 2 step 8 constrain the model *family*, because a family is a
#: property of the measurement while an ID is provider inventory that expires on someone
#: else's schedule. The Appendix anticipates precisely this - "judge and extractor via
#: `litellm` so model swaps are a config line - you will want that when a provider
#: rate-limits you at 11pm on day 12". So: one line, and the family requirement is met more
#: strongly than before. Judge (llama), AUT (qwen), adversary (mistral) and extractor (gpt)
#: are now four distinct families rather than the previous three, since the old judge and
#: adversary were both Llama and differed only in size.
#:
#: A local judge also takes the provider key out of the judge path entirely, which is worth
#: something: the 2026-08-23 rotation was forced by a pytest traceback printing the
#: `Authorization` header litellm passes as a frame argument.
#:
#: WHY `ollama_chat/` AND NOT `ollama/`
#: litellm routes `ollama/` at the legacy `/api/generate` completion endpoint and
#: `ollama_chat/` at `/api/chat`. `InstructorJudgeClient` needs the chat endpoint: instructor
#: defaults to TOOLS mode, i.e. function calling, which `/api/generate` does not carry.
#: `llama3.1` is one of the Ollama tags with native tool support, so the default mode holds
#: and SCHEMA_REPAIR_RETRIES stays a safety net rather than the mechanism.
DEFAULT_JUDGE_MODEL: Final = "ollama_chat/llama3.1:8b"
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

#: A ceiling on a failure, not a target. Raised from 120.0 on 2026-08-23 when the judge
#: moved to a local model: a cold Ollama load of a 7-8B model was measured at 81s on this
#: machine, and Ollama serialises generation by default, so a queued call behind a cold load
#: can plausibly exceed two minutes without anything being wrong.
#:
#: This is in tension with DESIGN.md 2 step 11's "under 45 seconds for an incremental run",
#: which a serialised local judge will not meet at ~37 probes - and step 8's k=3 majority on
#: the consequential class triples the calls that matter most. That is a Step 7 decision
#: about where the judge runs, not a reason to set a timeout that turns slowness into a
#: JudgeError: a timeout here loses the row entirely rather than abstaining (see "WHAT THE
#: ABSTAIN RATE IS ALLOWED TO MEAN"), so it must be generous and the speed problem must be
#: solved somewhere it can be seen.
DEFAULT_TIMEOUT_S: Final = 300.0


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
    ) -> None:
        self._model = model or resolve_judge_model()
        self._timeout_s = timeout_s
        self._client = None  # built lazily; importing litellm is slow and noisy

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
        try:
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
        except Exception as exc:  # noqa: BLE001 - provider detail belongs in the message
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
    * `span_verified` is None when no span was ever offered (L0 answered). False means
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
            return JudgeOutcome(
                source="llm",
                prefilter=prefilter_result,
                agent_stance=judgment.agent_stance,
                judgment=judgment,
                span_verified=True,
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
