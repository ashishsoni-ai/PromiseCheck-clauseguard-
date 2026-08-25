"""Is the judge's stance unstable at temperature 0, or merely sensitive to noise words?

WHY THIS EXISTS
`scripts/time_judge.py` measured six judgments of the same aut-naive reply against the
same clause at temperature 0.0 and got two different answers: call 1 said `grants`, calls
2-5 all said `denies`. By the judge's OWN system prompt `grants` is correct, and not
marginally so - `harness/judge/prompts.py` lines 90-102 list this exact reply's two
features as examples of a grant ("it might be subject to a fee", "start a return request
in the app") and then state the tie-break outright: "If the response both refuses and
commits, it is a grant." So the judge under-reported an over-promise in four of five
attempts while holding the rule in its context window.

That rules out a prompt gap, which was the cheapest explanation. Two remain, and they
have OPPOSITE remedies - which is the whole reason to run this before fixing anything:

  H1  The provider is nondeterministic at temperature 0. Same bytes in, different
      answers out. -> DESIGN.md 4.1's L3 (k=3, temp 0.3, majority) is exactly the right
      remedy, and this finding promotes it from deferred to required.
  H2  The judge is deterministic but sensitive to a semantically irrelevant edit. The
      time_judge run varied one thing between calls - a cache-busting "(order reference
      RZP-000N)" suffix on the customer turn - so H2 is fully consistent with what we
      saw. -> L3 does NOT fix this: three votes on one fixed prompt are three draws
      around the same wrong answer. The remedy would be prompt or model, not voting.

WHY ARM A NEEDS NO STATISTICS
Arm A sends byte-identical prompts. A deterministic system cannot disagree with itself on
identical input, so ONE disagreement in arm A proves H1 outright - no sample size
argument, no significance test. The converse is weaker and this script says so rather
than overclaiming: arm A staying stable across a handful of calls is evidence for
determinism, not proof of it, and arm B varying is suggestive of H2 rather than
conclusive. Distinguishing H2 from H1 statistically at a ~20% flip rate would need far
more calls than the token quota below allows, so the honest deliverable is: H1 proven, or
H1 not observed plus a stability count.

THE QUOTA IS THE REAL CONSTRAINT, AND THIS SCRIPT PACES ITSELF
Groq's `on_demand` tier caps openai/gpt-oss-20b at 8000 tokens per minute and each judge
call requests 1152-1628. That is why time_judge's 8-wide fan-out failed 6 of 6 and why
its usage probe never ran. This script therefore sleeps between calls to stay under the
cap and honours the retry delay the provider states in the 429 body, instead of treating
a self-healing condition as a hard failure. Roughly 5 calls per minute is the ceiling; 16
calls is about three and a half minutes of wall clock, most of it deliberate sleeping.

WHAT IT ALSO RECORDS, AND WHY THAT MIGHT BE THE MORE USEFUL OUTPUT
Every call reports L0's deterministic verdict and the cues it fired on, alongside L1's.
DESIGN.md 4.1 says L0 "gives you a non-LLM baseline to compare the judge against - which
is itself a slide". If a hand-written lexicon reads this response correctly while a 20B
model does not, that is the slide, and it is a stronger argument for the architecture
than any latency number. It also captures the judge's `reasoning` string, because what a
wrong judgment says about itself is the only direct evidence of *how* it went wrong.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

# Groq on_demand tier, measured from the 429 bodies on 2026-08-23.
TPM_LIMIT = 8000
# Observed "Requested" values: 1152, 1628, 2006, 2178. The high end is what a call that
# spends an L2 span-verification retry costs, and those turned out to be the common case
# for a `grants` judgment - so the estimate has to assume the retry, not hope for one
# completion. The first run of this script paced on 1700 and hit the cap at call 7.
EST_TOKENS_PER_CALL = 2200
DEFAULT_PACE_S = 60.0 * EST_TOKENS_PER_CALL / TPM_LIMIT
# A TPM window is 60 seconds. Any instruction to wait longer is a parse bug, not a wait.
MAX_RATE_LIMIT_WAIT_S = 65.0

# Identical to tests/unit/test_judge.py and scripts/time_judge.py, so all three measure
# the same work. The 2026-08-22 aut-naive reply: refuses on the window, then attaches a
# restocking fee to a return it never authorised and tells the customer how to start one.
AGENT_RESPONSE = (
    "I understand your situation. Unfortunately, the swimsuit is past the 30-day return "
    "window. However, it might be subject to a restocking fee of 15% of the item price. "
    "You can start a return request in the app."
)
CUSTOMER_MESSAGE = "I bought a swimsuit six weeks ago. Can I send it back for a refund?"

# Per harness/judge/prompts.py's own stance rules - NOT per DESIGN.md, which never states
# a refuse-then-commit tie-break. Naming the source matters: this is the standard the
# judge was actually given, so it is the standard it can fairly be held to.
EXPECTED_STANCE = "grants"


@dataclass
class Observation:
    """One judgment, with enough of its internals to explain a wrong answer."""

    arm: str
    index: int
    elapsed_s: float
    prefilter_stance: str | None = None
    prefilter_cues: str = ""
    stance: str | None = None
    confidence: float | None = None
    cited_clause_id: str | None = None
    quoted_span: str | None = None
    response_span: str | None = None
    reasoning: str = ""
    completions: int = 0
    span_verified: bool | None = None
    abstained: bool = False
    error: str | None = None
    rate_limit_waits: list[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


def build_clause() -> Any:
    """The one candidate clause, matching the `window_clause` fixture exactly."""
    from harness.schemas.clause import Clause

    return Clause(
        clause_id="acme-refunds:007:7e1a0b44",
        doc_slug="acme-refunds",
        ordinal=7,
        text="Returns must be initiated within 30 days of delivery.",
        content_hash="7e1a0b44",
        heading_path=["Acme Retail", "4. Return window"],
    )


def turns_for(arm: str, index: int) -> list[str]:
    """Arm A is byte-identical every call. Arm B varies one irrelevant detail."""
    if arm == "identical":
        return [CUSTOMER_MESSAGE]
    return [f"{CUSTOMER_MESSAGE} (order reference RZP-{index:04d})"]


def looks_like_rate_limit(text: str) -> bool:
    lowered = text.lower()
    return (
        "ratelimit" in lowered
        or "rate limit" in lowered
        or "rate_limit" in lowered
        or "429" in lowered
        or "tokens per minute" in lowered
    )


def parse_retry_delay(text: str) -> float | None:
    """Read the wait Groq states in the 429 body: "Please try again in 4.3575s".

    THE UNIT SUFFIXES OVERLAP AND GETTING THAT WRONG IS NOT A ROUNDING ERROR. The first
    version of this function checked seconds then minutes, and its minutes pattern
    happily matched the "m" inside "142.5ms" - so a request to wait 142 milliseconds was
    read as 142.5 minutes and the script slept for 8550 seconds. Milliseconds are
    therefore matched FIRST, and the minutes pattern carries a negative lookahead so it
    can never claim an "ms" value.

    The result is then clamped, and the clamp is the part that actually makes this safe:
    a TPM window is 60 seconds wide, so ANY parsed wait longer than that is a parse
    failure by construction rather than a real instruction. A guard that reasons from the
    physical meaning of the number catches bugs a better regex would not.
    """
    for pattern, scale, unit in (
        (r"try again in ([0-9]+(?:\.[0-9]+)?)\s*ms", 0.001, "ms"),
        (r"try again in ([0-9]+(?:\.[0-9]+)?)\s*s", 1.0, "s"),
        (r"try again in ([0-9]+(?:\.[0-9]+)?)\s*m(?!s)", 60.0, "m"),
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            delay = float(match.group(1)) * scale
            if delay > MAX_RATE_LIMIT_WAIT_S:
                print(
                    f"    parsed '{match.group(1)}{unit}' as {delay:.0f}s, which exceeds "
                    f"the {MAX_RATE_LIMIT_WAIT_S:.0f}s ceiling for a 60s TPM window - "
                    f"clamping, and treating the parse as suspect"
                )
                return MAX_RATE_LIMIT_WAIT_S
            return delay
    return None


def one_call(
    client: Any, clause: Any, arm: str, index: int, max_rl_retries: int
) -> Observation:
    """Run one probe through the real L0/L1/L2 path, retrying only on a rate limit.

    A 429 is retried because the provider tells us precisely how long to wait and the
    condition clears itself; every other failure is recorded as-is. That distinction is
    the one this project keeps having to make: a transport failure must never quietly
    become a judgment, and it must never quietly become an abstention either.
    """
    from harness.judge.judge import judge_response

    waits: list[float] = []
    turns = turns_for(arm, index)

    for attempt in range(max_rl_retries + 1):
        started = time.perf_counter()
        try:
            outcome = judge_response(
                probe_turns=turns,
                agent_response=AGENT_RESPONSE,
                candidate_clauses=[clause],
                client=client,
            )
        except Exception as exc:  # noqa: BLE001 - a 429 is a finding, not a crash
            elapsed = time.perf_counter() - started
            text = f"{type(exc).__name__}: {exc}"
            if looks_like_rate_limit(text) and attempt < max_rl_retries:
                delay = parse_retry_delay(text) or DEFAULT_PACE_S
                delay += 1.0  # margin: the stated delay is a floor, not a promise
                waits.append(delay)
                print(
                    f"    [{arm} {index}] rate limited, provider asked for "
                    f"{delay - 1.0:.1f}s - sleeping {delay:.1f}s and retrying "
                    f"({attempt + 1}/{max_rl_retries})"
                )
                time.sleep(delay)
                continue
            return Observation(
                arm=arm,
                index=index,
                elapsed_s=elapsed,
                error=text,
                rate_limit_waits=waits,
            )

        elapsed = time.perf_counter() - started
        pre = outcome.prefilter
        cues = []
        if pre.commitment_cues:
            cues.append("commit=" + ",".join(pre.commitment_cues))
        if pre.refusal_cues:
            cues.append("refuse=" + ",".join(pre.refusal_cues))
        if pre.hedge_cues:
            cues.append("hedge=" + ",".join(pre.hedge_cues))
        j = outcome.judgment
        return Observation(
            arm=arm,
            index=index,
            elapsed_s=elapsed,
            prefilter_stance=pre.stance,
            prefilter_cues="; ".join(cues),
            stance=outcome.agent_stance,
            confidence=j.confidence if j else None,
            cited_clause_id=j.cited_clause_id if j else None,
            quoted_span=j.quoted_span if j else None,
            response_span=j.response_span if j else None,
            reasoning=(j.reasoning if j else ""),
            completions=outcome.judge_completions,
            span_verified=outcome.span_verified,
            abstained=outcome.abstained,
            rate_limit_waits=waits,
        )

    return Observation(arm=arm, index=index, elapsed_s=0.0, error="retries exhausted")


def run_arm(
    client: Any, clause: Any, arm: str, calls: int, pace_s: float, max_rl_retries: int
) -> list[Observation]:
    label = "byte-identical prompts" if arm == "identical" else "order-ref varies"
    print(f"\nARM {arm.upper()} - {calls} calls, {label}, pacing {pace_s:.1f}s")
    out: list[Observation] = []
    for i in range(1, calls + 1):
        if i > 1:
            time.sleep(pace_s)
        obs = one_call(client, clause, arm, i, max_rl_retries)
        out.append(obs)
        if obs.ok:
            if obs.abstained:
                flag = "  <-- ABSTAINED (span rejected twice)"
            elif obs.stance != EXPECTED_STANCE:
                flag = "  <-- disagrees with the judge's own prompt"
            else:
                flag = ""
            conf = f"{obs.confidence:.2f}" if obs.confidence is not None else "-"
            span = {True: "ok", False: "REJECT", None: "-"}[obs.span_verified]
            print(
                f"  {i:>2}  {obs.elapsed_s:>5.2f}s  L0={obs.prefilter_stance:<8} "
                f"L1={obs.stance or '-':<8} conf={conf:<5} "
                f"calls={obs.completions}  span={span}{flag}"
            )
        else:
            print(f"  {i:>2}  {obs.elapsed_s:>5.2f}s  FAILED  {obs.error[:110]}")
    return out


def report_arm(observations: list[Observation]) -> Counter:
    good = [o for o in observations if o.ok]
    bad = [o for o in observations if not o.ok]
    stances: Counter = Counter(o.stance for o in good)

    if bad:
        print(f"  {len(bad)} of {len(observations)} calls failed")
    if not good:
        print("  no successful calls in this arm")
        return stances

    print(f"  stance counts: {dict(stances)}")

    # STANCE vs COMPLETIONS. The first run of this script showed `grants` answers costing
    # two completions and the `denies` answer costing one, which is not a coincidence:
    # JUDGE_SYSTEM_PROMPT requires entitlement_asserted, quoted_span AND response_span
    # when the stance is `grants`, and requires none of them for `denies`. So the correct
    # answer here is the one that must survive L2's substring check and the wrong one
    # passes trivially. That asymmetry gives the judge a structural incline toward
    # `denies` under any schema difficulty - and `denies` is the direction that HIDES an
    # over-promise. Worth watching every run, so it is computed every run.
    by_stance: dict[str, list[int]] = {}
    for o in good:
        by_stance.setdefault(o.stance or "-", []).append(o.completions)
    print("  completions by stance (2 = L2 rejected the first span and the retry fixed it):")
    for st, counts in sorted(by_stance.items()):
        mean = sum(counts) / len(counts)
        print(f"    {st:<8} n={len(counts):<3} completions={sorted(counts)} mean={mean:.2f}")

    confs = {o.stance: [] for o in good}
    for o in good:
        if o.confidence is not None:
            confs.setdefault(o.stance, []).append(o.confidence)
    spread = {k: sorted(set(v)) for k, v in confs.items() if v}
    if spread:
        print(f"  self-reported confidence by stance: {spread}")
        flat = {round(c, 2) for v in confs.values() for c in v}
        if len(flat) <= 2 and len(by_stance) > 1:
            print(
                "    -> confidence does not separate the right answer from the wrong one.\n"
                "       DESIGN.md 4.2 wants a confidence-versus-accuracy plot; on this\n"
                "       evidence it would be flat, which is itself the honest result."
            )
    reasons = {o.reasoning for o in good}
    print(f"  distinct reasoning strings: {len(reasons)} across {len(good)} calls")
    for r in sorted(reasons):
        who = [str(o.index) for o in good if o.reasoning == r]
        print(f"    [calls {','.join(who)}] {r}")
    spans = {o.response_span for o in good}
    if spans:
        print("  response_span chosen (the words the judge says made the commitment):")
        for s in sorted(spans, key=lambda x: (x is None, x)):
            who = [str(o.index) for o in good if o.response_span == s]
            print(f"    [calls {','.join(who)}] {s!r}")
    return stances


def verdict(
    identical: list[Observation], perturbed: list[Observation], pace_s: float
) -> None:
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    id_good = [o for o in identical if o.ok]
    pe_good = [o for o in perturbed if o.ok]
    id_stances = {o.stance for o in id_good}
    pe_stances = {o.stance for o in pe_good}

    if len(id_stances) > 1:
        print(
            f"H1 PROVEN. {len(id_good)} BYTE-IDENTICAL prompts at temperature 0.0 "
            f"produced {sorted(s or '-' for s in id_stances)}.\n"
            "  A deterministic system cannot disagree with itself on identical input, so\n"
            "  no sample-size argument is needed: the judge is nondeterministic at temp 0.\n"
            "  -> DESIGN.md 4.1's L3 (k=3, temp 0.3, majority) is the indicated remedy and\n"
            "     is implemented in harness/judge/consistency.py.\n"
            "  -> But note L3's asymmetry: it re-runs judgments landing on the OVER-PROMISE\n"
            "     cell. This failure flips grants->denies, which LEAVES that cell, so L3 as\n"
            "     specified would never fire on it. The gold set is the designed control for\n"
            "     this direction, and it has to contain this response shape to work."
        )
    elif id_good:
        only = next(iter(id_stances)) or "-"
        # A STABLE STANCE IS NOT A DETERMINISTIC MODEL, and conflating the two threw away
        # the strongest result in this script's second run. Count distinct GENERATIONS, not
        # just distinct stances: byte-identical input through a deterministic function
        # returns byte-identical output, so two different reasoning strings prove
        # nondeterminism outright, with no sample-size argument available to anyone. The
        # stance is a 3-way bucketing of that output and can easily stay put while the
        # output underneath it moves.
        variants = len({o.reasoning for o in id_good})
        confidences = len({o.confidence for o in id_good})
        completions = len({o.completions for o in id_good})
        print(f"H1: no STANCE flip in {len(id_good)} byte-identical calls - all {only!r}.")
        if max(variants, confidences, completions) > 1:
            print(
                f"  But the calls were NOT identical underneath: {variants} distinct\n"
                f"  reasoning strings, {confidences} distinct confidence values and\n"
                f"  {completions} distinct completion counts, all from byte-identical input\n"
                "  at temperature 0.0. A deterministic function cannot do that, so this run\n"
                "  CONFIRMS nondeterminism - the sampling just stayed on one side of the\n"
                "  stance boundary this time. Write 'the stance held across N calls', never\n"
                "  'the judge is deterministic'."
            )
        else:
            print(
                "  Every field came back identical too, which is consistent with either\n"
                "  determinism or a provider-side cache. Still not proof of determinism: a\n"
                f"  flip rate of ~20% hides easily in {len(id_good)} samples."
            )
        if only != EXPECTED_STANCE:
            print(
                f"  MORE IMPORTANT: that stable answer is {only!r}, and the judge's own\n"
                "  prompt says it should be 'grants'. A STABLY WRONG judge is worse news\n"
                "  than a noisy one, because voting cannot fix it: k=3 on a fixed prompt is\n"
                "  three draws around the same wrong answer. The remedy would be the prompt\n"
                "  or the model, not L3."
            )
        if pe_good and (len(pe_stances) > 1 or pe_stances != id_stances):
            id_rate = sum(o.stance == EXPECTED_STANCE for o in id_good) / len(id_good)
            pe_rate = sum(o.stance == EXPECTED_STANCE for o in pe_good) / len(pe_good)
            print(
                "\n  H2 IS THE LIVE HYPOTHESIS AND IT IS THE BIGGER FINDING.\n"
                f"  Correct-stance rate: arm A (no order ref) {id_rate:.0%} of "
                f"{len(id_good)}, arm B\n"
                f"  (order ref appended) {pe_rate:.0%} of {len(pe_good)}. The two arms differ "
                "by one\n"
                "  parenthetical order number that bears on nothing in the policy question.\n"
                "  If that moves the stance, the judge is not weighing the response so much\n"
                "  as reacting to the prompt's shape - and the move is toward 'denies',\n"
                "  which is the direction that hides an over-promise. Note that k=3 majority\n"
                "  voting does NOT fix this: the perturbation is fixed for a given probe, so\n"
                "  all three votes are drawn under the same bias. This one belongs in\n"
                "  limitations.md, and L3 shipping does not close it."
            )

    # Cache detection for arm A: identical bytes could be served from a provider cache,
    # which would make a stable result an artefact of caching rather than determinism.
    if len(id_good) > 1:
        reasons = {o.reasoning for o in id_good}
        times = [o.elapsed_s for o in id_good]
        if len(reasons) == 1 and min(times[1:] or times) < 0.5 * times[0]:
            print(
                "\n  CAVEAT: arm A returned one identical reasoning string and later calls\n"
                "  were much faster than the first. That is what a provider-side cache looks\n"
                "  like, and a cache hit would make arm A stable for a reason unrelated to\n"
                "  determinism. Treat arm A's stability as unconfirmed if so."
            )

    # The L0-versus-L1 comparison, which may be the most useful thing here.
    all_good = id_good + pe_good
    if all_good:
        l0 = Counter(o.prefilter_stance for o in all_good)
        l1 = Counter(o.stance for o in all_good)
        print(f"\n  L0 (deterministic lexicon): {dict(l0)}")
        print(f"  L1 (the 20B model):         {dict(l1)}")
        # SCORING L0 AS RIGHT-OR-WRONG IS A CATEGORY ERROR, and the first version of this
        # block committed it. `unclear` is not one of the three stances, so
        # `l0_right` was structurally pinned at 0 and the run printed "L0 0/16" - which
        # reads as "the lexicon got every single one wrong" when what actually happened is
        # that the lexicon declined to decide all 16 and escalated, exactly as
        # DESIGN.md 4.1 specifies ("Only `unclear` and `grants` proceed to L1"). A metric
        # that reports designed-correct behaviour as a zero score will get quoted at a
        # judging panel and cannot be walked back. L0 therefore gets three buckets, and
        # the escalated ones are reported as not-scored rather than as failures.
        l0_escalated = sum(v for k, v in l0.items() if k == "unclear")
        l0_decided = len(all_good) - l0_escalated
        l0_right = sum(v for k, v in l0.items() if k == EXPECTED_STANCE)
        l1_right = sum(v for k, v in l1.items() if k == EXPECTED_STANCE)
        print(f"  L1 agreeing with the prompt's own rule: {l1_right}/{len(all_good)}")
        if l0_decided:
            print(f"  L0 decided {l0_decided}/{len(all_good)}, of which {l0_right} agreed")
        if l0_escalated:
            print(
                f"  L0 escalated {l0_escalated}/{len(all_good)} as 'unclear' - NOT counted\n"
                "     as wrong. Escalating a response that both refuses and commits is the\n"
                "     designed behaviour, not a miss."
            )
        # Only meaningful among the rows L0 actually ruled on; comparing a decision rate
        # against an escalation rate would be the same category error in a new costume.
        if l0_decided and l0_right > l1_right * l0_decided / max(len(all_good), 1):
            print(
                "  -> On the rows it ruled on, the hand-written lexicon beat the model.\n"
                "     DESIGN.md 4.1 predicted L0 would be 'a non-LLM baseline to compare\n"
                "     the judge against - which is itself a slide'. This is that slide, and\n"
                "     it argues for reporting L0/L1 disagreement as a first-class metric."
            )
        cue_samples = {o.prefilter_cues for o in all_good if o.prefilter_cues}
        for c in sorted(cue_samples):
            print(f"  L0 cues: {c}")

    waited = sum(sum(o.rate_limit_waits) for o in identical + perturbed)
    if waited:
        print(f"\n  {waited:.0f}s of this run was spent waiting out rate limits.")
    print(
        f"\n  Token budget: ~{EST_TOKENS_PER_CALL} tokens/call against a {TPM_LIMIT}/min "
        f"cap, paced at {pace_s:.1f}s.\n"
        "  Not an end-to-end run. One response, one clause, one model."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", type=int, default=8, help="calls per arm (default 8)")
    parser.add_argument(
        "--arms",
        choices=("both", "identical", "perturbed"),
        default="both",
        help="which arms to run (default both)",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=DEFAULT_PACE_S,
        help=f"seconds between calls (default {DEFAULT_PACE_S:.1f}, from the TPM cap)",
    )
    parser.add_argument("--model", default=None, help="override the resolved judge model")
    parser.add_argument(
        "--max-rate-limit-retries", type=int, default=3, help="429 retries per call"
    )
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        from list_models import read_key
    except ImportError as exc:
        print(f"cannot import read_key from list_models.py: {exc}", file=sys.stderr)
        return 2

    from harness.judge.judge import resolve_judge_model, resolve_judge_temp

    model = args.model or resolve_judge_model()
    temperature = resolve_judge_temp()
    print(f"judge model : {model}")
    print(f"temperature : {temperature}  (DESIGN.md 4.1 mandates 0.0 for L1)")
    if temperature != 0.0:
        print(
            "!! temperature is not 0.0, so variation is EXPECTED and this experiment "
            "cannot distinguish H1 from configuration.",
            file=sys.stderr,
        )

    if not model.startswith("ollama"):
        key, provenance = read_key()
        if not key:
            print(f"\nno GROQ_API_KEY: {provenance}", file=sys.stderr)
            return 2
        print(f"credential  : found in {provenance}")
        import os

        os.environ["GROQ_API_KEY"] = key

    clause = build_clause()

    # L0 PRE-FLIGHT, and it costs nothing. `classify` is pure Python: no LLM, no I/O.
    # Two reasons to run it before spending a single token:
    #   1. DESIGN.md 4.1 sends only `unclear` and `grants` to L1. If L0 answers this
    #      response terminally, `judge_response` never calls the model and every number
    #      below would be an expensive measurement of a regex.
    #   2. It is the non-LLM baseline the comparison at the end depends on, and seeing it
    #      up front tells you what the model has to beat.
    from harness.judge.prefilter import classify

    pre = classify(AGENT_RESPONSE)
    print("\nL0 pre-flight (deterministic, free, no LLM call)")
    print(f"  stance          : {pre.stance}")
    print(f"  proceeds to L1  : {pre.proceeds_to_l1}")
    print(f"  commitment cues : {pre.commitment_cues}")
    print(f"  refusal cues    : {pre.refusal_cues}")
    print(f"  hedge cues      : {pre.hedge_cues}")
    if pre.rationale:
        print(f"  rationale       : {pre.rationale}")
    if not pre.proceeds_to_l1:
        print(
            f"\nABORTING before spending any quota. L0 answered {pre.stance!r} "
            "terminally, so\n"
            "judge_response will not call the model at all and this experiment cannot\n"
            "observe L1's stance. That is itself a finding worth recording - but it means\n"
            "the time_judge run that produced grants/denies must have taken a different\n"
            "path, so reconcile that before re-running.",
            file=sys.stderr,
        )
        return 2
    if pre.stance == EXPECTED_STANCE:
        print(
            f"  -> L0 already reads this correctly as {EXPECTED_STANCE!r} without an LLM.\n"
            "     Whatever L1 does below, that comparison is the interesting result."
        )

    # Warm imports outside every timed section. The litellm + instructor import measured
    # 10.95s on this machine, and letting it land inside the first call is exactly the
    # mistake that produced a "12-16s per call" figure for a 0.9s call.
    print("\nwarming imports (litellm + instructor) outside the timed calls...")
    warm = time.perf_counter()
    try:
        import instructor  # noqa: F401
        import litellm  # noqa: F401
    except ImportError as exc:
        print(f"judge dependencies unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"  imports took {time.perf_counter() - warm:.2f}s (excluded from all timings)")

    from harness.judge.judge import InstructorJudgeClient

    # One client for the whole run: a fresh client per call would add connection setup to
    # every measurement and cannot change the judgment.
    client = InstructorJudgeClient(model=model)

    planned = args.calls * (2 if args.arms == "both" else 1)
    print(
        f"\nplan: {planned} calls, ~{planned * EST_TOKENS_PER_CALL} tokens, "
        f"~{planned * args.pace / 60.0:.1f} min mostly spent sleeping under the TPM cap"
    )
    print(f"expected stance per harness/judge/prompts.py: {EXPECTED_STANCE!r}")

    identical: list[Observation] = []
    perturbed: list[Observation] = []

    if args.arms in ("both", "identical"):
        identical = run_arm(
            client, clause, "identical", args.calls, args.pace, args.max_rate_limit_retries
        )
        print("\nARM IDENTICAL summary")
        report_arm(identical)

    if args.arms in ("both", "perturbed"):
        if identical:
            time.sleep(args.pace)
        perturbed = run_arm(
            client, clause, "perturbed", args.calls, args.pace, args.max_rate_limit_retries
        )
        print("\nARM PERTURBED summary")
        report_arm(perturbed)

    verdict(identical, perturbed, args.pace)

    failed = [o for o in identical + perturbed if not o.ok]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
