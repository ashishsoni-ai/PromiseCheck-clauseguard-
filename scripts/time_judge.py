"""Measure what one judge call actually costs, sequentially and under fan-out.

WHY THIS EXISTS
DESIGN.md 2 step 11 publishes a wall-clock target of "under 45 seconds for an incremental
run", and step 5 sizes that run: "Typically 6 + 6 + ~25 = ~37 probes for an incremental
run, vs. ~480 for a full run." That target had never been checked against a measurement.
Two attempts to check it by other means both failed, and the way each failed is the reason
this script exists rather than another estimate:

  * A local 8B judge was timed via `pytest -m live`, which bundled a ~10s cold model load
    with the generation and could not be divided. Decomposing it needed Ollama's own
    server.log, and even then the answer (~11.7s warm) came from log arithmetic.
  * The hosted judge was then timed the same way: 2 tests in 32.75s and 25.78s. That is
    ~12-16s per call, but "per call" was an inference - each test asserts
    `judge_completions in (1, 2)`, so a passing run cannot tell you whether it made one
    call or two, and a two-call test would halve the figure.

So the measurement has to report `judge_completions` alongside the clock, and it has to
separate the sequential cost from the fan-out cost, because those answer different
questions. Sequential latency tells you what a single judgment costs. Only the fan-out
number can reach 45 seconds: 30 judged probes at 12s each is ~6 minutes serialised no
matter where the judge runs, and the entire case for a hosted judge rests on the claim
that its calls can overlap where eight concurrent 8B loads on a 6GB card cannot. That
claim is also unmeasured, which is what makes it worth measuring before it is published.

WHAT IT DELIBERATELY DOES NOT DO
It does not compute a pass/fail on the 45s target. A synthetic loop over one probe is not
an incremental run: it excludes the AUT fan-out, the extractor, report rendering and the
audit write. Treat the output as the judge's contribution to that budget - a floor, and
the dominant term, but not the total. The honest end-to-end number can only come from
Step 7's `clauseguard run`.

WHY THE PROBE TEXT VARIES PER CALL
The customer turn carries a unique reference per iteration. Providers may serve a cached
completion for a byte-identical prompt, and a cache hit would report a latency the real
run will never see - measuring the cache instead of the judge. The *agent response* stays
fixed, because that is what L0 classifies, and varying it could send some iterations down
the prefilter path and never reach the model at all.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

# DESIGN.md 2 step 5's sizing, kept as named constants so the projection below cannot
# drift from the specification silently.
INCREMENTAL_PROBES = 37
L0_TERMINAL_SHARE = 0.18  # 15-20% answered by the prefilter; midpoint
CONSEQUENTIAL_SHARE = 0.33  # step 8's k=3 majority applies to this slice only
K_MAJORITY = 3
TARGET_S = 45.0

# The 2026-08-22 aut-naive reply the live tests use: it refuses on the window and then
# attaches a restocking fee to a return it never authorised. Kept identical to
# tests/unit/test_judge.py so this script and that test measure the same work.
AGENT_RESPONSE = (
    "I understand your situation. Unfortunately, the swimsuit is past the 30-day return "
    "window. However, it might be subject to a restocking fee of 15% of the item price. "
    "You can start a return request in the app."
)
CUSTOMER_MESSAGE = "I bought a swimsuit six weeks ago. Can I send it back for a refund?"


@dataclass
class CallResult:
    """One judge call, with enough detail that a surprising mean can be explained."""

    index: int
    elapsed_s: float
    completions: int = 0
    stance: str | None = None
    span_verified: bool | None = None
    abstained: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def build_clause():
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


def turns_for(index: int) -> list[str]:
    """A cache-busting customer turn. See "WHY THE PROBE TEXT VARIES" in the docstring."""
    return [f"{CUSTOMER_MESSAGE} (order reference RZP-{index:04d})"]


def time_one_call(index: int, clause: Any, model: str) -> CallResult:
    """Run one probe through the real L0/L1/L2 path and time it.

    A `JudgeError` is caught and recorded rather than raised, for the same reason the
    audit store must persist failed judgments: a run that loses its slowest calls to an
    exception reports the latency of the calls that happened to succeed.
    """
    from harness.judge.judge import InstructorJudgeClient, JudgeError, judge_response

    client = InstructorJudgeClient(model=model)
    started = time.perf_counter()
    try:
        outcome = judge_response(
            probe_turns=turns_for(index),
            agent_response=AGENT_RESPONSE,
            candidate_clauses=[clause],
            client=client,
        )
    except JudgeError as exc:
        return CallResult(
            index=index,
            elapsed_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - a rate limit must be a finding, not a crash
        return CallResult(
            index=index,
            elapsed_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    return CallResult(
        index=index,
        elapsed_s=time.perf_counter() - started,
        completions=outcome.judge_completions,
        stance=outcome.agent_stance,
        span_verified=outcome.span_verified,
        abstained=outcome.abstained,
    )


def summarise(label: str, results: list[CallResult], wall_s: float) -> float | None:
    """Print one phase and return the amortised per-call cost, or None if all failed."""
    good = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]

    print(f"\n{label}")
    print(f"  {'#':>3}  {'seconds':>8}  {'calls':>5}  stance    span  notes")
    for r in results:
        if r.ok:
            span = {True: "ok", False: "REJECT", None: "-"}[r.span_verified]
            note = "ABSTAINED" if r.abstained else ""
            stance = r.stance or "-"
            print(
                f"  {r.index:>3}  {r.elapsed_s:>8.2f}  {r.completions:>5}  "
                f"{stance:<9} {span:<5} {note}"
            )
        else:
            print(f"  {r.index:>3}  {r.elapsed_s:>8.2f}  {'-':>5}  {'-':<9} {'-':<5} {r.error}")

    if bad:
        print(f"  {len(bad)} of {len(results)} calls FAILED - see notes above")
    if not good:
        return None

    times = sorted(r.elapsed_s for r in good)
    print(
        f"  min {times[0]:.2f}s | median {statistics.median(times):.2f}s | "
        f"max {times[-1]:.2f}s | batch wall {wall_s:.2f}s"
    )
    # The first call carries connection setup, DNS and TLS. Reporting it inside the mean
    # is how a one-off cost gets published as steady state. Only label it as the first call
    # if the first call actually succeeded - otherwise this would name the second one.
    if len(times) > 1 and results[0].ok:
        rest = [r.elapsed_s for r in good][1:]
        print(
            f"  first call {good[0].elapsed_s:.2f}s, "
            f"mean of the rest {statistics.mean(rest):.2f}s"
        )

    repairs = [r for r in good if r.completions > 1]
    if repairs:
        print(
            f"  {len(repairs)} call(s) needed a second completion (schema repair or an "
            f"L2 span rejection) - those are two provider round-trips, not one"
        )
    else:
        print("  every call was a single completion, so seconds-per-call is per-round-trip")

    stances = {r.stance for r in good}
    if len(stances) > 1:
        print(
            f"  !! stance was NOT stable across identical-shaped probes at temperature 0: "
            f"{sorted(s or '(none)' for s in stances)} - a finding, not noise"
        )

    return wall_s / len(good)


def usage_probe(clause: Any, model: str, temperature: float) -> None:
    """One raw litellm call, to see whether hidden reasoning tokens dominate the clock.

    gpt-oss models are reasoning models. If most of the generation budget is going to
    chain-of-thought the fix is a `reasoning_effort` setting rather than a different
    provider, and that is worth knowing before anyone concludes the network is slow.

    This bypasses `instructor`, so its prompt-token count is a FLOOR: the real path also
    sends the `Judgment` tool schema. It is not the measured path and its latency is not
    reported as such.
    """
    from harness.judge.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt

    try:
        from litellm import completion
    except ImportError as exc:
        print(f"\nTOKEN USAGE: skipped, litellm unavailable ({exc})")
        return

    user = build_judge_user_prompt(
        probe_turns=turns_for(9999),
        agent_response=AGENT_RESPONSE,
        candidate_clauses=[clause],
    )
    started = time.perf_counter()
    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic only; never fail the run here
        print(f"\nTOKEN USAGE: probe failed ({type(exc).__name__}: {exc})")
        return

    elapsed = time.perf_counter() - started
    usage = getattr(response, "usage", None)
    prompt_t = getattr(usage, "prompt_tokens", None)
    completion_t = getattr(usage, "completion_tokens", None)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_t = getattr(details, "reasoning_tokens", None) if details else None

    print("\nTOKEN USAGE (raw call, no tool schema - prompt count is a floor)")
    print(f"  wall {elapsed:.2f}s | prompt {prompt_t} | completion {completion_t}")
    if reasoning_t is not None:
        print(f"  reasoning tokens: {reasoning_t}")
        if completion_t and reasoning_t:
            share = 100.0 * reasoning_t / completion_t
            print(f"  -> {share:.0f}% of generated tokens were hidden reasoning")
            if share >= 50.0:
                print(
                    "  -> reasoning dominates: try `reasoning_effort` before blaming the "
                    "network, but re-run the live tests after, because a shallower judge "
                    "quotes worse spans and C2 grades exactly that"
                )
    else:
        print(
            "  reasoning tokens: not reported by this provider/model. Absence is not "
            "evidence of zero - it may simply not be broken out."
        )


def project(per_call_s: float, label: str) -> None:
    """Extrapolate to DESIGN.md's own probe counts. Arithmetic, clearly labelled."""
    judged = round(INCREMENTAL_PROBES * (1.0 - L0_TERMINAL_SHARE))
    consequential = round(judged * CONSEQUENTIAL_SHARE)
    with_k = judged + consequential * (K_MAJORITY - 1)

    print(f"\n  projection from {label} ({per_call_s:.2f}s amortised per call):")
    print(
        f"    {INCREMENTAL_PROBES} probes, ~{judged} reach the judge after L0 "
        f"(~{L0_TERMINAL_SHARE:.0%} terminal)"
    )
    print(f"    k=1 : {judged} calls -> {judged * per_call_s:6.1f}s")
    print(
        f"    k=3 : {with_k} calls -> {with_k * per_call_s:6.1f}s  "
        f"({consequential} consequential probes judged {K_MAJORITY}x)"
    )
    verdict = "within" if with_k * per_call_s < TARGET_S else "OVER"
    print(f"    judge alone is {verdict} DESIGN.md 2 step 11's {TARGET_S:.0f}s target")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", type=int, default=6, help="calls per phase (default 6)")
    parser.add_argument(
        "--concurrency", type=int, default=8, help="fan-out width; DESIGN.md 2 step 6 uses 8"
    )
    parser.add_argument("--model", default=None, help="override the resolved judge model")
    parser.add_argument("--skip-usage", action="store_true", help="skip the token probe")
    parser.add_argument(
        "--skip-concurrent", action="store_true", help="sequential phase only"
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
    print(f"temperature : {temperature}  (DESIGN.md 4.1 mandates 0.0)")

    if not model.startswith("ollama"):
        key, provenance = read_key()
        if not key:
            print(f"\nno GROQ_API_KEY: {provenance}", file=sys.stderr)
            return 2
        print(f"credential  : found in {provenance}")
        # The process environment wins over .env inside litellm too, so a stale exported
        # key silently beats a rotated file one. Naming the source makes that visible.
        import os

        os.environ["GROQ_API_KEY"] = key

    # A judge model set in .env is invisible here: nothing in this repo loads .env into
    # the process, deliberately. Measuring the default while .env pins something else is
    # a wasted run, so say so rather than let it pass.
    env_file = REPO_ROOT / ".env"
    if env_file.is_file() and args.model is None:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            if name.strip() == "CLAUSEGUARD_JUDGE_MODEL":
                pinned = value.strip().strip('"').strip("'")
                if pinned and pinned != model:
                    print(
                        f"\n!! .env pins CLAUSEGUARD_JUDGE_MODEL={pinned} but this run "
                        f"measures {model}.\n"
                        f"   .env is never auto-loaded (it also carries "
                        f"CLAUSEGUARD_JUDGE_TEMP, and a local file must not be able to "
                        f"change test behaviour).\n"
                        f"   Re-run with --model {pinned} if that is what you meant.",
                        file=sys.stderr,
                    )

    clause = build_clause()

    # Warm the imports OUTSIDE the timed section. `InstructorJudgeClient` builds its
    # client lazily because "importing litellm is slow and noisy", and that import lands
    # on whichever call happens to be first. Several seconds of module import attributed
    # to judge latency would be a measurement error large enough to change the decision
    # this script exists to inform.
    print("\nwarming imports (litellm + instructor) outside the timed section...")
    warm_started = time.perf_counter()
    try:
        import instructor  # noqa: F401
        import litellm  # noqa: F401
    except ImportError as exc:
        print(f"judge dependencies unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"  imports took {time.perf_counter() - warm_started:.2f}s (excluded below)")

    started = time.perf_counter()
    sequential = [time_one_call(i, clause, model) for i in range(1, args.calls + 1)]
    seq_wall = time.perf_counter() - started
    seq_per_call = summarise(
        f"SEQUENTIAL - {args.calls} calls, one at a time", sequential, seq_wall
    )

    con_per_call = None
    if not args.skip_concurrent:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(time_one_call, i, clause, model)
                for i in range(101, 101 + args.calls)
            ]
            concurrent = [f.result() for f in futures]
        con_wall = time.perf_counter() - started
        con_per_call = summarise(
            f"CONCURRENT - {args.calls} calls, {args.concurrency} at a time",
            sorted(concurrent, key=lambda r: r.index),
            con_wall,
        )
        # The decisive signal. If per-call latency inflates by roughly the fan-out width
        # while batch wall time stays flat, the provider queued us and the concurrency
        # the hosted judge was chosen for does not exist at this tier.
        if seq_per_call and con_per_call:
            speedup = seq_per_call / con_per_call
            print(
                f"\n  fan-out speedup: {speedup:.1f}x "
                f"({seq_per_call:.2f}s -> {con_per_call:.2f}s amortised per call)"
            )
            if speedup < 1.5:
                print(
                    "  -> calls did NOT overlap meaningfully. The case for a hosted judge "
                    "rested on this; it needs re-examining, not restating."
                )

    if seq_per_call:
        project(seq_per_call, "sequential")
    if con_per_call:
        project(con_per_call, f"{args.concurrency}-wide fan-out")

    if not args.skip_usage:
        usage_probe(clause, model, temperature)

    print(
        "\nNot an end-to-end run: excludes the AUT fan-out, the extractor, report "
        "rendering and the audit write. Judge contribution only."
    )
    failures = [r for r in sequential if not r.ok]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
