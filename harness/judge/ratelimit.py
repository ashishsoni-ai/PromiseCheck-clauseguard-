"""Rate-limit detection and provider-stated backoff for the hosted judge.

WHY THIS IS ITS OWN MODULE

Two reasons, both about keeping it testable.

It must not import litellm. `judge.py` builds its client lazily precisely
because importing litellm costs ~11 seconds per process, and detection here
therefore cannot do `except litellm.RateLimitError`. Everything below is
duck-typed against the exception's shape - status code, class name, message -
so the offline suite can raise a synthetic 429 and assert the whole retry path
without the provider SDK present at all.

And it must be exercisable without spending quota. A 429 is the one failure
mode that is impossible to provoke on demand and expensive to provoke by
accident, so the arithmetic - what we parse, how long we wait, when we give up
- is pure functions over an exception object plus an injected `sleep`.

WHY A 429 IS RETRIED HERE AND NOT BY THE L1 CONTROL

This is the load-bearing distinction in this file. DESIGN.md 4.1 gives the
judge one retry, and 4.2 publishes the abstain rate as a reported metric. That
retry budget is about JUDGMENT VALIDITY: L2 rejected the quoted spans, we name
the violation, we ask once more, and a second failure abstains.

A rate limit is not a judgment failure. It is the provider declining to be
asked. If a 429 consumed the L1 retry, then:

  * the published abstain rate would partly measure Groq's quota rather than
    the judge's behaviour, which makes a reported number mean two things; and
  * `judge_completions` would over-count, and since `runner.py` paces on
    `pace_s * max(1, judge_completions)`, a rejected call that burned no tokens
    would buy itself extra sleep it did not earn.

So retries live BELOW `judge()`'s return: invisible to the L1 loop, absent
from `completions`, absent from the abstain rate. What the operator sees if we
exhaust them is a `JudgeError` naming the quota, because at that point the fix
is to wait or change tier, not to debug the harness.

WHY WE HONOUR THE STATED DELAY RATHER THAN GUESSING

Groq states the exact wait in the error body - "Please try again in 4.3575s" -
which is strictly better information than any backoff curve we could invent,
because it is computed from our own token window. Blind exponential backoff
either sleeps too long (wasting a run that is already ~8 minutes) or too short
(burning an attempt on a window that provably has not reopened). We add a
small fixed margin because sleeping the stated duration exactly races the
boundary, and we fall back to exponential only when nothing parseable is
offered.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Final, TypeVar

T = TypeVar("T")

#: Total attempts, so this many minus one actual waits. Three is chosen against
#: the run it has to survive: a 30-probe run is ~8 minutes of deliberate pacing,
#: and the pacing is what makes a 429 unlikely in the first place. If three
#: provider-stated waits in a row do not clear it, the budget is gone rather
#: than momentarily tight, and sleeping longer just delays the same answer.
RATE_LIMIT_ATTEMPTS: Final = 3

#: Added to whatever the provider asks for. Sleeping exactly the stated wait
#: races the window boundary - the provider computes it at response time, and
#: our clock is not theirs. A quarter second costs nothing against an 8-minute
#: run and removes an entire class of "retried too early" failures.
RETRY_MARGIN_S: Final = 0.25

#: Refuse to sleep longer than this per attempt however large the stated wait.
#: A provider that asks for ten minutes is telling us the run is not viable
#: now, and silently blocking for ten minutes inside what looks like a judge
#: call is worse than failing with a message that says so.
MAX_WAIT_S: Final = 60.0

#: Used only when the error carries no parseable delay. Deliberately coarse:
#: this is the ignorant path, and the whole point of the module is that we
#: normally are not on it.
FALLBACK_WAITS_S: Final = (5.0, 15.0)

# "Please try again in 4.3575s" / "try again in 1m2.646s" / "in 500ms".
# Milliseconds are matched first because `1m` in the seconds pattern is guarded
# by a negative lookahead against `ms`, and checking ms first keeps that
# subtlety from being the only thing standing between us and a 500-second sleep.
_WAIT_MS = re.compile(r"try\s+again\s+in\s+([\d.]+)\s*ms", re.IGNORECASE)
_WAIT_S = re.compile(
    r"try\s+again\s+in\s+(?:(\d+)\s*m(?!s))?\s*([\d.]+)\s*s",
    re.IGNORECASE,
)


def _status_code(exc: BaseException) -> int | None:
    """Dig a status code out of whichever shape the SDK used this week."""
    for probe in (exc, getattr(exc, "response", None)):
        code = getattr(probe, "status_code", None)
        if isinstance(code, int):
            return code
        code = getattr(probe, "status", None)
        if isinstance(code, int):
            return code
    return None


def is_rate_limited(exc: BaseException) -> bool:
    """True if `exc` looks like a 429 from any of the layers we sit under.

    Three tells, tried in order of authority, because the exception we actually
    see depends on whether litellm, instructor or httpx got to wrap it first and
    a false negative here is a lost run.

    An explicit status code decides on its own, and is allowed to say NO as well
    as yes. That asymmetry matters: a class called `RateLimitError` carrying a
    403 is an auth problem wearing the wrong name, and retrying it burns three
    provider-stated waits inside a run that is already ~8 minutes to learn
    something the status code said immediately. Only when no status is available
    do we fall back to guessing from the class name and the message.

    Note what is deliberately NOT a tell: the bare phrase "rate limit" anywhere
    in the message. A judge whose *prompt* mentioned rate limits, or a policy
    clause about limits, would otherwise make an unrelated failure look
    retryable. We accept the provider's machine-readable code string instead,
    which does not occur by accident.
    """
    status = _status_code(exc)
    if status is not None:
        return status == 429
    if "ratelimit" in type(exc).__name__.lower().replace("_", ""):
        return True
    return "rate_limit_exceeded" in str(exc).lower()


def retry_after_s(exc: BaseException) -> float | None:
    """The provider's own stated wait in seconds, or None if it did not say.

    Prefers a `Retry-After` header when present because it is the standardised
    channel, then falls back to parsing the message body, which is where Groq
    actually puts the useful number.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
        except AttributeError:  # pragma: no cover - not a mapping
            raw = None
        if raw is not None:
            try:
                return max(0.0, float(str(raw).strip()))
            except ValueError:
                pass  # HTTP-date form; the message body is more precise anyway

    message = str(exc)
    ms = _WAIT_MS.search(message)
    if ms is not None:
        return max(0.0, float(ms.group(1)) / 1000.0)

    sec = _WAIT_S.search(message)
    if sec is not None:
        minutes = float(sec.group(1) or 0.0)
        return max(0.0, minutes * 60.0 + float(sec.group(2)))

    return None


def wait_for(exc: BaseException, *, attempt: int) -> float:
    """How long to sleep before retrying, given what the provider told us.

    `attempt` is 1-based and only selects a fallback, so a provider that states
    its delay gets honoured identically on every attempt rather than having our
    guesswork added on top of its arithmetic.
    """
    stated = retry_after_s(exc)
    if stated is None:
        index = min(attempt, len(FALLBACK_WAITS_S)) - 1
        return FALLBACK_WAITS_S[index]
    return min(stated + RETRY_MARGIN_S, MAX_WAIT_S)


class RateLimitExhausted(RuntimeError):
    """Every attempt was refused for quota. Carries the last provider error.

    A distinct type so the caller can say "quota, not defect" in the message it
    surfaces. An operator who reads "the judge failed" goes looking for a bug;
    one who reads "the tier's token budget refused three paced attempts" waits
    or changes tier, which is the actual remedy.
    """

    def __init__(self, attempts: int, slept_s: float, last: BaseException) -> None:
        super().__init__(
            f"the judge provider refused {attempts} attempts for rate limiting "
            f"after waiting {slept_s:.1f}s in total. This is a token budget, not "
            f"a harness defect: DESIGN.md 2's 45-second target is not reachable "
            f"on Groq's on_demand tier and the run is paced for it, so a refusal "
            f"here means the budget is spent rather than momentarily tight "
            f"({type(last).__name__}: {last})"
        )
        self.attempts = attempts
        self.slept_s = slept_s
        self.last = last


def call_with_rate_limit_retry(
    call: Callable[[], T],
    *,
    attempts: int = RATE_LIMIT_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `call`, honouring provider-stated waits on 429 and nothing else.

    Anything that is not a rate limit propagates on the first raise, untouched.
    That matters: a schema failure, a timeout or a bad key must not be silently
    retried into a delay, because each has a different remedy and only one of
    them gets better by waiting.

    `sleep` is injected so the offline suite asserts the arithmetic - which
    waits, in which order, totalling what - without any real elapsed time.
    """
    slept = 0.0
    last: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - re-raised or wrapped below
            if not is_rate_limited(exc):
                raise
            last = exc
            if attempt == attempts:
                break
            pause = wait_for(exc, attempt=attempt)
            sleep(pause)
            slept += pause

    assert last is not None  # only reachable via the rate-limited break
    raise RateLimitExhausted(attempts, slept, last) from last
