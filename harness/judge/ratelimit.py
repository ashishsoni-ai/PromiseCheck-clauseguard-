"""Transient provider-failure retries for the hosted judge. DESIGN.md 4.1.

Two failures are retried below `judge()`: a 429 for quota, and Groq rejecting
its own model's tool call. The module name predates the second and is kept
anyway, because renaming it would churn every import and every test for no gain
in accuracy - what it actually means is "the retries the L1 control never sees".

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

AND WHY A MALFORMED TOOL CALL IS RETRIED HERE BUT ENDS SOMEWHERE ELSE

The second failure this module retries is Groq rejecting its own model's tool
call: `code: tool_use_failed`. It belongs here for the same reason a 429 does -
resending an identical prompt is a transport concern, and L1's one retry means
something specific that does not apply (name the L2 violation, ask again).

But it ends in the opposite place. A quota refusal that outlives its retries is
a `JudgeError`, because no judgment was ever attempted. A malformed tool call
that outlives its retries is an ABSTENTION, because the judge was asked three
times and could not return something readable - which is what L2's double
rejection already means, arriving one layer earlier. Ruled 2026-08-24, after a
live 30-probe run lost two rows to this, both on `expected=denies` probes, which
is exactly where over-promises live: losing them silently understated the
headline number, and abstaining at least reports them.

Two consequences, stated rather than left to be discovered. The published
abstain rate now partly measures the provider's JSON reliability - a real cost,
accepted deliberately, and paid only AFTER retries fail. And these retries are
visible to `judge_completions`, which is why `MalformedToolCallExhausted`
carries its attempt count: unlike a 429, every attempt here BURNED a generation,
and a burned generation the pacing cannot see is how a run walks into the rate
limit it was paced to avoid.

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

#: Total attempts for a malformed tool call, so this many minus one retries.
#: "Retry once or twice" was the ruling; two retries is the top of that range,
#: and the budget is set against token cost rather than wall clock. A 429 is free
#: to retry - the request was refused before the model ran - but a malformed tool
#: call means the model DID generate, so every attempt spends 1152-2178 tokens
#: out of a window holding 8000 a minute. Two rows retried twice on a 30-probe
#: run costs about one extra minute. A larger budget would trade the run's
#: headline latency for a recovery that, on the truncation case, either works
#: immediately or does not work at all.
MALFORMED_TOOL_CALL_ATTEMPTS: Final = 3

#: A malformed tool call has nothing to wait for: no amount of sleeping makes the
#: next sample better-formed, so this is a courtesy pause and not a backoff. It
#: is deliberately not `FALLBACK_WAITS_S`, which exists for the opposite case -
#: a quota window that genuinely does reopen with time.
MALFORMED_TOOL_CALL_WAIT_S: Final = 0.5

#: Groq's machine-readable code for "the model did not emit a usable tool call".
#: Matched as a literal token rather than by parsing the JSON body, because the
#: body reaches us already stringified inside a litellm message and its exact
#: shape - spacing, escaping, nesting depth - is a property of whichever layer
#: wrapped it rather than of the provider's contract.
MALFORMED_TOOL_CALL_CODE: Final = "tool_use_failed"

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


def is_malformed_tool_call(exc: BaseException) -> bool:
    """True if the provider rejected its OWN model's tool call as unusable.

    Two shapes were observed in the live run of 2026-08-24, both carrying
    `"type":"invalid_request_error","code":"tool_use_failed"`:

      * "Failed to parse tool call arguments as JSON", with a `failed_generation`
        that was valid JSON up to a missing closing brace; and
      * "Tool choice is required, but model did not call a tool", with
        `failed_generation` empty.

    Different symptoms, one class: the model was asked for a structured judgment
    and did not produce one the provider would forward. Both are resample-shaped,
    which is why both are retryable.

    WHY THE PREDICATE KEYS ON THE CODE AND NOT ON 400

    This is the part that has to stay narrow. Almost every other 400 the judge can
    provoke is a harness defect - a model id that no longer exists, a malformed
    request, a context length exceeded - and each is a bug we want to hear about
    loudly and immediately. Retrying those would spend quota to learn nothing and
    then, because exhausting these retries ABSTAINS rather than raising, file the
    bug away as judicial humility. That is strictly worse than crashing. So the
    provider's own code string is required, and a bare status of 400 is not
    enough.

    An explicit status is still allowed to veto: a body quoting `tool_use_failed`
    under a 401 or a 500 is not this failure wearing a costume, it is something
    else entirely, and the same asymmetry is argued at length in
    `is_rate_limited`. When no status is available we accept the code on its own.

    Note the residual risk, since it is the mirror of the one `is_rate_limited`
    names: the token could in principle arrive inside quoted prose rather than as
    a code. `tool_use_failed` is not English and does not occur in policy text or
    in a judge prompt by accident, so the exposure is a provider that starts
    echoing our request body back in unrelated errors - at which point the retry
    is harmless and the abstention is the thing to notice.
    """
    if MALFORMED_TOOL_CALL_CODE not in str(exc):
        return False
    status = _status_code(exc)
    return status is None or status == 400


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


class MalformedToolCallExhausted(RuntimeError):
    """The provider rejected its own model's tool call on every attempt.

    A distinct type because the caller's response has to be distinct. This is the
    one provider failure that ends in an abstention rather than a `JudgeError`,
    and `judge.py` can only tell the two apart if the exception does.

    `attempts` is not decoration. Every attempt counted here burned a generation
    against the token window, and `judge_response` adds this number to
    `judge_completions` so `runner.py` paces for tokens that were actually spent.
    Omitting it would make a recovered-then-abstained row look one call cheaper
    than it was, which is precisely the accounting error the 429 path is written
    to avoid in the other direction.

    There is no `slept_s` counterpart, unlike `RateLimitExhausted`: the waits here
    are a fixed courtesy pause carrying no information about the provider's state,
    so reporting their total would dress a constant up as a measurement.
    """

    def __init__(self, attempts: int, last: BaseException) -> None:
        super().__init__(
            f"the judge provider rejected its own model's tool call on all "
            f"{attempts} attempts. This is not a harness defect and not a quota "
            f"refusal: the model was asked for a structured judgment and did not "
            f"return one that could be read. DESIGN.md 4.1's retry-then-abstain "
            f"applies, so the row abstains rather than being lost "
            f"({type(last).__name__}: {last})"
        )
        self.attempts = attempts
        self.last = last


def call_with_malformed_tool_call_retry(
    call: Callable[[], T],
    *,
    attempts: int = MALFORMED_TOOL_CALL_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `call`, resampling when the provider rejects its own tool call.

    Anything that is not a malformed tool call propagates on the first raise,
    untouched - including `RateLimitExhausted`, which must reach `judge()` as
    itself so the operator is told about quota rather than about JSON.

    WHY THIS WRAPS THE RATE-LIMIT RETRY AND NOT THE OTHER WAY AROUND

    `judge()` nests these as malformed-outside, rate-limited-inside, and the
    ordering is load-bearing rather than arbitrary. This way each fresh sample
    gets the full provider-stated backoff budget, and the worst case is three
    token-burning attempts wrapped in at most nine refusals that cost nothing.
    Inverted, a 429 arriving on the second sample would restart the whole
    malformed loop, so the worst case becomes nine token-burning attempts for one
    row - roughly two and a half minutes of the token window to abstain once.
    """
    last: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - re-raised or wrapped below
            if not is_malformed_tool_call(exc):
                raise
            last = exc
            if attempt == attempts:
                break
            sleep(MALFORMED_TOOL_CALL_WAIT_S)

    assert last is not None  # only reachable via the malformed break
    raise MalformedToolCallExhausted(attempts, last) from last
