"""Tests for the judge's transient-provider-failure retries: 429s and malformed
tool calls.

Everything here runs without litellm, without a key and without elapsed time,
which is the whole point of `ratelimit.py` being a separate module: a 429 is
impossible to provoke on demand and expensive to provoke by accident, so the
arithmetic has to be testable from synthetic exceptions.

`FakeProviderError` mimics the shape litellm hands us rather than subclassing
anything real, precisely to prove the detection is duck-typed. If someone later
makes `is_rate_limited` do `except litellm.RateLimitError`, these tests stop
being runnable without the SDK, which is the regression to notice.

Note the fakes are named so that they do NOT contain the substring "ratelimit".
That is not cosmetic. Detection has three independent tells, and a fake called
`FakeRateLimit` satisfies the class-name tell by accident - so the tests that
mean to exercise the status code would have gone green without the status code
ever being read. The first draft of this file had exactly that bug.

The same discipline applies to the malformed-tool-call fakes below, for the
opposite reason: that detection has exactly one positive tell, the literal token
`tool_use_failed`, so a fake whose class name or message mentions it in passing
would pass the negative tests for the wrong reason. The negatives here carry no
such token unless the test is specifically about a status code vetoing one.
"""

import pytest

from harness.judge.ratelimit import (
    FALLBACK_WAITS_S,
    MALFORMED_TOOL_CALL_ATTEMPTS,
    MALFORMED_TOOL_CALL_WAIT_S,
    MAX_WAIT_S,
    RETRY_MARGIN_S,
    MalformedToolCallExhausted,
    RateLimitExhausted,
    call_with_malformed_tool_call_retry,
    call_with_rate_limit_retry,
    is_malformed_tool_call,
    is_rate_limited,
    retry_after_s,
    wait_for,
)

# The real thing, copied from a Groq refusal on 2026-08-23.
GROQ_BODY = (
    "litellm.RateLimitError: RateLimitError: GroqException - "
    '{"error":{"message":"Rate limit reached for model '
    "`openai/gpt-oss-20b` in organization `org_x` service tier `on_demand` on "
    ": Limit 8000, Used 7684, Requested 1152. Please try again in 4.3575s. "
    'Need more tokens?","type":"tokens","code":"rate_limit_exceeded"}}'
)

# Both real bodies, copied out of `runs.db` after the live run of 2026-08-24 lost
# two rows to them. Different messages, one code, one class of failure: the model
# was asked for a structured judgment and did not produce one the provider would
# forward. The first is a truncated generation, the second is no generation at all.
#
# Kept whole, including the `failed_generation` payload, rather than trimmed to the
# `code` the predicate actually greps for. A shortened fixture would still pass every
# test here and would quietly stop being evidence of what the provider sends - and it
# is the payload that shows what these two rows cost: a stance was decided, a clause
# was cited, a confidence was set, and all of it was discarded for a missing brace.
TRUNCATED_TOOL_CALL = (
    "litellm.BadRequestError: GroqException - "
    '{"error":{"message":"Failed to parse tool call arguments as JSON",'
    '"type":"invalid_request_error","code":"tool_use_failed",'
    '"failed_generation":"'
    '{\\"name\\": \\"Judgment\\", \\"arguments\\": {\\"agent_stance\\":\\"grants\\",'
    '\\"cited_clause_id\\":\\"acme-refunds:006:0dbae093\\",'
    '\\"confidence\\":0.9,\\"entitlement_asserted\\":\\"refund\\"}"}}'
)
NO_TOOL_CALL_AT_ALL = (
    "litellm.BadRequestError: GroqException - "
    '{"error":{"message":"Tool choice is required, but model did not call a tool",'
    '"type":"invalid_request_error","code":"tool_use_failed","failed_generation":""}}'
)

# An opaque message, for tests that must not pass via the message tell.
OPAQUE = "the provider declined to elaborate"


def raiser(exc):
    """A zero-arg callable that always raises `exc`.

    Spelled out rather than using the `(_ for _ in ()).throw(exc)` trick, which
    is compact and unreadable and depends on generator-throw semantics that are
    not the thing under test here.
    """

    def call():
        raise exc

    return call


class FakeProviderError(Exception):
    """Carries a status code the way the SDK exceptions do.

    The name matters: it must not contain "ratelimit", or every test below
    passes through the class-name tell whatever else is broken.
    """

    def __init__(self, message=GROQ_BODY, status_code=429):
        super().__init__(message)
        self.status_code = status_code


class FakeResponse:
    def __init__(self, headers=None, status_code=429):
        self.headers = headers or {}
        self.status_code = status_code


class WrappedProviderError(Exception):
    """No status code of its own - only a nested response, as httpx wrappers do."""

    def __init__(self, message=GROQ_BODY, headers=None):
        super().__init__(message)
        self.response = FakeResponse(headers)


class Recorder:
    """An injected sleep that records instead of waiting."""

    def __init__(self):
        self.waits = []

    def __call__(self, seconds):
        self.waits.append(seconds)

    @property
    def total(self):
        return sum(self.waits)


class TestDetection:
    def test_a_status_code_of_429_is_enough(self):
        """OPAQUE, not GROQ_BODY: the body carries `rate_limit_exceeded`, so
        using it here would let the message tell answer for the status tell."""
        assert is_rate_limited(FakeProviderError(OPAQUE, status_code=429))

    def test_a_nested_response_status_is_found(self):
        assert is_rate_limited(WrappedProviderError(OPAQUE))

    def test_the_class_name_is_enough_without_any_status(self):
        class RateLimitError(Exception):
            pass

        assert is_rate_limited(RateLimitError("no status, no code"))

    def test_the_providers_machine_readable_code_is_enough(self):
        assert is_rate_limited(Exception(GROQ_BODY))

    @pytest.mark.parametrize(
        "exc",
        [
            TimeoutError("timed out"),
            ValueError("malformed judgment"),
            Exception("authentication failed: invalid api key"),
            Exception("500 internal server error"),
        ],
    )
    def test_other_failures_are_not_rate_limits(self, exc):
        """Each of these has a different remedy and none improves by waiting."""
        assert not is_rate_limited(exc)

    def test_prose_mentioning_rate_limits_is_not_a_rate_limit(self):
        """Deliberately excluded. A policy clause or judge prompt discussing rate
        limits must not make an unrelated failure look retryable, so detection
        keys on the machine-readable code, not the English phrase."""
        assert not is_rate_limited(ValueError("the clause mentions a rate limit"))

    def test_a_403_is_not_retried(self):
        assert not is_rate_limited(FakeProviderError("forbidden", status_code=403))

    def test_an_explicit_status_overrides_the_weaker_tells(self):
        """The asymmetry in `is_rate_limited`, pinned. A 403 whose class is
        literally named RateLimitError and whose body quotes the 429 code is
        still not retried, because the status code is the one signal the
        provider set on purpose. Found by a fake that was accidentally named
        `FakeRateLimit` and passed three tests without its status being read."""

        class RateLimitError(Exception):
            def __init__(self):
                super().__init__(GROQ_BODY)
                self.status_code = 403

        assert not is_rate_limited(RateLimitError())


class TestParsingTheStatedWait:
    def test_it_reads_the_real_groq_body(self):
        assert retry_after_s(FakeProviderError()) == pytest.approx(4.3575)

    def test_it_reads_minutes_and_seconds(self):
        exc = Exception("Please try again in 1m2.646s.")
        assert retry_after_s(exc) == pytest.approx(62.646)

    def test_it_reads_milliseconds(self):
        """`500ms` must not be read as 500 seconds. The seconds pattern guards
        against it with a negative lookahead, and ms is checked first."""
        assert retry_after_s(Exception("try again in 500ms")) == pytest.approx(0.5)

    def test_a_retry_after_header_wins(self):
        exc = WrappedProviderError(GROQ_BODY, headers={"retry-after": "9"})
        assert retry_after_s(exc) == pytest.approx(9.0)

    def test_an_http_date_header_falls_through_to_the_body(self):
        """`Retry-After` may be an HTTP date. Rather than parse a date against a
        clock that is not the provider's, fall through to the body, which states
        a duration and needs no clock at all."""
        exc = WrappedProviderError(
            GROQ_BODY, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
        )
        assert retry_after_s(exc) == pytest.approx(4.3575)

    def test_nothing_parseable_is_none_not_zero(self):
        """None and 0.0 must stay distinguishable: 0.0 would mean "retry now",
        which is exactly the wrong response to an unparseable refusal."""
        assert retry_after_s(Exception("rate_limit_exceeded")) is None


class TestWaitFor:
    def test_the_stated_wait_gets_a_margin(self):
        assert wait_for(FakeProviderError(), attempt=1) == pytest.approx(
            4.3575 + RETRY_MARGIN_S
        )

    def test_the_margin_does_not_compound_across_attempts(self):
        """A provider that states its delay is honoured identically each time.
        Adding our own escalation on top would double-count arithmetic the
        provider computed from our own token window."""
        exc = FakeProviderError()
        assert wait_for(exc, attempt=1) == wait_for(exc, attempt=2)

    def test_an_absurd_stated_wait_is_capped(self):
        exc = Exception("rate_limit_exceeded. Please try again in 10m0s.")
        assert wait_for(exc, attempt=1) == MAX_WAIT_S

    def test_an_unparseable_refusal_escalates(self):
        exc = Exception("rate_limit_exceeded")
        first = wait_for(exc, attempt=1)
        second = wait_for(exc, attempt=2)
        assert (first, second) == FALLBACK_WAITS_S
        assert second > first

    def test_the_fallback_does_not_index_past_its_table(self):
        exc = Exception("rate_limit_exceeded")
        assert wait_for(exc, attempt=99) == FALLBACK_WAITS_S[-1]


class TestRetryLoop:
    def test_a_call_that_succeeds_never_sleeps(self):
        sleep = Recorder()
        assert call_with_rate_limit_retry(lambda: "ok", sleep=sleep) == "ok"
        assert sleep.waits == []

    def test_it_returns_the_value_once_the_window_reopens(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise FakeProviderError()
            return "judged"

        sleep = Recorder()
        assert call_with_rate_limit_retry(flaky, sleep=sleep) == "judged"
        assert len(calls) == 3
        assert sleep.waits == [
            pytest.approx(4.3575 + RETRY_MARGIN_S),
            pytest.approx(4.3575 + RETRY_MARGIN_S),
        ]

    def test_it_does_not_sleep_after_the_final_attempt(self):
        """Sleeping before giving up wastes the operator's time to no purpose:
        there is no attempt left to protect."""
        sleep = Recorder()
        with pytest.raises(RateLimitExhausted):
            call_with_rate_limit_retry(
                raiser(FakeProviderError()), sleep=sleep, attempts=3
            )
        assert len(sleep.waits) == 2

    def test_a_non_rate_limit_propagates_on_the_first_raise(self):
        """Never retried, never delayed. A bad key or a schema failure does not
        get better by waiting, and hiding it behind two sleeps makes it slower
        to diagnose."""
        sleep = Recorder()
        with pytest.raises(TimeoutError):
            call_with_rate_limit_retry(raiser(TimeoutError("nope")), sleep=sleep)
        assert sleep.waits == []

    def test_exhaustion_reports_the_total_wait_and_keeps_the_cause(self):
        sleep = Recorder()
        with pytest.raises(RateLimitExhausted) as caught:
            call_with_rate_limit_retry(
                raiser(FakeProviderError()), sleep=sleep, attempts=3
            )
        exc = caught.value
        assert exc.attempts == 3
        assert exc.slept_s == pytest.approx(sleep.total)
        assert isinstance(exc.last, FakeProviderError)
        assert isinstance(exc.__cause__, FakeProviderError)

    def test_the_message_names_the_budget_not_a_defect(self):
        """An operator who reads "the judge failed" hunts a bug that is not
        there. The remedy for this failure is to wait or change tier, so the
        message has to say which of the two situations it is."""
        sleep = Recorder()
        with pytest.raises(RateLimitExhausted) as caught:
            call_with_rate_limit_retry(
                raiser(FakeProviderError()), sleep=sleep, attempts=2
            )
        text = str(caught.value).lower()
        assert "token budget" in text
        assert "not a harness defect" in text

    def test_one_attempt_means_no_retry_at_all(self):
        sleep = Recorder()
        with pytest.raises(RateLimitExhausted):
            call_with_rate_limit_retry(
                raiser(FakeProviderError()), sleep=sleep, attempts=1
            )
        assert sleep.waits == []


class SchemaRefusal(Exception):
    """A 400 from the provider, shaped the way litellm hands one over.

    Named for what happened rather than for what detects it: nothing in
    "SchemaRefusal" contains `tool_use_failed`, so a test using it can only pass
    through the message body it was given, which is the tell under test.
    """

    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


class TestMalformedToolCallDetection:
    @pytest.mark.parametrize(
        "body",
        [TRUNCATED_TOOL_CALL, NO_TOOL_CALL_AT_ALL],
        ids=["truncated-generation", "no-generation"],
    )
    def test_both_real_bodies_are_recognised(self, body):
        """Two different messages, one `code`. Parametrised over both because the
        second was the one that returned nothing at all, and a predicate that only
        handled the truncation case would have silently kept losing it."""
        assert is_malformed_tool_call(SchemaRefusal(body))

    def test_it_does_not_require_a_status_code(self):
        """Whichever of litellm, instructor or httpx wrapped it last may not have
        carried one forward, and a false negative here loses a row."""
        assert is_malformed_tool_call(Exception(TRUNCATED_TOOL_CALL))

    @pytest.mark.parametrize(
        "exc",
        [
            SchemaRefusal("context_length_exceeded: 8192 < 9001"),
            SchemaRefusal("Invalid value for 'response_format'"),
            SchemaRefusal("model `openai/gpt-oss-20b` does not exist", status_code=404),
            TimeoutError("timed out"),
            Exception(GROQ_BODY),
        ],
        ids=["context-length", "bad-request", "dead-model", "timeout", "rate-limit"],
    )
    def test_no_other_failure_is_retried_as_one(self, exc):
        """THE POINT OF KEEPING THE PREDICATE NARROW. Three of these are 400s, and
        every one of them is a harness defect: a prompt that outgrew the context
        window, a malformed request, a model id that expired on someone else's
        schedule. Retrying them would spend quota to learn nothing and then - because
        exhausting these retries ABSTAINS rather than raises - file the bug away as
        judicial humility. Crashing is strictly better."""
        assert not is_malformed_tool_call(exc)

    @pytest.mark.parametrize("status", [401, 403, 500, 502])
    def test_an_explicit_non_400_status_vetoes_the_code(self, status):
        """Same asymmetry `is_rate_limited` argues for at length. A body quoting
        `tool_use_failed` under a 401 is not this failure in disguise, it is
        something else entirely, and resampling it three times delays the real
        answer by three generations."""
        assert not is_malformed_tool_call(SchemaRefusal(TRUNCATED_TOOL_CALL, status))

    def test_the_two_detectors_do_not_overlap(self):
        """They lead to opposite endings - `JudgeError` for quota, an abstention for
        a malformed tool call - and `judge()` nests them, so one body satisfying both
        would make which ending you get depend on nesting order."""
        for body in (TRUNCATED_TOOL_CALL, NO_TOOL_CALL_AT_ALL):
            assert not is_rate_limited(SchemaRefusal(body))
        assert not is_malformed_tool_call(FakeProviderError())


class TestMalformedToolCallRetryLoop:
    def test_a_call_that_succeeds_never_sleeps(self):
        sleep = Recorder()
        assert call_with_malformed_tool_call_retry(lambda: "ok", sleep=sleep) == "ok"
        assert sleep.waits == []

    def test_a_resample_can_recover_the_row(self):
        """The case the whole change exists for: the truncation was transient, the
        second sample parses, and no abstention is booked at all."""
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise SchemaRefusal(TRUNCATED_TOOL_CALL)
            return "judged"

        sleep = Recorder()
        assert call_with_malformed_tool_call_retry(flaky, sleep=sleep) == "judged"
        assert len(calls) == 2

    def test_the_pause_is_flat_and_short(self):
        """Not `FALLBACK_WAITS_S`, and not escalating. There is nothing to wait for:
        no amount of sleeping makes the next sample better-formed, so a backoff curve
        here would only lengthen a run that is already about eight minutes."""
        sleep = Recorder()
        with pytest.raises(MalformedToolCallExhausted):
            call_with_malformed_tool_call_retry(
                raiser(SchemaRefusal(TRUNCATED_TOOL_CALL)), sleep=sleep, attempts=3
            )
        assert sleep.waits == [MALFORMED_TOOL_CALL_WAIT_S] * 2

    def test_it_does_not_sleep_after_the_final_attempt(self):
        sleep = Recorder()
        with pytest.raises(MalformedToolCallExhausted):
            call_with_malformed_tool_call_retry(
                raiser(SchemaRefusal(NO_TOOL_CALL_AT_ALL)), sleep=sleep, attempts=2
            )
        assert len(sleep.waits) == 1

    def test_exhaustion_reports_the_attempts_and_keeps_the_cause(self):
        """`attempts` is not decoration: `judge_response` adds it to
        `judge_completions`, which paces the run. Unlike a 429, every attempt counted
        here burned a generation against the token window, and pacing that cannot see
        a burned generation is how a run walks into the rate limit it was paced to
        avoid."""
        sleep = Recorder()
        with pytest.raises(MalformedToolCallExhausted) as caught:
            call_with_malformed_tool_call_retry(
                raiser(SchemaRefusal(TRUNCATED_TOOL_CALL)), sleep=sleep, attempts=3
            )
        exc = caught.value
        assert exc.attempts == 3
        assert isinstance(exc.last, SchemaRefusal)
        assert isinstance(exc.__cause__, SchemaRefusal)

    def test_the_message_says_it_is_neither_a_defect_nor_a_quota_refusal(self):
        """It is the third thing, and an operator who cannot tell which of the three
        they have goes looking in the wrong place: a defect means read the diff, a
        quota refusal means wait or change tier, and this means the judge could not
        answer."""
        sleep = Recorder()
        with pytest.raises(MalformedToolCallExhausted) as caught:
            call_with_malformed_tool_call_retry(
                raiser(SchemaRefusal(NO_TOOL_CALL_AT_ALL)), sleep=sleep, attempts=2
            )
        text = str(caught.value).lower()
        assert "not a harness defect" in text
        assert "not a quota refusal" in text
        assert "abstains rather than being lost" in text

    def test_anything_else_propagates_on_the_first_raise(self):
        sleep = Recorder()
        with pytest.raises(SchemaRefusal):
            call_with_malformed_tool_call_retry(
                raiser(SchemaRefusal("context_length_exceeded")), sleep=sleep
            )
        assert sleep.waits == []

    def test_a_rate_limit_passes_straight_through_it(self):
        """`judge()` nests the 429 budget INSIDE this one, so a `RateLimitExhausted`
        surfacing here has already spent its own retries and must reach the operator
        as itself. Flattening it would tell someone whose token budget is spent to go
        and read the JSON."""
        sleep = Recorder()
        with pytest.raises(RateLimitExhausted):
            call_with_malformed_tool_call_retry(
                raiser(RateLimitExhausted(3, 12.0, FakeProviderError())), sleep=sleep
            )
        assert sleep.waits == []

    def test_the_default_budget_allows_two_retries(self):
        """The ruling was to retry "once or twice". Pinned as a number because the
        cost is tokens, not seconds: every attempt spends 1152-2178 of the 8000 a
        minute this tier allows, so raising it trades run latency for a recovery that
        on the truncation case either works immediately or does not work at all."""
        assert MALFORMED_TOOL_CALL_ATTEMPTS == 3

        calls = []

        def always_truncated():
            calls.append(1)
            raise SchemaRefusal(TRUNCATED_TOOL_CALL)

        sleep = Recorder()
        with pytest.raises(MalformedToolCallExhausted):
            call_with_malformed_tool_call_retry(always_truncated, sleep=sleep)
        assert len(calls) == MALFORMED_TOOL_CALL_ATTEMPTS
