"""Generation backend for aut-strong. STEP 4. Zero imports from harness/.

Mirrors aut-naive/backends.py's interface - `BackendError`, a `Generator` protocol with
`complete(messages, *, temperature)` / `name` / `backend`, and a `build_generator` that
refuses an unknown value rather than defaulting past it. app.py depends on that shape,
and every `/chat` reply carries `model` and `backend` out of it so rows can be split by
what actually answered them instead of by someone's memory of how the run was launched.

THE MODEL, AND THE TWO THINGS IT COLLIDES WITH
DESIGN.md 1.4 puts aut-strong on "a frontier API model" and the pin chosen for it is
`openai/gpt-oss-120b` on Groq - the largest chat model this account can see. That choice
is deliberate and it is not free:

  1. IT SHARES A FAMILY WITH THE JUDGE. `CLAUSEGUARD_JUDGE_MODEL` is
     `groq/openai/gpt-oss-20b`, and DESIGN.md 1.5 requires the judge to come from a
     different family than the AUT, on the grounds that a judge sharing pretraining with
     the agent finds a fluent over-promise reasonable exactly when the agent's own priors
     produced it.
  2. IT IS THE SAME MODEL AS THE EXTRACTOR, not merely the same family.
     `CLAUSEGUARD_EXTRACTOR_MODEL` is `groq/openai/gpt-oss-120b`. The extractor produced
     the clause set that C1's ground-truth labels derive from and that C2's span check
     cites against, so with this pin the agent under test and the source of its own answer
     key are one model.

Both are now asserted rather than described: tests/unit/test_aut_strong_backends.py
::TestTheCollisionsAreDeclaredNotDiscovered fails if either overlap stops matching what
docs/limitations.md records, and fails equally if the overlap silently disappears - a pin
change that removed it would invalidate the lower-bound reading below, which is a result
worth noticing rather than inheriting. The remedy
TestTheKnownFamilyCollisionIsTheDocumentedOne's docstring prescribes - record the overlap
"in DESIGN.md 8's limitations" - is not available, because DESIGN.md is authoritative and
never edited; docs/limitations.md is where it went.

The bias has a direction, and it is the one that must be stated before the run rather
than after it. Shared blind spots between agent, extractor and judge make over-promises
HARDER to detect, never easier, so the aut-strong number this pin produces is a LOWER
BOUND. That is survivable for DESIGN.md 1.4's actual thesis - "non-zero here is the entire
thesis" holds a fortiori if a downward-biased measurement still finds over-promises. It is
fatal for the other branch: a 0% result cannot be reported as "a well-prompted frontier
model does not over-promise" when the model grading it and the model that wrote its answer
key are its own siblings. STEP 6 pre-commits to that reading before seeing the number.

THERE IS NO LOCAL BACKEND, AND THAT IS THE CONFIGURATION, NOT A GAP
aut-naive carries a second backend because its documented model is local and a hosted
fallback lets it run where Ollama cannot. aut-strong is the mirror image: its documented
model IS the hosted one, so a local pin here would be an *undocumented* configuration -
DESIGN.md 1.4 names no local model for this agent, and choosing one means colliding with
something (a local qwen with aut-naive's own model, a local mistral with the adversary).

So `OLLAMA_MODEL` stays `None` and asking for that backend fails closed with a message
that says why. The alternative - pin something plausible and let it run - is worse than it
looks: it produces rows that are indistinguishable from hosted rows at a glance and get
pooled with them, and pooling two models under one agent name reports a system that does
not exist. An agent that refuses to start is a bad afternoon; an agent that answers as
somebody else is a bad result.

IT DOES NOT READ `LLM_BACKEND`, FOR THE SAME REASON IT DOES NOT READ `GROQ_MODEL`
`.env.example:116` pins `LLM_BACKEND=ollama` and docker-compose.yml passes it to
aut-naive; both say in their comments that it is aut-naive's switch, whose default is
local because DESIGN.md 1.4 makes it local. Reading that variable here would mean the
repository's own documented environment asks aut-strong for a backend it does not have,
on every start. And the way that failure lands is the bad way: `/health` does not build a
generator - deliberately, so a missing key cannot mark the container unhealthy - so the
container would go green and 503 every `/chat`, which is precisely the "starts, passes its
healthcheck and serves nonsense" state app.py's scaffold was written to prevent.

`AUT_STRONG_LLM_BACKEND` is therefore its own variable, defaulting to `groq`, and setting
`LLM_BACKEND` has no effect here at all. That is a real trap for whoever debugs this next,
so `/health` reports the variable it read by name rather than only the value it resolved.

`GROQ_MODEL` IS NOT THE VARIABLE TO READ EITHER
aut-naive's backends.py reads `GROQ_MODEL` ahead of its own constant, because its frozen
default `qwen/qwen3-32b` was decommissioned on 2026-08-23 and the constant is inside the
frozen tree. `.env.example` therefore pins `GROQ_MODEL` to `qwen/qwen3.6-27b`, and
test_the_env_override_for_the_dead_fallback_stays_in_the_agents_family asserts it stays in
the AGENT's family - meaning Qwen. If this file read the same variable it would silently
run as qwen3.6-27b while reporting itself as aut-strong, which is the exact species-swap
that test exists to prevent.

AND THE `groq/` PREFIX IS NOT PART OF THE MODEL ID
The harness talks to providers through litellm, where `groq/openai/gpt-oss-120b` means
"route openai/gpt-oss-120b to Groq". This module uses the native `groq` client, as
aut-naive does, so the id it sends is `openai/gpt-oss-120b`. A prefixed id is a 404 at
request time rather than a startup error, and the id most likely to be pasted into
`AUT_STRONG_GROQ_MODEL` is the one sitting in `.env` next to the extractor - prefix
included. So the prefix is rejected at construction, with the fix in the message, instead
of being stripped: stripping would also silently accept `groq/some-other-model` and answer
as a model nobody chose.
"""

from __future__ import annotations

import os
from typing import Protocol, Sequence, runtime_checkable

#: The pin, in native-client form - no litellm routing prefix. See the docstring for the
#: family and extractor-identity collisions this creates, and for why they are recorded
#: rather than resolved.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

#: aut-strong's own model override. Deliberately NOT `GROQ_MODEL`, which an existing test
#: pins to the agent's Qwen family, and deliberately not `AUT_MODEL`, which would read as
#: applying to both agents.
GROQ_MODEL_ENV = "AUT_STRONG_GROQ_MODEL"

#: aut-strong's own backend switch, for the reasons in the docstring. `LLM_BACKEND` is
#: aut-naive's and is not consulted here.
BACKEND_ENV = "AUT_STRONG_LLM_BACKEND"

#: DESIGN.md 1.4's documented configuration for this agent, so it is also the default.
DEFAULT_BACKEND = "groq"

#: No local pin, on purpose - see the docstring. Kept as a declared `None` rather than
#: omitted so that the absence is a stated property of this agent that a test can assert,
#: instead of something a future reader fills in because the constant looked missing.
OLLAMA_MODEL: str | None = None

#: aut-naive defines `OLLAMA_HOST_ENV` / `DEFAULT_OLLAMA_HOST`; this module deliberately
#: does not. There is no Ollama client here to point at a host, so those constants would
#: be configuration for a code path that does not exist - and the reader most likely to
#: look for them is the one about to add that path, who should read the docstring instead.

DEFAULT_TIMEOUT_S = 120.0

Messages = Sequence[dict[str, str]]


class BackendError(RuntimeError):
    """The backend could not produce a response.

    Surfaced as a 502 rather than swallowed into an empty string: a silent empty answer
    would be scored `evasive` by the harness (DESIGN.md 2 step 7), which on this agent
    would credit the defensive prompt with a non-answer on the exact metric under test.
    """


@runtime_checkable
class Generator(Protocol):
    def complete(self, messages: Messages, *, temperature: float) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def backend(self) -> str: ...


class GroqGenerator:
    """The documented configuration, not a fallback. DESIGN.md 1.4's frontier model."""

    backend = "groq"

    def __init__(
        self,
        model: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        # THE ORDER OF THESE FOUR CHECKS IS DELIBERATE, and it is not aut-naive's order.
        # Configuration is validated before the key is read and before the SDK is
        # imported. Two reasons. The first is what the errors mean: a prefixed model id
        # reported as "GROQ_API_KEY is not set" sends the reader to the wrong file, and
        # the config mistakes are the ones a person actually makes. The second is that
        # the checks below are then reachable from a test environment that has neither a
        # key nor the `groq` package - and `groq` is an AUT-only dependency, absent from
        # the harness venv on purpose, so any check sitting behind that import is a check
        # this project cannot assert.
        resolved = (model or "").strip() or configured_model()
        if resolved.startswith("groq/"):
            raise BackendError(
                f"model id {resolved!r} carries litellm's routing prefix. This module "
                f"uses the native groq client, so the id must be "
                f"{resolved[len('groq/'):]!r}; the prefixed form is a 404 per request, "
                f"not a startup error - check {GROQ_MODEL_ENV}"
            )
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise BackendError(
                "aut-strong requires GROQ_API_KEY: its documented model "
                f"({DEFAULT_GROQ_MODEL}) is hosted and this agent has no local backend"
            )
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - present in the frozen image
            # A BackendError rather than a bare ImportError so app.py's `except
            # BackendError` turns it into a 503 with a reason, instead of a 500 with a
            # traceback. This path cannot occur in the frozen container - the Dockerfile
            # installs requirements.txt, which pins groq - so it exists for bare-venv
            # runs only and cannot affect a measured row.
            raise BackendError(
                f"the groq client is not installed, so backend {self.backend!r} cannot "
                "be constructed. aut-strong/requirements.txt pins it; a bare-venv run "
                "needs `pip install -r aut-strong/requirements.txt`"
            ) from exc
        self._model = resolved
        self._client = Groq(api_key=api_key, timeout=timeout_s)

    @property
    def name(self) -> str:
        return self._model

    def complete(self, messages: Messages, *, temperature: float) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001 - transport detail belongs in the message
            raise BackendError(f"groq {self._model}: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise BackendError(f"groq {self._model} returned no content")
        return content


def configured_backend(backend: str | None = None) -> str:
    """The backend name this agent would use, resolved without constructing a client.

    Split out from `build_generator` so `/health` can report the configuration on an
    agent with no provider key. Reporting the *resolved* name matters more than it looks:
    the value is chosen by a variable aut-naive does not read, so an operator who set
    `LLM_BACKEND` and got groq anyway needs somewhere to see that.
    """
    return (backend or os.getenv(BACKEND_ENV) or "").strip().lower() or DEFAULT_BACKEND


def configured_model() -> str:
    """The model id this agent would send, without constructing a client.

    Blank-safe rather than truthiness-only: `AUT_STRONG_GROQ_MODEL=` (or a value that is
    all spaces) reads as "unset" here, because an empty override means someone commented
    a line out badly, not that the agent should ask the provider for the model named "".

    Deliberately does NOT validate the `groq/` prefix - `/health` should show the operator
    what is configured, including a value that will fail, rather than refuse to answer and
    leave them guessing which of the two variables they got wrong.
    """
    return (os.getenv(GROQ_MODEL_ENV) or "").strip() or DEFAULT_GROQ_MODEL


def build_generator(backend: str | None = None) -> Generator:
    """Construct the configured backend. Defaults to the documented hosted path."""
    choice = configured_backend(backend)
    if choice == "groq":
        return GroqGenerator()
    if choice == "ollama":
        raise BackendError(
            f"{BACKEND_ENV}=ollama, but aut-strong pins no local model (OLLAMA_MODEL is "
            f"None). DESIGN.md 1.4 documents a hosted frontier model for this agent, so "
            f"a local run here would be an undocumented configuration whose rows would "
            f"be pooled with hosted ones. Note that LLM_BACKEND is aut-naive's switch "
            f"and is not read here; unset {BACKEND_ENV} to get {DEFAULT_BACKEND}"
        )
    raise BackendError(f"unknown {BACKEND_ENV} {choice!r}; expected 'groq'")
