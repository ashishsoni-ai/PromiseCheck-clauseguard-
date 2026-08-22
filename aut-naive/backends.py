"""Generation backend for aut-naive. STEP 4. Zero imports from harness/.

DESIGN.md 1.4 names Ollama `qwen2.5:7b-instruct` for this agent, and the Appendix repeats
it. That is the default and the documented configuration.

WHY THERE IS A SECOND BACKEND
The sandbox this project is partly developed in has no Ollama and no outbound network, so
a hosted fallback exists behind `LLM_BACKEND=groq`. It is a development affordance, not a
config choice to make casually: the frozen agent is defined by its model as much as by its
code.

What that does NOT mean is that the git tag becomes invalid. The tag records
`git rev-parse HEAD:aut-naive`, which does not move when an environment variable changes,
so re-cutting it on a backend switch would be theatre. What must actually happen is that
rows are never pooled across backends - the same code answering through a 7B local model
and through a hosted model is two different agents for measurement purposes, and averaging
them reports a system that does not exist. `/health` and every `/chat` reply carry the
backend and model in use so that split is possible from the audit trail alone rather than
from someone's memory of how the run was launched.

WHY host.docker.internal IS THE DEFAULT HOST
aut-naive runs in a container and Ollama runs on the developer's machine. Inside a
container `localhost` is the container, so the default points at the host gateway.

The override is `AUT_OLLAMA_HOST`, deliberately *not* `OLLAMA_HOST`. Ollama reads
`OLLAMA_HOST` itself, to decide what address its server binds to, and on Windows it must
often be set to `0.0.0.0:11434` before a container can reach it at all. That value is
meaningless as a client target - inside a container `0.0.0.0` is the container - so sharing
the name puts a footgun in the one deployment this agent is documented to run in: both
sides look correctly configured and every request 502s. Different meanings, different name.
"""

from __future__ import annotations

import os
from typing import Protocol, Sequence, runtime_checkable

#: DESIGN.md 1.4, verbatim.
OLLAMA_MODEL = "qwen2.5:7b-instruct"

#: Reachable from inside a container; override for a bare venv run.
DEFAULT_OLLAMA_HOST = "http://host.docker.internal:11434"

#: Not `OLLAMA_HOST` - that name belongs to the Ollama server's bind address. See module
#: docstring; sharing it silently breaks the containerised path.
OLLAMA_HOST_ENV = "AUT_OLLAMA_HOST"

#: Only used when LLM_BACKEND=groq. Kept out of the frozen default path.
#:
#: MUST stay in the Qwen family. DESIGN.md 1.5 and the 2 step 8 role table both require the
#: judge to come from a different family than the AUT, and the judge is a Llama model
#: (`llama-3.3-70b-versatile`). A Llama fallback here would collapse that separation
#: silently - and `llama-3.1-8b-instant` specifically is the *adversary* model, which would
#: have the adversary writing probes against itself. Enforced by a test, not by this comment.
DEFAULT_GROQ_MODEL = "qwen/qwen3-32b"

DEFAULT_TIMEOUT_S = 120.0

Messages = Sequence[dict[str, str]]


class BackendError(RuntimeError):
    """The backend could not produce a response.

    Surfaced as a 502 rather than swallowed into an empty string: a silent empty answer
    would be scored as `evasive` by the harness (DESIGN.md 2 step 7) and a transport
    failure would masquerade as agent behaviour.
    """


@runtime_checkable
class Generator(Protocol):
    def complete(self, messages: Messages, *, temperature: float) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def backend(self) -> str: ...


class OllamaGenerator:
    """The frozen default."""

    backend = "ollama"

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        import ollama

        self._model = model
        self._host = host or os.getenv(OLLAMA_HOST_ENV) or DEFAULT_OLLAMA_HOST
        self._client = ollama.Client(host=self._host, timeout=timeout_s)

    @property
    def name(self) -> str:
        return self._model

    @property
    def host(self) -> str:
        return self._host

    def complete(self, messages: Messages, *, temperature: float) -> str:
        try:
            response = self._client.chat(
                model=self._model,
                messages=list(messages),
                options={"temperature": temperature},
            )
        except Exception as exc:  # noqa: BLE001 - transport detail belongs in the message
            raise BackendError(f"ollama {self._model} at {self._host}: {exc}") from exc

        content = (response.get("message") or {}).get("content")
        if not content:
            raise BackendError(f"ollama {self._model} returned no content")
        return content


class GroqGenerator:
    """Development fallback. Not the documented frozen configuration."""

    backend = "groq"

    def __init__(
        self,
        model: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise BackendError("LLM_BACKEND=groq but GROQ_API_KEY is not set")
        self._model = model or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
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
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"groq {self._model}: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise BackendError(f"groq {self._model} returned no content")
        return content


def build_generator(backend: str | None = None) -> Generator:
    """Construct the configured backend. Defaults to the frozen Ollama path."""
    choice = (backend or os.getenv("LLM_BACKEND") or "ollama").strip().lower()
    if choice == "ollama":
        return OllamaGenerator()
    if choice == "groq":
        return GroqGenerator()
    raise BackendError(f"unknown LLM_BACKEND {choice!r}; expected 'ollama' or 'groq'")
