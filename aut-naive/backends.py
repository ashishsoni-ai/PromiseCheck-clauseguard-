"""Generation backend for aut-naive. STEP 4. Zero imports from harness/.

DESIGN.md 1.4 names Ollama `qwen2.5:7b-instruct` for this agent, and the Appendix repeats
it. That is the default and the documented configuration.

WHY THERE IS A SECOND BACKEND
The sandbox this project is partly developed in has no Ollama and no outbound network, so
a hosted fallback exists behind `LLM_BACKEND=groq`. It is a development affordance, not a
config choice to make casually: the frozen agent is defined by its model as much as by its
code, so switching backends after freezing means the tag has to be re-cut and the audit
rows split. `/health` reports the backend and model actually in use for exactly this
reason - a run whose rows disagree with the tag should be visible, not inferred.

WHY host.docker.internal IS THE DEFAULT HOST
aut-naive runs in a container and Ollama runs on the developer's machine. Inside a
container `localhost` is the container, so the default points at the host gateway. Override
with `OLLAMA_HOST` when running the app directly in a venv.
"""

from __future__ import annotations

import os
from typing import Protocol, Sequence, runtime_checkable

#: DESIGN.md 1.4, verbatim.
OLLAMA_MODEL = "qwen2.5:7b-instruct"

#: Reachable from inside a container; override for a bare venv run.
DEFAULT_OLLAMA_HOST = "http://host.docker.internal:11434"

#: Only used when LLM_BACKEND=groq. Kept out of the frozen default path.
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

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
        self._host = host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
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
