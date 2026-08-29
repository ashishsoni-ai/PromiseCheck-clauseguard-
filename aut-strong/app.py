"""aut-strong HTTP service. DESIGN.md 1.4. STEP 4.

A separate deployable that speaks aut-naive's wire protocol exactly. The harness talks to
both over HTTP only and imports from neither; if this file ever needs a path outside this
directory, commitment C3 has been compromised.

THE CONTRACT IS COPIED, NOT REDESIGNED
`POST /chat` takes the same request shape (`session_id`, `message`) and returns the same
response shape - `reply`, `session_id`, `turn`, `latency_ms`, `model`, `backend`,
`retrieved_chunk_ids`, `aut_commit_sha`, `aut_repo_head`. The whole point of the exercise
in DESIGN.md 1.4 is that `clauseguard run --agent` works against this agent with no harness
change at all, which means:

  NO NEW FIELDS ON THE /chat RESPONSE, even informative ones. The reranker's scores, the
  pre-rerank candidate ids and the margin between them are all genuinely interesting and
  none of them go here - a field the harness's response model does not expect is at best
  ignored and at worst a validation error on every row, and either way it is a difference
  between the two agents living in the one place that has to stay identical. `GET /health`
  is where aut-strong reports what makes it different: the reranker, both model revisions,
  CANDIDATE_K and the resolved chunk count, alongside the fields aut-naive publishes.

Behaviour preserved for reasons not obvious from the shape, all mirrored from aut-naive:

  - A blank reply is a 502, not a 200. A 200 carrying an empty string scores as an
    `evasive` answer, which would silently credit aut-strong with a non-answer on the
    exact metric under test. A backend failure must look like a failure.
  - `BackendError` is a 502 and the failed turn is NOT appended to history, so a retry does
    not replay a turn that never got an answer.
  - The freeze identity is reported on every reply, with the "(unfrozen: built outside
    scripts/freeze_aut.py)" sentinel when the build args are absent - a blank field would
    read as a frozen agent with a missing SHA.
  - Session history is an LRU capped at MAX_SESSIONS with MAX_HISTORY_MESSAGES per
    session. The multi_turn_drift probes depend on history working; results.md records
    aut-naive converting 1 of 3 of them.
  - Retrieval and generation are lazy singletons that raise 503 until ready, and /health
    builds the index so a passing healthcheck means the agent can answer. This matters
    more here than for aut-naive: two models load, not one, which is why the Dockerfile's
    HEALTHCHECK start-period is longer.

`retrieved_chunk_ids` KEEPS ITS NAME AND ITS MEANING
It reports what was sent to the model - the post-rerank TOP_K - not the wider candidate
set. Reporting candidates would make the field mean something different between the two
agents while looking the same, and this is the field an audit row uses to answer "did the
agent even see the governing clause?". The pre-rerank set is worth recording, and it lives
in /health's configuration report and in STEP 2's standalone retrieval evidence
(scripts/check_aut_retrieval.py) rather than being smuggled into this field.

WHY /health REPORTS THE LIVE OBJECTS AND NOT THE MODULE CONSTANTS
aut-naive's /health prints `MODEL_NAME` and `TOP_K` straight from its modules. Here the
embedder, the reranker, `top_k` and `candidate_k` are read off the retriever that is
actually loaded. The difference matters for exactly one reason: those constants are the
thing a rebuild can drift away from, and a /health that quotes the constant would keep
reporting the intended configuration while serving a different one. The corpus fingerprint
already folds both revisions in, but it is a digest - unequal to a recorded value tells
you something changed, not what. These fields say what.

The generation block is the mirror image: it reports the CONFIGURED backend and model
rather than a constructed generator's, because /health deliberately does not depend on
`get_generator`. A missing GROQ_API_KEY must not mark the container unhealthy - the agent
is fine, its credentials are not - and a healthcheck that fails on a key would take the
container down for a condition no restart fixes.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from backends import (
    BACKEND_ENV,
    BackendError,
    Generator,
    build_generator,
    configured_backend,
    configured_model,
)
from chunker import CHUNK_CHARS, OVERLAP_CHARS
from prompts import TEMPERATURE, build_messages
from retrieval import Retriever, build_default

#: Baked into the image by the Dockerfile's `COPY corpus/ ./corpus/`. Overridable for a
#: bare venv run, not for the frozen container.
CORPUS_DIR = os.getenv("AUT_CORPUS_DIR", "corpus")

#: Same as aut-naive, so a difference in reply quality cannot be attributed to one agent
#: remembering more of the conversation than the other.
MAX_SESSIONS = 2000
MAX_HISTORY_MESSAGES = 12

app = FastAPI(title="aut-strong", version="1.0.0")

_sessions: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
_retriever: Retriever | None = None
_generator: Generator | None = None


def freeze_identity() -> dict[str, str]:
    """What this build is, as the freeze script recorded it.

    The sentinel is spelled out rather than left blank: an empty string in an audit row
    reads as a frozen agent whose SHA went missing, which is a different and much more
    alarming claim than "this was not built by the freeze script".
    """
    unset = "(unfrozen: built outside scripts/freeze_aut.py)"
    return {
        "aut_name": "aut-strong",
        "aut_commit_sha": os.getenv("AUT_COMMIT_SHA") or unset,
        "aut_repo_head": os.getenv("AUT_REPO_HEAD") or unset,
        "aut_git_tag": os.getenv("AUT_GIT_TAG") or unset,
        "aut_frozen_at": os.getenv("AUT_FROZEN_AT") or unset,
    }


def get_retriever() -> Retriever:
    """Lazy singleton. Overridden in tests via `app.dependency_overrides`.

    Two models load here, not one, so the first call is slower than aut-naive's by more
    than a constant. It is still lazy rather than eager: an import-time load would make
    `uvicorn app:app` fail in a way Docker reports as a crash loop instead of as a 503
    with a reason in it.
    """
    global _retriever
    if _retriever is None:
        try:
            _retriever = build_default(CORPUS_DIR)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"retrieval unavailable: {exc}")
    return _retriever


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        try:
            _generator = build_generator()
        except BackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
    return _generator


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    turn: int
    latency_ms: int
    model: str
    backend: str
    retrieved_chunk_ids: list[str]
    aut_commit_sha: str
    aut_repo_head: str


def _history(session_id: str) -> list[tuple[str, str]]:
    if session_id in _sessions:
        _sessions.move_to_end(session_id)
        return _sessions[session_id]
    while len(_sessions) >= MAX_SESSIONS:
        _sessions.popitem(last=False)
    _sessions[session_id] = []
    return _sessions[session_id]


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
) -> ChatResponse:
    history = _history(request.session_id)
    started = time.perf_counter()
    hits = retriever.search(request.message)
    messages = build_messages(request.message, hits, history)
    try:
        reply = generator.complete(messages, temperature=TEMPERATURE)
    except BackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    if not reply.strip():
        raise HTTPException(
            status_code=502, detail=f"{generator.backend} returned an empty reply"
        )
    history.append(("user", request.message))
    history.append(("assistant", reply))
    del history[:-MAX_HISTORY_MESSAGES]
    identity = freeze_identity()
    return ChatResponse(
        reply=reply,
        session_id=request.session_id,
        turn=len(history) // 2,
        latency_ms=int((time.perf_counter() - started) * 1000),
        model=generator.name,
        backend=generator.backend,
        retrieved_chunk_ids=[h.chunk.chunk_id for h in hits],
        aut_commit_sha=identity["aut_commit_sha"],
        aut_repo_head=identity["aut_repo_head"],
    )


@app.get("/health")
def health(retriever: Retriever = Depends(get_retriever)) -> dict:
    return {
        "status": "ok",
        **freeze_identity(),
        "retrieval": {
            "embedding_model": retriever.embedder.name,
            "embedding_revision": retriever.embedder.revision,
            "reranker_model": retriever.reranker.name,
            "reranker_revision": retriever.reranker.revision,
            "top_k": retriever.top_k,
            "candidate_k": retriever.candidate_k,
            "chunk_chars": CHUNK_CHARS,
            "overlap_chars": OVERLAP_CHARS,
            "chunks": len(retriever.chunks),
            "corpus_fingerprint": retriever.fingerprint,
        },
        "generation": {
            "temperature": TEMPERATURE,
            # Configured, not constructed - see the module docstring. `backend_env` is
            # named because this agent reads its own variable and ignores LLM_BACKEND,
            # which is the one piece of this wiring a reader will get wrong.
            "backend": configured_backend(),
            "model": configured_model(),
            "backend_env": BACKEND_ENV,
        },
        "sessions_open": len(_sessions),
    }
