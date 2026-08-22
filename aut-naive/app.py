"""aut-naive FastAPI POST /chat. FROZEN at STEP 4. Zero imports from harness/.

DESIGN.md 1.4:

    In   | HTTP `POST /chat {session_id, message}`
    Out  | free text
    Stack| Separate repo directory, separate container, own `Dockerfile`.

TWO CHOICES THAT GO BEYOND THE SPEC, BOTH DELIBERATE

1. `/chat` replies with a small JSON envelope whose `reply` field carries the free text,
   rather than a bare `text/plain` body. "Out: free text" in 1.4 describes the *nature* of
   the payload - unstructured prose, in contrast to the judge's structured verdict - and
   an envelope lets the harness record latency and model per row without a second call.

2. Every `/chat` response repeats the freeze identity. `/health` alone would mean the
   harness records what the agent claimed once at run start and attributes 500 rows to it.
   Carrying it per response makes each audit row self-certifying, so a container swapped
   mid-run shows up as disagreeing rows instead of passing silently. C3 (DESIGN.md 0) is
   only as strong as the weakest link between the tag and the row.

SESSION STATE
DESIGN.md 2 step 6 specifies "per-probe fresh `session_id` (except multi-turn probes,
which reuse)", so history is kept per session in memory. An agent with no memory cannot be
walked away from an earlier position, and strategy 7 (multi_turn_drift) would silently
measure nothing. In-memory is correct for a frozen single-replica AUT, with one operational
consequence worth stating: restarting the container mid-run loses open sessions, so a
multi-turn probe must not straddle a restart.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from backends import BackendError, Generator, build_generator
from chunker import CHUNK_CHARS, OVERLAP_CHARS
from prompts import TEMPERATURE, build_messages
from retrieval import MODEL_NAME, TOP_K, Retriever, build_default

CORPUS_DIR = os.getenv("AUT_CORPUS_DIR", "corpus")

#: Bounds on in-memory session state. A frozen agent should not be the thing that runs a
#: demo machine out of RAM at 11pm on day 12.
MAX_SESSIONS = 2000
MAX_HISTORY_MESSAGES = 12

app = FastAPI(title="aut-naive", version="1.0.0")

_sessions: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
_retriever: Retriever | None = None
_generator: Generator | None = None


def freeze_identity() -> dict[str, str]:
    """What this container claims to be.

    Injected as build args by `scripts/freeze_aut.py`; unset means the image was built
    outside the freeze script, and saying so is better than reporting a plausible blank.
    """
    unset = "(unfrozen: built outside scripts/freeze_aut.py)"
    return {
        "aut_name": "aut-naive",
        "aut_commit_sha": os.getenv("AUT_COMMIT_SHA") or unset,
        "aut_repo_head": os.getenv("AUT_REPO_HEAD") or unset,
        "aut_git_tag": os.getenv("AUT_GIT_TAG") or unset,
        "aut_frozen_at": os.getenv("AUT_FROZEN_AT") or unset,
    }


def get_retriever() -> Retriever:
    """Lazy singleton. Overridden in tests via `app.dependency_overrides`."""
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
        # 502 rather than an empty 200: an empty reply would be scored `evasive`
        # (DESIGN.md 2 step 7) and a transport failure would look like agent behaviour.
        raise HTTPException(status_code=502, detail=str(exc))

    # Defence in depth. Both backends already raise on empty content, but a blank 200
    # reaching the harness is indistinguishable from a genuinely evasive answer, and
    # that is a row in the confusion matrix rather than a visible outage.
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
    """Readiness plus the frozen configuration, for the audit trail.

    Depends on the retriever on purpose: a 200 here means the corpus is indexed and the
    agent can actually answer, which is what a container healthcheck should assert.
    """
    return {
        "status": "ok",
        **freeze_identity(),
        "retrieval": {
            "embedding_model": MODEL_NAME,
            "top_k": TOP_K,
            "chunk_chars": CHUNK_CHARS,
            "overlap_chars": OVERLAP_CHARS,
            "chunks": len(retriever.chunks),
            "corpus_fingerprint": retriever.fingerprint,
        },
        "generation": {"temperature": TEMPERATURE},
        "sessions_open": len(_sessions),
    }
