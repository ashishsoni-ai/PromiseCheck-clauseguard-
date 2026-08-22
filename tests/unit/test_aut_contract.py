"""STEP 4: the aut-naive HTTP contract, DESIGN.md 1.4.

    In   | HTTP `POST /chat {session_id, message}`
    Out  | free text

What is real here and what is a stand-in matters, so: the FastAPI app, the request and
response models, the session bookkeeping, the prompt assembly and the error mapping are all
the real code. Only the *collaborators* at the edges are substituted - a retriever that
returns fixed chunks and a generator that returns a fixed string - because embedding a
130MB model and calling a 7B model would test neither of them and would not run in CI.

The retrieval path and the model call are proven by `docker build` plus a live query, not
here. This file exists so that a failure in the container can be localised: if these pass
and the container misbehaves, the fault is in retrieval, the model, or the image.

Two classes are worth reading even if the rest is routine. `TestThePromptIsNotRigged`
mechanically enforces DESIGN.md 1.4's "no conformance instruction, no citation requirement"
*and* the opposite failure - that the agent has not been prompted into over-promising -
because DESIGN.md 10 names the strawman objection as the most dangerous attack on the
project. `TestTheFreezeTravelsWithEveryReply` checks C3's weakest link: not the tag, but
whether a given audit row can prove which agent produced it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest
from fastapi.testclient import TestClient

AUT_DIR = Path(__file__).resolve().parents[2] / "aut-naive"
if str(AUT_DIR) not in sys.path:
    sys.path.insert(0, str(AUT_DIR))

import app as aut_app  # noqa: E402
from app import app, get_generator, get_retriever  # noqa: E402
from backends import BackendError  # noqa: E402
from chunker import CHUNK_CHARS, OVERLAP_CHARS, Chunk  # noqa: E402
from prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    TEMPERATURE,
    build_messages,
    format_context,
)
from retrieval import MODEL_NAME, TOP_K, Hit  # noqa: E402

REPLY = "Yes, you can absolutely return that - I'll get it sorted for you."


def make_chunk(n: int, text: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"acme-refunds#{n:04d}",
        doc_id="acme-refunds",
        ordinal=n,
        text=text or f"Refund window paragraph {n}. Items may be returned.",
        start=(n - 1) * 100,
        end=(n - 1) * 100 + 50,
    )


class FakeRetriever:
    top_k = TOP_K

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = tuple(chunks)
        self.queries: list[str] = []

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        self.queries.append(query)
        wanted = min(k or self.top_k, len(self.chunks))
        return [
            Hit(chunk=c, score=0.9 - 0.1 * i)
            for i, c in enumerate(self.chunks[:wanted])
        ]

    @property
    def fingerprint(self) -> str:
        return "sha256:" + "fa" * 32


class FakeGenerator:
    backend = "fake"

    def __init__(self, reply: str = REPLY) -> None:
        self.reply = reply
        self.calls: list[tuple[list[dict[str, str]], float]] = []

    @property
    def name(self) -> str:
        return "fake-model-1"

    def complete(self, messages, *, temperature: float) -> str:
        self.calls.append(([dict(m) for m in messages], temperature))
        return self.reply


class ExplodingGenerator(FakeGenerator):
    def complete(self, messages, *, temperature: float) -> str:
        raise BackendError("ollama qwen2.5:7b-instruct at http://x:11434: refused")


@dataclass
class Rig:
    client: TestClient
    retriever: FakeRetriever
    generator: FakeGenerator


def build_rig(generator: FakeGenerator | None = None) -> Rig:
    retriever = FakeRetriever([make_chunk(i) for i in range(1, 6)])
    gen = generator or FakeGenerator()
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = lambda: gen
    aut_app._sessions.clear()
    return Rig(client=TestClient(app), retriever=retriever, generator=gen)


@pytest.fixture(autouse=True)
def _clean_state():
    yield
    app.dependency_overrides.clear()
    aut_app._sessions.clear()


@pytest.fixture
def rig() -> Rig:
    return build_rig()


class TestTheChatContract:
    def test_a_question_gets_a_reply(self, rig: Rig):
        response = rig.client.post(
            "/chat", json={"session_id": "s1", "message": "Can I return my order?"}
        )
        assert response.status_code == 200
        assert response.json()["reply"] == REPLY

    def test_the_envelope_carries_what_an_audit_row_needs(self, rig: Rig):
        body = rig.client.post(
            "/chat", json={"session_id": "s1", "message": "Refund please"}
        ).json()
        assert body["session_id"] == "s1"
        assert body["turn"] == 1
        assert body["model"] == "fake-model-1"
        assert body["backend"] == "fake"
        assert body["latency_ms"] >= 0

    def test_the_retrieved_chunk_ids_are_reported_in_the_aut_s_own_format(self, rig: Rig):
        body = rig.client.post(
            "/chat", json={"session_id": "s1", "message": "Refund please"}
        ).json()
        assert body["retrieved_chunk_ids"] == [
            "acme-refunds#0001",
            "acme-refunds#0002",
            "acme-refunds#0003",
        ]
        assert all(":" not in cid for cid in body["retrieved_chunk_ids"])

    def test_top_k_is_three(self, rig: Rig):
        """DESIGN.md 1.4, verbatim."""
        body = rig.client.post(
            "/chat", json={"session_id": "s1", "message": "Refund please"}
        ).json()
        assert len(body["retrieved_chunk_ids"]) == 3

    def test_the_user_message_is_what_gets_embedded(self, rig: Rig):
        rig.client.post(
            "/chat", json={"session_id": "s1", "message": "Day 40, can I refund?"}
        )
        assert rig.retriever.queries == ["Day 40, can I refund?"]

    def test_generation_runs_at_the_frozen_temperature(self, rig: Rig):
        rig.client.post("/chat", json={"session_id": "s1", "message": "Hello"})
        _, temperature = rig.generator.calls[0]
        assert temperature == TEMPERATURE == 0.7

    def test_the_retrieved_text_reaches_the_model(self, rig: Rig):
        rig.client.post("/chat", json={"session_id": "s1", "message": "Hello"})
        messages, _ = rig.generator.calls[0]
        assert messages[0]["role"] == "system"
        assert "Refund window paragraph 1." in messages[0]["content"]
        assert messages[-1] == {"role": "user", "content": "Hello"}


class TestTheRequestIsValidated:
    def test_an_empty_message_is_rejected(self, rig: Rig):
        response = rig.client.post("/chat", json={"session_id": "s1", "message": ""})
        assert response.status_code == 422

    def test_a_missing_session_id_is_rejected(self, rig: Rig):
        assert rig.client.post("/chat", json={"message": "hi"}).status_code == 422

    def test_an_empty_session_id_is_rejected(self, rig: Rig):
        response = rig.client.post("/chat", json={"session_id": "", "message": "hi"})
        assert response.status_code == 422

    def test_an_absurd_session_id_is_rejected(self, rig: Rig):
        response = rig.client.post(
            "/chat", json={"session_id": "x" * 5000, "message": "hi"}
        )
        assert response.status_code == 422

    def test_nothing_was_generated_for_a_rejected_request(self, rig: Rig):
        rig.client.post("/chat", json={"session_id": "s1", "message": ""})
        assert rig.generator.calls == []


class TestSessionsCarryHistory:
    """DESIGN.md 2 step 6: fresh session per probe, reused for multi-turn probes.

    Strategy 7 (multi_turn_drift) needs an agent that can be walked away from an earlier
    position. An amnesiac AUT would make that strategy silently measure nothing.
    """

    def test_a_second_turn_replays_the_first(self, rig: Rig):
        rig.client.post("/chat", json={"session_id": "s1", "message": "First question"})
        rig.client.post("/chat", json={"session_id": "s1", "message": "Second question"})
        messages, _ = rig.generator.calls[1]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]
        assert messages[1]["content"] == "First question"
        assert messages[2]["content"] == REPLY
        assert messages[3]["content"] == "Second question"

    def test_the_turn_counter_advances(self, rig: Rig):
        turns = [
            rig.client.post(
                "/chat", json={"session_id": "s1", "message": f"q{i}"}
            ).json()["turn"]
            for i in range(3)
        ]
        assert turns == [1, 2, 3]

    def test_a_different_session_starts_clean(self, rig: Rig):
        rig.client.post("/chat", json={"session_id": "s1", "message": "First"})
        body = rig.client.post("/chat", json={"session_id": "s2", "message": "Other"}).json()
        assert body["turn"] == 1
        messages, _ = rig.generator.calls[1]
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_history_is_bounded(self, rig: Rig):
        for i in range(12):
            rig.client.post("/chat", json={"session_id": "s1", "message": f"q{i}"})
        messages, _ = rig.generator.calls[-1]
        assert len(messages) - 1 <= aut_app.MAX_HISTORY_MESSAGES + 1

    def test_the_oldest_session_is_evicted_at_the_cap(self, rig: Rig, monkeypatch):
        monkeypatch.setattr(aut_app, "MAX_SESSIONS", 3)
        for i in range(4):
            rig.client.post("/chat", json={"session_id": f"s{i}", "message": "hi"})
        assert "s0" not in aut_app._sessions
        assert len(aut_app._sessions) == 3

    def test_a_reused_session_is_kept_alive_by_use(self, rig: Rig, monkeypatch):
        monkeypatch.setattr(aut_app, "MAX_SESSIONS", 3)
        rig.client.post("/chat", json={"session_id": "keep", "message": "hi"})
        for i in range(2):
            rig.client.post("/chat", json={"session_id": f"s{i}", "message": "hi"})
        rig.client.post("/chat", json={"session_id": "keep", "message": "again"})
        rig.client.post("/chat", json={"session_id": "new", "message": "hi"})
        assert "keep" in aut_app._sessions


class TestABackendFailureIsNotAnAnswer:
    """A transport failure must not be scorable as agent behaviour."""

    def test_a_backend_error_is_a_502(self):
        rig = build_rig(ExplodingGenerator())
        response = rig.client.post("/chat", json={"session_id": "s1", "message": "hi"})
        assert response.status_code == 502
        assert "ollama" in response.json()["detail"]

    def test_an_empty_reply_is_a_502_not_an_evasive_answer(self):
        rig = build_rig(FakeGenerator(reply="   \n  "))
        response = rig.client.post("/chat", json={"session_id": "s1", "message": "hi"})
        assert response.status_code == 502
        assert "empty" in response.json()["detail"]

    def test_a_failed_turn_is_not_recorded_as_history(self):
        rig = build_rig(ExplodingGenerator())
        rig.client.post("/chat", json={"session_id": "s1", "message": "hi"})
        assert aut_app._sessions.get("s1") == []


class TestTheFreezeTravelsWithEveryReply:
    """C3's weakest link is not the tag, it is whether a row can name its producer."""

    def test_an_unfrozen_build_says_so_rather_than_reporting_a_blank(
        self, rig: Rig, monkeypatch
    ):
        for var in ("AUT_COMMIT_SHA", "AUT_REPO_HEAD", "AUT_GIT_TAG", "AUT_FROZEN_AT"):
            monkeypatch.delenv(var, raising=False)
        body = rig.client.post("/chat", json={"session_id": "s1", "message": "hi"}).json()
        assert "unfrozen" in body["aut_commit_sha"]

    def test_the_build_args_are_reported_on_every_reply(self, rig: Rig, monkeypatch):
        monkeypatch.setenv("AUT_COMMIT_SHA", "tree123")
        monkeypatch.setenv("AUT_REPO_HEAD", "head456")
        body = rig.client.post("/chat", json={"session_id": "s1", "message": "hi"}).json()
        assert body["aut_commit_sha"] == "tree123"
        assert body["aut_repo_head"] == "head456"

    def test_chat_and_health_agree_about_who_is_answering(self, rig: Rig, monkeypatch):
        monkeypatch.setenv("AUT_COMMIT_SHA", "tree123")
        chat = rig.client.post("/chat", json={"session_id": "s1", "message": "hi"}).json()
        health = rig.client.get("/health").json()
        assert chat["aut_commit_sha"] == health["aut_commit_sha"] == "tree123"

    def test_the_two_hashes_are_distinct_fields(self, rig: Rig, monkeypatch):
        """Recording only repo HEAD would churn on every harness commit; recording only
        the tree hash gives a reviewer nothing to look up. Both, separately."""
        monkeypatch.setenv("AUT_COMMIT_SHA", "tree123")
        monkeypatch.setenv("AUT_REPO_HEAD", "head456")
        health = rig.client.get("/health").json()
        assert health["aut_commit_sha"] != health["aut_repo_head"]


class TestHealthReportsTheFrozenConfiguration:
    def test_it_is_ok_when_the_index_is_built(self, rig: Rig):
        body = rig.client.get("/health").json()
        assert body["status"] == "ok"
        assert body["aut_name"] == "aut-naive"

    def test_it_names_the_documented_retrieval_stack(self, rig: Rig):
        retrieval = rig.client.get("/health").json()["retrieval"]
        assert retrieval["embedding_model"] == MODEL_NAME == "BAAI/bge-small-en-v1.5"
        assert retrieval["top_k"] == 3
        assert retrieval["chunk_chars"] == CHUNK_CHARS
        assert retrieval["overlap_chars"] == OVERLAP_CHARS

    def test_it_pins_the_corpus_that_was_baked_in(self, rig: Rig):
        """The tree hash proves which code was frozen; this proves which policy text was
        frozen beside it. A stale agent and a wrong agent are different findings."""
        retrieval = rig.client.get("/health").json()["retrieval"]
        assert retrieval["corpus_fingerprint"].startswith("sha256:")
        assert retrieval["chunks"] == 5

    def test_it_reports_the_frozen_temperature(self, rig: Rig):
        assert rig.client.get("/health").json()["generation"]["temperature"] == 0.7

    def test_it_fails_loudly_when_retrieval_is_unavailable(self):
        from fastapi import HTTPException

        def broken():
            raise HTTPException(status_code=503, detail="retrieval unavailable: no model")

        app.dependency_overrides[get_retriever] = broken
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 503
        assert "retrieval unavailable" in response.json()["detail"]


class TestThePromptIsNotRigged:
    """DESIGN.md 10: "your agent under test is a strawman" is the most dangerous attack.

    Tested in both directions, because rigging can go either way: the prompt must lack the
    defensive scaffolding aut-strong will have, and it must not have been told to
    over-promise. The finding only means something if the failure emerges.
    """

    @pytest.mark.parametrize(
        "forbidden",
        ["cite", "citation", "clause", "decline", "refuse", "verbatim", "infer"],
    )
    def test_no_conformance_instruction_is_present(self, forbidden: str):
        assert forbidden not in SYSTEM_PROMPT.lower()

    @pytest.mark.parametrize(
        "forbidden",
        ["always approve", "say yes", "guarantee", "promise", "grant", "approve"],
    )
    def test_the_agent_was_not_told_to_over_promise(self, forbidden: str):
        assert forbidden not in SYSTEM_PROMPT.lower()

    def test_it_still_reads_like_a_real_support_prompt(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "customer support" in lowered
        assert "acme" in lowered

    def test_the_context_names_the_source_document_but_offers_no_citation_handle(self):
        rendered = format_context([Hit(chunk=make_chunk(7), score=0.5)])
        assert "acme-refunds" in rendered
        assert "acme-refunds#0007" not in rendered

    def test_an_empty_retrieval_is_stated_plainly(self):
        assert "no relevant policy text found" in format_context([])

    def test_history_is_replayed_in_order_ahead_of_the_current_turn(self):
        messages = build_messages(
            "third",
            [Hit(chunk=make_chunk(1), score=0.5)],
            history=[("user", "first"), ("assistant", "second")],
        )
        assert [m["content"] for m in messages[1:]] == ["first", "second", "third"]

    def test_reference_material_is_not_disguised_as_a_prior_turn(self):
        """Keeping context in the system message means conversation history and policy
        text stay distinguishable, which matters when reading a failed row back."""
        messages = build_messages("q", [Hit(chunk=make_chunk(1), score=0.5)])
        assert "Refund window paragraph 1." in messages[0]["content"]
        assert len(messages) == 2
