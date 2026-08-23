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

from tests.model_families import family_of

AUT_DIR = Path(__file__).resolve().parents[2] / "aut-naive"
if str(AUT_DIR) not in sys.path:
    sys.path.insert(0, str(AUT_DIR))

import app as aut_app  # noqa: E402
from app import app, get_generator, get_retriever  # noqa: E402
from backends import (  # noqa: E402
    DEFAULT_GROQ_MODEL,
    OLLAMA_HOST_ENV,
    OLLAMA_MODEL,
    BackendError,
    build_generator,
)
from chunker import CHUNK_CHARS, OVERLAP_CHARS, Chunk  # noqa: E402
from prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    TEMPERATURE,
    build_messages,
    format_context,
)
from retrieval import MODEL_NAME, TOP_K, Hit  # noqa: E402

REPLY = "Yes, you can absolutely return that - I'll get it sorted for you."

REPO_ROOT = AUT_DIR.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"
COMPOSE = REPO_ROOT / "docker-compose.yml"


def env_example_models() -> dict[str, str]:
    """The `*_MODEL` settings as `.env.example` documents them."""
    assert ENV_EXAMPLE.is_file(), (
        f"{ENV_EXAMPLE.name} is missing from the working tree. It is tracked in git, so "
        "this is a local deletion, not an absence - it went missing around the 2026-08-23 "
        "key rotation, most likely `.env` being created from it by rename rather than "
        "copy. Restore with `git show HEAD:.env.example > .env.example` rather than "
        "committing the deletion, which would take the documented config with it. This "
        "check exists because the bare FileNotFoundError below reads like a broken test."
    )
    found: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.endswith("_MODEL") and value:
            found[key] = value
    return found


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


class TestTheJudgeStaysInADifferentFamilyFromTheAgent:
    """DESIGN.md 1.5: "judge model deliberately from a **different family** than the AUT",
    repeated in the 2 step 8 role table as "Different family from AUT; span-verified".

    An AUT and a judge from one family share pretraining and therefore share blind spots:
    the judge is most likely to find a fluent over-promise reasonable exactly when the
    agent's own priors produced it. That is grading with the marker that wrote the answer.
    Asserted here because the constraint spans two files and a comment cannot enforce it.

    THE COMMENT IN `backends.py` ABOVE `DEFAULT_GROQ_MODEL` IS STALE, AND MUST STAY STALE.
    It explains the Qwen requirement by naming the judge as `llama-3.3-70b-versatile` and the
    adversary as `llama-3.1-8b-instant`. Both were re-pinned on 2026-08-23 - the judge is now
    hosted gpt-oss and the adversary is local mistral - so the comment's *reason* is out of
    date while its *rule* is still exactly right.

    It cannot be corrected. `aut-naive/backends.py` is inside the frozen tree, the
    `aut-naive-v1` tag records `git rev-parse HEAD:aut-naive`, and editing any byte under
    `aut-naive/` changes that hash and makes every audit row citing the tag unverifiable
    (commitment C3). Fixing a comment is not worth breaking the freeze, and a freeze that
    bends for a comment is not a freeze.

    So this class is the live source of truth for the rule, which is what that comment's own
    last line already says: "Enforced by a test, not by this comment." The next legitimate
    re-freeze of the agent is when the wording gets updated.
    """

    def test_the_frozen_default_is_qwen(self):
        assert family_of(OLLAMA_MODEL) == "qwen"

    def test_the_groq_fallback_does_not_silently_switch_family(self):
        """The fallback exists so the agent can run without Ollama - not so it can run as
        a different species of model while still reporting itself as aut-naive."""
        assert family_of(DEFAULT_GROQ_MODEL) == family_of(OLLAMA_MODEL)

    @pytest.mark.parametrize(
        "role",
        [
            "CLAUSEGUARD_JUDGE_MODEL",
            "CLAUSEGUARD_EXTRACTOR_MODEL",
            "CLAUSEGUARD_ADVERSARY_MODEL",
        ],
    )
    def test_no_harness_role_shares_a_family_with_the_agent(self, role: str):
        models = env_example_models()
        assert role in models, f"{role} is missing from .env.example"
        assert family_of(models[role]) != family_of(OLLAMA_MODEL)

    def test_the_fallback_is_not_the_adversary_model_itself(self):
        """An adversary writing probes against itself is not adversarial, it is a mirror."""
        adversary = env_example_models()["CLAUSEGUARD_ADVERSARY_MODEL"]
        assert not adversary.endswith(DEFAULT_GROQ_MODEL)

    def test_the_env_override_for_the_dead_fallback_stays_in_the_agents_family(self):
        """`DEFAULT_GROQ_MODEL` (qwen/qwen3-32b) was decommissioned on 2026-08-23, but it
        lives inside the frozen tree and cannot be edited: the `aut-naive-v1` tag records
        `git rev-parse HEAD:aut-naive`, and a moved tag makes every audit row citing it
        unverifiable (commitment C3). The revival therefore happens through the
        `GROQ_MODEL` env var, which `backends.py` reads ahead of its own constant.

        Which means the family constraint has moved out of frozen code and into a config
        file, where it can be changed by anyone - so it needs asserting here. Equality, not
        difference: the fallback exists so the agent can run without Ollama, not so it can
        run as a different species of model while still reporting itself as the same agent.
        """
        models = env_example_models()
        assert "GROQ_MODEL" in models, "GROQ_MODEL is missing from .env.example"
        assert family_of(models["GROQ_MODEL"]) == family_of(OLLAMA_MODEL)


class TestTheJudgeDoesNotShareAFamilyWithTheAdversary:
    """DESIGN.md 2 (lines 253-256), verbatim:

        "Using one prompt for all three is the most common way this project quietly fails: a
        model that generated a probe is measurably more likely to accept a response that
        pattern-matches its own generation. Different roles, different prompts, and where
        budget allows different providers."

    Note what the spec actually requires and what it merely prefers. Different *prompts* is
    unconditional; different *providers* is hedged with "where budget allows". So unlike the
    judge-vs-agent rule in 1.5, this assertion enforces more than the specification demands,
    and that was a deliberate call (user-approved 2026-08-23) rather than a reading of the
    text.

    The reason for going beyond the spec here: the named failure - a model accepting what
    pattern-matches its own generation - is a property of the *model*, not of the prompt. A
    different prompt does not give a model different priors about which refund story sounds
    reasonable. Prompt separation is necessary and cheap; family separation is what actually
    addresses the mechanism the spec describes. The current pins satisfy it at no cost, so
    the budget hedge does not apply and there is nothing to trade away by pinning it.

    Since the judge moved to a hosted model later the same day, this pair now also satisfies
    the spec's hedged preference in full: judge `groq/openai/gpt-oss-20b` and adversary
    `ollama_chat/mistral:7b` are different *providers* as well as different families. That is
    a side effect of a latency decision rather than something aimed at, and it is the one
    pair for which the "different providers" wording was written.

    If a future provider outage forces both roles back into one family, this test failing is
    the correct outcome: it is a real weakening of the measurement and belongs in 8's
    limitations, not in a config diff nobody reads.
    """

    ROLES = ("CLAUSEGUARD_JUDGE_MODEL", "CLAUSEGUARD_ADVERSARY_MODEL")

    def test_the_judge_and_the_adversary_are_different_families(self):
        models = env_example_models()
        for role in self.ROLES:
            assert role in models, f"{role} is missing from .env.example"
        judge, adversary = (models[role] for role in self.ROLES)
        assert family_of(judge) != family_of(adversary), (
            f"judge {judge!r} and adversary {adversary!r} are both "
            f"{family_of(judge)!r}. The adversary writes the probe and the judge grades "
            "the response to it; from one family that is closer to self-assessment than "
            "to review. See this class's docstring before relaxing it."
        )

    def test_this_assertion_would_have_failed_under_the_pins_it_was_written_for(self):
        """Control, in the same spirit as `naive_family_of` in test_model_families.py.

        Until the 2026-08-23 re-pin the judge was `groq/llama-3.3-70b-versatile` and the
        adversary `groq/llama-3.1-8b-instant` - one family, differing only in parameter
        count, which is exactly the arrangement 2 warns about. The test above passes today
        only because the pins moved, so without this control there is no way to tell it
        apart from an assertion that cannot fail. That distinction has already bitten this
        project three times.
        """
        assert family_of("groq/llama-3.3-70b-versatile") == family_of(
            "groq/llama-3.1-8b-instant"
        ), "the control is broken: these two really were one family"


def ptext(pairs: set[frozenset[str]]) -> str:
    """Render colliding role pairs readably in an assertion message.

    Bare frozensets print as `frozenset({'judge', 'extractor'})` in an unstable order, which
    is a poor thing to hand someone reading a failure for the first time. Sorted twice - within
    each pair and across them - so the message is identical on every run and can be diffed.
    """
    if not pairs:
        return "{none}"
    return "{" + ", ".join(sorted("+".join(sorted(pair)) for pair in pairs)) + "}"


class TestTheKnownFamilyCollisionIsTheDocumentedOne:
    """Four roles, three families. The judge and the extractor are both gpt-oss, and that is
    an accepted limitation rather than an oversight - so it is asserted, not merely written
    down, and asserted in a form that pins *which* collision is the accepted one.

    WHY THE COLLISION EXISTS
    Of the 13 models this account can see, five are chat models: three gpt-oss, one Qwen -
    the agent's own family, closed to the judge by DESIGN.md 1.5 - and `allam-2-7b`. The
    remaining eight are ASR, TTS, tiny prompt-injection classifiers, or agentic systems with
    built-in web search that 4.1 disqualifies. So no hosted model in a fourth family is a
    *suitable* judge: `allam-2-7b` is Arabic-first at 7B with unverified tool-calling, and
    the judge is the one role graded mechanically by C2's exact-substring span check. The
    judge also had to be hosted at all: a local 8B judge measured ~11.7s per call on this
    machine against 2 step 11's "under 45 seconds for an incremental run", and the judge is
    the only harness role on the incremental path. Full reasoning sits above
    `DEFAULT_JUDGE_MODEL` in `harness/judge/judge.py`.

    WHY THIS TEST IS AN EQUALITY ON A SET AND NOT A "count >= 3"
    A count would let the collision *move* silently. If someone re-pinned the extractor to a
    Mistral model, the count of distinct families would stay at three and this file would go
    on passing while the collision landed on the adversary/judge pair - the one pair DESIGN.md
    2 actually warns about, and the one `TestTheJudgeDoesNotShareAFamilyWithTheAdversary`
    exists to forbid. Comparing the full set of colliding pairs against a single named pair
    means any change to which roles overlap fails here and has to be re-argued.

    So this test failing is not a bug to route around. It means the family topology moved, and
    the correct response is to decide whether the new overlap is acceptable and record that
    decision - in 8's limitations and in this docstring - before changing the expectation.

    `GROQ_MODEL`, the agent's hosted fallback, is deliberately absent from the table below. It
    is pinned to the agent's family *on purpose* so the agent cannot change species by
    changing backend, which `test_the_env_override_for_the_dead_fallback_stays_in_the_agents_family`
    asserts as an equality. Including it here would book that intended match as a collision.
    """

    HARNESS_ROLES = {
        "extractor": "CLAUSEGUARD_EXTRACTOR_MODEL",
        "judge": "CLAUSEGUARD_JUDGE_MODEL",
        "adversary": "CLAUSEGUARD_ADVERSARY_MODEL",
    }

    #: The overlap that has been argued for and accepted. Any other, or none, fails.
    ACCEPTED = {frozenset(("extractor", "judge"))}

    def _families(self) -> dict[str, str]:
        """Role -> family, reading each role from wherever it is really pinned.

        The agent's comes from the frozen tree rather than `.env.example`: its model is a
        constant in `backends.py` precisely because a frozen agent whose behaviour moves with
        an env var is not frozen (commitment C3), so `.env.example` is not authoritative for it.
        """
        models = env_example_models()
        for role, key in self.HARNESS_ROLES.items():
            assert key in models, f"{key} ({role}) is missing from .env.example"
        families = {"agent": family_of(OLLAMA_MODEL)}
        families.update(
            {role: family_of(models[key]) for role, key in self.HARNESS_ROLES.items()}
        )
        return families

    @staticmethod
    def _colliding_pairs(families: dict[str, str]) -> set[frozenset[str]]:
        return {
            frozenset((one, other))
            for one, first in families.items()
            for other, second in families.items()
            if one != other and first == second
        }

    def test_the_only_family_overlap_is_judge_and_extractor(self):
        families = self._families()
        collisions = self._colliding_pairs(families)
        assert collisions == self.ACCEPTED, (
            f"the family topology moved. Roles now map to {families!r}, giving overlapping "
            f"pairs {ptext(collisions)} where only {ptext(self.ACCEPTED)} has been argued "
            "for and accepted. Read this class's docstring: the fix is to decide whether the "
            "new overlap is acceptable and record it in DESIGN.md 8's limitations, not to "
            "edit ACCEPTED until the test passes."
        )

    def test_the_check_is_sensitive_in_both_directions(self):
        """Control, in the same spirit as `naive_family_of` in test_model_families.py.

        The assertion above compares against a hand-written expectation, which is exactly the
        shape of test that can quietly stop discriminating. These two hypotheticals prove the
        comparison notices both a *wider* overlap than the accepted one and a *narrower* one -
        so passing above means the topology really is the documented one, not that the helper
        returns the accepted set regardless of input.
        """
        collapsed = {
            "agent": "qwen",
            "extractor": "gpt",
            "judge": "gpt",
            "adversary": "gpt",
        }
        assert len(self._colliding_pairs(collapsed)) == 3, (
            "the control is broken: three roles in one family is three overlapping pairs"
        )
        assert self._colliding_pairs(collapsed) != self.ACCEPTED

        four_distinct = {
            "agent": "qwen",
            "extractor": "gpt",
            "judge": "llama",
            "adversary": "mistral",
        }
        assert self._colliding_pairs(four_distinct) == set()
        assert self._colliding_pairs(four_distinct) != self.ACCEPTED


class TestTheOllamaHostVariableDoesNotCollide:
    """Ollama reads `OLLAMA_HOST` to decide what address its *server* binds to. On Windows
    it has to be `0.0.0.0:11434` before a container can reach it at all - and that value is
    meaningless as a client target, because inside the container `0.0.0.0` is the container.
    Reusing the name would make both sides look correct while every request 502s.
    """

    def test_the_client_variable_is_not_ollama_s_own_bind_variable(self):
        assert OLLAMA_HOST_ENV == "AUT_OLLAMA_HOST"
        assert OLLAMA_HOST_ENV != "OLLAMA_HOST"

    def test_compose_passes_the_client_variable_and_defaults_to_the_host_gateway(self):
        lines = [
            line.strip()
            for line in COMPOSE.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith(f"{OLLAMA_HOST_ENV}:")
        ]
        assert lines, f"docker-compose.yml does not pass {OLLAMA_HOST_ENV}"
        assert "host.docker.internal:11434" in lines[0]

    def test_env_example_documents_the_name_the_code_reads(self):
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        assert f"\n{OLLAMA_HOST_ENV}=" in text

    def test_an_unknown_backend_is_refused_rather_than_defaulted(self):
        with pytest.raises(BackendError, match="unknown LLM_BACKEND"):
            build_generator("vllm")

