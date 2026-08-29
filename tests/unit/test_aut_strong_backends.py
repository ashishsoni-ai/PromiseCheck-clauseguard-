"""aut-strong's HTTP contract and its backend. STEP 4.

Three subjects, one file, because they share a loader and the loader is the hard part.

WHY THIS FILE EXECUTES MODULES BY PATH AND NEVER TOUCHES sys.path
`tests/unit/test_aut_contract.py:36-38` does `sys.path.insert(0, aut-naive)` and then bare
`from app import ...`, which binds aut-naive's modules in `sys.modules` under the names
`chunker`, `prompts`, `retrieval`, `backends` and `app`. aut-strong ships files with all
five of those names, and collection is alphabetical, so a second insert here would receive
aut-naive's modules and assert against the wrong agent while passing.

`aut-strong/prompts.py` could be loaded under an alias because it imports no siblings
(`tests/unit/test_aut_strong_prompts.py` does exactly that). `app.py` cannot: it runs as
`uvicorn app:app` with no package around it, so its imports are bare by necessity, and
`from backends import ...` resolves through `sys.modules` and `sys.path` like any other.
The loader below therefore *borrows* the five bare names - saves whatever is under them,
executes one agent's modules in dependency order, then puts the originals back. sys.path
is never touched, so nothing about which test ran first can change what gets loaded.

`test_the_loader_restored_the_bare_names` and `test_app_caught_the_backend_error_class_this
_file_raises` exist to fail if that machinery is wrong, because every assertion below is
meaningless if it is - a fake raising some *other* module's `BackendError` would sail past
app.py's `except BackendError` and show up as a 500 that looks like a bug in the agent.

WHAT IS NOT HERE
The freeze discipline for this agent's Dockerfile - digest pin, both baked revisions
agreeing with `retrieval.py`, the offline guarantee - is STEP 5's, and it wants a mirror of
`test_aut_freeze.py` rather than a corner of this file. Three comments in
`aut-strong/Dockerfile` already cite tests that do not exist yet; task #90 records that.
The two byte-equality claims that file makes about `requirements.txt` and `corpus/` are
asserted here, because they are what makes "same corpus, different retrieval" true and
they cost four lines.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator, Sequence

import pytest
from fastapi.testclient import TestClient

from tests.model_families import family_of, strip_provider_prefix

REPO_ROOT = Path(__file__).resolve().parents[2]
STRONG_DIR = REPO_ROOT / "aut-strong"
NAIVE_DIR = REPO_ROOT / "aut-naive"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
COMPOSE = REPO_ROOT / "docker-compose.yml"
LIMITATIONS = REPO_ROOT / "docs" / "limitations.md"

#: Dependency order. Each module must be findable under its bare name before the next
#: one's `from <name> import ...` executes.
AGENT_MODULES = ("chunker", "prompts", "retrieval", "backends", "app")


@contextmanager
def _bare_names_borrowed(names: Sequence[str]) -> Iterator[None]:
    """Lend `sys.modules` the bare module names, then hand back exactly what was there."""
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    for name in names:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def _load_agent(agent_dir: Path) -> dict[str, ModuleType]:
    """Execute one agent's five modules against each other and nothing else.

    Registration under the bare name is not cosmetic: `from backends import BackendError`
    inside `app.py` resolves through `sys.modules`, and dataclass and pydantic class
    creation want the defining module findable by its own `__name__` while the class body
    runs. Both happen inside the borrow window, so the names are only needed there.
    """
    loaded: dict[str, ModuleType] = {}
    with _bare_names_borrowed(AGENT_MODULES):
        for name in AGENT_MODULES:
            path = agent_dir / f"{name}.py"
            assert path.is_file(), f"expected {path} to exist"
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    return loaded


_MODULES_BEFORE = {name: sys.modules.get(name) for name in AGENT_MODULES}

STRONG = _load_agent(STRONG_DIR)
NAIVE = _load_agent(NAIVE_DIR)

backends = STRONG["backends"]
prompts = STRONG["prompts"]
retrieval = STRONG["retrieval"]
strong_app = STRONG["app"]
naive_backends = NAIVE["backends"]
naive_app = NAIVE["app"]

#: A reply in aut-strong's own register - hedged and pointing at the exclusions - so that
#: a test reading as "the agent answered" is not also quietly asserting an over-promise.
REPLY = "Section 5 lists exclusions, so let me quote the clause that applies to your item."


def env_example_models() -> dict[str, str]:
    """The active `*_MODEL` settings in `.env.example`.

    Reimplemented rather than imported from `test_aut_contract.py`: importing that module
    would execute its `sys.path.insert`, which is the one thing this file must not cause.
    """
    assert ENV_EXAMPLE.is_file(), f"{ENV_EXAMPLE.name} is missing from the working tree"
    found: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.endswith("_MODEL") and value:
            found[key] = value
    return found


def compose_service(name: str) -> list[str]:
    """The lines of one active service block in `docker-compose.yml`.

    Hand-parsed rather than `yaml.safe_load`ed: PyYAML is not a declared dependency of
    this harness, and a test that skips when an optional import is missing is a test that
    stops running the day someone trims the venv. Two-space indentation is what this file
    uses and what `test_aut_contract.py` already assumes.
    """
    lines = COMPOSE.read_text(encoding="utf-8").splitlines()
    block: list[str] = []
    inside = False
    for line in lines:
        if line.startswith(f"  {name}:"):
            inside = True
            continue
        if inside:
            stripped = line.strip()
            if line.startswith("  ") and not line.startswith("   ") and stripped:
                break  # the next service at the same depth
            block.append(line)
    return block


def source_of(module: str, agent_dir: Path = STRONG_DIR) -> str:
    return (agent_dir / f"{module}.py").read_text(encoding="utf-8")


def getenv_literals(module: str, agent_dir: Path = STRONG_DIR) -> set[str]:
    """Every name passed to `os.getenv` in one of an agent's modules.

    A literal argument comes back as itself; a variable argument comes back as `<NAME>`,
    so a read routed through a constant is visible as a read rather than vanishing.

    An AST walk, not a substring scan: `backends.py`'s docstring names `LLM_BACKEND` and
    `GROQ_MODEL` repeatedly while explaining why it does not read them, so a grep would
    find the prose - and a grep tuned to skip the prose would be one reformat away from
    missing a real read.
    """
    tree = ast.parse(source_of(module, agent_dir))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        target = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if target != "getenv":
            continue
        for arg in node.args[:1]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                names.add(arg.value)
            elif isinstance(arg, ast.Name):
                names.add(f"<{arg.id}>")
    return names


# --------------------------------------------------------------------------------------
# Fakes. Duck-typed against the surface app.py actually touches, and no wider: a fake that
# implements more than the real object is a fake that can hide a missing attribute.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class FakeHit:
    chunk: FakeChunk
    score: float


def make_hit(n: int) -> FakeHit:
    return FakeHit(
        chunk=FakeChunk(
            chunk_id=f"acme-refunds#{n:04d}",
            doc_id="acme-refunds",
            text=f"Clause paragraph {n}. Items may be returned within the stated window.",
        ),
        score=1.0 - n / 100,
    )


class FakeModel:
    """Stands in for the embedder and for the reranker. Both expose name + revision."""

    def __init__(self, name: str, revision: str) -> None:
        self._name = name
        self._revision = revision

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return self._revision


class FakeRetriever:
    """Every attribute here is one `/chat` or `/health` reads. Nothing spare."""

    def __init__(self, hits: Sequence[FakeHit]) -> None:
        self._hits = list(hits)
        self.chunks = tuple(hit.chunk for hit in hits)
        self.top_k = retrieval.TOP_K
        self.candidate_k = retrieval.CANDIDATE_K
        self.embedder = FakeModel(retrieval.MODEL_NAME, retrieval.MODEL_REVISION)
        self.reranker = FakeModel(retrieval.RERANKER_NAME, retrieval.RERANKER_REVISION)
        self.fingerprint = "sha256:" + "0" * 64
        self.queries: list[str] = []

    def search(self, query: str, k: int | None = None) -> list[FakeHit]:
        self.queries.append(query)
        return self._hits[: k or self.top_k]


class FakeGenerator:
    backend = "groq"

    def __init__(self, reply: str = REPLY, error: Exception | None = None) -> None:
        self.name = backends.DEFAULT_GROQ_MODEL
        self._reply = reply
        self._error = error
        self.calls: list[tuple[list[dict[str, str]], float]] = []

    def complete(self, messages: Sequence[dict[str, str]], *, temperature: float) -> str:
        self.calls.append(([dict(m) for m in messages], temperature))
        if self._error is not None:
            raise self._error
        return self._reply


@dataclass
class Rig:
    client: TestClient
    retriever: FakeRetriever
    generator: FakeGenerator


@pytest.fixture(autouse=True)
def _no_ambient_overrides(monkeypatch):
    """Every test here starts from "this agent's own variables are unset".

    Without this, a developer who exported `AUT_STRONG_GROQ_MODEL` to try something would
    see `/health` and default-resolution tests fail for a reason that has nothing to do
    with the code - and, worse, a stale export in CI would make the default path untested
    while everything stayed green. Tests that want a value set one explicitly.
    """
    for name in (backends.BACKEND_ENV, backends.GROQ_MODEL_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_rig():
    """Factory, so one test can build a failing backend and another a working one."""

    def _make(
        reply: str = REPLY,
        error: Exception | None = None,
        hits: Sequence[FakeHit] | None = None,
        with_generator: bool = True,
    ) -> Rig:
        retriever = FakeRetriever(
            list(hits) if hits is not None else [make_hit(n) for n in range(1, 9)]
        )
        generator = FakeGenerator(reply=reply, error=error)
        strong_app.app.dependency_overrides[strong_app.get_retriever] = lambda: retriever
        if with_generator:
            strong_app.app.dependency_overrides[strong_app.get_generator] = (
                lambda: generator
            )
        strong_app._sessions.clear()
        return Rig(TestClient(strong_app.app), retriever, generator)

    yield _make
    strong_app.app.dependency_overrides.clear()
    strong_app._sessions.clear()


@pytest.fixture
def rig(make_rig):
    return make_rig()


class TestTheLoaderDidWhatItClaims:
    """If these fail, nothing else in this file means anything."""

    def test_the_two_agents_modules_are_different_objects(self):
        for name in AGENT_MODULES:
            assert STRONG[name] is not NAIVE[name], name

    def test_each_agent_loaded_its_own_file(self):
        assert backends.OLLAMA_MODEL is None
        assert naive_backends.OLLAMA_MODEL == "qwen2.5:7b-instruct"
        assert strong_app.freeze_identity()["aut_name"] == "aut-strong"
        assert naive_app.freeze_identity()["aut_name"] == "aut-naive"

    def test_the_loader_restored_the_bare_names(self):
        """The borrow is reversible, so collection order stays irrelevant."""
        assert {name: sys.modules.get(name) for name in AGENT_MODULES} == _MODULES_BEFORE

    def test_app_caught_the_backend_error_class_this_file_raises(self):
        """`except BackendError` in app.py must name the class the fakes raise.

        Two copies of the same module define two unrelated exception classes, and the
        failure is silent in the worst direction: the 502 path would never run, the
        exception would escape as a 500, and the tests below would be asserting that a
        transport failure looks like a crash.
        """
        assert strong_app.BackendError is backends.BackendError
        assert strong_app.BackendError is not naive_backends.BackendError


class TestThereIsNoLocalBackend:
    """aut-strong is hosted-only, by decision, and the decision is asserted here.

    DESIGN.md 1.4 documents a frontier API model for this agent and names no local one.
    Pinning a plausible local model anyway would be an undocumented configuration that
    produces rows indistinguishable at a glance from hosted rows - and pooling two models
    under one agent name reports a system that does not exist. So the local path does not
    exist, and asking for it fails closed with the reason in the message.
    """

    def test_no_local_model_is_pinned(self):
        assert backends.OLLAMA_MODEL is None

    def test_there_is_no_ollama_generator_at_all(self):
        """Not merely unreachable - absent. An unused class is an invitation."""
        assert not hasattr(backends, "OllamaGenerator")
        assert not hasattr(backends, "OLLAMA_HOST_ENV")
        assert not hasattr(backends, "DEFAULT_OLLAMA_HOST")

    def test_asking_for_ollama_fails_with_the_reason(self):
        with pytest.raises(backends.BackendError) as caught:
            backends.build_generator("ollama")
        message = str(caught.value)
        assert "OLLAMA_MODEL" in message
        assert backends.BACKEND_ENV in message
        # The trap this message exists for: the repository's own .env.example sets
        # LLM_BACKEND=ollama, for the other agent.
        assert "LLM_BACKEND" in message

    def test_an_unknown_backend_names_the_variable_to_fix(self):
        with pytest.raises(backends.BackendError) as caught:
            backends.build_generator("vllm")
        assert backends.BACKEND_ENV in str(caught.value)
        assert "vllm" in str(caught.value)

    def test_the_default_path_is_the_hosted_one(self, monkeypatch):
        """Behavioural, not a constant read: with no key the default backend must fail on
        the *key*, which is only true if the default resolved to groq rather than ollama.
        """
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(backends.BackendError) as caught:
            backends.build_generator()
        assert "GROQ_API_KEY" in str(caught.value)

    def test_the_two_agents_defaults_are_mirror_images(self):
        assert backends.DEFAULT_BACKEND == "groq"
        assert naive_backends.OLLAMA_MODEL is not None


class TestItReadsItsOwnEnvironmentVariables:
    """The species-swap guard. See backends.py's docstring for the mechanism.

    `.env.example` pins `GROQ_MODEL` to a Qwen model on purpose, and an existing test
    asserts it stays in aut-naive's family. If this agent read the same variable it would
    run as qwen3.6-27b while reporting itself as aut-strong.
    """

    def test_the_variable_names_are_this_agents_own(self):
        assert backends.BACKEND_ENV == "AUT_STRONG_LLM_BACKEND"
        assert backends.GROQ_MODEL_ENV == "AUT_STRONG_GROQ_MODEL"
        assert backends.BACKEND_ENV != "LLM_BACKEND"
        assert backends.GROQ_MODEL_ENV != "GROQ_MODEL"

    def test_aut_naive_has_no_such_variable(self):
        """Proves the asymmetry is real rather than a rename of a shared thing."""
        assert not hasattr(naive_backends, "BACKEND_ENV")
        assert not hasattr(naive_backends, "GROQ_MODEL_ENV")

    def test_the_shared_variables_are_never_read(self):
        assert getenv_literals("backends") == {
            "GROQ_API_KEY",
            "<BACKEND_ENV>",
            "<GROQ_MODEL_ENV>",
        }

    @pytest.mark.parametrize("module", AGENT_MODULES)
    def test_no_module_of_this_agent_reads_them(self, module):
        """Wider than backends.py, because app.py is where a convenience read would go -
        and a `/health` that reported `LLM_BACKEND` would document a wiring that does not
        exist."""
        assert not {"LLM_BACKEND", "GROQ_MODEL"} & getenv_literals(module)

    def test_aut_naive_really_does_read_them(self):
        """Control. Without it, the assertions above could pass because the AST walk is
        broken rather than because this agent abstains."""
        assert {"LLM_BACKEND", "GROQ_MODEL"} <= getenv_literals("backends", NAIVE_DIR)

    @pytest.mark.parametrize("value", ["ollama", "groq", "anything"])
    def test_the_other_agents_backend_switch_is_ignored(self, monkeypatch, value):
        monkeypatch.setenv("LLM_BACKEND", value)
        monkeypatch.delenv(backends.BACKEND_ENV, raising=False)
        assert backends.configured_backend() == "groq"

    def test_the_other_agents_model_pin_is_ignored(self, monkeypatch):
        monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.6-27b")
        monkeypatch.delenv(backends.GROQ_MODEL_ENV, raising=False)
        assert backends.configured_model() == backends.DEFAULT_GROQ_MODEL
        assert family_of(backends.configured_model()) != family_of("qwen/qwen3.6-27b")

    def test_its_own_variables_do_take_effect(self, monkeypatch):
        monkeypatch.setenv(backends.GROQ_MODEL_ENV, "openai/gpt-oss-20b")
        assert backends.configured_model() == "openai/gpt-oss-20b"
        monkeypatch.setenv(backends.BACKEND_ENV, "GROQ")
        assert backends.configured_backend() == "groq"

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_a_blank_override_reads_as_unset(self, monkeypatch, blank):
        """`AUT_STRONG_GROQ_MODEL=` is a badly commented-out line, not a request for the
        model whose name is the empty string."""
        monkeypatch.setenv(backends.GROQ_MODEL_ENV, blank)
        monkeypatch.setenv(backends.BACKEND_ENV, blank)
        assert backends.configured_model() == backends.DEFAULT_GROQ_MODEL
        assert backends.configured_backend() == backends.DEFAULT_BACKEND

    def test_env_example_documents_both_without_pinning_either(self):
        """Documented as commented lines, deliberately.

        An active `AUT_STRONG_GROQ_MODEL=` line would move a frozen agent's model out of
        code and into a config file - the exact drift `.env.example` warns about two
        paragraphs above, and the reason aut-naive's constants live in `backends.py`.
        """
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        for name in (backends.BACKEND_ENV, backends.GROQ_MODEL_ENV):
            assert f"# {name}=" in text, f"{name} is not documented in .env.example"
            assert f"\n{name}=" not in text, f"{name} must not be pinned in .env.example"
        assert backends.GROQ_MODEL_ENV not in env_example_models()

    def test_compose_hands_this_agent_only_its_own_variables(self):
        """The wiring half of the same point, and the one that fails silently.

        `/health` does not construct a generator, so an agent handed a backend it cannot
        serve would pass its healthcheck and 502 every `/chat` - green container, no
        answers. Compose must therefore not pass aut-naive's switch to this service.
        """
        service = compose_service("aut-strong")
        assert service, "docker-compose.yml has no active aut-strong service"
        environment = {
            line.split(":", 1)[0].strip()
            for line in service
            if ":" in line and not line.lstrip().startswith("#")
        }
        assert backends.BACKEND_ENV in environment
        assert backends.GROQ_MODEL_ENV in environment
        assert "GROQ_API_KEY" in environment
        assert "LLM_BACKEND" not in environment
        assert "AUT_OLLAMA_HOST" not in environment

    def test_compose_still_passes_aut_naive_its_own(self):
        """Control, and a guard against fixing the above by deleting both variables."""
        service = compose_service("aut-naive")
        rendered = "\n".join(service)
        assert "LLM_BACKEND:" in rendered
        assert "AUT_OLLAMA_HOST:" in rendered
        assert "AUT_STRONG_LLM_BACKEND:" not in rendered


class TestTheRoutingPrefixIsRejectedAtConstruction:
    """`groq/openai/gpt-oss-120b` is litellm's address for this model. The native client
    wants `openai/gpt-oss-120b`, and the prefixed form is a 404 per request rather than a
    startup error - so it is caught at construction, where it is one message instead of
    thirty identical failures.

    Rejected rather than stripped: stripping would also silently accept
    `groq/some-other-model` and answer as a model nobody chose.
    """

    def test_a_prefixed_model_is_refused_and_the_fix_is_in_the_message(self):
        with pytest.raises(backends.BackendError) as caught:
            backends.GroqGenerator(model="groq/openai/gpt-oss-120b")
        message = str(caught.value)
        assert "openai/gpt-oss-120b" in message
        assert backends.GROQ_MODEL_ENV in message

    def test_the_env_override_is_checked_too(self, monkeypatch):
        monkeypatch.setenv(backends.GROQ_MODEL_ENV, "groq/openai/gpt-oss-20b")
        with pytest.raises(backends.BackendError, match="routing prefix"):
            backends.GroqGenerator()

    def test_health_still_reports_the_broken_value(self, monkeypatch):
        """`configured_model` deliberately does not validate: /health showing the value
        that will fail is how the operator finds out which variable they got wrong."""
        monkeypatch.setenv(backends.GROQ_MODEL_ENV, "groq/openai/gpt-oss-20b")
        assert backends.configured_model() == "groq/openai/gpt-oss-20b"

    def test_the_check_is_not_a_catch_all(self, monkeypatch):
        """Control. An unprefixed id must get *past* this check, which is only visible
        because the next failure names something else."""
        monkeypatch.setenv(backends.GROQ_MODEL_ENV, "openai/gpt-oss-120b")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(backends.BackendError) as caught:
            backends.GroqGenerator()
        assert "GROQ_API_KEY" in str(caught.value)
        assert "routing prefix" not in str(caught.value)

    def test_a_missing_client_is_a_backend_error_not_an_import_error(self, monkeypatch):
        """`groq` is an AUT-only dependency and is absent from the harness venv on
        purpose, so this path is reachable here and is the reason the import is wrapped: a
        bare ImportError escapes app.py's `except BackendError` as a 500."""
        monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
        if importlib.util.find_spec("groq") is not None:
            pytest.skip("the groq client is installed in this environment")
        with pytest.raises(backends.BackendError, match="not installed"):
            backends.GroqGenerator()


class TestTheGeneratorMatchesTheProtocolAutNaiveDefines:
    """The harness never imports either module, so the protocol is the whole contract."""

    def test_the_real_generator_class_satisfies_it(self):
        assert isinstance(backends.GroqGenerator, type)
        for member in ("complete", "name", "backend"):
            assert hasattr(backends.GroqGenerator, member), member

    def test_the_fake_satisfies_it_too(self):
        assert isinstance(FakeGenerator(), backends.Generator)

    def test_the_protocol_surface_is_the_same_in_both_agents(self):
        def surface(protocol: type) -> set[str]:
            # `vars`, not `dir`: `dir` on a Protocol also reports `register` and `mro` off
            # ABCMeta, which are the metaclass's and not part of the contract.
            return {name for name in vars(protocol) if not name.startswith("_")}

        assert surface(backends.Generator) == surface(naive_backends.Generator)
        assert surface(backends.Generator) == {"complete", "name", "backend"}


class TestTheChatContractIsCopiedNotRedesigned:
    """`clauseguard run --agent` must work against this agent with no harness change."""

    EXPECTED_FIELDS = {
        "reply",
        "session_id",
        "turn",
        "latency_ms",
        "model",
        "backend",
        "retrieved_chunk_ids",
        "aut_commit_sha",
        "aut_repo_head",
    }

    def test_a_reply_carries_exactly_the_fields_aut_naive_carries(self, rig):
        response = rig.client.post("/chat", json={"session_id": "s1", "message": "hi?"})
        assert response.status_code == 200, response.text
        assert set(response.json()) == self.EXPECTED_FIELDS

    def test_the_response_model_is_field_for_field_aut_naives(self):
        """Compared against the other agent rather than against a hand-written list, so a
        field added to one and not the other fails here whichever one moved."""
        for model in ("ChatRequest", "ChatResponse"):
            strong_fields = getattr(strong_app, model).model_fields
            naive_fields = getattr(naive_app, model).model_fields
            assert {k: str(v.annotation) for k, v in strong_fields.items()} == {
                k: str(v.annotation) for k, v in naive_fields.items()
            }, model

    def test_the_routes_are_the_same_two(self):
        def routes(module: ModuleType) -> set[tuple[str, str]]:
            return {
                (route.path, method)
                for route in module.app.routes
                for method in getattr(route, "methods", set())
                if not route.path.startswith("/openapi")
                and route.path not in {"/docs", "/redoc", "/docs/oauth2-redirect"}
            }

        assert routes(strong_app) == routes(naive_app) == {("/chat", "POST"), ("/health", "GET")}

    def test_the_session_limits_are_identical(self):
        """Named in app.py as the reason a reply-quality difference cannot be blamed on
        one agent remembering more of the conversation."""
        assert strong_app.MAX_SESSIONS == naive_app.MAX_SESSIONS
        assert strong_app.MAX_HISTORY_MESSAGES == naive_app.MAX_HISTORY_MESSAGES

    def test_the_reply_and_the_identity_come_back(self, rig):
        body = rig.client.post("/chat", json={"session_id": "s", "message": "?"}).json()
        assert body["reply"] == REPLY
        assert body["session_id"] == "s"
        assert body["turn"] == 1
        assert body["latency_ms"] >= 0
        assert body["model"] == backends.DEFAULT_GROQ_MODEL
        assert body["backend"] == "groq"

    def test_retrieved_chunk_ids_are_the_post_rerank_top_k(self, rig):
        body = rig.client.post("/chat", json={"session_id": "s", "message": "?"}).json()
        assert body["retrieved_chunk_ids"] == [c.chunk_id for c in rig.retriever.chunks]
        assert len(body["retrieved_chunk_ids"]) == retrieval.TOP_K
        # Not the candidate set. Same field name as aut-naive, same meaning: what the
        # model was shown. This is the field an audit row uses to ask whether the agent
        # ever saw the governing clause.
        assert len(body["retrieved_chunk_ids"]) < retrieval.CANDIDATE_K

    def test_the_retrieved_text_reaches_the_model(self, rig):
        rig.client.post("/chat", json={"session_id": "s", "message": "can I return?"})
        messages, _ = rig.generator.calls[0]
        rendered = "\n".join(m["content"] for m in messages)
        for chunk in rig.retriever.chunks:
            assert chunk.text in rendered
        assert rig.retriever.queries == ["can I return?"]

    def test_the_temperature_is_the_defensive_one(self, rig):
        rig.client.post("/chat", json={"session_id": "s", "message": "?"})
        _, temperature = rig.generator.calls[0]
        assert temperature == prompts.TEMPERATURE == 0.1
        # DESIGN.md 1.4's independent variable, so it must not have drifted to aut-naive's.
        assert temperature != 0.7

    @pytest.mark.parametrize(
        "payload",
        [
            {"message": "no session"},
            {"session_id": "s"},
            {"session_id": "", "message": "empty session"},
            {"session_id": "s", "message": ""},
            {"session_id": "s" * 201, "message": "session too long"},
        ],
    )
    def test_a_malformed_request_is_a_422(self, rig, payload):
        assert rig.client.post("/chat", json=payload).status_code == 422


class TestSessionsCarryHistory:
    def test_the_previous_turn_is_replayed(self, rig):
        rig.client.post("/chat", json={"session_id": "s", "message": "first"})
        rig.client.post("/chat", json={"session_id": "s", "message": "second"})
        messages, _ = rig.generator.calls[1]
        rendered = [(m["role"], m["content"]) for m in messages]
        assert ("user", "first") in rendered
        assert ("assistant", REPLY) in rendered

    def test_the_turn_counter_counts_turns(self, rig):
        for expected in (1, 2, 3):
            body = rig.client.post(
                "/chat", json={"session_id": "s", "message": f"turn {expected}"}
            ).json()
            assert body["turn"] == expected

    def test_sessions_do_not_leak_into_each_other(self, rig):
        rig.client.post("/chat", json={"session_id": "a", "message": "secret"})
        rig.client.post("/chat", json={"session_id": "b", "message": "other"})
        messages, _ = rig.generator.calls[1]
        assert "secret" not in "\n".join(m["content"] for m in messages)

    def test_history_is_capped(self, rig):
        for n in range(10):
            rig.client.post("/chat", json={"session_id": "s", "message": f"m{n}"})
        assert len(strong_app._sessions["s"]) == strong_app.MAX_HISTORY_MESSAGES

    def test_the_oldest_session_is_evicted_first(self, rig, monkeypatch):
        monkeypatch.setattr(strong_app, "MAX_SESSIONS", 3)
        for name in ("s0", "s1", "s2"):
            rig.client.post("/chat", json={"session_id": name, "message": "?"})
        rig.client.post("/chat", json={"session_id": "s1", "message": "touch"})
        rig.client.post("/chat", json={"session_id": "s3", "message": "?"})
        assert "s0" not in strong_app._sessions
        assert "s1" in strong_app._sessions
        assert len(strong_app._sessions) == 3


class TestABackendFailureIsNotAnAnswer:
    """A 200 carrying an empty string scores `evasive`, which would credit the defensive
    prompt with a non-answer on the exact metric DESIGN.md 1.4 is measuring."""

    def test_a_backend_error_is_a_502(self, make_rig):
        rig = make_rig(error=backends.BackendError("groq exploded"))
        response = rig.client.post("/chat", json={"session_id": "s", "message": "?"})
        assert response.status_code == 502
        assert "groq exploded" in response.json()["detail"]

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
    def test_a_blank_reply_is_a_502(self, make_rig, blank):
        rig = make_rig(reply=blank)
        response = rig.client.post("/chat", json={"session_id": "s", "message": "?"})
        assert response.status_code == 502
        assert "groq" in response.json()["detail"]

    def test_a_failed_turn_is_not_written_to_history(self, make_rig):
        rig = make_rig(error=backends.BackendError("down"))
        rig.client.post("/chat", json={"session_id": "s", "message": "?"})
        assert strong_app._sessions.get("s") == []

    def test_a_retry_after_a_failure_is_still_turn_one(self, make_rig):
        rig = make_rig(error=backends.BackendError("down"))
        rig.client.post("/chat", json={"session_id": "s", "message": "?"})
        working = FakeGenerator()
        strong_app.app.dependency_overrides[strong_app.get_generator] = lambda: working
        body = rig.client.post("/chat", json={"session_id": "s", "message": "?"}).json()
        assert body["turn"] == 1

    def test_an_unbuildable_retriever_is_a_503(self):
        def broken() -> None:
            raise strong_app.HTTPException(status_code=503, detail="retrieval unavailable")

        strong_app.app.dependency_overrides[strong_app.get_retriever] = broken
        try:
            with TestClient(strong_app.app) as client:
                assert client.get("/health").status_code == 503
        finally:
            strong_app.app.dependency_overrides.clear()


class TestTheFreezeTravelsWithEveryReply:
    SENTINEL = "(unfrozen: built outside scripts/freeze_aut.py)"

    def test_an_unfrozen_build_says_so_rather_than_leaving_it_blank(
        self, rig, monkeypatch
    ):
        for name in ("AUT_COMMIT_SHA", "AUT_REPO_HEAD", "AUT_GIT_TAG", "AUT_FROZEN_AT"):
            monkeypatch.delenv(name, raising=False)
        body = rig.client.post("/chat", json={"session_id": "s", "message": "?"}).json()
        assert body["aut_commit_sha"] == self.SENTINEL
        assert body["aut_repo_head"] == self.SENTINEL

    def test_the_freeze_args_pass_through(self, rig, monkeypatch):
        monkeypatch.setenv("AUT_COMMIT_SHA", "a" * 40)
        monkeypatch.setenv("AUT_REPO_HEAD", "b" * 40)
        body = rig.client.post("/chat", json={"session_id": "s", "message": "?"}).json()
        assert body["aut_commit_sha"] == "a" * 40
        assert body["aut_repo_head"] == "b" * 40

    def test_the_agent_names_itself_correctly(self, rig):
        """The copy-paste guard. This file's whole subject is a directory cloned from
        aut-naive, and an identity that still said "aut-naive" would mislabel every row
        of the comparison DESIGN.md 1.4 exists to make."""
        assert strong_app.freeze_identity()["aut_name"] == "aut-strong"
        assert rig.client.get("/health").json()["aut_name"] == "aut-strong"


class TestHealthReportsWhatMakesThisAgentDifferent:
    def test_the_retrieval_block_names_both_models_and_both_depths(self, rig):
        block = rig.client.get("/health").json()["retrieval"]
        assert set(block) == {
            "embedding_model",
            "embedding_revision",
            "reranker_model",
            "reranker_revision",
            "top_k",
            "candidate_k",
            "chunk_chars",
            "overlap_chars",
            "chunks",
            "corpus_fingerprint",
        }
        assert block["reranker_model"] == retrieval.RERANKER_NAME
        assert block["reranker_revision"] == retrieval.RERANKER_REVISION
        assert block["embedding_model"] == retrieval.MODEL_NAME
        assert block["top_k"] == retrieval.TOP_K == 8
        assert block["candidate_k"] == retrieval.CANDIDATE_K
        assert block["candidate_k"] > block["top_k"]

    def test_the_reranker_is_what_aut_naives_health_cannot_report(self, rig):
        """The point of the extension: the two payloads differ exactly where the two
        agents differ, and the naive shape is read off the other agent rather than
        hand-copied here, so a change to either one lands on this assertion.
        """
        naive_app.app.dependency_overrides[naive_app.get_retriever] = (
            lambda: FakeRetriever([make_hit(1)])
        )
        try:
            naive_health = TestClient(naive_app.app).get("/health").json()
        finally:
            naive_app.app.dependency_overrides.clear()
        strong_health = rig.client.get("/health").json()

        assert set(naive_health) == set(strong_health)
        assert set(naive_health["retrieval"]) < set(strong_health["retrieval"])
        assert set(naive_health["generation"]) < set(strong_health["generation"])
        assert {"reranker_model", "reranker_revision", "candidate_k"} <= set(
            strong_health["retrieval"]
        )
        assert not {"reranker_model", "candidate_k"} & set(naive_health["retrieval"])

    def test_the_live_objects_are_reported_not_the_constants(self, make_rig):
        """A /health that quoted the module constant would keep describing the intended
        configuration while serving a different one - the one drift a git SHA cannot see.
        """
        rig = make_rig()
        rig.retriever.reranker = FakeModel("some/other-reranker", "f" * 40)
        block = rig.client.get("/health").json()["retrieval"]
        assert block["reranker_model"] == "some/other-reranker"
        assert block["reranker_model"] != retrieval.RERANKER_NAME

    def test_the_generation_block_names_the_variable_it_read(self, rig):
        block = rig.client.get("/health").json()["generation"]
        assert block["temperature"] == prompts.TEMPERATURE
        assert block["backend"] == "groq"
        assert block["model"] == backends.DEFAULT_GROQ_MODEL
        assert block["backend_env"] == backends.BACKEND_ENV == "AUT_STRONG_LLM_BACKEND"

    def test_health_does_not_need_a_provider_key(self, make_rig, monkeypatch):
        """A missing GROQ_API_KEY must not mark the container unhealthy. The agent is
        fine, its credentials are not, and a healthcheck that fails on a key takes the
        container down for a condition no restart fixes.
        """
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        rig = make_rig(with_generator=False)
        response = rig.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["generation"]["model"] == backends.DEFAULT_GROQ_MODEL

    def test_the_fingerprint_and_the_chunk_count_are_reported(self, rig):
        block = rig.client.get("/health").json()["retrieval"]
        assert block["corpus_fingerprint"].startswith("sha256:")
        assert block["chunks"] == len(rig.retriever.chunks)


class TestNothingReachesOutsideThisDirectory:
    """Commitment C3. An import graph check, not a substring match."""

    @pytest.mark.parametrize("module", AGENT_MODULES)
    def test_no_module_imports_the_harness(self, module):
        tree = ast.parse(source_of(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] != "harness", alias.name
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"{module} uses a relative import"
                root = (node.module or "").split(".")[0]
                assert root != "harness", node.module

    @pytest.mark.parametrize("module", AGENT_MODULES)
    def test_every_import_is_stdlib_a_sibling_or_a_pinned_dependency(self, module):
        """The drift guard behind the byte-identical requirements.txt.

        `aut-strong/requirements.txt` is byte-for-byte aut-naive's, which is what lets the
        two agents differ only in code. So an import of anything outside this set is a
        package the frozen image does not install, and the failure lands at container
        start - after the freeze, after the tag, and one step before a measured run.
        """
        allowed = (
            set(sys.stdlib_module_names)
            | set(AGENT_MODULES)
            | {"fastapi", "pydantic", "groq", "numpy", "faiss", "sentence_transformers"}
        )
        tree = ast.parse(source_of(module))
        roots = {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert roots <= allowed, f"{module} imports {sorted(roots - allowed)}"
        assert "tests" not in roots and "scripts" not in roots

    def test_the_check_would_notice_a_harness_import(self):
        """Control, in the same spirit as `naive_family_of` in test_model_families.py."""
        tree = ast.parse("from harness.judge import judge\n")
        roots = {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert roots == {"harness"}


class TestTheTwoAgentsShareTheirCorpusAndTheirPins:
    """Both claims are made in `aut-strong/Dockerfile` and neither was asserted anywhere.

    DESIGN.md 1.4 says "same corpus", and the way that is guaranteed is byte equality of
    two independent snapshots - not one directory reaching into the other, which would
    make the two agents share a failure and compromise C3.
    """

    def test_the_corpus_is_byte_identical(self):
        strong = sorted(p.name for p in (STRONG_DIR / "corpus").iterdir())
        naive = sorted(p.name for p in (NAIVE_DIR / "corpus").iterdir())
        assert strong == naive and strong, strong
        for name in strong:
            assert (STRONG_DIR / "corpus" / name).read_bytes() == (
                NAIVE_DIR / "corpus" / name
            ).read_bytes(), name

    def test_the_corpus_is_a_snapshot_not_a_symlink(self):
        for path in (STRONG_DIR / "corpus").iterdir():
            assert not path.is_symlink(), path

    def test_requirements_are_byte_identical(self):
        """The pins were harvested by `pip freeze` from a real build rather than written
        by hand, so copying the file preserves that provenance. The reranker needed no new
        package - CrossEncoder ships inside the pinned sentence-transformers.
        """
        assert (STRONG_DIR / "requirements.txt").read_bytes() == (
            NAIVE_DIR / "requirements.txt"
        ).read_bytes()

    def test_the_embedder_is_held_equal_and_only_retrieval_moved(self):
        naive_retrieval = NAIVE["retrieval"]
        assert retrieval.MODEL_NAME == naive_retrieval.MODEL_NAME
        assert retrieval.MODEL_REVISION == naive_retrieval.MODEL_REVISION
        assert retrieval.TOP_K == 8 and naive_retrieval.TOP_K == 3
        assert not hasattr(naive_retrieval, "RERANKER_NAME")


class TestTheCollisionsAreDeclaredNotDiscovered:
    """aut-strong's pin breaks DESIGN.md 1.5's judge-versus-AUT family rule, and it is the
    extractor's own model. Both are accepted, and acceptance means asserted.

    WHY THESE ARE EQUALITIES AND NOT INEQUALITIES
    Every other family test in this repository asserts separation. These assert the
    overlap, because the overlap is what makes aut-strong's number a LOWER BOUND, and that
    reading is pre-committed in writing before STEP 6's run. If a later re-pin removed the
    overlap, these tests failing is the correct outcome: the recorded limitation and the
    pre-commitment would both be wrong - too pessimistic rather than too generous, but
    wrong - and `docs/limitations.md` has to be rewritten before the numbers are read.

    WHY THE PIN IS THIS ANYWAY
    Of the chat models this account can see, gpt-oss is the only family that is both
    frontier-scale and suitable; qwen is aut-naive's own family and reusing it would make
    the two agents one species, which is a worse confound than the judge overlap. The full
    inventory is in docs/limitations.md's first entry.
    """

    #: agent-strong overlaps the extractor and the judge; those two already overlapped.
    ACCEPTED = {
        frozenset(("agent_strong", "extractor")),
        frozenset(("agent_strong", "judge")),
        frozenset(("extractor", "judge")),
    }

    HARNESS_ROLES = {
        "extractor": "CLAUSEGUARD_EXTRACTOR_MODEL",
        "judge": "CLAUSEGUARD_JUDGE_MODEL",
        "adversary": "CLAUSEGUARD_ADVERSARY_MODEL",
    }

    def _families(self) -> dict[str, str]:
        models = env_example_models()
        for role, key in self.HARNESS_ROLES.items():
            assert key in models, f"{key} ({role}) is missing from .env.example"
        families = {
            "agent_naive": family_of(naive_backends.OLLAMA_MODEL),
            "agent_strong": family_of(backends.DEFAULT_GROQ_MODEL),
        }
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

    def test_the_agent_shares_a_family_with_the_judge(self):
        judge = env_example_models()["CLAUSEGUARD_JUDGE_MODEL"]
        assert family_of(backends.DEFAULT_GROQ_MODEL) == family_of(judge), (
            "the aut-strong/judge overlap has gone. That is an improvement, and it "
            "invalidates the lower-bound reading recorded in docs/limitations.md - "
            "rewrite the entry before reading STEP 6's number, do not delete this test."
        )

    def test_the_agent_is_the_extractor_model_itself(self):
        extractor = env_example_models()["CLAUSEGUARD_EXTRACTOR_MODEL"]
        assert strip_provider_prefix(extractor) == backends.DEFAULT_GROQ_MODEL, (
            "aut-strong and the extractor are no longer the same model. Same note as "
            "above: the recorded limitation is now wrong and has to be rewritten."
        )

    def test_the_agent_is_not_the_judge_model_itself(self):
        """The one thing that would be worse. A judge grading its own generations at the
        same parameter count is self-assessment, and no lower-bound argument survives it.
        """
        judge = env_example_models()["CLAUSEGUARD_JUDGE_MODEL"]
        assert strip_provider_prefix(judge) != backends.DEFAULT_GROQ_MODEL

    def test_the_two_agents_are_different_families(self):
        """The confound that would break the comparison itself, rather than the judging."""
        assert family_of(backends.DEFAULT_GROQ_MODEL) != family_of(
            naive_backends.OLLAMA_MODEL
        )

    def test_the_full_topology_is_the_documented_one(self):
        families = self._families()
        collisions = self._colliding_pairs(families)
        assert collisions == self.ACCEPTED, (
            f"the family topology moved. Roles now map to {families!r}, giving "
            f"overlapping pairs {sorted('+'.join(sorted(p)) for p in collisions)} where "
            f"only {sorted('+'.join(sorted(p)) for p in self.ACCEPTED)} has been argued "
            "for. Read this class's docstring: the fix is to decide whether the new "
            "overlap is acceptable and record it in docs/limitations.md, not to edit "
            "ACCEPTED until the test passes."
        )

    def test_the_check_is_sensitive_in_both_directions(self):
        """Control. A hand-written expectation is exactly the shape of test that quietly
        stops discriminating."""
        everything = dict.fromkeys(self._families(), "gpt")
        assert len(self._colliding_pairs(everything)) == 10
        assert self._colliding_pairs(everything) != self.ACCEPTED

        all_distinct = {
            "agent_naive": "qwen",
            "agent_strong": "claude",
            "extractor": "gpt",
            "judge": "llama",
            "adversary": "mistral",
        }
        assert self._colliding_pairs(all_distinct) == set()

    def test_the_lower_bound_reading_is_written_down_before_the_run(self):
        """The pre-commitment. Written before STEP 6, because a bias direction argued
        after the number is known is not a limitation, it is an explanation.
        """
        assert LIMITATIONS.is_file()
        text = LIMITATIONS.read_text(encoding="utf-8")
        section = [
            block
            for block in text.split("\n## ")
            if block.lower().startswith("aut-strong")
            and "extractor" in block.lower()
        ]
        assert section, (
            "docs/limitations.md has no aut-strong entry covering the extractor "
            "identity. backends.py's docstring promises one and this is the assertion "
            "that keeps that promise honest."
        )
        entry = section[0]
        assert "lower bound" in entry.lower()
        assert backends.DEFAULT_GROQ_MODEL in entry
        for phrase in ("harder to detect", "0%"):
            assert phrase in entry.lower(), phrase
