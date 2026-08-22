"""STEP 4 tripwires: the inputs a git SHA does NOT cover.

Commitment C3 (DESIGN.md 0) freezes an agent under test "by commit SHA before any probe
exists". A commit SHA covers tracked bytes and nothing else, so the freeze leaks through
every input resolved at build time:

  * floating pip dependencies  - new `transformers` changes tokenisation, new
                                 `sentence-transformers` changes pooling, so top-k moves
  * a floating HF model revision - `BAAI/bge-small-en-v1.5` is a moving reference; a push
                                 by BAAI changes the weights outright
  * a floating base image tag   - `python:3.12-slim` is rebuilt on every patch release, so
                                 the interpreter and system libs move
  * the policy corpus snapshot  - covered separately, in test_aut_chunker.py

Each of those can change the agent's answers while `git status` stays clean and the tag
stays valid. The failure mode is not a broken build; it is a *quiet* one, where the audit
trail attributes drift to the agent and the C3 claim in the writeup becomes false without
anyone touching the repo.

These tests read the Dockerfile and requirements.txt as text on purpose. They assert the
freeze is *written down*, which is checkable in CI on any machine with no Docker daemon,
no network and no model download. Whether the pins are the RIGHT versions is a different
question, answered by `docker build` + `pip freeze` (see the requirements.txt header).

The load-bearing test here is
`TestTheModelRevisionIsPinnedAndAgrees::test_the_baked_revision_equals_the_runtime_revision`:
the Dockerfile downloads the weights and retrieval.py asks for them, and if those two
drift apart the image either fails closed under HF_HUB_OFFLINE=1 or - worse - silently
serves a revision nobody recorded.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

AUT_DIR = Path(__file__).resolve().parents[2] / "aut-naive"
if str(AUT_DIR) not in sys.path:
    sys.path.insert(0, str(AUT_DIR))

from retrieval import MODEL_NAME, MODEL_REVISION  # noqa: E402

DOCKERFILE = AUT_DIR / "Dockerfile"
REQUIREMENTS = AUT_DIR / "requirements.txt"

FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def requirement_lines() -> list[str]:
    """Non-comment, non-blank lines only."""
    return [
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def package_name(line: str) -> str:
    """`uvicorn[standard]==0.52.4` -> `uvicorn`. Normalised per PEP 503."""
    name = re.split(r"[=<>!~;]", line, maxsplit=1)[0].strip()
    name = re.sub(r"\[.*\]$", "", name)
    return name.lower().replace("_", "-").replace(".", "-")


class TestEveryPipDependencyIsPinned:
    def test_there_is_something_to_check(self):
        assert len(requirement_lines()) > 20, "a full pin should be dozens of lines"

    def test_every_requirement_is_an_exact_pin(self):
        """`>=` is the failure this whole file exists to prevent."""
        loose = [line for line in requirement_lines() if "==" not in line]
        assert loose == [], f"unpinned requirements: {loose}"

    def test_no_requirement_carries_a_range_alongside_the_pin(self):
        bad = [
            line
            for line in requirement_lines()
            if any(op in line for op in (">=", "<=", "~=", "!=", ">", "<"))
        ]
        assert bad == [], f"ranges defeat the pin: {bad}"

    def test_no_package_is_listed_twice(self):
        """A duplicate means pip resolves whichever came last - order-dependent freeze."""
        names = [package_name(line) for line in requirement_lines()]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        assert duplicates == [], f"listed more than once: {duplicates}"

    def test_torch_is_absent_because_pypi_has_no_cpu_build(self):
        """The resolved build is 2.13.0+cpu, and that local-version segment exists only
        on the PyTorch CPU index. Listing it here would 404 against PyPI, so it is pinned
        on the Dockerfile's own --index-url line instead."""
        assert "torch" not in [package_name(line) for line in requirement_lines()]

    def test_the_aut_only_appendix_dependencies_are_present(self):
        """DESIGN.md Appendix marks these "(AUT only)"; they are the retrieval half."""
        names = {package_name(line) for line in requirement_lines()}
        assert {"sentence-transformers", "faiss-cpu"} <= names


class TestTheBaseImageIsDigestPinned:
    def test_the_from_line_carries_a_digest(self):
        match = re.search(r"^FROM\s+(\S+)", dockerfile_text(), re.MULTILINE)
        assert match, "no FROM line found"
        assert "@sha256:" in match.group(1), (
            "python:3.12-slim is a moving tag; pin the digest or the interpreter can "
            "change under a fixed git SHA"
        )

    def test_the_digest_is_a_full_sha256(self):
        match = re.search(r"^FROM\s+\S+@sha256:([0-9a-f]+)", dockerfile_text(), re.M)
        assert match
        assert FULL_SHA256.match(match.group(1)), "truncated digests do not pin anything"

    def test_the_image_is_still_named_so_a_human_can_read_it(self):
        """Digest-only would be opaque; `name:tag@digest` keeps both."""
        match = re.search(r"^FROM\s+(\S+)", dockerfile_text(), re.MULTILINE)
        assert match
        assert match.group(1).startswith("python:3.12-slim@sha256:")


class TestTorchIsPinnedOnItsOwnIndexLine:
    def _torch_line(self) -> str:
        lines = [
            line
            for line in dockerfile_text().splitlines()
            if line.startswith("RUN pip install") and "torch" in line
        ]
        assert len(lines) == 1, f"expected exactly one torch install line, got {lines}"
        return lines[0]

    def test_the_cpu_index_is_used(self):
        """The default index serves CUDA wheels on Linux - gigabytes this agent
        never uses, and a different build of the same version number."""
        assert "download.pytorch.org/whl/cpu" in self._torch_line()

    def test_the_version_is_pinned(self):
        assert re.search(r"torch==\d+\.\d+(\.\d+)?", self._torch_line()), (
            "torch must be pinned even though it lives outside requirements.txt"
        )


class TestTheModelRevisionIsPinnedAndAgrees:
    def _baked_revisions(self) -> list[str]:
        return re.findall(r"revision='([0-9a-f]{40})'", dockerfile_text())

    def test_the_runtime_constant_is_a_full_commit_hash(self):
        assert FULL_GIT_SHA.match(MODEL_REVISION), MODEL_REVISION

    def test_a_branch_name_would_not_be_a_pin(self):
        """`main` resolves to whatever HEAD is today, which is the bug, not the fix."""
        assert MODEL_REVISION not in {"main", "master", "refs/heads/main", ""}

    def test_the_dockerfile_bake_passes_a_revision(self):
        assert self._baked_revisions(), (
            "the model-bake layer must pass revision=; without it a rebuild can fetch "
            "different weights at the same git SHA"
        )

    def test_the_baked_revision_equals_the_runtime_revision(self):
        """The build downloads the weights; retrieval.py asks for them. If these differ,
        the image fails closed under HF_HUB_OFFLINE=1 - or silently serves a revision
        that was never recorded anywhere."""
        assert self._baked_revisions() == [MODEL_REVISION], (
            f"Dockerfile bakes {self._baked_revisions()} but retrieval.MODEL_REVISION "
            f"is {MODEL_REVISION}"
        )

    def test_the_model_id_agrees_too(self):
        assert MODEL_NAME == "BAAI/bge-small-en-v1.5"
        assert f"'{MODEL_NAME}'" in dockerfile_text()


class TestTheOfflineGuaranteeIsNotAccidental:
    def _cache_vars(self) -> dict[str, str]:
        found = {}
        for name in ("HF_HOME", "HF_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
            match = re.search(rf"\b{name}=(\S+)", dockerfile_text())
            if match:
                found[name] = match.group(1).rstrip("\\").strip()
        return found

    def test_the_runtime_cannot_reach_the_hub(self):
        """A frozen agent that can still download is not frozen; it is merely warmed up."""
        assert re.search(r"\bHF_HUB_OFFLINE=1", dockerfile_text())

    def test_all_three_cache_variables_name_the_same_root(self):
        """sentence-transformers treats SENTENCE_TRANSFORMERS_HOME as the cache root,
        while huggingface_hub appends /hub to HF_HOME. Setting HF_HUB_CACHE explicitly is
        what stops those conventions disagreeing - which they did, silently, until the
        weights turned up in /opt/models/models--BAAI--... and not /opt/models/hub."""
        found = self._cache_vars()
        assert set(found) == {"HF_HOME", "HF_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"}, (
            f"missing cache variable(s): {found}"
        )
        assert len(set(found.values())) == 1, f"cache roots disagree: {found}"
