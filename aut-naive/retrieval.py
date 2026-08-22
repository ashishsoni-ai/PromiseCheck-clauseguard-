"""bge-small-en-v1.5 + FAISS, top-k=3 (DESIGN.md 1.4). STEP 4. Zero imports from harness/.

The retrieval half of aut-naive. Two things worth knowing before reading the code.

WHY THE bge QUERY PREFIX IS USED
The BAAI/bge-small-en-v1.5 model card documents a short instruction prefix on the
*query* side (not the passage side) for short-query-to-passage retrieval. Including it is
the neutral choice: following the model card is what a competent-but-unspecialised
deployment does, and deviating in either direction - omitting it to weaken retrieval, or
hand-tuning something better - would be a thumb on the scale (DESIGN.md 7.3).

WHY THERE IS A SEAM AT `Embedder`
The HTTP contract, the session handling and the prompt all need testing, and none of them
should require a 130MB model download or a FAISS build to test. So embedding sits behind
a tiny protocol and the real implementation is one class. The seam is for injecting a
stand-in *collaborator* in tests; the retrieval path itself is the real thing, and it is
proven by `docker build` + a live query, not by a stand-in.

Imports of numpy, faiss and sentence_transformers are deferred into the functions that
need them so this module can be imported bare.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from chunker import CHUNK_CHARS, OVERLAP_CHARS, Chunk, load_corpus

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

#: DESIGN.md 1.4, verbatim.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 3

#: From the bge-small-en-v1.5 model card. Query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into unit-norm row vectors."""

    def encode_documents(self, texts: Sequence[str]) -> "np.ndarray": ...

    def encode_query(self, text: str) -> "np.ndarray": ...

    @property
    def name(self) -> str: ...


class BGEEmbedder:
    """The real one. `sentence_transformers` is imported on construction."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def name(self) -> str:
        return self._model_name

    def encode_documents(self, texts: Sequence[str]) -> "np.ndarray":
        return self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )

    def encode_query(self, text: str) -> "np.ndarray":
        return self._model.encode(
            [QUERY_PREFIX + text], normalize_embeddings=True, convert_to_numpy=True
        )


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    """FAISS inner-product search over unit-norm vectors, i.e. cosine similarity."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        *,
        top_k: int = TOP_K,
    ) -> None:
        if not chunks:
            raise ValueError("cannot build a retriever over zero chunks")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        import faiss
        import numpy as np

        self.chunks = tuple(chunks)
        self.embedder = embedder
        self.top_k = top_k

        vectors = np.asarray(
            embedder.encode_documents([c.text for c in self.chunks]), dtype="float32"
        )
        if vectors.ndim != 2 or len(vectors) != len(self.chunks):
            raise ValueError(
                f"embedder returned {vectors.shape} for {len(self.chunks)} chunks"
            )

        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self.dim = int(vectors.shape[1])

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        """Top-k chunks for `query`, best first."""
        import numpy as np

        wanted = min(k or self.top_k, len(self.chunks))
        vector = np.asarray(self.embedder.encode_query(query), dtype="float32")
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        scores, indices = self._index.search(vector, wanted)
        return [
            Hit(chunk=self.chunks[int(i)], score=float(s))
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    @property
    def fingerprint(self) -> str:
        """Identifies exactly what is in this index.

        Reported at `/health` alongside the freeze SHAs. The commit hash proves which
        *code* was frozen; this proves which *corpus* was baked in beside it, which is
        the other half of a claim that the agent has not moved.
        """
        digest = hashlib.sha256()
        digest.update(f"{self.embedder.name}|{self.top_k}|{self.dim}\n".encode())
        for chunk in self.chunks:
            digest.update(f"{chunk.chunk_id}:{len(chunk.text)}\n".encode())
        return "sha256:" + digest.hexdigest()


def build_default(
    corpus_dir: Path | str = "corpus",
    *,
    top_k: int = TOP_K,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> Retriever:
    """The frozen configuration: baked-in corpus, bge-small, k=3."""
    chunks = load_corpus(
        corpus_dir, chunk_chars=chunk_chars, overlap_chars=overlap_chars
    )
    return Retriever(chunks, BGEEmbedder(), top_k=top_k)
