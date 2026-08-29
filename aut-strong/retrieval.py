"""Retrieval for aut-strong: widened candidate set, then a cross-encoder rerank.
Implemented in STEP 2. Zero imports from harness/.

Mirrors aut-naive/retrieval.py's interface - `Embedder`, `Hit`, `Retriever`,
`search`, `fingerprint`, `build_default` - because app.py consumes it the same way
and /health reports the same fields. numpy, faiss and sentence_transformers
imports stay deferred into functions so this module imports bare, which is what
lets the offline test suite read it without the ML stack installed.

WHAT MAKES THIS THE STRONG AGENT'S RETRIEVAL (DESIGN.md 1.4: k=8, reranking)
aut-naive takes the top 3 by cosine similarity and stops. Two things change here:

  1. The candidate set widens past what will be shown to the model, so a clause
     that cosine similarity ranks 6th is still in contention. Embedding similarity
     is a poor proxy for "this clause governs this request", because an exclusion is
     usually lexically dissimilar to the request it excludes.
  2. A cross-encoder scores each candidate jointly with the query rather than
     comparing two independently-computed vectors, so it can IN PRINCIPLE rank a
     clause highly for contradicting the request. That is a property a bi-encoder
     cannot have. Whether it does so on this corpus is a separate question, answered
     below, and the answer is no.

WHAT AN EARLIER VERSION OF THIS DOCSTRING CLAIMED, AND WHY IT IS GONE
It said aut-naive's flagship failure was a category_smuggling probe whose
hygiene-seal carve-out "was never retrieved", and then that STEP 2 had sharpened the
diagnosis to "the chunk was whole and simply ranked outside the top 3". Those are
opposite mechanisms and neither was measured:

  - The flagship over-promise in docs/results.md is
    `P-acme-018-multi_turn_drift-003`, a dispatch/cancellation probe, not a
    category_smuggling one.
  - No run can adjudicate between them. aut-naive returns `retrieved_chunk_ids` on
    every reply (aut-naive/app.py:108,162) and nothing in harness/ ever reads it, so
    audit_rows carries no retrieval column. For run 01a032fd a ranking failure and a
    "had the text and ignored it" reasoning failure are indistinguishable. Recorded
    in docs/limitations.md under "No run can tell a retrieval failure from a
    reasoning failure".

What IS measured is the negative: at aut-naive's 800/150 window the corpus is 7
chunks and the entire hygiene chain sits whole inside one of them, so all ten
protected spans survive. Segmentation is exonerated; the ranking half is unknown.
This file therefore justifies its design from the SHAPE of the failure - exclusions
read unlike the requests they exclude - and not from a trace of one.

WHERE THE WORK ACTUALLY HAPPENS ON THIS CORPUS, MEASURED IN STEP 2
22 chunks, CANDIDATE_K=16, TOP_K=8. The dense stage discards 6 of 22 windows, so it
is close to a pass-through, and on a 4.6KB single document it cannot be otherwise.

The pre-registered check (scripts/check_aut_retrieval.py, evidence at
docs/evidence/aut-strong-retrieval-step2.jsonl) prints the counterfactuals on every
run, and they are blunter than this file first assumed. Over ten governing
span-instances on three probes: aut-naive's dense top 3 reaches 6, dense top 8 with
the cross-encoder OFF reaches 8, and the reranked top 8 as shipped also reaches 8.
The last two are the same set - +0 promoted, -0 demoted. **So the cross-encoder's
contribution to whether the governing clause reaches the model is measured at zero,
and every measured gain over aut-naive is the DEPTH increase.** It is not idle: it
re-selects 2, 1 and 3 of the 8 slots on the three probes and reorders all three. But
membership is what was measured, ordering is not, and nothing here measures whether
position changes the model's answer. STEP 7 must not write "reranking surfaced the
governing clause".

Volume, corrected: eight 500-char windows are NOT 86% of a 4,625-char policy. That
figure assumed the returned chunks were disjoint. With OVERLAP_CHARS=300 neighbours
share three fifths of their text and the returned set is usually a run of adjacent
windows, so measured UNIQUE coverage of the reranked top 8 is 55% / 51% / 53% on the
three probes, against 30% / 23% / 19% for aut-naive's top 3. Presence is therefore a
real constraint at this depth rather than a formality - which is also why one
span-instance is a recorded FAIL: both chunks holding it fell in the bottom 6 of 22
by cosine, so the cross-encoder was never offered them and raising TOP_K would not
have reached it either. See chunker.py for the segmentation measurement.

THE ONE THING THIS FILE MUST NOT BECOME
A rules engine. No clause ids, no keyword lists, no "if the query mentions
hygiene, boost chunk 4". The question under test is whether ordinary good
retrieval engineering reduces over-promises; hardcoding the answers measures
nothing except whether Clauseguard can be gamed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from chunker import CHUNK_CHARS, OVERLAP_CHARS, Chunk, load_corpus

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

#: Unchanged from aut-naive, and deliberately so: DESIGN.md 1.4 changes retrieval
#: depth and adds a reranker, it does not change the embedder. Holding this equal
#: keeps the comparison between the two agents interpretable.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

#: bge's asymmetric query prefix. Documents are embedded bare; only queries carry
#: this. Dropping it costs real recall and is a silent error, not a loud one.
#:
#: It is NOT applied on the cross-encoder side. bge-reranker-base takes a raw
#: (query, passage) pair and was not trained with an instruction prefix, so adding
#: one there would be a made-up input format. The asymmetry between the two stages
#: is intentional, not an oversight.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

#: DESIGN.md 1.4, verbatim. What is returned to the caller after reranking.
TOP_K = 8

#: How many chunks the bi-encoder hands to the reranker. Must be strictly greater
#: than TOP_K or the rerank can only reorder what it was already going to return,
#: which changes the prompt's ordering and nothing about its content.
#:
#: 16 is twice TOP_K, the conventional rerank ratio, and it fits the 22 chunks
#: chunker.py's sweep settled on. Raising it to 22 would make the dense stage a
#: literal no-op and the agent would be "cross-encode the entire policy", which is
#: not a retrieval architecture anyone deploys; lowering it toward 8 reinstates
#: aut-naive's failure, where a governing clause ranked 6th by cosine never reaches
#: the stage that could promote it.
CANDIDATE_K = 16

#: The cross-encoder. Chosen to pair with the bge embeddings above rather than for
#: novelty, and it adds no dependency - CrossEncoder ships inside the already
#: pinned sentence-transformers.
RERANKER_NAME = "BAAI/bge-reranker-base"

#: MUST EQUAL THE `RERANKER_REVISION` ARG DEFAULT IN THIS DIRECTORY'S Dockerfile,
#: the same discipline test_aut_freeze.py already enforces between MODEL_REVISION
#: and the embedder's `revision=`. Without it the model id is a moving reference
#: and a rebuild at a fixed git SHA can fetch different weights, which is the one
#: form of drift commitment C3 cannot see.
#:
#: Resolved 2026-08-28 against the live hub, since the sandbox that wrote this file
#: has no network and a guessed hash would have been indistinguishable from a real
#: one until the day the weights moved:
#:   HfApi().model_info("BAAI/bge-reranker-base").sha
RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into unit-norm row vectors."""

    def encode_documents(self, texts: Sequence[str]) -> "np.ndarray": ...

    def encode_query(self, text: str) -> "np.ndarray": ...

    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str: ...


@runtime_checkable
class Reranker(Protocol):
    """Anything that scores (query, passage) pairs jointly.

    Same seam rationale as `Embedder`: the HTTP contract and the prompt need
    testing without a model download. The seam is for injecting a stand-in
    collaborator; the real reranking path is proven by `docker build` plus a live
    query, not by a stand-in.
    """

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]: ...

    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str: ...


class BGEEmbedder:
    """The real one. `sentence_transformers` is imported on construction."""

    def __init__(
        self, model_name: str = MODEL_NAME, revision: str = MODEL_REVISION
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._revision = revision
        self._model = SentenceTransformer(model_name, revision=revision)

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str:
        return self._revision

    def encode_documents(self, texts: Sequence[str]) -> "np.ndarray":
        return self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )

    def encode_query(self, text: str) -> "np.ndarray":
        return self._model.encode(
            [QUERY_PREFIX + text], normalize_embeddings=True, convert_to_numpy=True
        )


class BGECrossEncoder:
    """The real reranker. `sentence_transformers` is imported on construction.

    Scores are raw logits, not probabilities: bge-reranker-base emits a single
    unbounded relevance logit per pair. Nothing here needs them calibrated, since
    only the ordering is consumed, and passing them through a sigmoid would make
    them look like probabilities they are not.
    """

    def __init__(
        self, model_name: str = RERANKER_NAME, revision: str = RERANKER_REVISION
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._revision = revision
        self._model = CrossEncoder(model_name, revision=revision)

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str:
        return self._revision

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        if not passages:
            return []
        raw = self._model.predict([(query, p) for p in passages])
        return [float(x) for x in raw]


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float
    #: The bi-encoder's cosine score and its 1-based rank in the candidate set,
    #: kept so STEP 2's standalone evidence can show a clause moving from dense
    #: rank 6 into the reranked top 8 - the exact claim DESIGN.md 1.4 rests on.
    #: `None` only when a `Hit` is constructed without a dense stage, which is a
    #: test stand-in, never the frozen path. These do NOT reach /chat; see app.py.
    dense_score: float | None = None
    dense_rank: int | None = None


class Retriever:
    """FAISS candidate search over unit-norm vectors, then a cross-encoder rerank."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        reranker: Reranker,
        *,
        top_k: int = TOP_K,
        candidate_k: int = CANDIDATE_K,
    ) -> None:
        if not chunks:
            raise ValueError("cannot build a retriever over zero chunks")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if candidate_k <= top_k:
            raise ValueError(
                f"candidate_k ({candidate_k}) must exceed top_k ({top_k}), or the "
                f"rerank can only reorder chunks that were already being returned"
            )
        if reranker is None:
            # Fail closed rather than degrading to dense-only. A silent fallback
            # would make aut-strong a copy of aut-naive at k=8 while still
            # reporting itself as the reranked agent, and DESIGN.md 1.4's headline
            # difference would be attributed to a component that never ran.
            raise ValueError("aut-strong requires a reranker; refusing dense-only")

        import faiss
        import numpy as np

        self.chunks = tuple(chunks)
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_k = candidate_k

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

    def dense_search(self, query: str, k: int) -> list[Hit]:
        """The bi-encoder stage on its own, best first. Exposed for STEP 2 evidence.

        Reranking is only interesting against a baseline, and the baseline has to
        come from the same index and the same embedder or the comparison measures
        two things at once.
        """
        import numpy as np

        wanted = min(k, len(self.chunks))
        vector = np.asarray(self.embedder.encode_query(query), dtype="float32")
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        scores, indices = self._index.search(vector, wanted)
        hits: list[Hit] = []
        for rank, (s, i) in enumerate(zip(scores[0], indices[0]), start=1):
            if i < 0:
                continue
            hits.append(
                Hit(
                    chunk=self.chunks[int(i)],
                    score=float(s),
                    dense_score=float(s),
                    dense_rank=rank,
                )
            )
        return hits

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        """Top-k chunks for `query` after reranking, best first.

        `Hit.score` is the cross-encoder logit, not the cosine score, because it is
        the value the returned ordering is actually sorted by; the cosine score
        survives on `dense_score`. Reporting the cosine score as `score` while
        ordering by something else would make /health's numbers unfalsifiable.
        """
        wanted = min(k or self.top_k, len(self.chunks))
        candidates = self.dense_search(
            query, min(max(self.candidate_k, wanted), len(self.chunks))
        )
        if not candidates:
            return []

        scores = self.reranker.score(query, [h.chunk.text for h in candidates])
        if len(scores) != len(candidates):
            raise ValueError(
                f"reranker returned {len(scores)} scores for "
                f"{len(candidates)} candidates"
            )

        # `sorted` is stable, so equal cross-encoder scores keep dense order rather
        # than resolving arbitrarily. The agent is frozen; identical inputs have to
        # produce an identical prompt across rebuilds, and a tie broken by list
        # order would be a reproducibility hole that only shows up on some corpora.
        reranked = sorted(
            (
                Hit(
                    chunk=h.chunk,
                    score=float(s),
                    dense_score=h.dense_score,
                    dense_rank=h.dense_rank,
                )
                for h, s in zip(candidates, scores)
            ),
            key=lambda h: -h.score,
        )
        return reranked[:wanted]

    @property
    def fingerprint(self) -> str:
        """Identifies exactly what is in this index.

        Reported at `/health` alongside the freeze SHAs. The commit hash proves which
        *code* was frozen; this proves which *corpus* was baked in beside it, which is
        the other half of a claim that the agent has not moved.

        Both models' revisions are folded in, not just their names, and so is
        `candidate_k`. Without the reranker's revision, swapping the cross-encoder
        weights - the one drift a git SHA cannot see - would leave this value
        unchanged, and a fingerprint that survives the change it exists to detect is
        worse than none. Without `candidate_k`, the same holds for the one number
        that decides whether the rerank has anything to promote.
        """
        digest = hashlib.sha256()
        digest.update(
            f"{self.embedder.name}@{self.embedder.revision}"
            f"|{self.reranker.name}@{self.reranker.revision}"
            f"|{self.candidate_k}|{self.top_k}|{self.dim}\n".encode()
        )
        for chunk in self.chunks:
            digest.update(f"{chunk.chunk_id}:{len(chunk.text)}\n".encode())
        return "sha256:" + digest.hexdigest()


def build_default(
    corpus_dir: Path | str = "corpus",
    *,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> Retriever:
    """The frozen configuration: baked-in corpus, bge-small, rerank 16 -> 8."""
    chunks = load_corpus(
        corpus_dir, chunk_chars=chunk_chars, overlap_chars=overlap_chars
    )
    return Retriever(
        chunks,
        BGEEmbedder(),
        BGECrossEncoder(),
        top_k=top_k,
        candidate_k=candidate_k,
    )


if __name__ == "__main__":  # pragma: no cover - evidence run, not a test
    # STEP 2's standalone retrieval proof, run INSIDE the container because that is
    # the only place both models exist at their pinned revisions:
    #
    #   docker run --rm -i --entrypoint python <image> retrieval.py < queries.txt
    #
    # Queries arrive on stdin, one per line, and nothing here knows what a probe or
    # a clause is - the file would be a rules engine if it did. It emits JSON lines
    # so the judgement can be made mechanically by scripts/check_aut_retrieval.py,
    # which owns the answer key and lives outside this frozen tree.
    #
    # Both stages are reported, with each chunk's character span. The span is what
    # makes the output checkable without re-running the chunker: a governing clause
    # is "retrieved" exactly when some returned [start, end) contains its verbatim
    # source_span, which is offset arithmetic on the policy text and needs no model.
    import hashlib
    import json
    import sys
    from pathlib import Path

    corpus = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    retriever = build_default(corpus)

    docs = sorted(corpus.glob("*.md"))
    header = {
        "kind": "header",
        "chunk_chars": CHUNK_CHARS,
        "overlap_chars": OVERLAP_CHARS,
        "n_chunks": len(retriever.chunks),
        "top_k": retriever.top_k,
        "candidate_k": retriever.candidate_k,
        "embedder": f"{retriever.embedder.name}@{retriever.embedder.revision}",
        "reranker": f"{retriever.reranker.name}@{retriever.reranker.revision}",
        "fingerprint": retriever.fingerprint,
        # EVERY chunk, not just the ones that placed. Without the full table the
        # checker cannot tell "no window contains this clause whole" (a chunking
        # result) from "the window holding it was never a candidate" (a ranking
        # result), and those two have opposite fixes.
        "chunks": [
            {"ordinal": c.ordinal, "start": c.start, "end": c.end}
            for c in retriever.chunks
        ],
        # The checker resolves source_span offsets against its own copy of the
        # policy, so it has to prove that copy is the bytes retrieval actually saw.
        # Without this the offsets could silently refer to a different document.
        "corpus": {
            p.name: "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest() for p in docs
        },
    }
    print(json.dumps(header), flush=True)

    def as_json(hit: Hit) -> dict[str, object]:
        return {
            "chunk_id": hit.chunk.chunk_id,
            "ordinal": hit.chunk.ordinal,
            "start": hit.chunk.start,
            "end": hit.chunk.end,
            "score": hit.score,
            "dense_score": hit.dense_score,
            "dense_rank": hit.dense_rank,
        }

    for line in sys.stdin:
        query = line.strip()
        if not query:
            continue
        candidates = retriever.dense_search(query, retriever.candidate_k)
        print(
            json.dumps(
                {
                    "kind": "query",
                    "query": query,
                    # The dense stage alone, so the rerank has a baseline from the
                    # same index. Its first 3 entries are also what aut-naive's k=3
                    # would have returned had it chunked this way, which is not the
                    # same as what aut-naive actually returned - that comparison
                    # belongs to the 30-probe run, not here.
                    "dense": [as_json(h) for h in candidates],
                    "reranked": [as_json(h) for h in retriever.search(query)],
                }
            ),
            flush=True,
        )
