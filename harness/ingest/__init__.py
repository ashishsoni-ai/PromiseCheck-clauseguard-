"""Ingest: policy source -> ordered, content-addressed `Clause[]`.

DESIGN.md 1.1 and 2 step ①. The pipeline is four stages, deliberately separable:

    load (loaders)  ->  segment (segmenter)  ->  hash (hashing)  ->  PolicyDocument

No LLM is involved at any point. This is the property that makes step ① free and
makes the gate's change detection auditable - a human can re-run it and get the
same clause IDs, which is not true of anything downstream of the extractor.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from harness.ingest.hashing import (
    CONTENT_HASH_LENGTH,
    collapse_whitespace,
    content_hash,
    make_clause_id,
    normalize,
    policy_version,
    slugify,
)
from harness.ingest.loaders import (
    CACHE_DIR,
    LoadedSource,
    load,
    load_markdown,
    load_pdf,
    load_url,
    raw_sha256,
)
from harness.ingest.manifest import (
    MANIFEST_PATH,
    ClauseChange,
    ClauseMove,
    DocumentDiff,
    diff_against_manifest,
    diff_document,
    fingerprint_document,
    load_manifest,
    update_manifest,
    write_manifest,
)
from harness.ingest.segmenter import (
    MAX_CLAUSE_TOKENS,
    MIN_CLAUSE_TOKENS,
    RawSegment,
    estimate_tokens,
    length_audit,
    segment_markdown,
)
from harness.schemas import Clause, CorpusRole, PolicyDocument

__all__ = [
    "CACHE_DIR",
    "CONTENT_HASH_LENGTH",
    "MANIFEST_PATH",
    "MAX_CLAUSE_TOKENS",
    "MIN_CLAUSE_TOKENS",
    "Clause",
    "ClauseChange",
    "ClauseMove",
    "DocumentDiff",
    "LoadedSource",
    "PolicyDocument",
    "RawSegment",
    "collapse_whitespace",
    "content_hash",
    "diff_against_manifest",
    "diff_document",
    "estimate_tokens",
    "fingerprint_document",
    "ingest",
    "ingest_text",
    "length_audit",
    "load",
    "load_manifest",
    "load_markdown",
    "load_pdf",
    "load_url",
    "make_clause_id",
    "normalize",
    "policy_version",
    "raw_sha256",
    "segment_markdown",
    "slugify",
    "update_manifest",
    "write_manifest",
]


def clauses_from_segments(
    doc_slug: str, segments: list[RawSegment]
) -> list[Clause]:
    """Assign ordinals, hashes and composite IDs to ordered segments.

    Ordinals are 1-based positions in document order and are assigned here rather
    than by the segmenter, because the segmenter's job ends at "what is a clause"
    and ordinal assignment belongs with ID construction. Keeping them together
    means there is exactly one place where a clause ID can be built wrongly.
    """
    clauses: list[Clause] = []
    for ordinal, segment in enumerate(segments, start=1):
        h = content_hash(segment.text)
        clauses.append(
            Clause(
                clause_id=make_clause_id(doc_slug, ordinal, h),
                doc_slug=doc_slug,
                ordinal=ordinal,
                text=segment.text,
                content_hash=h,
                heading_path=list(segment.heading_path),
                token_estimate=segment.token_estimate,
            )
        )
    return clauses


def ingest_text(
    text: str,
    *,
    doc_slug: str,
    source: str,
    corpus_role: CorpusRole,
    fetched_at: datetime | None = None,
    is_holdout: bool = False,
) -> PolicyDocument:
    """Segment and hash already-loaded text. The I/O-free core of `ingest`.

    Split out from `ingest` so the pipeline is testable without touching the
    filesystem or the network, and so a caller who already has the text (a cached
    fetch, a fixture, a paste in the review UI) does not have to write it to a
    temp file to get clause IDs.
    """
    segments = segment_markdown(text)
    clauses = clauses_from_segments(doc_slug, segments)
    return PolicyDocument(
        doc_slug=doc_slug,
        source=source,
        policy_version=policy_version(doc_slug, [c.content_hash for c in clauses]),
        fetched_at=fetched_at or datetime.now(timezone.utc),
        clauses=clauses,
        corpus_role=corpus_role,
        is_holdout=is_holdout,
    )


def ingest(
    source: str | Path,
    *,
    corpus_role: CorpusRole,
    doc_slug: str | None = None,
    is_holdout: bool = False,
    **loader_kwargs,
) -> PolicyDocument:
    """Load, segment and hash one policy source into a `PolicyDocument`.

    `corpus_role` is keyword-only and required, mirroring the schema. DESIGN.md
    7.1 forbids pooling real and synthetic results, and a default here would let a
    caller quietly promote a fixture into evidence - so the caller has to say.
    """
    loaded = load(source, **loader_kwargs)
    return ingest_text(
        loaded.text,
        doc_slug=doc_slug or loaded.doc_slug,
        source=loaded.source,
        corpus_role=corpus_role,
        fetched_at=loaded.fetched_at,
        is_holdout=is_holdout,
    )
