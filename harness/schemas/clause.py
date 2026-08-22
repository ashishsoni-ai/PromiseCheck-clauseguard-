"""PolicyDocument + Clause schemas.

Specified by DESIGN.md 1.1: a policy document becomes an ordered list of clause
units, each with a stable content-hash ID.

    clause_id = f"{doc_slug}:{ordinal:03d}:{sha256(normalize(text))[:8]}"
    # e.g.  acme-refunds:014:a3f91c22

The ordinal/hash pairing is the change-detection primitive for the gate
(DESIGN.md 1.1, 6.2): the ordinal survives edits, the hash does not. That is what
lets the gate say "clause 14 changed" rather than "the document changed", which
in turn is what makes the incremental run cheap.

NOTE ON `Clause.text`: it holds the clause VERBATIM, never normalised. Commitment
C2 requires the judge to quote a span that is an exact substring of the cited
clause, so the verbatim text is the ground truth that check runs against.
Normalisation exists only to compute the hash. Normalising `text` in place would
quietly destroy span verification.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# sha256(...)[:8] -> exactly 8 lowercase hex characters.
_CONTENT_HASH_RE = re.compile(r"^[0-9a-f]{8}$")
# doc slugs appear inside a colon-delimited ID, so they may not contain a colon.
_DOC_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

DocSlug = Annotated[str, Field(pattern=_DOC_SLUG_RE.pattern, min_length=1)]

#: Provenance of a policy document. DESIGN.md 7.1 describes two buckets - 8 real
#: fetched policies and 2 synthetic stress fixtures - and forbids pooling their
#: results. A third role is needed in practice: `policies/acme-refunds.md` is
#: authored in-repo, so it is not real-world evidence, but it is also not one of
#: the deliberately-nasty stress fixtures. Collapsing it into either bucket would
#: misreport the corpus, so it gets its own role.
#:
#:   real            fetched from a live merchant policy page. Evidence.
#:   synthetic_stress  deliberately nasty nesting / internal contradiction.
#:                     A fixture, never evidence (DESIGN.md 7.1).
#:   worked_example    authored in-repo for demos, docs and tests. Realistic but
#:                     not observed in the wild, so also never evidence.
CorpusRole = Literal["real", "synthetic_stress", "worked_example"]


class Clause(BaseModel):
    """One addressable clause unit of a policy document.

    Clause boundaries come from structure (headings, numbered items, bullets),
    not naive chunking, with a recursive splitter only as a fallback for
    wall-of-text paragraphs. Target length is 40-400 tokens (DESIGN.md 1.1).
    """

    model_config = ConfigDict(extra="forbid")

    clause_id: str = Field(
        description="Composite ID: {doc_slug}:{ordinal:03d}:{content_hash}. "
        "Validated below for internal consistency."
    )
    doc_slug: DocSlug
    ordinal: int = Field(
        ge=1,
        description="1-based position in the document. Survives content edits, "
        "which is what makes 'clause 14 changed' expressible.",
    )
    text: str = Field(
        min_length=1,
        description="VERBATIM clause text. Never normalised - commitment C2's "
        "exact-substring span check runs against this exact string.",
    )
    content_hash: str = Field(
        description="sha256(normalize(text))[:8]. Changes iff meaningful content "
        "changes; whitespace- and punctuation-only edits do not move it."
    )
    heading_path: list[str] = Field(
        default_factory=list,
        description="Structural breadcrumb from the segmenter, outermost first, "
        "e.g. ['Returns', 'Timelines']. Gives the extractor and the review UI "
        "context a bare chunk would lose.",
    )
    token_estimate: int | None = Field(
        default=None,
        ge=0,
        description="Approximate token count, used to audit the 40-400 target "
        "in DESIGN.md 1.1. Advisory only; never a correctness input.",
    )

    @field_validator("content_hash")
    @classmethod
    def _hash_is_8_lowercase_hex(cls, v: str) -> str:
        if not _CONTENT_HASH_RE.match(v):
            raise ValueError(
                f"content_hash must be 8 lowercase hex chars (sha256[:8]), got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _clause_id_matches_its_parts(self) -> Clause:
        """A clause_id that disagrees with its own components is a silent
        corruption of the gate's change-detection primitive, so it is rejected
        rather than tolerated."""
        expected = f"{self.doc_slug}:{self.ordinal:03d}:{self.content_hash}"
        if self.clause_id != expected:
            raise ValueError(
                f"clause_id {self.clause_id!r} is inconsistent with its parts; "
                f"expected {expected!r}"
            )
        return self


class PolicyDocument(BaseModel):
    """An ingested policy document and its ordered clauses.

    `corpus_role` and `is_holdout` are schema fields rather than bookkeeping kept
    elsewhere, because DESIGN.md 7.1 requires real and synthetic results to be
    broken out separately and never pooled, and 7.3 requires two policies to stay
    fully held out. Encoding them here means a reporting path cannot pool them by
    accident.
    """

    model_config = ConfigDict(extra="forbid")

    doc_slug: DocSlug
    source: str = Field(
        min_length=1, description="Originating URL or file path."
    )
    policy_version: str = Field(
        description="Whole-document hash, 'sha256:<hex>'. Recorded on every "
        "audit row (DESIGN.md 5.1) so the regression story is provable."
    )
    fetched_at: datetime
    clauses: list[Clause] = Field(default_factory=list)
    corpus_role: CorpusRole = Field(
        description="Provenance. Required rather than defaulted: a default of "
        "'real' would fail open by silently promoting authored fixtures into "
        "evidence, and any other default would be a lie about half the corpus. "
        "DESIGN.md 7.1 makes this distinction a reporting requirement, so it is "
        "stated per document, not inferred from a filename."
    )
    is_holdout: bool = Field(
        default=False,
        description="True for the two policies never opened during prompt "
        "iteration (DESIGN.md 7.3).",
    )

    @field_validator("policy_version")
    @classmethod
    def _policy_version_is_prefixed_sha256(cls, v: str) -> str:
        if not re.match(r"^sha256:[0-9a-f]{64}$", v):
            raise ValueError(
                "policy_version must look like 'sha256:<64 lowercase hex>', "
                f"got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _clauses_are_ordered_and_belong_here(self) -> PolicyDocument:
        for i, clause in enumerate(self.clauses, start=1):
            if clause.doc_slug != self.doc_slug:
                raise ValueError(
                    f"clause {clause.clause_id!r} has doc_slug "
                    f"{clause.doc_slug!r}, expected {self.doc_slug!r}"
                )
            if clause.ordinal != i:
                raise ValueError(
                    "clauses must be contiguous and 1-based in document order; "
                    f"position {i} carries ordinal {clause.ordinal}"
                )
        return self

    def by_id(self, clause_id: str) -> Clause | None:
        """Look up a clause by its full composite ID."""
        for clause in self.clauses:
            if clause.clause_id == clause_id:
                return clause
        return None

    @property
    def counts_as_evidence(self) -> bool:
        """True only for policies actually observed in the wild.

        Reporting paths gate on this rather than testing `corpus_role` inline, so
        that "never pool real with synthetic" (DESIGN.md 7.1) is enforced in one
        place instead of re-derived at every call site.
        """
        return self.corpus_role == "real"
