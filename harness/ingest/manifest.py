"""The committed clause-hash baseline, and the diff the gate runs against it.

DESIGN.md 2 step ①:

    "trafilatura/file read -> normalise -> segment -> 47 clauses, each hashed.
    Compare against `policies/.clauseguard/manifest.json`. Result: clause ordinal
    014 has a new hash (a3f91c22 -> b7d0e419), everything else unchanged.
    *No LLM call.*"

Everything expensive downstream is gated on this file. Extraction (LLM call #1)
runs only on clauses this module reports as changed or added, so a manifest that
over-reports costs money on every push and one that under-reports lets a policy
edit reach production unexamined.

THE MANIFEST CONTAINS NO CLAUSE TEXT
Only ordinals, hashes, heading paths and token counts. This is not an
optimisation: the manifest is COMMITTED, and DESIGN.md 7.1 says to ship "URLs,
fetch timestamps, content hashes and the fetcher - not the policy corpus". Storing
`text` here would republish eight merchants' policy pages in the repo through the
back door. The gitignored `policies/.cache/` is where text is allowed to live.

WHY THERE IS A `moved` CATEGORY
Diffing on ordinal alone has a bad failure mode: insert one clause at the top and
every following ordinal shifts, so all 47 clauses look changed and the whole
document goes to the extractor. Matching unchanged hashes at new ordinals first
collapses that to one added clause plus 46 moves, and a move needs no LLM call
because the content is byte-identical. This is what keeps the incremental run
cheap in the one edit shape most likely to occur.

WHY TIMESTAMPS DO NOT CHURN
`content_fetched_at` is the timestamp of the fetch that last *changed* the
content, not of the most recent fetch. If it were the latter, every no-op CI run
would rewrite the committed manifest, and "the manifest diff is empty" would stop
being the same statement as "the policy did not change" - which is the entire
value of committing it. Re-verification timestamps are per-run facts and belong in
the append-only audit store (DESIGN.md 5.1), not in a committed baseline.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from harness.schemas import PolicyDocument

#: Committed, unlike `policies/.cache/`. This file IS the gate's baseline; losing
#: it means the next run cannot tell an edit from a first sighting.
MANIFEST_PATH = Path("policies/.clauseguard/manifest.json")

#: Bumped only on a breaking layout change, so an old manifest is refused loudly
#: rather than misread as "every clause changed".
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ClauseChange:
    """One clause whose hash moved. Carries both hashes for the gate's report."""

    ordinal: int
    old_hash: str
    new_hash: str
    clause_id: str


@dataclass(frozen=True)
class ClauseMove:
    """Byte-identical content that changed position. Needs no LLM call."""

    content_hash: str
    old_ordinal: int
    new_ordinal: int
    clause_id: str


@dataclass
class DocumentDiff:
    """Per-document result of comparing a fresh ingest against the baseline."""

    doc_slug: str
    is_new_document: bool = False
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[ClauseChange] = field(default_factory=list)
    moved: list[ClauseMove] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def needs_extraction(self) -> list[str]:
        """Clause IDs that must go to the EXTRACTOR (DESIGN.md 2 step ②).

        Added and changed only. A move is byte-identical content at a new
        position, so re-extracting it would spend a model call to rediscover a
        rule already in `rules.lock.json`.
        """
        return [*self.added, *(c.clause_id for c in self.changed)]

    @property
    def is_clean(self) -> bool:
        """True when nothing needs extraction and nothing disappeared.

        Moves are tolerated here: they change clause IDs, which downstream probe
        invalidation must handle, but they are not a policy change.
        """
        return not (self.added or self.changed or self.removed)

    def summary(self) -> str:
        return (
            f"{self.doc_slug}: {len(self.unchanged)} unchanged, "
            f"{len(self.moved)} moved, {len(self.changed)} changed, "
            f"{len(self.added)} added, {len(self.removed)} removed"
        )


def fingerprint_document(doc: PolicyDocument) -> dict[str, Any]:
    """Reduce a PolicyDocument to its committable fingerprint. No clause text."""
    return {
        "doc_slug": doc.doc_slug,
        "source": doc.source,
        "policy_version": doc.policy_version,
        "corpus_role": doc.corpus_role,
        "is_holdout": doc.is_holdout,
        "content_fetched_at": doc.fetched_at.isoformat(),
        "clauses": [
            {
                "ordinal": c.ordinal,
                "content_hash": c.content_hash,
                "clause_id": c.clause_id,
                "heading_path": list(c.heading_path),
                "token_estimate": c.token_estimate,
            }
            for c in doc.clauses
        ],
    }


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    """Read the baseline, or an empty manifest if it does not exist yet.

    A missing manifest is a legitimate first run, not an error - but a manifest
    from an incompatible schema version is an error, because silently misreading
    it would report every clause as changed and trigger a full re-extraction.
    """
    p = Path(path)
    if not p.exists():
        return {"schema_version": SCHEMA_VERSION, "documents": {}}

    raw = p.read_text(encoding="utf-8").strip()
    if not raw or raw == "{}":
        # The scaffolded placeholder. Distinct from a version mismatch: an empty
        # object carries no claims, so reading it as "no baseline yet" cannot
        # cause a wrong answer, whereas refusing it would block a first run.
        return {"schema_version": SCHEMA_VERSION, "documents": {}}

    data = json.loads(raw)
    found = data.get("schema_version")
    if found != SCHEMA_VERSION:
        raise ValueError(
            f"{p} has schema_version {found!r}, this build expects "
            f"{SCHEMA_VERSION}. Refusing to guess: re-baseline explicitly."
        )
    data.setdefault("documents", {})
    return data


def write_manifest(
    documents: dict[str, Any], path: str | Path = MANIFEST_PATH
) -> Path:
    """Write the baseline deterministically.

    `sort_keys` plus a fixed indent plus a trailing newline so that two runs over
    an unchanged policy produce byte-identical files. Without that, the committed
    manifest churns on dict ordering and its git diff stops being readable
    evidence of what changed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "documents": documents}
    p.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        # LF explicitly: this file is committed evidence, and without it Windows writes
        # CRLF while Linux writes LF, so the same manifest hashes two ways.
        newline="\n",
    )
    return p


def diff_document(
    previous: dict[str, Any] | None, current: PolicyDocument
) -> DocumentDiff:
    """Compare a fresh ingest against this document's manifest entry.

    Resolution order is unchanged, then moved, then changed, then added/removed.
    Cheapest and most certain classification first: an exact ordinal+hash match is
    unambiguous, a hash match elsewhere is still certain about content, and only
    what neither explains is treated as an edit.
    """
    diff = DocumentDiff(doc_slug=current.doc_slug)

    if not previous:
        diff.is_new_document = True
        diff.added = [c.clause_id for c in current.clauses]
        return diff

    prev_entries = previous.get("clauses", [])
    prev_by_ordinal: dict[int, dict[str, Any]] = {
        int(e["ordinal"]): e for e in prev_entries
    }
    prev_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in prev_entries:
        prev_by_hash[str(entry["content_hash"])].append(entry)

    consumed_ordinals: set[int] = set()

    # Pass 1: exact ordinal + hash match.
    pending = []
    for clause in current.clauses:
        prior = prev_by_ordinal.get(clause.ordinal)
        if prior is not None and str(prior["content_hash"]) == clause.content_hash:
            diff.unchanged.append(clause.clause_id)
            consumed_ordinals.add(clause.ordinal)
        else:
            pending.append(clause)

    # Pass 2: same content, different position.
    still_pending = []
    for clause in pending:
        candidates = [
            e
            for e in prev_by_hash.get(clause.content_hash, [])
            if int(e["ordinal"]) not in consumed_ordinals
        ]
        if candidates:
            match = candidates[0]
            consumed_ordinals.add(int(match["ordinal"]))
            diff.moved.append(
                ClauseMove(
                    content_hash=clause.content_hash,
                    old_ordinal=int(match["ordinal"]),
                    new_ordinal=clause.ordinal,
                    clause_id=clause.clause_id,
                )
            )
        else:
            still_pending.append(clause)

    # Pass 3: an edit in place, or a genuinely new clause.
    for clause in still_pending:
        prior = prev_by_ordinal.get(clause.ordinal)
        if prior is not None and int(prior["ordinal"]) not in consumed_ordinals:
            consumed_ordinals.add(clause.ordinal)
            diff.changed.append(
                ClauseChange(
                    ordinal=clause.ordinal,
                    old_hash=str(prior["content_hash"]),
                    new_hash=clause.content_hash,
                    clause_id=clause.clause_id,
                )
            )
        else:
            diff.added.append(clause.clause_id)

    # Whatever the baseline had and nothing claimed is gone.
    for ordinal, entry in sorted(prev_by_ordinal.items()):
        if ordinal not in consumed_ordinals:
            diff.removed.append(str(entry["clause_id"]))

    return diff


def diff_against_manifest(
    document: PolicyDocument, path: str | Path = MANIFEST_PATH
) -> DocumentDiff:
    """Convenience: load the baseline and diff one document against it."""
    manifest = load_manifest(path)
    return diff_document(manifest["documents"].get(document.doc_slug), document)


def update_manifest(
    document: PolicyDocument,
    diff: DocumentDiff,
    path: str | Path = MANIFEST_PATH,
) -> Path:
    """Fold one document's fresh fingerprint into the committed baseline.

    When the diff is clean AND nothing moved, the previous `content_fetched_at` is
    preserved - see the module docstring on timestamp churn. A move still rewrites
    clause IDs in the baseline, so it does count as a content-layout change worth
    a new timestamp.
    """
    manifest = load_manifest(path)
    entry = fingerprint_document(document)

    prior = manifest["documents"].get(document.doc_slug)
    if prior and diff.is_clean and not diff.moved:
        entry["content_fetched_at"] = prior.get(
            "content_fetched_at", entry["content_fetched_at"]
        )

    manifest["documents"][document.doc_slug] = entry
    return write_manifest(manifest["documents"], path)


def manifest_fetched_at(entry: dict[str, Any]) -> datetime:
    """Parse a manifest entry's timestamp back into a datetime."""
    return datetime.fromisoformat(str(entry["content_fetched_at"]))
