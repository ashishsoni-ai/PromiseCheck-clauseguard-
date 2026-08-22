"""`coverage.json` - which clauses produced no rule, by name and by number.

DESIGN.md 1.2:

    "**Unextractable clauses are logged, not dropped.** A `coverage.json` records
    every clause with zero rules. You will be asked 'what did you miss' and the
    answer must be a number, not a shrug."

DESIGN.md 8 attaches the number to a band: rule extraction coverage, measured as
clauses with at least one rule, targets 70-85% with 90% aspirational, because
"the uncovered clauses are a named limitation, not a hidden one."

That framing decides the shape of this module. A bare percentage would satisfy
neither sentence: the demo answer to "what did you miss" is a list of clause ids
with their heading breadcrumbs, so a reviewer can look at the four clauses that
produced nothing and see for themselves that three are list stems and one is a
shipping-address paragraph. So the report carries every clause either way, and
the percentage is derived from it rather than stored beside it.

TWO DIRECTIONS OF FAILURE, BOTH REPORTED
An uncovered clause is a rule the harness will never probe - a false negative in
the policy's coverage, and the one DESIGN.md 1.2 asks about. The reverse also
happens and is worse: a rule citing a clause id that no document contains. That
means the policy was edited after extraction and the rule is now grounded in text
that has ceased to exist, because the content hash in a clause id moves when the
clause does (DESIGN.md 1.1). Such a rule still evaluates, still labels probes, and
is silently wrong. It surfaces here as `orphan_clause_ids`.

WHERE THE FILE LIVES: `policies/.clauseguard/coverage.json`, beside the ingest
manifest, and committed. DESIGN.md names the file but not its path; the choice of
directory is this repo's, on the grounds that coverage is a fact about a specific
`policy_version` and belongs next to the baseline that pins it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from harness.schemas.clause import PolicyDocument
from harness.schemas.rule import EntitlementRule

#: Committed. Sits beside policies/.clauseguard/manifest.json.
COVERAGE_PATH = Path("policies/.clauseguard/coverage.json")

#: DESIGN.md 8: target 70-85%, aspirational 90%.
TARGET_MIN = 0.70
TARGET_MAX = 0.85
ASPIRATIONAL = 0.90

SCHEMA_VERSION = 1


def _doc_slug_of(clause_id: str) -> str:
    """`acme-refunds:014:a3f91c22` -> `acme-refunds`.

    A doc slug cannot contain a colon (see the slug pattern in
    harness/schemas/clause.py), so the first segment is unambiguous.
    """
    return clause_id.split(":", 1)[0]


def rules_by_clause(rules: Iterable[EntitlementRule]) -> dict[str, list[str]]:
    """Map clause_id -> the rule_ids grounded in it, nested exceptions included.

    Exceptions count. A clause whose only rule is a depth-2 carve-out is still a
    clause the harness can probe, and DESIGN.md 3.2 strategy 3 exists to probe
    exactly those - counting it as uncovered would understate coverage on the
    clauses that matter most.
    """
    index: dict[str, set[str]] = {}
    for root in rules:
        for node in root.walk():
            for clause_id in node.clause_ids:
                index.setdefault(clause_id, set()).add(node.rule_id)
    return {cid: sorted(ids) for cid, ids in index.items()}


@dataclass(frozen=True, slots=True)
class ClauseCoverage:
    """One clause and the rules extracted from it, if any."""

    clause_id: str
    ordinal: int
    heading_path: tuple[str, ...]
    token_estimate: int | None
    rule_ids: tuple[str, ...]

    @property
    def is_covered(self) -> bool:
        return bool(self.rule_ids)

    @property
    def breadcrumbs(self) -> str:
        """The reviewer-facing locator, matching how the extractor sees a clause
        (see harness/extract/prompts.py)."""
        return " > ".join(self.heading_path)

    def to_dict(self) -> dict:
        return {
            "clause_id": self.clause_id,
            "ordinal": self.ordinal,
            "heading_path": list(self.heading_path),
            "token_estimate": self.token_estimate,
            "rule_ids": list(self.rule_ids),
        }


@dataclass(frozen=True, slots=True)
class DocumentCoverage:
    """Extraction coverage for one policy document at one `policy_version`."""

    doc_slug: str
    policy_version: str
    clauses: tuple[ClauseCoverage, ...]
    orphan_clause_ids: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return len(self.clauses)

    @property
    def covered(self) -> tuple[ClauseCoverage, ...]:
        return tuple(c for c in self.clauses if c.is_covered)

    @property
    def uncovered(self) -> tuple[ClauseCoverage, ...]:
        """Ordinal order, so the report reads down the document."""
        return tuple(c for c in self.clauses if not c.is_covered)

    @property
    def rule_count(self) -> int:
        """Distinct rules grounded anywhere in this document."""
        return len({rid for c in self.clauses for rid in c.rule_ids})

    @property
    def fraction(self) -> float:
        """Clauses with >=1 rule, as a fraction. 0.0 for an empty document.

        Zero rather than undefined because the caller asking is a gate, and a
        gate cannot act on None. `band` reports "empty" so the zero is never
        mistaken for a failed extraction.
        """
        if not self.clauses:
            return 0.0
        return len(self.covered) / self.total

    @property
    def pct(self) -> float:
        return round(self.fraction * 100, 1)

    @property
    def band(self) -> str:
        """Where this sits against DESIGN.md 8's target."""
        if not self.clauses:
            return "empty"
        if self.fraction >= ASPIRATIONAL:
            return "aspirational"
        if self.fraction > TARGET_MAX:
            return "above_target"
        if self.fraction >= TARGET_MIN:
            return "in_band"
        return "below_target"

    @property
    def meets_target(self) -> bool:
        return bool(self.clauses) and self.fraction >= TARGET_MIN

    def summary(self) -> str:
        """One line for the CLI checkpoint."""
        return (
            f"{self.doc_slug}: {len(self.covered)}/{self.total} clauses covered "
            f"({self.pct}%, {self.band}) by {self.rule_count} rule(s)"
            + (
                f"; {len(self.orphan_clause_ids)} orphan citation(s)"
                if self.orphan_clause_ids
                else ""
            )
        )

    def to_dict(self) -> dict:
        return {
            "doc_slug": self.doc_slug,
            "policy_version": self.policy_version,
            "total_clauses": self.total,
            "covered_clauses": len(self.covered),
            "uncovered_clauses": len(self.uncovered),
            "rule_count": self.rule_count,
            "coverage_pct": self.pct,
            "band": self.band,
            "meets_target": self.meets_target,
            "orphan_clause_ids": list(self.orphan_clause_ids),
            "clauses": [c.to_dict() for c in self.clauses],
        }


@dataclass(frozen=True, slots=True)
class CorpusCoverage:
    """Roll-up across documents. The number quoted in the report is this one."""

    documents: tuple[DocumentCoverage, ...]

    @property
    def total(self) -> int:
        return sum(d.total for d in self.documents)

    @property
    def covered(self) -> int:
        return sum(len(d.covered) for d in self.documents)

    @property
    def fraction(self) -> float:
        """Pooled over clauses, not averaged over documents.

        A mean of per-document percentages would let a 3-clause synthetic
        fixture at 100% offset a 60-clause real policy at 55%, and DESIGN.md 8
        measures clauses.
        """
        return self.covered / self.total if self.total else 0.0

    @property
    def pct(self) -> float:
        return round(self.fraction * 100, 1)

    @property
    def band(self) -> str:
        if not self.total:
            return "empty"
        if self.fraction >= ASPIRATIONAL:
            return "aspirational"
        if self.fraction > TARGET_MAX:
            return "above_target"
        if self.fraction >= TARGET_MIN:
            return "in_band"
        return "below_target"

    @property
    def meets_target(self) -> bool:
        return bool(self.total) and self.fraction >= TARGET_MIN

    @property
    def orphan_clause_ids(self) -> tuple[str, ...]:
        seen: set[str] = set()
        for d in self.documents:
            seen.update(d.orphan_clause_ids)
        return tuple(sorted(seen))

    def summary(self) -> str:
        return (
            f"corpus: {self.covered}/{self.total} clauses covered "
            f"({self.pct}%, {self.band}) across {len(self.documents)} document(s)"
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "target_band_pct": [TARGET_MIN * 100, TARGET_MAX * 100],
            "aspirational_pct": ASPIRATIONAL * 100,
            "total_clauses": self.total,
            "covered_clauses": self.covered,
            "coverage_pct": self.pct,
            "band": self.band,
            "meets_target": self.meets_target,
            "orphan_clause_ids": list(self.orphan_clause_ids),
            "documents": {d.doc_slug: d.to_dict() for d in self.documents},
        }


def compute_coverage(
    document: PolicyDocument, rules: Sequence[EntitlementRule]
) -> DocumentCoverage:
    """Coverage for one document. Every clause appears, covered or not.

    Rules citing another document's clauses are ignored rather than counted as
    orphans: a cross-reference rule (DESIGN.md 1.2, "sometimes 2+ for
    cross-refs") legitimately cites two documents, and flagging it here would
    make every such rule look broken from both sides. Corpus-level orphan
    detection is `compute_corpus_coverage`'s job, where the full clause universe
    is known.
    """
    index = rules_by_clause(rules)
    known = {c.clause_id for c in document.clauses}

    clauses = tuple(
        ClauseCoverage(
            clause_id=clause.clause_id,
            ordinal=clause.ordinal,
            heading_path=tuple(clause.heading_path),
            token_estimate=clause.token_estimate,
            rule_ids=tuple(index.get(clause.clause_id, ())),
        )
        for clause in document.clauses
    )

    orphans = tuple(
        sorted(
            cid
            for cid in index
            if cid not in known and _doc_slug_of(cid) == document.doc_slug
        )
    )

    return DocumentCoverage(
        doc_slug=document.doc_slug,
        policy_version=document.policy_version,
        clauses=clauses,
        orphan_clause_ids=orphans,
    )


def compute_corpus_coverage(
    documents: Sequence[PolicyDocument], rules: Sequence[EntitlementRule]
) -> CorpusCoverage:
    """Coverage across the corpus, with orphan citations resolved globally.

    A clause id cited by a rule but present in no document given here is an
    orphan regardless of which slug it names - at corpus scope there is no
    other document it could belong to.
    """
    index = rules_by_clause(rules)
    known = {c.clause_id for doc in documents for c in doc.clauses}
    global_orphans = sorted(cid for cid in index if cid not in known)

    per_doc: list[DocumentCoverage] = []
    for doc in documents:
        base = compute_coverage(doc, rules)
        mine = tuple(
            cid for cid in global_orphans if _doc_slug_of(cid) == doc.doc_slug
        )
        per_doc.append(
            DocumentCoverage(
                doc_slug=base.doc_slug,
                policy_version=base.policy_version,
                clauses=base.clauses,
                orphan_clause_ids=mine,
            )
        )

    # Orphans naming a slug no document here claims would otherwise vanish, so
    # they are parked on a synthetic entry rather than dropped - the whole point
    # of this module is that nothing goes unreported.
    unclaimed = tuple(
        cid
        for cid in global_orphans
        if _doc_slug_of(cid) not in {d.doc_slug for d in documents}
    )
    if unclaimed:
        per_doc.append(
            DocumentCoverage(
                doc_slug="(unknown document)",
                policy_version="",
                clauses=(),
                orphan_clause_ids=unclaimed,
            )
        )

    return CorpusCoverage(documents=tuple(per_doc))


def write_coverage(
    report: DocumentCoverage | CorpusCoverage, path: Path = COVERAGE_PATH
) -> Path:
    """Write `coverage.json` deterministically.

    Same discipline as the ingest manifest and the fetch lockfile: sorted keys
    and a trailing newline, so re-running extraction on an unchanged policy
    produces no git diff and "coverage.json changed" keeps meaning something.
    """
    payload = (
        report.to_dict()
        if isinstance(report, CorpusCoverage)
        else CorpusCoverage(documents=(report,)).to_dict()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        # LF explicitly, so this report is byte-identical on every host. See manifest.py.
        newline="\n",
    )
    return path


def load_coverage(path: Path = COVERAGE_PATH) -> dict:
    """Read the committed report back. Tolerates the `{}` placeholder."""
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw or raw == "{}":
        return {}
    data = json.loads(raw)
    version = data.get("schema_version")
    if version is not None and version != SCHEMA_VERSION:
        raise ValueError(
            f"{path} is schema_version {version}, this build writes "
            f"{SCHEMA_VERSION}; regenerate rather than merging by hand"
        )
    return data
