"""Step 3 checkpoint: `coverage.json` - "what did you miss", as a number.

DESIGN.md 1.2:

    "**Unextractable clauses are logged, not dropped.** A `coverage.json` records
    every clause with zero rules. You will be asked 'what did you miss' and the
    answer must be a number, not a shrug."

So the tests here are mostly about the two ways that number can lie. It can
overstate coverage by not noticing a clause produced nothing, and it can overstate
it by counting a rule that cites a clause id no longer present in the document -
which happens whenever a policy is edited, because the content hash inside a
clause id moves when the clause does (DESIGN.md 1.1). The second is worse: such a
rule still evaluates and still labels probes, confidently and against text that no
longer exists.

The band assertions come from DESIGN.md 8: coverage targets 70-85% with 90%
aspirational, "the uncovered clauses are a named limitation, not a hidden one."
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.extract.coverage import (
    ASPIRATIONAL,
    COVERAGE_PATH,
    SCHEMA_VERSION,
    TARGET_MAX,
    TARGET_MIN,
    compute_corpus_coverage,
    compute_coverage,
    load_coverage,
    rules_by_clause,
    write_coverage,
)
from harness.ingest import ingest_text
from harness.schemas import Clause, EntitlementRule, PolicyDocument

POLICY_PATH = Path(__file__).resolve().parents[2] / "policies" / "acme-refunds.md"

VERSION = "sha256:" + "ab" * 32
T0 = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)


def make_doc(
    hashes: list[str], *, doc_slug: str = "acme-refunds", headings=None
) -> PolicyDocument:
    """A document with one clause per supplied hash.

    Hashes are hand-written rather than computed, for the same reason the
    conftest clause factory does it: a coverage test must not move because the
    hasher changed.
    """
    clauses = [
        Clause(
            clause_id=f"{doc_slug}:{ordinal:03d}:{h}",
            doc_slug=doc_slug,
            ordinal=ordinal,
            text=f"Clause body {h}.",
            content_hash=h,
            heading_path=list(headings or ["Acme Retail", "4. Return window"]),
            token_estimate=12,
        )
        for ordinal, h in enumerate(hashes, start=1)
    ]
    return PolicyDocument(
        doc_slug=doc_slug,
        source=f"policies/{doc_slug}.md",
        policy_version=VERSION,
        fetched_at=T0,
        clauses=clauses,
        corpus_role="worked_example",
    )


def hexes(n: int, *, start: int = 1) -> list[str]:
    """`n` distinct 8-char hashes: 11111111, 22222222, ..."""
    return [f"{(start + i) % 10}" * 8 for i in range(n)]


def make_rule(
    rule_id: str, clause_ids: list[str], *, exceptions=(), entitlement: str = "refund"
) -> EntitlementRule:
    return EntitlementRule(
        rule_id=rule_id,
        clause_ids=clause_ids,
        entitlement=entitlement,
        polarity="grants",
        conditions=[],
        exceptions=list(exceptions),
        precedence=10,
        extraction_confidence=0.9,
        needs_human_review=False,
    )


class TestEveryClauseIsAccountedFor:
    """The central promise: a clause cannot fall out of the report silently."""

    def test_a_clause_with_no_rules_is_listed_not_omitted(self):
        doc = make_doc(["11111111", "22222222"])
        report = compute_coverage(doc, [make_rule("R-1", [doc.clauses[0].clause_id])])
        assert len(report.clauses) == 2
        assert [c.clause_id for c in report.uncovered] == [doc.clauses[1].clause_id]

    def test_uncovered_clauses_come_back_in_document_order(self):
        doc = make_doc(hexes(5))
        report = compute_coverage(doc, [make_rule("R-1", [doc.clauses[2].clause_id])])
        assert [c.ordinal for c in report.uncovered] == [1, 2, 4, 5]

    def test_an_uncovered_clause_carries_the_breadcrumbs_a_reviewer_needs(self):
        """DESIGN.md's demo answer to "what did you miss" is a list a human can
        scan. A bare clause id is not that; the heading path is what makes an
        uncovered list stem recognisable as a list stem (see
        harness/extract/prompts.py)."""
        doc = make_doc(["11111111"], headings=["Acme Retail", "5. Categories excluded"])
        report = compute_coverage(doc, [])
        missed = report.uncovered[0]
        assert missed.heading_path == ("Acme Retail", "5. Categories excluded")
        assert missed.breadcrumbs == "Acme Retail > 5. Categories excluded"
        assert missed.token_estimate == 12

    def test_a_covered_clause_names_the_rules_that_cover_it(self):
        doc = make_doc(["11111111"])
        cid = doc.clauses[0].clause_id
        report = compute_coverage(doc, [make_rule("R-2", [cid]), make_rule("R-1", [cid])])
        assert report.clauses[0].rule_ids == ("R-1", "R-2")
        assert report.clauses[0].is_covered is True

    def test_rule_ids_are_deduplicated(self):
        doc = make_doc(["11111111", "22222222"])
        cids = [c.clause_id for c in doc.clauses]
        report = compute_coverage(doc, [make_rule("R-1", cids)])
        assert report.clauses[0].rule_ids == ("R-1",)
        assert report.rule_count == 1


class TestNestedExceptionsCount:
    """A clause whose only rule is a depth-2 carve-out is still probeable.

    DESIGN.md 3.2 strategy 3 exists to probe exactly those, so counting them as
    uncovered would understate coverage on the clauses that matter most.
    """

    def test_a_clause_cited_only_by_an_exception_is_covered(self):
        doc = make_doc(["11111111", "22222222"])
        root_cid, exc_cid = (c.clause_id for c in doc.clauses)
        tree = make_rule(
            "R-1", [root_cid], exceptions=[make_rule("R-1-x1", [exc_cid])]
        )
        report = compute_coverage(doc, [tree])
        assert report.uncovered == ()
        assert report.clauses[1].rule_ids == ("R-1-x1",)

    def test_a_depth_two_exception_is_reached(self):
        doc = make_doc(hexes(3))
        a, b, c = (cl.clause_id for cl in doc.clauses)
        tree = make_rule(
            "R-1",
            [a],
            exceptions=[make_rule("R-1-x1", [b], exceptions=[make_rule("R-1-x1-x1", [c])])],
        )
        assert compute_coverage(doc, [tree]).uncovered == ()

    def test_rules_by_clause_indexes_the_whole_subtree(self):
        tree = make_rule(
            "R-1",
            ["d:001:11111111"],
            exceptions=[make_rule("R-1-x1", ["d:002:22222222"])],
        )
        index = rules_by_clause([tree])
        assert index == {
            "d:001:11111111": ["R-1"],
            "d:002:22222222": ["R-1-x1"],
        }


class TestTheNumberAndItsBand:
    """DESIGN.md 8: 70-85% target, 90% aspirational."""

    @staticmethod
    def report_with(covered: int, total: int):
        doc = make_doc([f"{i:08x}" for i in range(total)])
        rules = [
            make_rule(f"R-{i}", [doc.clauses[i].clause_id]) for i in range(covered)
        ]
        return compute_coverage(doc, rules)

    def test_the_fraction_and_percentage_agree(self):
        report = self.report_with(3, 4)
        assert report.fraction == 0.75
        assert report.pct == 75.0

    def test_exactly_the_lower_bound_is_in_band(self):
        report = self.report_with(7, 10)
        assert report.fraction == TARGET_MIN
        assert report.band == "in_band"
        assert report.meets_target is True

    def test_exactly_the_upper_bound_is_still_in_band(self):
        report = self.report_with(17, 20)
        assert report.fraction == TARGET_MAX
        assert report.band == "in_band"

    def test_between_the_upper_bound_and_the_aspiration_is_above_target(self):
        report = self.report_with(7, 8)
        assert report.pct == 87.5
        assert report.band == "above_target"

    def test_the_aspiration_is_reported_as_such(self):
        report = self.report_with(9, 10)
        assert report.fraction == ASPIRATIONAL
        assert report.band == "aspirational"

    def test_below_the_lower_bound_fails_the_target(self):
        report = self.report_with(2, 3)
        assert report.band == "below_target"
        assert report.meets_target is False

    def test_full_coverage_is_aspirational_not_an_error(self):
        assert self.report_with(4, 4).band == "aspirational"

    def test_zero_coverage_is_below_target_rather_than_empty(self):
        report = self.report_with(0, 4)
        assert report.pct == 0.0
        assert report.band == "below_target"
        assert len(report.uncovered) == 4

    def test_summary_names_the_counts_and_the_band(self):
        summary = self.report_with(3, 4).summary()
        assert "3/4" in summary and "75.0%" in summary and "in_band" in summary


class TestADocumentWithNoClauses:
    """`PolicyDocument.clauses` defaults to empty and the schema permits it, so
    the arithmetic has to survive it. Zero rather than None because the caller
    asking is a gate and a gate cannot act on None."""

    def test_the_fraction_is_zero_and_the_band_says_empty(self):
        report = compute_coverage(make_doc([]), [])
        assert report.total == 0
        assert report.fraction == 0.0
        assert report.band == "empty"

    def test_an_empty_document_does_not_meet_the_target(self):
        assert compute_coverage(make_doc([]), []).meets_target is False


class TestOrphanCitationsAreReported:
    """The failure mode that matters more than the headline number.

    A rule citing a clause id absent from the document is grounded in text that
    no longer exists - the policy was edited and the hash moved. Such a rule
    still evaluates and still labels probes, so it has to be named.
    """

    def test_a_rule_citing_a_vanished_clause_is_flagged(self):
        doc = make_doc(["11111111"])
        stale = make_rule("R-stale", ["acme-refunds:099:deadbeef"])
        report = compute_coverage(doc, [stale])
        assert report.orphan_clause_ids == ("acme-refunds:099:deadbeef",)

    def test_an_edited_clause_orphans_the_rule_that_cited_its_old_hash(self):
        """The realistic shape: same document, same ordinal, new content hash."""
        doc = make_doc(["9e9e9e9e"])
        report = compute_coverage(doc, [make_rule("R-1", ["acme-refunds:001:11111111"])])
        assert report.orphan_clause_ids == ("acme-refunds:001:11111111",)
        assert report.uncovered[0].clause_id == "acme-refunds:001:9e9e9e9e"

    def test_a_cross_reference_to_another_document_is_not_an_orphan_here(self):
        """DESIGN.md 1.2 allows "2+ for cross-refs". Flagging the far side would
        make every cross-reference rule look broken from both documents."""
        doc = make_doc(["11111111"])
        rule = make_rule("R-1", [doc.clauses[0].clause_id, "other-policy:003:33333333"])
        report = compute_coverage(doc, [rule])
        assert report.orphan_clause_ids == ()
        assert report.uncovered == ()

    def test_the_orphan_count_appears_in_the_summary(self):
        doc = make_doc(["11111111"])
        report = compute_coverage(doc, [make_rule("R-1", ["acme-refunds:099:deadbeef"])])
        assert "1 orphan citation" in report.summary()

    def test_a_clean_report_does_not_mention_orphans(self):
        doc = make_doc(["11111111"])
        report = compute_coverage(doc, [make_rule("R-1", [doc.clauses[0].clause_id])])
        assert "orphan" not in report.summary()


class TestCorpusRollUp:
    def test_coverage_is_pooled_over_clauses_not_averaged_over_documents(self):
        """A 1-clause fixture at 100% must not offset a 10-clause policy at 10%.
        DESIGN.md 8 measures clauses."""
        small = make_doc(["11111111"], doc_slug="tiny")
        large = make_doc([f"{i:08x}" for i in range(10)], doc_slug="large")
        rules = [
            make_rule("R-1", [small.clauses[0].clause_id]),
            make_rule("R-2", [large.clauses[0].clause_id]),
        ]
        corpus = compute_corpus_coverage([small, large], rules)
        assert corpus.total == 11
        assert corpus.covered == 2
        assert corpus.pct == 18.2

    def test_a_cross_document_citation_is_resolved_at_corpus_scope(self):
        a = make_doc(["11111111"], doc_slug="doc-a")
        b = make_doc(["22222222"], doc_slug="doc-b")
        rule = make_rule("R-1", [a.clauses[0].clause_id, b.clauses[0].clause_id])
        corpus = compute_corpus_coverage([a, b], [rule])
        assert corpus.orphan_clause_ids == ()
        assert corpus.covered == 2

    def test_an_orphan_naming_an_absent_document_is_still_reported(self):
        """It would otherwise vanish between the per-document and corpus views,
        and the point of this module is that nothing goes unreported."""
        a = make_doc(["11111111"], doc_slug="doc-a")
        rule = make_rule("R-1", ["doc-z:001:33333333"])
        corpus = compute_corpus_coverage([a], [rule])
        assert corpus.orphan_clause_ids == ("doc-z:001:33333333",)
        assert "(unknown document)" in [d.doc_slug for d in corpus.documents]

    def test_the_synthetic_entry_contributes_no_clauses_to_the_total(self):
        a = make_doc(["11111111"], doc_slug="doc-a")
        corpus = compute_corpus_coverage([a], [make_rule("R-1", ["doc-z:001:33333333"])])
        assert corpus.total == 1

    def test_an_empty_corpus_reports_empty(self):
        corpus = compute_corpus_coverage([], [])
        assert corpus.total == 0
        assert corpus.band == "empty"
        assert corpus.meets_target is False

    def test_the_summary_counts_documents(self):
        a = make_doc(["11111111"], doc_slug="doc-a")
        b = make_doc(["22222222"], doc_slug="doc-b")
        assert "2 document(s)" in compute_corpus_coverage([a, b], []).summary()


class TestTheFileOnDisk:
    """Same write discipline as the ingest manifest and the fetch lockfile:
    deterministic bytes, so "coverage.json changed" keeps meaning something."""

    def test_the_real_committed_path_is_never_touched_by_these_tests(self):
        """Tripwire. Every test below writes to tmp_path; a test that wrote to
        the committed report would re-baseline it and hide a real regression."""
        assert COVERAGE_PATH == Path("policies/.clauseguard/coverage.json")

    def test_writing_a_document_report_wraps_it_as_a_corpus(self, tmp_path):
        doc = make_doc(["11111111", "22222222"])
        path = write_coverage(
            compute_coverage(doc, [make_rule("R-1", [doc.clauses[0].clause_id])]),
            tmp_path / "coverage.json",
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["total_clauses"] == 2
        assert payload["coverage_pct"] == 50.0
        assert list(payload["documents"]) == ["acme-refunds"]

    def test_the_uncovered_clause_survives_the_round_trip_by_name(self, tmp_path):
        doc = make_doc(["11111111", "22222222"])
        write_coverage(compute_coverage(doc, []), tmp_path / "coverage.json")
        payload = load_coverage(tmp_path / "coverage.json")
        listed = payload["documents"]["acme-refunds"]["clauses"]
        assert [c["clause_id"] for c in listed] == [
            "acme-refunds:001:11111111",
            "acme-refunds:002:22222222",
        ]
        assert all(c["rule_ids"] == [] for c in listed)

    def test_rewriting_an_unchanged_report_produces_identical_bytes(self, tmp_path):
        doc = make_doc(["11111111", "22222222"])
        report = compute_coverage(doc, [make_rule("R-1", [doc.clauses[0].clause_id])])
        first = (tmp_path / "a.json")
        second = (tmp_path / "b.json")
        write_coverage(report, first)
        write_coverage(compute_coverage(doc, [make_rule("R-1", [doc.clauses[0].clause_id])]), second)
        assert first.read_bytes() == second.read_bytes()

    def test_the_file_ends_with_a_newline(self, tmp_path):
        path = write_coverage(compute_coverage(make_doc(["11111111"]), []), tmp_path / "c.json")
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_a_missing_file_reads_as_no_report(self, tmp_path):
        assert load_coverage(tmp_path / "absent.json") == {}

    def test_the_placeholder_reads_as_no_report(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("{}\n", encoding="utf-8")
        assert load_coverage(path) == {}

    def test_a_future_schema_version_refuses_to_be_read(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1}), encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            load_coverage(path)
        assert "regenerate" in str(excinfo.value)


class TestAgainstTheRealWorkedExample:
    """The claim is about real policy text, so at least one test uses it.

    `policies/acme-refunds.md` segments into 20 clauses, four of which are the
    non-normative list stems and bullets recorded in harness/extract/prompts.py.
    Extraction has not run yet, so coverage is legitimately zero here - and that
    is the point: the report answers "what did you miss" with 20 named clause ids
    rather than a shrug, before a single rule exists.
    """

    @staticmethod
    def document():
        return ingest_text(
            POLICY_PATH.read_text(encoding="utf-8"),
            doc_slug="acme-refunds",
            source="policies/acme-refunds.md",
            corpus_role="worked_example",
        )

    def test_the_worked_example_is_on_disk(self):
        assert POLICY_PATH.is_file()

    def test_with_no_rules_every_real_clause_is_named_as_uncovered(self):
        report = compute_coverage(self.document(), [])
        assert report.total == 20
        assert len(report.uncovered) == 20
        assert report.pct == 0.0
        assert all(c.clause_id.startswith("acme-refunds:") for c in report.uncovered)

    def test_the_report_pins_the_policy_version_it_measured(self):
        """Coverage is a fact about one `policy_version`; without it, a number
        quoted in the report cannot be tied to the text it describes."""
        doc = self.document()
        assert compute_coverage(doc, []).policy_version == doc.policy_version

    def test_one_rule_against_one_real_clause_moves_the_number(self):
        doc = self.document()
        report = compute_coverage(doc, [make_rule("R-1", [doc.clauses[0].clause_id])])
        assert report.pct == 5.0
        assert len(report.uncovered) == 19
