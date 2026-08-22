"""Tests for the committed clause-hash baseline and its diff (DESIGN.md 2 step ①).

This module decides which clauses cost an LLM call, so the tests are organised
around the two ways it can be wrong, which have very different costs:

- OVER-REPORTING wastes money. Insert one clause and every following ordinal
  shifts; a diff that called all of those "changed" would re-extract a whole
  document on a one-line edit. `TestInsertionIsCheap` is the guard.
- UNDER-REPORTING is a correctness failure. A clause wrongly called "unchanged"
  keeps a stale rule in `rules.lock.json`, and the gate then passes a policy edit
  it never examined. `TestEditInPlace` and `TestRemoval` are the guards.

Every test writes to `tmp_path`. Nothing here may touch the real
`policies/.clauseguard/manifest.json` - it is the committed baseline the gate
measures against, and a test that rewrote it would silently re-baseline the repo
and make the next real diff come back clean.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from harness.ingest.hashing import policy_version
from harness.ingest.manifest import (
    MANIFEST_PATH,
    SCHEMA_VERSION,
    diff_against_manifest,
    diff_document,
    fingerprint_document,
    load_manifest,
    manifest_fetched_at,
    update_manifest,
    write_manifest,
)
from harness.schemas import Clause, PolicyDocument

H1 = "11111111"
H2 = "22222222"
H3 = "33333333"
H4 = "44444444"
EDITED = "9e9e9e9e"

T0 = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)
T1 = T0 + timedelta(days=7)


def make_doc(
    hashes: list[str],
    *,
    doc_slug: str = "acme-refunds",
    fetched_at: datetime = T0,
    corpus_role: str = "worked_example",
    source: str = "policies/acme-refunds.md",
    is_holdout: bool = False,
) -> PolicyDocument:
    """Build a PolicyDocument from an ordered list of content hashes.

    Clause text is derived from the hash rather than from the ordinal, so a clause
    that "moves" really is byte-identical content at a new position. The manifest
    never stores or compares text, but keeping the two consistent means these
    fixtures stay honest if that ever changes.
    """
    clauses = [
        Clause(
            clause_id=f"{doc_slug}:{ordinal:03d}:{h}",
            doc_slug=doc_slug,
            ordinal=ordinal,
            text=f"Clause body for content {h}.",
            content_hash=h,
            heading_path=["Acme Retail", "4. Return window"],
            token_estimate=12,
        )
        for ordinal, h in enumerate(hashes, start=1)
    ]
    return PolicyDocument(
        doc_slug=doc_slug,
        source=source,
        policy_version=policy_version(doc_slug, hashes),
        fetched_at=fetched_at,
        clauses=clauses,
        corpus_role=corpus_role,
        is_holdout=is_holdout,
    )


def baseline_of(doc: PolicyDocument) -> dict:
    """The manifest entry a prior run would have committed for `doc`."""
    return fingerprint_document(doc)


class TestTheRealBaselineIsNeverTouched:
    def test_manifest_path_is_the_committed_location(self):
        """Tripwire. If this constant moves, every test below is writing somewhere
        other than where the gate reads, and would stop proving anything."""
        assert MANIFEST_PATH == Path("policies/.clauseguard/manifest.json")


class TestFingerprintCarriesNoClauseText:
    """DESIGN.md 7.1: ship "content hashes and the fetcher - not the policy
    corpus". The manifest is committed, so text here would republish the corpus."""

    def test_no_clause_text_appears_anywhere_in_the_fingerprint(self):
        doc = make_doc([H1, H2, H3])
        dumped = json.dumps(fingerprint_document(doc))
        for clause in doc.clauses:
            assert clause.text not in dumped

    def test_no_entry_has_a_text_key(self):
        fp = fingerprint_document(make_doc([H1, H2]))
        assert all("text" not in entry for entry in fp["clauses"])

    def test_it_keeps_what_the_diff_needs(self):
        fp = fingerprint_document(make_doc([H1, H2]))
        assert [e["content_hash"] for e in fp["clauses"]] == [H1, H2]
        assert [e["ordinal"] for e in fp["clauses"]] == [1, 2]
        assert all(e["clause_id"] for e in fp["clauses"])

    def test_it_keeps_provenance_and_role(self):
        fp = fingerprint_document(make_doc([H1], corpus_role="real"))
        assert fp["corpus_role"] == "real"
        assert fp["source"] == "policies/acme-refunds.md"
        assert fp["policy_version"].startswith("sha256:")
        assert fp["content_fetched_at"] == T0.isoformat()

    def test_heading_path_survives_as_a_list(self):
        """Stored because the gate's report names the section a change landed in.
        A tuple would not survive the JSON round trip as a tuple."""
        fp = fingerprint_document(make_doc([H1]))
        assert fp["clauses"][0]["heading_path"] == [
            "Acme Retail",
            "4. Return window",
        ]


class TestLoadManifest:
    def test_a_missing_file_is_a_first_run_not_an_error(self, tmp_path):
        loaded = load_manifest(tmp_path / "nope.json")
        assert loaded == {"schema_version": SCHEMA_VERSION, "documents": {}}

    @pytest.mark.parametrize("placeholder", ["{}", "", "   \n"])
    def test_the_scaffolded_placeholder_is_tolerated(self, tmp_path, placeholder):
        """Step 0 committed `{}`. Refusing it would block the first real run."""
        p = tmp_path / "manifest.json"
        p.write_text(placeholder, encoding="utf-8")
        assert load_manifest(p)["documents"] == {}

    def test_a_version_mismatch_is_refused_loudly(self, tmp_path):
        """Silently misreading an old layout would report every clause changed and
        trigger a full re-extraction - expensive and misleading."""
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps({"schema_version": 999, "documents": {}}))
        with pytest.raises(ValueError, match="schema_version"):
            load_manifest(p)

    def test_a_manifest_without_documents_still_loads(self, tmp_path):
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps({"schema_version": SCHEMA_VERSION}))
        assert load_manifest(p)["documents"] == {}

    def test_corrupt_json_is_not_silently_swallowed(self, tmp_path):
        p = tmp_path / "manifest.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_manifest(p)


class TestWriteManifest:
    def test_round_trips(self, tmp_path):
        doc = make_doc([H1, H2])
        p = write_manifest({doc.doc_slug: baseline_of(doc)}, tmp_path / "m.json")
        assert load_manifest(p)["documents"]["acme-refunds"] == baseline_of(doc)

    def test_two_writes_are_byte_identical(self, tmp_path):
        """Without sorted keys and a fixed indent the committed file churns on dict
        ordering, and its git diff stops being readable evidence of what changed."""
        doc = make_doc([H1, H2, H3])
        a = write_manifest({"acme-refunds": baseline_of(doc)}, tmp_path / "a.json")
        b = write_manifest({"acme-refunds": baseline_of(doc)}, tmp_path / "b.json")
        assert a.read_bytes() == b.read_bytes()

    def test_it_ends_with_a_newline(self, tmp_path):
        doc = make_doc([H1])
        p = write_manifest({"acme-refunds": baseline_of(doc)}, tmp_path / "m.json")
        assert p.read_text(encoding="utf-8").endswith("\n")

    def test_it_creates_the_parent_directory(self, tmp_path):
        doc = make_doc([H1])
        p = write_manifest({"a": baseline_of(doc)}, tmp_path / "deep" / "m.json")
        assert p.is_file()

    def test_it_stamps_the_schema_version(self, tmp_path):
        p = write_manifest({}, tmp_path / "m.json")
        assert json.loads(p.read_text())["schema_version"] == SCHEMA_VERSION


class TestFirstSighting:
    def test_no_baseline_means_every_clause_is_added(self):
        diff = diff_document(None, make_doc([H1, H2, H3]))
        assert diff.is_new_document
        assert len(diff.added) == 3
        assert diff.changed == [] and diff.moved == [] and diff.removed == []

    def test_a_first_sighting_is_not_clean(self):
        assert not diff_document(None, make_doc([H1])).is_clean

    def test_everything_needs_extraction(self):
        diff = diff_document(None, make_doc([H1, H2, H3]))
        assert len(diff.needs_extraction) == 3

    def test_an_empty_dict_baseline_is_also_a_first_sighting(self):
        """`load_manifest` returns `{}` for a doc it has never seen."""
        assert diff_document({}, make_doc([H1])).is_new_document


class TestUnchanged:
    def test_an_identical_document_is_entirely_unchanged(self):
        doc = make_doc([H1, H2, H3])
        diff = diff_document(baseline_of(doc), doc)
        assert len(diff.unchanged) == 3
        assert diff.is_clean
        assert diff.needs_extraction == []

    def test_a_refetch_with_a_new_timestamp_is_still_clean(self):
        """The gate's question is "did the policy change", not "did we look again"."""
        before = make_doc([H1, H2], fetched_at=T0)
        after = make_doc([H1, H2], fetched_at=T1)
        assert diff_document(baseline_of(before), after).is_clean

    def test_summary_reads_correctly(self):
        doc = make_doc([H1, H2])
        assert diff_document(baseline_of(doc), doc).summary() == (
            "acme-refunds: 2 unchanged, 0 moved, 0 changed, 0 added, 0 removed"
        )


class TestEditInPlace:
    """The under-reporting guard. DESIGN.md 2 traces exactly this: ordinal 014 has
    a new hash, everything else unchanged, no LLM call for the rest."""

    def test_one_edited_clause_is_reported_as_changed(self):
        before = make_doc([H1, H2, H3])
        after = make_doc([H1, EDITED, H3])
        diff = diff_document(baseline_of(before), after)
        assert len(diff.changed) == 1
        assert len(diff.unchanged) == 2
        assert diff.added == [] and diff.removed == [] and diff.moved == []

    def test_the_change_carries_both_hashes_and_the_ordinal(self):
        before = make_doc([H1, H2, H3])
        after = make_doc([H1, EDITED, H3])
        change = diff_document(baseline_of(before), after).changed[0]
        assert (change.ordinal, change.old_hash, change.new_hash) == (2, H2, EDITED)

    def test_an_edit_is_not_clean_and_needs_extraction(self):
        before = make_doc([H1, H2])
        after = make_doc([H1, EDITED])
        diff = diff_document(baseline_of(before), after)
        assert not diff.is_clean
        assert diff.needs_extraction == ["acme-refunds:002:9e9e9e9e"]

    def test_only_the_edited_clause_needs_extraction(self):
        """The cost claim: a one-clause edit must not re-extract the document."""
        before = make_doc([H1, H2, H3, H4])
        after = make_doc([H1, EDITED, H3, H4])
        assert len(diff_document(baseline_of(before), after).needs_extraction) == 1

    def test_every_clause_edited_is_reported_as_changed_not_added(self):
        before = make_doc([H1, H2])
        after = make_doc(["aaaaaaa1", "bbbbbbb2"])
        diff = diff_document(baseline_of(before), after)
        assert len(diff.changed) == 2
        assert diff.added == [] and diff.removed == []


class TestInsertionIsCheap:
    """The over-reporting guard, and the reason `moved` exists at all."""

    def test_inserting_at_the_top_yields_one_add_and_the_rest_moves(self):
        before = make_doc([H1, H2, H3])
        after = make_doc([H4, H1, H2, H3])
        diff = diff_document(baseline_of(before), after)
        assert len(diff.added) == 1
        assert len(diff.moved) == 3
        assert diff.changed == [] and diff.removed == []

    def test_only_the_inserted_clause_costs_an_llm_call(self):
        before = make_doc([H1, H2, H3])
        after = make_doc([H4, H1, H2, H3])
        assert diff_document(baseline_of(before), after).needs_extraction == [
            "acme-refunds:001:44444444"
        ]

    def test_a_move_records_where_the_content_went(self):
        before = make_doc([H1, H2])
        after = make_doc([H4, H1, H2])
        diff = diff_document(baseline_of(before), after)
        moves = {m.content_hash: (m.old_ordinal, m.new_ordinal) for m in diff.moved}
        assert moves == {H1: (1, 2), H2: (2, 3)}

    def test_appending_at_the_end_moves_nothing(self):
        before = make_doc([H1, H2])
        after = make_doc([H1, H2, H3])
        diff = diff_document(baseline_of(before), after)
        assert len(diff.unchanged) == 2
        assert len(diff.added) == 1
        assert diff.moved == []

    def test_a_pure_reorder_is_all_moves(self):
        before = make_doc([H1, H2])
        after = make_doc([H2, H1])
        diff = diff_document(baseline_of(before), after)
        assert len(diff.moved) == 2
        assert diff.changed == [] and diff.added == [] and diff.removed == []

    def test_a_reorder_needs_no_extraction_but_is_still_a_layout_change(self):
        """`is_clean` tolerates moves - no policy text changed - but the clause IDs
        moved, so probe invalidation downstream still has work to do."""
        before = make_doc([H1, H2])
        diff = diff_document(baseline_of(before), make_doc([H2, H1]))
        assert diff.needs_extraction == []
        assert diff.is_clean
        assert diff.moved


class TestRemoval:
    def test_a_deleted_clause_is_reported_as_removed(self):
        before = make_doc([H1, H2, H3])
        after = make_doc([H1, H3])
        diff = diff_document(baseline_of(before), after)
        assert diff.removed == ["acme-refunds:002:22222222"]

    def test_the_survivors_are_not_reported_as_changed(self):
        before = make_doc([H1, H2, H3])
        diff = diff_document(baseline_of(before), make_doc([H1, H3]))
        assert diff.changed == []
        assert len(diff.unchanged) + len(diff.moved) == 2

    def test_a_removal_needs_no_extraction_but_is_not_clean(self):
        """Nothing new to read, but a rule now cites a clause that is gone - the
        gate must not wave that through."""
        before = make_doc([H1, H2])
        diff = diff_document(baseline_of(before), make_doc([H1]))
        assert diff.needs_extraction == []
        assert not diff.is_clean

    def test_removing_everything_reports_every_clause(self):
        before = make_doc([H1, H2, H3])
        diff = diff_document(baseline_of(before), make_doc([H1]))
        assert len(diff.removed) == 2


class TestDuplicateContent:
    """Two clauses can legitimately hash alike - a repeated boilerplate sentence.
    A move must not consume a baseline slot twice, or a clause vanishes."""

    def test_identical_duplicates_are_both_unchanged(self):
        doc = make_doc([H1, H1, H2])
        diff = diff_document(baseline_of(doc), doc)
        assert len(diff.unchanged) == 3

    def test_a_reorder_among_duplicates_does_not_double_consume(self):
        before = make_doc([H1, H1, H2])
        after = make_doc([H2, H1, H1])
        diff = diff_document(baseline_of(before), after)
        assert len(diff.unchanged) + len(diff.moved) == 3
        assert diff.added == [] and diff.removed == [] and diff.changed == []

    def test_losing_one_of_two_duplicates_reports_exactly_one_removal(self):
        before = make_doc([H1, H1])
        diff = diff_document(baseline_of(before), make_doc([H1]))
        assert len(diff.removed) == 1
        assert len(diff.unchanged) == 1


class TestDegenerateDocuments:
    def test_an_empty_document_against_an_empty_baseline_is_clean(self):
        empty = make_doc([])
        assert diff_document(baseline_of(empty), empty).is_clean

    def test_emptying_a_document_reports_every_clause_removed(self):
        before = make_doc([H1, H2])
        diff = diff_document(baseline_of(before), make_doc([]))
        assert len(diff.removed) == 2
        assert not diff.is_clean

    def test_populating_an_empty_document_reports_adds(self):
        diff = diff_document(baseline_of(make_doc([])), make_doc([H1, H2]))
        assert len(diff.added) == 2


class TestUpdateManifest:
    def test_it_writes_a_first_baseline(self, tmp_path):
        p = tmp_path / "m.json"
        doc = make_doc([H1, H2])
        update_manifest(doc, diff_document(None, doc), p)
        assert load_manifest(p)["documents"]["acme-refunds"]["clauses"][0][
            "content_hash"
        ] == H1

    def test_a_clean_refetch_preserves_the_content_timestamp(self, tmp_path):
        """Otherwise every no-op CI run dirties a committed file, and "the manifest
        changed" stops meaning "the policy changed"."""
        p = tmp_path / "m.json"
        first = make_doc([H1, H2], fetched_at=T0)
        update_manifest(first, diff_document(None, first), p)

        refetched = make_doc([H1, H2], fetched_at=T1)
        update_manifest(refetched, diff_against_manifest(refetched, p), p)

        entry = load_manifest(p)["documents"]["acme-refunds"]
        assert manifest_fetched_at(entry) == T0

    def test_a_clean_refetch_leaves_the_file_byte_identical(self, tmp_path):
        p = tmp_path / "m.json"
        first = make_doc([H1, H2], fetched_at=T0)
        update_manifest(first, diff_document(None, first), p)
        before_bytes = p.read_bytes()

        refetched = make_doc([H1, H2], fetched_at=T1)
        update_manifest(refetched, diff_against_manifest(refetched, p), p)
        assert p.read_bytes() == before_bytes

    def test_a_real_change_advances_the_timestamp(self, tmp_path):
        p = tmp_path / "m.json"
        first = make_doc([H1, H2], fetched_at=T0)
        update_manifest(first, diff_document(None, first), p)

        edited = make_doc([H1, EDITED], fetched_at=T1)
        update_manifest(edited, diff_against_manifest(edited, p), p)

        entry = load_manifest(p)["documents"]["acme-refunds"]
        assert manifest_fetched_at(entry) == T1

    def test_a_move_advances_the_timestamp(self, tmp_path):
        """A move rewrites clause IDs in the baseline, so the file did change even
        though no policy text did."""
        p = tmp_path / "m.json"
        first = make_doc([H1, H2], fetched_at=T0)
        update_manifest(first, diff_document(None, first), p)

        reordered = make_doc([H2, H1], fetched_at=T1)
        update_manifest(reordered, diff_against_manifest(reordered, p), p)

        entry = load_manifest(p)["documents"]["acme-refunds"]
        assert manifest_fetched_at(entry) == T1

    def test_updating_one_document_does_not_disturb_another(self, tmp_path):
        p = tmp_path / "m.json"
        acme = make_doc([H1], doc_slug="acme-refunds")
        globex = make_doc([H2], doc_slug="globex-returns")
        update_manifest(acme, diff_document(None, acme), p)
        update_manifest(globex, diff_document(None, globex), p)

        docs = load_manifest(p)["documents"]
        assert set(docs) == {"acme-refunds", "globex-returns"}
        assert docs["acme-refunds"]["clauses"][0]["content_hash"] == H1

    def test_the_new_baseline_diffs_clean_against_itself(self, tmp_path):
        """The property that makes the gate usable: after accepting a change, the
        next run must report nothing."""
        p = tmp_path / "m.json"
        doc = make_doc([H1, H2, H3])
        update_manifest(doc, diff_document(None, doc), p)
        assert diff_against_manifest(doc, p).is_clean


class TestDiffAgainstManifest:
    def test_an_unknown_document_is_a_first_sighting(self, tmp_path):
        p = tmp_path / "m.json"
        write_manifest({}, p)
        assert diff_against_manifest(make_doc([H1]), p).is_new_document

    def test_it_reads_the_baseline_from_disk(self, tmp_path):
        p = tmp_path / "m.json"
        before = make_doc([H1, H2])
        write_manifest({"acme-refunds": baseline_of(before)}, p)
        diff = diff_against_manifest(make_doc([H1, EDITED]), p)
        assert len(diff.changed) == 1

    def test_documents_are_matched_by_slug_not_position(self, tmp_path):
        p = tmp_path / "m.json"
        acme = make_doc([H1], doc_slug="acme-refunds")
        globex = make_doc([H2], doc_slug="globex-returns")
        write_manifest(
            {"acme-refunds": baseline_of(acme), "globex-returns": baseline_of(globex)},
            p,
        )
        assert diff_against_manifest(globex, p).is_clean
        assert diff_against_manifest(acme, p).is_clean


class TestManifestFetchedAt:
    def test_it_parses_back_to_the_original_datetime(self):
        entry = fingerprint_document(make_doc([H1], fetched_at=T0))
        assert manifest_fetched_at(entry) == T0

    def test_it_keeps_the_timezone(self):
        entry = fingerprint_document(make_doc([H1], fetched_at=T0))
        assert manifest_fetched_at(entry).tzinfo is not None
