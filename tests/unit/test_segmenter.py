"""Tests for structural clause segmentation (DESIGN.md 1.1).

Two properties matter more than the rest and are tested hardest:

1. MARKDOWN SYNTAX MUST NOT REACH `Clause.text`. That string is what commitment
   C2's exact-substring span check runs against, so a stray `- ` or `**` can turn
   a correct judgment into a fabrication verdict. Markup must never be able to
   manufacture a C2 failure.
2. THE FALLBACK SPLITTER MUST NOT OVERLAP. Overlapping chunks would put one
   sentence in two clauses, so one edit moves two hashes and a span could verify
   against a clause the promise did not come from.

The fallback is exercised through an injected splitter wherever the assertion is
about wiring rather than about splitting, so these tests do not silently change
behaviour depending on whether `langchain-text-splitters` happens to be installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.ingest.segmenter import (
    MAX_CLAUSE_TOKENS,
    MIN_CLAUSE_TOKENS,
    RawSegment,
    _split_with_stdlib,
    estimate_tokens,
    length_audit,
    segment_markdown,
    split_long_block,
)

POLICY_PATH = Path(__file__).resolve().parents[2] / "policies" / "acme-refunds.md"


def texts(md: str) -> list[str]:
    return [s.text for s in segment_markdown(md)]


class TestHeadingsAreStructureNotClauses:
    def test_a_heading_does_not_become_a_clause(self):
        segs = segment_markdown("# Title\n\n## Section\n\nBody text here.\n")
        assert [s.text for s in segs] == ["Body text here."]

    def test_heading_path_is_ordered_outermost_first(self):
        segs = segment_markdown("# Doc\n\n## 4. Return window\n\nBody.\n")
        assert segs[0].heading_path == ("Doc", "4. Return window")

    def test_a_sibling_heading_pops_the_stack(self):
        md = "# Doc\n\n## A\n\nOne.\n\n## B\n\nTwo.\n"
        segs = segment_markdown(md)
        assert segs[0].heading_path == ("Doc", "A")
        assert segs[1].heading_path == ("Doc", "B")

    def test_a_deeper_heading_pushes_the_stack(self):
        md = "# Doc\n\n## A\n\nOne.\n\n### A1\n\nTwo.\n"
        segs = segment_markdown(md)
        assert segs[1].heading_path == ("Doc", "A", "A1")

    def test_a_shallower_heading_unwinds_several_levels(self):
        md = "# Doc\n\n## A\n\n### A1\n\nOne.\n\n## B\n\nTwo.\n"
        segs = segment_markdown(md)
        assert segs[0].heading_path == ("Doc", "A", "A1")
        assert segs[1].heading_path == ("Doc", "B")

    def test_closing_hashes_are_stripped_from_headings(self):
        segs = segment_markdown("## Section ##\n\nBody.\n")
        assert segs[0].heading_path == ("Section",)


class TestMarkdownSyntaxNeverReachesClauseText:
    """Commitment C2 depends on this. See the module docstring."""

    def test_bullet_markers_are_stripped(self):
        segs = segment_markdown("- Innerwear and sleepwear.\n- Swimwear.\n")
        assert [s.text for s in segs] == ["Innerwear and sleepwear.", "Swimwear."]

    @pytest.mark.parametrize("marker", ["-", "*", "+"])
    def test_every_bullet_marker_style_is_stripped(self, marker):
        assert texts(f"{marker} Innerwear.\n") == ["Innerwear."]

    @pytest.mark.parametrize("marker", ["1.", "2)", "10."])
    def test_ordered_list_markers_are_stripped(self, marker):
        assert texts(f"{marker} Innerwear.\n") == ["Innerwear."]

    def test_list_items_are_classified_as_list_items(self):
        segs = segment_markdown("- Innerwear.\n- Swimwear.\n")
        assert {s.kind for s in segs} == {"list_item"}

    def test_bold_markers_are_stripped(self):
        """A judge quoting "must be initiated" has to match the stored text."""
        assert texts("Returns **must** be initiated promptly.\n") == [
            "Returns must be initiated promptly."
        ]

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("A *strong* claim.", "A strong claim."),
            ("A _strong_ claim.", "A strong claim."),
            ("A __strong__ claim.", "A strong claim."),
            ("A `coded` claim.", "A coded claim."),
            ("See [our policy](https://a.example/p) now.", "See our policy now."),
            ("![alt text](img.png) follows.", "alt text follows."),
        ],
    )
    def test_inline_markup_is_removed_but_words_are_kept(self, raw, expected):
        assert texts(raw + "\n") == [expected]

    def test_blockquote_markers_are_stripped(self):
        assert texts("> Note: refunds take five days.\n") == [
            "Note: refunds take five days."
        ]

    def test_code_fence_markers_are_dropped_but_content_survives(self):
        assert texts("```\nrefund_window = 30\n```\n") == ["refund_window = 30"]

    def test_emphasis_markers_do_not_change_the_hash_either(self):
        """Stripping before hashing means bolding a word is a no-op for the gate."""
        from harness.ingest.hashing import content_hash

        plain = texts("Returns must be initiated promptly.\n")[0]
        bolded = texts("Returns **must** be initiated promptly.\n")[0]
        assert content_hash(plain) == content_hash(bolded)


class TestListStructure:
    def test_nested_sub_items_stay_with_their_parent(self):
        """A sub-bullet is a sub-condition; detached, it cannot be read alone -
        and the judge sees clauses in isolation (DESIGN.md 4.1)."""
        md = "- Excluded if sealed:\n    - unless unopened\n    - unless defective\n"
        segs = segment_markdown(md)
        assert len(segs) == 1
        assert "unless unopened" in segs[0].text
        assert "unless defective" in segs[0].text

    def test_sibling_top_level_items_are_separate_clauses(self):
        md = "- First item here.\n- Second item here.\n- Third item here.\n"
        assert len(segment_markdown(md)) == 3

    def test_a_wrapped_continuation_line_stays_with_its_item(self):
        md = "- A long item that wraps\n  onto a second line.\n- Another item.\n"
        segs = segment_markdown(md)
        assert len(segs) == 2
        assert segs[0].text == "A long item that wraps onto a second line."

    def test_a_blank_line_between_items_still_yields_separate_clauses(self):
        md = "- First item.\n\n- Second item.\n"
        assert len(segment_markdown(md)) == 2


class TestBlockBoundaries:
    def test_blank_lines_separate_paragraphs(self):
        assert texts("One.\n\nTwo.\n\nThree.\n") == ["One.", "Two.", "Three."]

    def test_a_soft_wrapped_paragraph_is_one_clause(self):
        assert texts("One sentence\nwrapped over lines.\n") == [
            "One sentence wrapped over lines."
        ]

    @pytest.mark.parametrize("rule", ["---", "***", "___", "- - -"])
    def test_a_horizontal_rule_separates_and_is_not_a_clause(self, rule):
        assert texts(f"One.\n{rule}\nTwo.\n") == ["One.", "Two."]

    def test_punctuation_only_blocks_are_dropped(self):
        """`Clause.text` requires non-empty content, so a block that normalises
        to nothing cannot become an addressable clause."""
        assert texts("Real clause.\n\n...\n\n???\n") == ["Real clause."]

    def test_an_empty_document_yields_no_clauses(self):
        assert segment_markdown("") == []

    def test_a_headings_only_document_yields_no_clauses(self):
        assert segment_markdown("# A\n\n## B\n\n### C\n") == []

    def test_table_rows_are_joined_and_the_delimiter_row_is_dropped(self):
        md = "| Category | Window |\n| --- | --- |\n| Clearance | 7 days |\n"
        segs = segment_markdown(md)
        assert len(segs) == 1
        assert segs[0].kind == "table"
        assert "Clearance" in segs[0].text
        assert "---" not in segs[0].text


class TestLongBlockFallback:
    def test_a_block_under_the_cap_is_not_split(self):
        called: list[str] = []

        def spy(text: str) -> list[str]:
            called.append(text)
            return [text]

        segment_markdown("Short clause.\n", long_block_splitter=spy)
        assert called == []

    def test_a_block_over_the_cap_goes_to_the_splitter(self):
        long_text = "word " * (MAX_CLAUSE_TOKENS * 2)
        segs = segment_markdown(
            long_text + "\n", long_block_splitter=lambda t: ["piece one", "piece two"]
        )
        assert [s.text for s in segs] == ["piece one", "piece two"]

    def test_split_pieces_are_marked_as_split(self):
        long_text = "word " * (MAX_CLAUSE_TOKENS * 2)
        segs = segment_markdown(
            long_text + "\n", long_block_splitter=lambda t: ["a piece"]
        )
        assert [s.kind for s in segs] == ["split"]

    def test_empty_pieces_from_a_splitter_are_dropped(self):
        long_text = "word " * (MAX_CLAUSE_TOKENS * 2)
        segs = segment_markdown(
            long_text + "\n", long_block_splitter=lambda t: ["kept", "  ", "..."]
        )
        assert [s.text for s in segs] == ["kept"]

    def test_split_pieces_inherit_the_heading_path(self):
        long_text = "word " * (MAX_CLAUSE_TOKENS * 2)
        segs = segment_markdown(
            f"# Doc\n\n## S\n\n{long_text}\n",
            long_block_splitter=lambda t: ["a", "b"],
        )
        assert all(s.heading_path == ("Doc", "S") for s in segs)

    def test_the_stdlib_splitter_does_not_overlap(self):
        """The correctness requirement: reassembling the pieces reproduces the
        input exactly once, so no sentence lives in two clauses."""
        text = " ".join(f"Sentence number {i} is here." for i in range(300))
        pieces = _split_with_stdlib(text)
        assert len(pieces) > 1
        assert " ".join(pieces) == text

    def test_the_stdlib_splitter_respects_the_cap(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(300))
        assert all(estimate_tokens(p) <= MAX_CLAUSE_TOKENS for p in _split_with_stdlib(text))

    def test_a_single_oversized_sentence_is_still_broken_up(self):
        """No sentence boundary to split on, so the character fallback must fire."""
        text = "word " * (MAX_CLAUSE_TOKENS * 3)
        pieces = _split_with_stdlib(text)
        assert len(pieces) > 1
        assert all(estimate_tokens(p) <= MAX_CLAUSE_TOKENS for p in pieces)

    def test_split_long_block_respects_the_cap_whichever_backend_runs(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(300))
        assert all(estimate_tokens(p) <= MAX_CLAUSE_TOKENS for p in split_long_block(text))


class TestLengthAudit:
    def test_estimate_tokens_is_never_zero_for_non_empty_text(self):
        assert estimate_tokens("a") >= 1

    def test_estimate_tokens_ignores_layout(self):
        assert estimate_tokens("a b") == estimate_tokens("a   \n b")

    def test_audit_buckets_add_up(self):
        segs = [
            RawSegment(text="x" * 4),  # 1 token
            RawSegment(text="x " * MIN_CLAUSE_TOKENS * 2),
            RawSegment(text="x " * MAX_CLAUSE_TOKENS * 3),
        ]
        audit = length_audit(segs)
        assert audit["count"] == 3
        assert audit["under_min"] + audit["in_band"] + audit["over_max"] == 3
        assert audit["under_min"] == 1
        assert audit["over_max"] == 1

    def test_audit_of_nothing_does_not_divide_by_zero(self):
        assert length_audit([]) == {
            "count": 0,
            "under_min": 0,
            "in_band": 0,
            "over_max": 0,
            "min": 0,
            "max": 0,
        }

    def test_audit_never_merges_segments(self):
        """Reporting, not repairing. Fusing a short item into its neighbour would
        make editing one item churn the other's clause hash."""
        segs = [RawSegment(text="Short."), RawSegment(text="Also short.")]
        assert length_audit(segs)["count"] == 2


class TestTheRealPolicyDocument:
    """Segmentation of the committed worked example must stay stable, because its
    clause IDs are referenced by fixtures, probes and the manifest baseline."""

    # staticmethod, not an instance method: a class-scoped fixture runs once per
    # class while each test gets a fresh instance, so pytest 10 removes the
    # instance-method form outright. These tests take `segments` as a parameter
    # rather than reading `self`, so dropping `self` is the whole fix.
    @staticmethod
    @pytest.fixture(scope="class")
    def segments():
        return segment_markdown(POLICY_PATH.read_text(encoding="utf-8"))

    def test_the_worked_example_exists(self):
        assert POLICY_PATH.is_file()

    def test_clause_count_is_stable(self, segments):
        assert len(segments) == 20

    def test_no_clause_is_empty(self, segments):
        assert all(s.text.strip() for s in segments)

    def test_no_clause_text_starts_with_a_list_marker(self, segments):
        assert not [s.text for s in segments if s.text[:2] in ("- ", "* ", "+ ")]

    def test_no_clause_text_contains_a_heading_marker(self, segments):
        assert not [s for s in segments if s.text.lstrip().startswith("#")]

    def test_every_clause_carries_a_heading_path(self, segments):
        assert all(s.heading_path for s in segments)

    def test_no_clause_exceeds_the_upper_bound(self, segments):
        assert length_audit(segments)["over_max"] == 0

    def test_most_clauses_are_in_the_target_band(self, segments):
        """Short clauses are tolerated - the excluded-category bullets are
        genuinely one line each - but the bulk must sit in DESIGN.md 1.1's band,
        or the extractor is being handed fragments without enough context."""
        audit = length_audit(segments)
        assert audit["in_band"] >= 0.75 * audit["count"]

    def test_the_thirty_day_window_is_addressable(self, segments):
        matches = [s for s in segments if "within 30 days of delivery" in s.text]
        assert len(matches) == 1

    def test_the_seven_day_clearance_window_is_a_separate_clause(self, segments):
        """The cross-clause conflict DESIGN.md 3.2 strategy 8 probes needs the two
        windows to live in different clauses."""
        thirty = [s for s in segments if "within 30 days of delivery" in s.text]
        seven = [s for s in segments if "within 7 days of delivery" in s.text]
        assert len(seven) == 1
        assert thirty[0].text != seven[0].text

    def test_the_hygiene_exception_is_addressable(self, segments):
        assert [s for s in segments if "hygiene seal is intact" in s.text]
