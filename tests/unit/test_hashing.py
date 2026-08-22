"""Tests for the clause change-detection primitive (DESIGN.md 1.1).

The four cases in the Step 2 brief - same text same hash, punctuation-only diff
same hash, content change different hash, ordinal survives edits but hash does not
- are the first four classes here. The rest defend specific design decisions in
`hashing.py` that would otherwise silently regress: punctuation becoming a space
rather than being deleted, decimal points surviving, and the deliberate separation
of the hashing normaliser from the span-verification one.

Assertions compare hashes to each other rather than to literal hex, except where
the ID FORMAT is under test. Pinning literals would make every test a tripwire on
the sha256 implementation instead of on the property actually being claimed.
"""

from __future__ import annotations

import pytest

from harness.ingest.hashing import (
    CONTENT_HASH_LENGTH,
    collapse_whitespace,
    content_hash,
    make_clause_id,
    normalize,
    policy_version,
    slugify,
)

CLAUSE = "Returns must be initiated within 30 days of delivery."


class TestSameTextSameHash:
    def test_identical_text_hashes_identically(self):
        assert content_hash(CLAUSE) == content_hash(CLAUSE)

    def test_hash_does_not_depend_on_string_identity(self):
        rebuilt = "".join(list(CLAUSE))
        assert content_hash(rebuilt) == content_hash(CLAUSE)

    def test_hash_shape_is_eight_lowercase_hex(self):
        h = content_hash(CLAUSE)
        assert len(h) == CONTENT_HASH_LENGTH == 8
        assert h == h.lower()
        assert all(c in "0123456789abcdef" for c in h)


class TestWhitespaceAndPunctuationOnlyDiffs:
    """DESIGN.md 1.1: "strip punctuation-only diffs"."""

    @pytest.mark.parametrize(
        "variant",
        [
            "Returns  must   be initiated within 30 days of delivery.",
            "Returns must be initiated within 30 days of delivery.\n",
            "\tReturns must be initiated within 30 days of delivery. ",
            "Returns must be initiated within 30 days of\ndelivery.",
        ],
    )
    def test_whitespace_only_diff_keeps_the_hash(self, variant):
        assert content_hash(variant) == content_hash(CLAUSE)

    @pytest.mark.parametrize(
        "variant",
        [
            "Returns must be initiated within 30 days of delivery",  # no full stop
            "Returns must be initiated within 30 days of delivery!",
            "Returns must be initiated, within 30 days of delivery.",
            "'Returns' must be initiated within 30 days of delivery.",
            "Returns must be initiated within 30 days of delivery -- ",
        ],
    )
    def test_punctuation_only_diff_keeps_the_hash(self, variant):
        assert content_hash(variant) == content_hash(CLAUSE)

    def test_case_only_diff_keeps_the_hash(self):
        assert content_hash(CLAUSE.upper()) == content_hash(CLAUSE)

    def test_digit_group_comma_is_a_punctuation_only_diff(self):
        assert content_hash("A fee of Rs 1,000 applies.") == content_hash(
            "A fee of Rs 1000 applies."
        )

    def test_nfkc_folds_compatibility_forms(self):
        """Full-width digits and non-breaking spaces come from PDF and HTML
        extraction; the same clause from two sources must hash alike."""
        assert content_hash("within ３０ days") == content_hash("within 30 days")
        # Escapes, not literals: an invisible NBSP in source is unreviewable.
        assert content_hash("within 30\u00a0days") == content_hash("within 30 days")
        assert content_hash("within\u200930 days") == content_hash("within 30 days")


class TestContentChangeMovesTheHash:
    def test_changed_threshold_changes_the_hash(self):
        """The exact edit DESIGN.md 2 traces: 30 days becomes 7 days."""
        edited = CLAUSE.replace("30 days", "7 days")
        assert content_hash(edited) != content_hash(CLAUSE)

    def test_negation_changes_the_hash(self):
        assert content_hash("Refunds are available.") != content_hash(
            "Refunds are not available."
        )

    def test_added_sentence_changes_the_hash(self):
        assert content_hash(CLAUSE + " Clearance items are excluded.") != content_hash(
            CLAUSE
        )

    def test_word_order_change_changes_the_hash(self):
        assert content_hash("innerwear and swimwear") != content_hash(
            "swimwear and innerwear"
        )


class TestOrdinalSurvivesEditsButHashDoesNot:
    """The pairing that lets the gate say "clause 14 changed" (DESIGN.md 1.1, 6.2)."""

    def test_edit_in_place_keeps_the_ordinal_and_moves_the_hash(self):
        before = make_clause_id("acme-refunds", 14, content_hash(CLAUSE))
        after = make_clause_id(
            "acme-refunds", 14, content_hash(CLAUSE.replace("30", "7"))
        )
        assert before != after
        slug_b, ord_b, hash_b = before.split(":")
        slug_a, ord_a, hash_a = after.split(":")
        assert (slug_b, ord_b) == (slug_a, ord_a) == ("acme-refunds", "014")
        assert hash_b != hash_a

    def test_same_content_at_a_new_ordinal_keeps_the_hash(self):
        """A clause inserted above shifts position without changing content, so
        the hash must be reusable to recognise the move without an LLM call."""
        h = content_hash(CLAUSE)
        assert make_clause_id("acme-refunds", 14, h).endswith(h)
        assert make_clause_id("acme-refunds", 15, h).endswith(h)


class TestPunctuationBecomesSpaceNotDeleted:
    """Deleting punctuation fuses tokens, and fused tokens collide."""

    def test_decimal_percentages_do_not_collide(self):
        """The motivating case: deleting the point makes "7.5%" into "75"."""
        assert normalize("A fee of 7.5% applies.") != normalize("A fee of 75% applies.")
        assert content_hash("A fee of 7.5% applies.") != content_hash(
            "A fee of 75% applies."
        )

    def test_decimal_point_is_preserved_as_content(self):
        assert "7.5" in normalize("a fee of 7.5% applies")

    def test_sentence_boundary_does_not_fuse_words(self):
        assert normalize("ends here.Starts there") == "ends here starts there"

    def test_hyphenated_and_unhyphenated_forms_fold_together(self):
        assert normalize("end-of-season sale") == "end of season sale"

    def test_currency_symbols_survive(self):
        """Sc, not P. Losing the symbol would make two fee clauses hash alike."""
        assert normalize("a fee of $5") != normalize("a fee of 5")
        assert "$" in normalize("a fee of $5")

    def test_percent_sign_is_treated_as_noise(self):
        """Documented, accepted trade-off: it buys "20 %" == "20%"."""
        assert normalize("a fee of 20%") == normalize("a fee of 20")


class TestNormalizeIsNotForSpanVerification:
    """Commitment C2 permits whitespace normalisation only (DESIGN.md 4.1 L2)."""

    def test_collapse_whitespace_preserves_punctuation(self):
        text = "Refunds are not available, except for defects."
        messy = "  Refunds are  not\navailable, except for defects.  "
        assert collapse_whitespace(messy) == text

    def test_the_two_normalisers_disagree_on_punctuation(self):
        """If span checks used `normalize`, these two different promises would
        verify against each other."""
        with_comma = "refunds are not available, except for defects"
        without = "refunds are not available except for defects"
        assert normalize(with_comma) == normalize(without)
        assert collapse_whitespace(with_comma) != collapse_whitespace(without)

    def test_collapse_whitespace_is_idempotent(self):
        once = collapse_whitespace("  a   b \n c ")
        assert collapse_whitespace(once) == once == "a b c"

    def test_normalize_is_idempotent(self):
        once = normalize(CLAUSE)
        assert normalize(once) == once


class TestMakeClauseId:
    def test_matches_the_design_md_example_shape(self):
        assert make_clause_id("acme-refunds", 14, "a3f91c22") == (
            "acme-refunds:014:a3f91c22"
        )

    @pytest.mark.parametrize(
        "ordinal,expected", [(1, "001"), (9, "009"), (47, "047"), (999, "999")]
    )
    def test_ordinal_is_zero_padded_to_three_digits(self, ordinal, expected):
        assert make_clause_id("d", ordinal, "abcdef01").split(":")[1] == expected

    def test_four_digit_ordinal_is_not_truncated(self):
        """Padding is a minimum width, not a cap; a 1000-clause document must
        still produce a unique, parseable ID."""
        assert make_clause_id("d", 1000, "abcdef01") == "d:1000:abcdef01"

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_ordinal_is_rejected(self, bad):
        with pytest.raises(ValueError, match="1-based"):
            make_clause_id("d", bad, "abcdef01")

    def test_id_has_exactly_three_colon_separated_parts(self):
        assert len(make_clause_id("acme-refunds", 14, "a3f91c22").split(":")) == 3


class TestPolicyVersion:
    def test_shape_is_prefixed_sha256(self):
        v = policy_version("acme-refunds", ["aaaaaaaa", "bbbbbbbb"])
        assert v.startswith("sha256:")
        assert len(v.split(":", 1)[1]) == 64

    def test_same_clause_set_same_version(self):
        assert policy_version("d", ["aaaaaaaa", "bbbbbbbb"]) == policy_version(
            "d", ["aaaaaaaa", "bbbbbbbb"]
        )

    def test_a_changed_clause_moves_the_version(self):
        assert policy_version("d", ["aaaaaaaa", "bbbbbbbb"]) != policy_version(
            "d", ["aaaaaaaa", "cccccccc"]
        )

    def test_reordering_moves_the_version(self):
        """Order is semantic: precedence and cross-references depend on it."""
        assert policy_version("d", ["aaaaaaaa", "bbbbbbbb"]) != policy_version(
            "d", ["bbbbbbbb", "aaaaaaaa"]
        )

    def test_insertion_moves_the_version(self):
        assert policy_version("d", ["aaaaaaaa"]) != policy_version(
            "d", ["aaaaaaaa", "bbbbbbbb"]
        )

    def test_different_documents_version_differently(self):
        """Two merchants with coincidentally identical clause sets must not share
        a policy_version, or audit rows from one would appear comparable to the
        other's."""
        assert policy_version("acme", ["aaaaaaaa"]) != policy_version(
            "globex", ["aaaaaaaa"]
        )

    def test_empty_document_is_versionable(self):
        assert policy_version("d", []).startswith("sha256:")


class TestSlugify:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("acme-refunds", "acme-refunds"),
            ("Acme Refunds", "acme-refunds"),
            ("ACME_Refunds", "acme_refunds"),
            ("acme.example", "acme.example"),
            ("Acme  Refunds  Policy", "acme-refunds-policy"),
        ],
    )
    def test_produces_valid_doc_slugs(self, raw, expected):
        assert slugify(raw) == expected

    def test_colons_are_removed(self):
        """A colon in a slug would break clause ID parsing outright."""
        assert ":" not in slugify("https://acme.example/returns")

    @pytest.mark.parametrize("raw", ["-leading", "_leading", ".leading", "!!!"])
    def test_result_always_opens_with_an_alphanumeric(self, raw):
        assert slugify(raw)[0].isalnum()

    def test_accents_are_folded_to_ascii(self):
        assert slugify("Réfunds") == "refunds"
