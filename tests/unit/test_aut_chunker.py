"""STEP 4: aut-naive's chunker - the only part of the AUT provable without Docker.

DESIGN.md 1.4 gives aut-naive "top-k=3 chunk retrieval". Everything downstream of this
file is a model call, so this is where determinism can actually be asserted: a frozen
agent that chunked its corpus differently on each rebuild would not be frozen in any sense
that matters, since top-k would move without the SHA moving.

The other thing tested here is the *separation*. aut-naive must not know the harness has an
addressable clause model (DESIGN.md 1.4, "no shared imports"), so there is a test that its
chunk boundaries and chunk ids genuinely do not line up with `clause_id`s. Retrieval
landing neatly on clause units would be a strawman in the opposite direction from the usual
worry - an AUT flattered by a benchmark built from the same segmentation.

aut-naive is loaded by path because `aut-naive` contains a hyphen and is therefore not a
legal Python module name. That is deliberate (see TestTheSeparationIsStructural): the
harness *cannot* import it by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AUT_DIR = Path(__file__).resolve().parents[2] / "aut-naive"
if str(AUT_DIR) not in sys.path:
    sys.path.insert(0, str(AUT_DIR))

from chunker import (  # noqa: E402  - path insertion must precede the import
    CHUNK_CHARS,
    OVERLAP_CHARS,
    SNAP_WINDOW,
    Chunk,
    chunk_text,
    load_corpus,
)

CORPUS_DIR = AUT_DIR / "corpus"


def body(words: int, *, word: str = "policy") -> str:
    return " ".join(f"{word}{i:04d}" for i in range(words))


class TestTheWindowsCoverTheSource:
    """Offsets must be truthful: a chunk claiming [start:end] has to be that slice."""

    def test_every_chunk_is_the_slice_it_claims(self):
        text = body(400)
        for chunk in chunk_text(text, doc_id="d"):
            assert text[chunk.start : chunk.end] == chunk.text

    def test_a_short_document_is_one_chunk(self):
        text = "Refunds are available within 30 days of delivery."
        chunks = chunk_text(text, doc_id="d")
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert (chunks[0].start, chunks[0].end) == (0, len(text))

    def test_no_character_of_content_is_lost(self):
        """Union of the windows must cover the whole document."""
        text = body(500)
        covered = bytearray(len(text))
        for chunk in chunk_text(text, doc_id="d"):
            for i in range(chunk.start, chunk.end):
                covered[i] = 1
        gaps = [i for i, seen in enumerate(covered) if not seen and not text[i].isspace()]
        assert gaps == []

    def test_windows_overlap_rather_than_abut(self):
        chunks = chunk_text(body(600), doc_id="d")
        assert len(chunks) > 2
        for earlier, later in zip(chunks, chunks[1:]):
            assert later.start < earlier.end

    def test_the_overlap_is_the_configured_size(self):
        chunks = chunk_text(body(600), doc_id="d", chunk_chars=500, overlap_chars=100)
        for earlier, later in zip(chunks, chunks[1:]):
            assert earlier.end - later.start == 100

    def test_char_len_matches_the_text(self):
        for chunk in chunk_text(body(300), doc_id="d"):
            assert chunk.char_len == len(chunk.text)


class TestChunkIdsAreOwnAddressing:
    def test_ids_are_sequential_and_zero_padded(self):
        chunks = chunk_text(body(400), doc_id="acme-refunds")
        assert chunks[0].chunk_id == "acme-refunds#0001"
        assert [c.ordinal for c in chunks] == list(range(1, len(chunks) + 1))

    def test_the_id_format_is_not_the_harness_clause_id_format(self):
        """`doc#0001`, not `doc:001:a3f91c22`. Different addressing, on purpose."""
        chunk = chunk_text("Some policy text about refunds.", doc_id="acme-refunds")[0]
        assert "#" in chunk.chunk_id
        assert chunk.chunk_id.count(":") == 0

    def test_the_doc_id_is_carried(self):
        chunk = chunk_text("Some policy text.", doc_id="acme-refunds")[0]
        assert chunk.doc_id == "acme-refunds"

    def test_a_chunk_is_immutable(self):
        chunk = chunk_text("Some policy text.", doc_id="d")[0]
        with pytest.raises((AttributeError, TypeError)):
            chunk.text = "tampered"  # type: ignore[misc]


class TestSnappingToWhitespace:
    """Fixed-size windows, but not mid-word when a space is close by (DESIGN.md 7.3)."""

    def test_a_boundary_lands_on_whitespace_when_one_is_in_reach(self):
        """The precise claim: for every non-final window, the character the window
        stopped before is whitespace, so no token is split."""
        text = body(400)
        chunks = chunk_text(text, doc_id="d", chunk_chars=200, overlap_chars=20)
        assert len(chunks) > 5
        for chunk in chunks[:-1]:
            assert text[chunk.end].isspace()

    def test_the_snap_shortens_the_window_rather_than_extending_it(self):
        chunks = chunk_text(body(400), doc_id="d", chunk_chars=200, overlap_chars=20)
        assert all(c.char_len <= 200 for c in chunks)

    def test_a_word_longer_than_the_snap_window_is_cut_hard(self):
        """Falling back to a mid-word cut is the documented last resort, not a hang."""
        text = "x" * 5000
        chunks = chunk_text(text, doc_id="d", chunk_chars=800, overlap_chars=150)
        assert chunks[0].char_len == 800
        assert chunks[1].start == 650
        assert "".join(c.text for c in chunks).count("x") >= 5000

    def test_no_whitespace_anywhere_still_terminates(self):
        chunks = chunk_text("y" * 3000, doc_id="d", chunk_chars=100, overlap_chars=99)
        assert 1 < len(chunks) < 4000

    def test_snapping_never_crosses_back_past_the_window_start(self):
        text = " " + "z" * 2000
        chunks = chunk_text(text, doc_id="d", chunk_chars=120, overlap_chars=10)
        for chunk in chunks:
            assert chunk.end > chunk.start


class TestWhitespaceOnlyWindowsAreDropped:
    def test_blank_input_yields_nothing(self):
        assert chunk_text("", doc_id="d") == []
        assert chunk_text("   \n\n\t  ", doc_id="d") == []

    def test_a_blank_run_does_not_consume_an_ordinal(self):
        text = "Refunds within 30 days." + "\n" * 2000 + "Innerwear is excluded."
        chunks = chunk_text(text, doc_id="d", chunk_chars=100, overlap_chars=10)
        assert all(c.text.strip() for c in chunks)
        assert [c.ordinal for c in chunks] == list(range(1, len(chunks) + 1))


class TestTheArgumentsAreChecked:
    def test_overlap_equal_to_chunk_size_is_refused(self):
        with pytest.raises(ValueError, match="cannot advance"):
            chunk_text("text", doc_id="d", chunk_chars=100, overlap_chars=100)

    def test_overlap_larger_than_chunk_size_is_refused(self):
        with pytest.raises(ValueError, match="cannot advance"):
            chunk_text("text", doc_id="d", chunk_chars=100, overlap_chars=101)

    def test_a_non_positive_chunk_size_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            chunk_text("text", doc_id="d", chunk_chars=0)

    def test_a_negative_overlap_is_refused(self):
        with pytest.raises(ValueError, match="must not be negative"):
            chunk_text("text", doc_id="d", overlap_chars=-1)


class TestDeterminism:
    """A frozen agent whose top-k moves between rebuilds is not frozen."""

    def test_the_same_text_chunks_identically(self):
        text = body(500)
        assert chunk_text(text, doc_id="d") == chunk_text(text, doc_id="d")

    def test_the_frozen_defaults_are_the_documented_ones(self):
        assert (CHUNK_CHARS, OVERLAP_CHARS, SNAP_WINDOW) == (800, 150, 60)

    def test_a_one_character_edit_changes_only_nearby_chunks(self):
        """The property that makes re-freezing cheap to reason about."""
        text = body(500)
        edited = text[:50] + "X" + text[51:]
        before = chunk_text(text, doc_id="d")
        after = chunk_text(edited, doc_id="d")
        assert len(before) == len(after)
        assert before[0].text != after[0].text
        assert before[-1].text == after[-1].text


class TestLoadingTheBakedCorpus:
    def test_the_corpus_snapshot_is_in_the_image_context(self):
        """COPY corpus/ in the Dockerfile depends on this existing."""
        assert (CORPUS_DIR / "acme-refunds.md").is_file()

    def test_the_worked_example_chunks(self):
        chunks = load_corpus(CORPUS_DIR)
        assert len(chunks) >= 3
        assert all(c.doc_id == "acme-refunds" for c in chunks)
        assert all(c.text.strip() for c in chunks)

    def test_files_are_read_in_sorted_order(self, tmp_path):
        (tmp_path / "b.md").write_text(body(200), encoding="utf-8")
        (tmp_path / "a.md").write_text(body(200), encoding="utf-8")
        doc_ids = [c.doc_id for c in load_corpus(tmp_path)]
        assert doc_ids == sorted(doc_ids)
        assert doc_ids[0] == "a"

    def test_a_missing_directory_is_an_error_not_an_empty_index(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_corpus(tmp_path / "absent")

    def test_a_directory_with_no_policy_text_is_an_error(self, tmp_path):
        (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="no \\*.md"):
            load_corpus(tmp_path)

    def test_loading_is_deterministic(self):
        assert load_corpus(CORPUS_DIR) == load_corpus(CORPUS_DIR)


class TestTheSeparationIsStructural:
    """C3 (DESIGN.md 0) and 1.4's "no shared imports", asserted rather than trusted."""

    def test_the_aut_directory_name_is_not_importable(self):
        """A hyphen makes `import aut-naive` a syntax error, so the harness cannot
        accidentally depend on the agent it is meant to reach only over HTTP."""
        assert "-" in AUT_DIR.name

    def test_no_aut_module_imports_the_harness(self):
        offenders = []
        for path in sorted(AUT_DIR.glob("*.py")):
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and "harness" in stripped:
                    offenders.append(f"{path.name}:{lineno}: {stripped}")
        assert offenders == []

    def test_no_harness_module_reaches_into_the_aut_directory(self):
        harness = AUT_DIR.parent / "harness"
        offenders = []
        for path in sorted(harness.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "aut-naive" in text or "aut_naive" in text:
                offenders.append(str(path.relative_to(harness.parent)))
        assert offenders == []

    def test_chunk_boundaries_do_not_align_with_harness_clause_units(self):
        """If retrieval landed on clause units, the benchmark would be flattering the
        agent with its own segmentation. Overlap of zero is the claim."""
        from harness.ingest import ingest_text

        document = ingest_text(
            (CORPUS_DIR / "acme-refunds.md").read_text(encoding="utf-8"),
            doc_slug="acme-refunds",
            source="corpus/acme-refunds.md",
            corpus_role="worked_example",
        )
        clause_texts = {c.text.strip() for c in document.clauses}
        chunk_texts = {c.text.strip() for c in load_corpus(CORPUS_DIR)}
        assert clause_texts & chunk_texts == set()

    def test_a_chunk_is_a_different_shape_of_unit_than_a_clause(self):
        chunks = load_corpus(CORPUS_DIR)
        assert isinstance(chunks[0], Chunk)
        assert not hasattr(chunks[0], "clause_id")
        assert not hasattr(chunks[0], "content_hash")
