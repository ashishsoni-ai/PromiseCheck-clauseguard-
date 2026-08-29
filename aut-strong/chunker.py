"""Corpus chunking for aut-strong. Implemented in STEP 2, alongside retrieval.

Mirrors aut-naive/chunker.py's interface exactly - `Chunk`, `chunk_text`,
`load_corpus` - because retrieval.py and app.py consume it the same way and
because a test compares the two agents' chunk id formats. What changes is the
window size, and that change is the reason this file exists at all rather than
being copied.

The ALGORITHM is held equal to aut-naive's on purpose, down to the snap-back and
the forward-progress guard: fixed-size overlapping character windows, snapped to
whitespace. DESIGN.md 1.4 varies retrieval depth and adds a reranker; it does not
say aut-strong segments differently. Switching to heading-aware or clause-aware
chunking here would land retrieval on self-contained clauses almost every time,
which flatters this agent AND quietly reimplements the harness segmenter - the
separation these directories exist to maintain. Only the two constants move.

WHY THE WINDOW MUST SHRINK (measured, not assumed)
Running aut-naive's chunker over this corpus produces exactly SEVEN chunks:
acme-refunds.md is 4,627 bytes - 4,625 characters, the difference being one
3-byte em-dash, and `chunk_chars` counts characters - and aut-naive uses
CHUNK_CHARS=800 with OVERLAP_CHARS=150. DESIGN.md 1.4 specifies k=8 for
aut-strong. Eight over seven is the whole document, so at aut-naive's window size
the retrieval half of 1.4 degenerates: nothing is ever excluded, so nothing can be
recovered by ranking it higher, and "reranking fixed the retrieval miss" becomes a
claim no run can falsify. Worse, it would not generalise - a reviewer can point out
that an agent handed its entire policy is not doing retrieval, and the finding would
not transfer to a corpus of realistic size.

THE CHOSEN WINDOW, AND THE MEASUREMENT BEHIND IT
CHUNK_CHARS=500 with OVERLAP_CHARS=300 yields exactly TWENTY-TWO chunks, so k=8 is
a selection of 36% of them. Three quantities were swept (chunk_chars 250-800 x
overlap 100-300) and they pull against each other:

  - SURVIVAL. Overlap exists so a clause split across a boundary survives intact in
    at least one window. Ten spans in this corpus were labelled PROTECTED - each one
    a sentence or adjacent pair where splitting separates an entitlement from the
    condition limiting it, which is what every over-promise in docs/results.md
    actually is. The longest is 288 characters: the excluded-categories header plus
    its three bullets. Consecutive windows share exactly `overlap` characters by
    construction, since window i+1 begins at end_i - overlap, so a span of length L
    is guaranteed to sit inside some window if and only if L <= overlap + 1. That
    makes 288 the binding number, and it is why the overlap is 300 rather than
    something proportional to aut-naive's 150.

    This was verified rather than argued: sweeping the document through every
    offset in one stride, 450/200 and 500/260 keep all ten spans only because of
    where the text happens to sit - shift by 11 characters and the excluded-
    categories span splits - while overlap >= 287 holds at every offset. Positional
    luck is exactly the wrong thing to rely on here, because a merchant editing one
    clause shifts every offset after it, and stale-policy drift is what this harness
    exists to catch. NOTE THAT aut-naive's 800/150 ALSO KEEPS ALL TEN INTACT, again
    by position. Its flagship failure was therefore NOT a chunking failure: the
    hygiene-seal carve-out existed whole inside a single window. What that rules out
    is SEGMENTATION, and nothing more. It does not relocate the failure onto ranking,
    which an earlier version of this paragraph asserted: aut-naive returns
    `retrieved_chunk_ids` and nothing in harness/ reads it, so no run distinguishes a
    ranking failure from a "had the text and ignored it" reasoning failure. See
    aut-strong/retrieval.py and docs/limitations.md.

  - SELECTIVITY CEILING. 8 * chunk_chars as a share of 4,625 characters. Read this as
    an UPPER BOUND and never as coverage: it is what eight windows could cover if they
    were disjoint, which they are not. As a ceiling it still rules configurations out,
    because at or above 100% eight windows can hold the whole policy and retrieval
    quality is unfalsifiable by volume even when the chunk count looks healthy. That
    is what rules out 600/300 (15 chunks looks fine; 8 x 600 = 4,800 > the document).

  - REDUNDANCY. High overlap makes top-8 return near-duplicate windows clustered on
    one region, which wastes the k=8 budget on the same text and is the failure mode
    that matters most here, because the clause governing a smuggled request sits
    somewhere ELSE in the document. Measured as distinct characters covered by the
    worst contiguous run of eight: 40% at 500/300, against 31% at 450/300.

500/300 is the only configuration that guarantees survival at any offset, keeps eight
windows below the document's size, and does not shred the corpus into near-duplicates.
Its selectivity ceiling is 86% (8 x 500 / 4,625, overlap ignored by construction).

An earlier version of this paragraph also claimed "78% of the document by volume is
the floor across every configuration that survives at all". That is WITHDRAWN rather
than restated, for two reasons. It was computed with the same disjointness error; and
it is not even a floor over the configurations the span guarantee admits, since an
overlap of 300 also admits windows only slightly larger than 300 - 350/300 has a
ceiling of 61% - which fail on redundancy rather than on volume. The set of
configurations the "floor" ranged over was never recorded, so the figure is not
recomputable and is gone.

STEP 2 then MEASURED the thing those ceilings were standing in for, and the honest
number is far lower. Unique characters actually reached by the reranked top 8 on the
three pre-registered probes: 55% / 51% / 53%, against 30% / 23% / 19% for aut-naive's
top 3. An earlier version of this paragraph concluded from the 86% ceiling that
"retrieval here is a test of ORDERING and of what reaches the prompt's top positions,
not of whether the model is shown the governing clause at all". That conclusion is
withdrawn - it read a disjoint-chunk upper bound as coverage. Presence is a real
constraint at this depth: one governing span-instance is a recorded FAIL because both
windows holding it ranked in the bottom 6 of 22 by cosine. What remains true is the
uncomfortable part, and it belongs in STEP 7's write-up: on a 4.6KB single-document
corpus, protecting a 288-char span forces a large overlap, which forces either a large
window or heavy duplication, so k=8 is never a small share of the policy.

Chunk ids stay in aut-naive's `f"{doc_id}#{ordinal:04d}"` form, deliberately
unlike the harness's `doc:007:hash`. If those two formats ever start looking
alike, something has leaked across the HTTP barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Chosen in STEP 2 by the sweep recorded above, not by scaling aut-naive's pair.
#: 22 chunks over this corpus; k=8 selects 36% of them.
#:
#: OVERLAP_CHARS is 300 because the longest protected span is 288 characters and
#: survival is guaranteed only while `overlap >= span_length - 1`. Do not lower it
#: to reduce duplication without re-running the offset sweep: 260 keeps every span
#: today and loses one if the policy text shifts by 11 characters, which is a bug
#: that appears months later as an unexplained over-promise.
#:
#: Frozen with the agent. Changing either value changes behaviour and invalidates
#: the aut-strong-v1 freeze.
CHUNK_CHARS = 500
OVERLAP_CHARS = 300

#: Unchanged from aut-naive: the distance to scan backwards for a sentence or
#: paragraph boundary before accepting a hard cut mid-word.
SNAP_WINDOW = 60


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable window of policy text."""

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    start: int
    end: int

    @property
    def char_len(self) -> int:
        return self.end - self.start


def _snap_back(text: str, end: int, floor: int, window: int) -> int:
    """Move `end` back to just before the last whitespace within `window` chars.

    Returns `end` unchanged if there is no whitespace to snap to, or if snapping
    would not leave forward progress past `floor`. Cutting mid-word is the fallback,
    not the norm.
    """
    lo = max(floor + 1, end - window)
    for i in range(end - 1, lo - 1, -1):
        if text[i].isspace():
            return i
    return end


def chunk_text(
    text: str,
    *,
    doc_id: str,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    snap_window: int = SNAP_WINDOW,
) -> list[Chunk]:
    """Split `text` into overlapping windows. Deterministic.

    Whitespace-only windows are dropped rather than embedded, since an empty vector
    occupies a top-k slot for nothing.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must not be negative")
    if overlap_chars >= chunk_chars:
        raise ValueError(
            f"overlap_chars ({overlap_chars}) must be smaller than chunk_chars "
            f"({chunk_chars}), otherwise chunking cannot advance"
        )

    chunks: list[Chunk] = []
    n = len(text)
    cursor = 0
    ordinal = 0

    while cursor < n:
        end = min(cursor + chunk_chars, n)
        if end < n:
            end = _snap_back(text, end, cursor, snap_window)

        window = text[cursor:end]
        if window.strip():
            ordinal += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{ordinal:04d}",
                    doc_id=doc_id,
                    ordinal=ordinal,
                    text=window,
                    start=cursor,
                    end=end,
                )
            )

        if end >= n:
            break

        # Forward progress is not optional: a step of zero would spin forever on a
        # pathological snap. The max() is the guard, not an optimisation.
        cursor = max(cursor + 1, end - overlap_chars)

    return chunks


def load_corpus(
    directory: Path | str,
    *,
    pattern: str = "*.md",
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk]:
    """Chunk every policy file in `directory`, sorted by filename.

    Sorted so the FAISS index is built in a reproducible order: the agent is frozen,
    so "same inputs, same index, same top-k" has to hold across rebuilds or the freeze
    means less than it claims.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"policy corpus directory not found: {root}")

    chunks: list[Chunk] = []
    for path in sorted(root.glob(pattern)):
        chunks.extend(
            chunk_text(
                path.read_text(encoding="utf-8"),
                doc_id=path.stem,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )
        )
    if not chunks:
        raise FileNotFoundError(f"no {pattern} files with content under {root}")
    return chunks


if __name__ == "__main__":  # pragma: no cover - eyeball check, not a test
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    got = load_corpus(target)
    print(f"{len(got)} chunk(s) from {target}")
    for c in got:
        head = " ".join(c.text.split())[:88]
        print(f"  {c.chunk_id}  [{c.start:>5}:{c.end:>5}]  {c.char_len:>4}c  {head}")
