"""Naive fixed-size chunking of policy markdown. STEP 4. Zero imports from harness/.

DESIGN.md 1.4 gives aut-naive "top-k=3 chunk retrieval" and describes it as "what a
merchant ships on a Friday". The word is *chunk*, not *clause*: this agent has no idea
the harness has an addressable clause model, and it must not.

WHY FIXED-SIZE CHARACTER WINDOWS
Two failure modes to avoid, and they pull in opposite directions (DESIGN.md 7.3, "not
looking rigged"):

  - Chunking on markdown headings would land retrieval on coherent, self-contained
    clauses almost every time. That flatters the AUT and quietly reimplements the
    harness segmenter, which is the separation this directory exists to maintain.
  - Slicing blindly at `text[i:i+800]` mid-word would degrade the embeddings for no
    reason a real deployment would accept, and a panelist would be right to call it
    sabotage.

So: fixed-size overlapping windows, snapped back to the nearest whitespace within a
small tolerance. That is approximately what every RAG tutorial does, which is the point
- it is neither tuned for this benchmark nor crippled for it. Chunk boundaries land
mid-clause and mid-list routinely, headings detach from the text they govern, and
retrieval is imperfect in the ordinary way real retrieval is imperfect.

Chunk ids are deliberately `doc#0007`, not the harness's `doc:007:a3f91c22`. The formats
are different because the addressing is unshared; if these two ever start looking alike,
something has leaked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Roughly the LangChain tutorial defaults, which is the honest reference point for
#: "what a merchant ships". Frozen with the agent - changing these changes behaviour.
CHUNK_CHARS = 800
OVERLAP_CHARS = 150

#: How far to look back for whitespace before giving up and cutting mid-word.
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
