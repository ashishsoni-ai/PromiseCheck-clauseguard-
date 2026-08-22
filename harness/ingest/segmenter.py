"""Structural clause segmentation.

DESIGN.md 1.1: "Clause segmentation is **not** naive chunking. Policy pages are
already structured - headings, numbered items, bullets. Split on structural
boundaries first (`MarkdownHeaderTextSplitter` + list-item detection), and only
fall back to a recursive splitter for wall-of-text paragraphs. Target clause
length 40-400 tokens."

WHY THE STRUCTURAL PASS IS HAND-WRITTEN
---------------------------------------
The structural pass is stdlib rather than `MarkdownHeaderTextSplitter`, for three
reasons:

1. Clause IDs are a COMMITTED baseline (`policies/.clauseguard/manifest.json`).
   If segmentation shifts because a third-party splitter changed behaviour in a
   minor release, every clause ID churns and the gate fires on a policy nobody
   edited. The change-detection primitive should not have a moving part in it.
2. That splitter does not do list-item detection, which DESIGN.md 1.1 requires
   anyway - so a hand-written line walk was needed regardless.
3. It returns headings as a `{"Header 1": ...}` metadata dict; `Clause.heading_path`
   is an ordered list. Converting one to the other is most of the work of just
   tracking the heading stack directly.

The recursive splitter from `langchain-text-splitters` IS used, for exactly the
job DESIGN.md assigns it: breaking up wall-of-text paragraphs that exceed the
clause length cap.

WHAT IS NOT A CLAUSE
Headings themselves are structure, not content; they are recorded in
`heading_path` instead. Horizontal rules and blocks that normalise to nothing are
dropped, because `Clause.text` requires non-empty content.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from harness.ingest.hashing import collapse_whitespace, normalize

#: DESIGN.md 1.1's target band. Advisory: used to audit segmentation quality, and
#: as the trigger for the recursive fallback. Not a correctness input.
MIN_CLAUSE_TOKENS = 40
MAX_CLAUSE_TOKENS = 400

#: Fallback chunk target, with headroom under MAX so that a chunk landing slightly
#: over the splitter's request still fits the cap.
FALLBACK_CHUNK_TOKENS = 320

#: Rough tokens-per-character for English prose. `tiktoken` is deliberately not a
#: dependency (it is absent from DESIGN.md's Appendix, and the estimate is only
#: ever advisory), so this is the standard ~4-chars-per-token rule of thumb.
CHARS_PER_TOKEN = 4

SegmentKind = Literal["paragraph", "list_item", "table", "split"]

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_ORDERED = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
_HRULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_TABLE_ROW = re.compile(r"^\s*\|")
_TABLE_DELIM = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
_BLOCKQUOTE = re.compile(r"^\s*>\s?")

# Inline markdown syntax, stripped from clause text. See _strip_inline_markdown.
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_CODE = re.compile(r"`([^`]+)`")
_MD_STRONG = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_MD_EMPHASIS = re.compile(r"(?<![\w*_])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*_])")


def _strip_inline_markdown(text: str) -> str:
    """Remove inline markdown syntax, keeping the words.

    THIS IS A COMMITMENT C2 REQUIREMENT, not cosmetics. `Clause.text` is the
    exact string that L2 span verification substring-matches against
    (DESIGN.md 4.1). If the stored text were "returns **must** be initiated",
    a judge quoting the clause as it reads - "returns must be initiated" - would
    fail verification, and a correct judgment would be thrown away as a
    fabrication. Markup must not be able to manufacture a C2 failure.

    Applied after block assembly and before hashing, so the emphasis markers are
    absent from both the stored text and the content hash. A merchant bolding a
    word therefore does not churn the clause ID either, which is the same
    punctuation-only-diff principle `normalize()` applies.
    """
    text = _MD_IMAGE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_CODE.sub(r"\1", text)
    text = _MD_STRONG.sub(r"\2", text)
    text = _MD_EMPHASIS.sub(r"\2", text)
    return collapse_whitespace(text)


@dataclass(frozen=True)
class RawSegment:
    """One candidate clause, before hashing and ID assignment."""

    text: str
    heading_path: tuple[str, ...] = ()
    kind: SegmentKind = "paragraph"

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.text)


def estimate_tokens(text: str) -> int:
    """Approximate token count. Advisory only; expect +/-20%."""
    return max(1, round(len(collapse_whitespace(text)) / CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# The wall-of-text fallback
# ---------------------------------------------------------------------------
def _split_with_langchain(text: str) -> list[str] | None:
    """Recursive character split via langchain-text-splitters, or None if absent.

    `chunk_overlap=0` is not a tuning choice, it is a correctness requirement.
    Overlapping chunks would put the same sentence inside two clauses, which
    means one edit changes two clause hashes, and a judge's quoted span could
    verify against a clause that is not the one the promise came from.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FALLBACK_CHUNK_TOKENS * CHARS_PER_TOKEN,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


def _split_with_stdlib(text: str) -> list[str]:
    """Sentence-greedy fallback for environments without langchain installed.

    Keeps the module importable and testable with zero third-party deps. Splits
    on sentence boundaries and packs greedily up to the chunk target.
    """
    budget = FALLBACK_CHUNK_TOKENS * CHARS_PER_TOKEN
    pieces = re.split(r"(?<=[.;:])\s+", collapse_whitespace(text))
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > budget:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)

    # A single sentence longer than the budget still has to be broken up.
    out: list[str] = []
    for chunk in chunks:
        while len(chunk) > budget:
            cut = chunk.rfind(" ", 0, budget)
            if cut <= 0:
                cut = budget
            out.append(chunk[:cut].strip())
            chunk = chunk[cut:].strip()
        if chunk:
            out.append(chunk)
    return out


def split_long_block(text: str) -> list[str]:
    """Break a block that exceeds MAX_CLAUSE_TOKENS into cap-sized pieces."""
    return _split_with_langchain(text) or _split_with_stdlib(text)


# ---------------------------------------------------------------------------
# The structural pass
# ---------------------------------------------------------------------------
@dataclass
class _Block:
    lines: list[str] = field(default_factory=list)
    heading_path: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not any(line.strip() for line in self.lines)


def _classify(lines: Sequence[str]) -> SegmentKind:
    body = [ln for ln in lines if ln.strip()]
    if not body:
        return "paragraph"
    if any(_TABLE_ROW.match(ln) for ln in body):
        return "table"
    if _BULLET.match(body[0]) or _ORDERED.match(body[0]):
        return "list_item"
    return "paragraph"


def _match_marker(line: str) -> tuple[int, str] | None:
    """`(indent_width, content_after_marker)` if the line opens a list item.

    The content is returned WITHOUT the `- ` / `1. ` marker. The marker is
    markdown syntax describing that this is a list, not a word the policy
    author wrote, so it must not reach `Clause.text` - which is both the string
    shown to the judge and the string C2 span-verifies against.
    """
    for pattern in (_BULLET, _ORDERED):
        m = pattern.match(line)
        if m:
            return len(m.group(1).expandtabs(4)), m.group(2)
    return None


def _split_list_block(lines: Sequence[str]) -> list[str]:
    """One string per top-level list item, markers removed.

    Nested sub-items stay attached to their parent rather than becoming clauses
    of their own. A sub-bullet is almost always a sub-condition of the bullet
    above it, so detaching it would produce a clause that cannot be read on its
    own - and the judge is shown clauses in isolation (DESIGN.md 4.1).
    """
    items: list[list[str]] = []
    base_indent: int | None = None

    for line in lines:
        if not line.strip():
            continue
        matched = _match_marker(line)
        if matched is not None:
            indent, content = matched
            if base_indent is None or indent <= base_indent:
                base_indent = indent if base_indent is None else min(base_indent, indent)
                items.append([content.strip()])
                continue
            # A nested item: keep its own marker text out, but preserve the
            # sub-condition wording as part of the parent clause.
            if items:
                items[-1].append(content.strip())
                continue
            items.append([content.strip()])
            continue
        if items:
            items[-1].append(line.strip())
        else:
            # Continuation text before any marker: treat as its own item.
            items.append([line.strip()])

    return [" ".join(item) for item in items]


def segment_markdown(
    text: str,
    *,
    long_block_splitter: Callable[[str], list[str]] | None = None,
) -> list[RawSegment]:
    """Segment markdown policy text into ordered candidate clauses.

    `long_block_splitter` is injectable so tests can pin the fallback path
    instead of depending on whether langchain happens to be installed.
    """
    splitter = long_block_splitter or split_long_block

    heading_stack: list[tuple[int, str]] = []
    blocks: list[_Block] = []
    current = _Block()
    in_fence = False

    def close_block() -> None:
        nonlocal current
        if not current.is_empty():
            blocks.append(current)
        current = _Block(heading_path=tuple(t for _, t in heading_stack))

    current.heading_path = ()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue

        if in_fence:
            current.lines.append(line)
            continue

        heading = _HEADING.match(line)
        if heading:
            close_block()
            level = len(heading.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading.group(2).strip()))
            current.heading_path = tuple(t for _, t in heading_stack)
            continue

        if _HRULE.match(line):
            close_block()
            continue

        if not line.strip():
            close_block()
            continue

        # Blockquote markers are container syntax, not clause words. Real fetched
        # policies wrap notes and callouts in them.
        line = _BLOCKQUOTE.sub("", line, count=1)
        if not line.strip():
            close_block()
            continue

        if not current.lines:
            current.heading_path = tuple(t for _, t in heading_stack)
        current.lines.append(line)

    close_block()

    # --- blocks -> segments -------------------------------------------------
    segments: list[RawSegment] = []
    for block in blocks:
        kind = _classify(block.lines)

        if kind == "list_item":
            candidates = [(item, "list_item") for item in _split_list_block(block.lines)]
        elif kind == "table":
            rows = [
                ln
                for ln in block.lines
                if ln.strip() and not _TABLE_DELIM.match(ln)
            ]
            candidates = [(collapse_whitespace(" ".join(rows)), "table")]
        else:
            candidates = [(collapse_whitespace(" ".join(block.lines)), "paragraph")]

        for body, candidate_kind in candidates:
            body = _strip_inline_markdown(body)
            if not normalize(body):
                # Punctuation-only or empty residue is not an addressable clause.
                continue
            if estimate_tokens(body) <= MAX_CLAUSE_TOKENS:
                segments.append(
                    RawSegment(
                        text=body,
                        heading_path=block.heading_path,
                        kind=candidate_kind,  # type: ignore[arg-type]
                    )
                )
                continue
            for piece in splitter(body):
                piece = _strip_inline_markdown(piece)
                if normalize(piece):
                    segments.append(
                        RawSegment(
                            text=piece, heading_path=block.heading_path, kind="split"
                        )
                    )

    return segments


def length_audit(segments: Sequence[RawSegment]) -> dict[str, int]:
    """Count segments outside DESIGN.md 1.1's 40-400 token target band.

    Short segments are reported, never merged. Fusing a short list item into its
    neighbour would mean editing one item changes another item's clause hash,
    which is precisely the false positive the ordinal/hash pairing exists to
    avoid. Reporting the distribution keeps the 40-400 target auditable while
    leaving addressability intact.
    """
    lengths = [s.token_estimate for s in segments]
    return {
        "count": len(lengths),
        "under_min": sum(1 for n in lengths if n < MIN_CLAUSE_TOKENS),
        "in_band": sum(1 for n in lengths if MIN_CLAUSE_TOKENS <= n <= MAX_CLAUSE_TOKENS),
        "over_max": sum(1 for n in lengths if n > MAX_CLAUSE_TOKENS),
        "min": min(lengths, default=0),
        "max": max(lengths, default=0),
    }
