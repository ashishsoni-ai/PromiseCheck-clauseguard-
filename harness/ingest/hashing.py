"""Clause normalisation, content hashing, and ID construction.

DESIGN.md 1.1 specifies:

    clause_id = f"{doc_slug}:{ordinal:03d}:{sha256(normalize(text))[:8]}"
    # e.g.  acme-refunds:014:a3f91c22

    Normalisation = lowercase, collapse whitespace, strip punctuation-only diffs.
    The hash suffix is the change-detection primitive for the gate (6). Ordinal
    survives edits; hash does not.

This module is the whole of that primitive, so it is written to be boring: pure
stdlib, no I/O, no configuration, deterministic across platforms and Python
versions. Everything downstream - the manifest diff, probe invalidation, the
gate's "clause 14 changed" claim - is only as trustworthy as these functions.

TWO NORMALISERS, DELIBERATELY NOT ONE
-------------------------------------
`normalize()` is for HASHING. It is lossy on purpose: punctuation is discarded so
that a merchant re-punctuating a sentence does not churn a clause ID.

`collapse_whitespace()` is for SPAN VERIFICATION (DESIGN.md 4.1 L2, commitment
C2), which matches quotes "after whitespace normalisation" only. Span checks must
NOT use `normalize()`. If they did, a judge could "quote" a clause with the
punctuation removed - and "refunds are not available, except for defects" and
"refunds are not available except for defects" are not the same promise. Using
the lossy normaliser for C2 would hand the judge a hole to walk through.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence

#: DESIGN.md 1.1 truncates sha256 to 8 hex characters.
CONTENT_HASH_LENGTH = 8

#: A comma between two digits is a digit-group separator, so "1,000" and "1000"
#: are the same number written two ways - a punctuation-only diff. Deleted rather
#: than spaced, or "1,000" would become "1 000" and never match "1000".
_DIGIT_GROUP_COMMA = re.compile(r"(?<=\d),(?=\d)")


def collapse_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends.

    This is the ONLY normalisation permitted before commitment C2's exact
    substring check (DESIGN.md 4.1 L2). It is information-preserving apart from
    layout, so a quote that survives it is still verbatim.
    """
    return " ".join(text.split())


def normalize(text: str) -> str:
    """Normalise clause text for hashing. Lossy by design; never for span checks.

    Steps, in order:

    1. Unicode NFKC. Folds compatibility forms - full-width digits, ligatures,
       non-breaking and thin spaces - so the same clause copied out of a PDF and
       out of an HTML page hashes identically.
    2. Case folding. `casefold()` rather than `lower()` because it is the
       Unicode-correct caseless comparison.
    3. Delete digit-group commas, so "Rs 1,000" == "Rs 1000".
    4. Replace every Unicode punctuation character with a SPACE, except a period
       between two digits, which is a decimal point and therefore content.
    5. Collapse whitespace.

    WHY PUNCTUATION BECOMES A SPACE RATHER THAN BEING DELETED
    Deleting it fuses neighbouring tokens, and fused tokens collide. "7.5%"
    would delete to "75" and collide with a clause saying "75%" - a real
    semantic difference in a tiered-fee policy, silently invisible to the gate.
    Spacing cannot fuse anything, and still gives punctuation-only diffs an
    identical hash, because the spaces collapse away in step 5.

    WHAT THIS DELIBERATELY TREATS AS NOISE
    The percent sign is Unicode category Po, so "20%" and "20" hash alike. That
    is accepted: no real policy sentence distinguishes them, whereas "20 %" vs
    "20%" - which this correctly folds together - is a common editorial edit.

    WHAT IS KEPT
    Currency symbols and mathematical operators are categories Sc and Sm, not P,
    so Rs, $ and < survive. Losing a currency symbol would make two different
    fee clauses hash alike.
    """
    s = unicodedata.normalize("NFKC", text)
    s = s.casefold()
    s = _DIGIT_GROUP_COMMA.sub("", s)

    chars: list[str] = []
    last = len(s) - 1
    for i, ch in enumerate(s):
        if unicodedata.category(ch).startswith("P"):
            is_decimal_point = (
                ch == "."
                and 0 < i < last
                and s[i - 1].isdigit()
                and s[i + 1].isdigit()
            )
            chars.append(ch if is_decimal_point else " ")
        else:
            chars.append(ch)

    return collapse_whitespace("".join(chars))


def content_hash(text: str) -> str:
    """sha256(normalize(text))[:8] - the change-detection primitive.

    ON TRUNCATING TO 32 BITS
    A short hash is only load-bearing within one (doc_slug, ordinal) slot, since
    the ordinal is a separate component of the clause ID. So the failure mode is
    not "two clauses collide" but the much narrower "a clause was edited and its
    new text happened to hash to the old value", which the gate would then miss.
    That is one chance in 2**32 per edit. At DESIGN.md 7.1's scale of ~300
    clauses the expected number of missed edits is around 7e-8, which is far
    below the error rate of every other component in the system.
    """
    digest = hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
    return digest[:CONTENT_HASH_LENGTH]


def make_clause_id(doc_slug: str, ordinal: int, clause_content_hash: str) -> str:
    """Assemble a clause ID. Mirrors the f-string in DESIGN.md 1.1 exactly.

    `Clause` re-derives and re-checks this in a validator, so a drift between
    this function and the schema surfaces as a ValidationError rather than as
    quietly mislabelled audit rows.
    """
    if ordinal < 1:
        raise ValueError(f"ordinal is 1-based, got {ordinal}")
    return f"{doc_slug}:{ordinal:03d}:{clause_content_hash}"


def policy_version(doc_slug: str, clause_hashes: Sequence[str]) -> str:
    """Whole-document version as 'sha256:<64 hex>' (DESIGN.md 5.1).

    Computed over the ordered (ordinal, content_hash) pairs rather than over the
    raw file bytes. This is a deliberate choice: it makes policy_version change
    if and only if the clause set changes, which is exactly the condition under
    which two runs stop being comparable. Hashing raw bytes instead would bump
    the version when someone adds a blank line, making the audit trail claim a
    new policy version for a semantically identical document and undermining the
    regression story the field exists to support.

    The doc_slug is included so that two documents with coincidentally identical
    clause sets still version distinctly.
    """
    payload = "\n".join(
        [doc_slug, *(f"{i:03d}:{h}" for i, h in enumerate(clause_hashes, start=1))]
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def slugify(name: str) -> str:
    """Turn a filename or title into a valid `DocSlug`.

    Must satisfy `^[a-z0-9][a-z0-9._-]*$` - in particular no colons, because the
    slug is the first colon-delimited component of every clause ID.
    """
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = s.casefold()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s or not s[0].isalnum():
        # The slug must open with an alphanumeric; prefix rather than drop
        # characters, so two different names cannot slugify to the same value.
        s = "doc-" + s.lstrip("-._")
    return s
