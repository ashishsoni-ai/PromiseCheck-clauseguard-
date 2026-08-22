"""Policy source loading: local markdown, local PDF, remote URL.

DESIGN.md 7.1 constrains this module more than it constrains most:

    "Ship URLs, fetch timestamps, content hashes and the fetcher - not the policy
    corpus."

So fetched text is cached under `policies/.cache/`, which is gitignored, while the
things that make a fetch REPRODUCIBLE - the URL, the timestamp, and the sha256 of
the canonical text we parsed - travel in `LoadedSource` and end up in the committed
manifest. Anyone can re-run the fetcher and compare hashes; nobody has to
redistribute a merchant's copyrighted policy page.

"Canonical" is load-bearing in that sentence, not a hedge: line endings are collapsed
to LF before hashing so the value does not depend on the ingest route or the host OS.
See `canonical_text`.

WHY `raw_sha256` IS NOT `content_hash`
`raw_sha256` is the full sha256 of the extracted text, unnormalised, and it
answers "did the fetch return the same bytes as last time?" - a provenance
question. `hashing.content_hash` is the truncated hash of NORMALISED clause text
and answers "did this clause's meaning change?" - the gate's question. A reflowed
HTML page moves `raw_sha256` and leaves every `content_hash` alone, which is
exactly the discrimination the gate needs and the reason these are two fields.

THIRD-PARTY IMPORTS ARE LAZY
`pypdf` and `trafilatura` are imported inside the functions that need them.
Ingesting the markdown worked example - the path every test and the Step 2
checkpoint take - therefore needs neither installed, so a missing optional
dependency can never turn into a mysterious collection error in the test suite.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from harness.ingest.hashing import slugify

#: Cache root for fetched policy text. GITIGNORED - see the module docstring and
#: DESIGN.md 7.1. Never read as a source of truth: it is a bandwidth and
#: politeness optimisation, and deleting it must change nothing but runtime.
CACHE_DIR = Path("policies/.cache")

#: Conservative default. A policy page that takes longer than this is a fetch
#: failure, not something to block a run on.
FETCH_TIMEOUT_SECONDS = 20.0

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


@dataclass(frozen=True)
class LoadedSource:
    """Raw text of one policy source, plus the provenance needed to re-fetch it.

    Frozen because everything downstream - clause hashes, the manifest, audit rows
    - is derived from these fields. If a later stage could mutate `text`, the
    `raw_sha256` recorded beside it would become a claim about bytes that no
    longer exist.
    """

    text: str
    """The policy text. LF-only by invariant - every loader runs `canonical_text`."""

    source: str
    fetched_at: datetime
    raw_sha256: str
    doc_slug: str
    from_cache: bool = False

    @property
    def is_remote(self) -> bool:
        return bool(_URL_RE.match(self.source))


def canonical_text(text: str) -> str:
    """Collapse CRLF and lone CR to LF. Applied by every loader before hashing.

    `LoadedSource.text` is LF-only by invariant, and this is where that is enforced.
    Three separate things go wrong without it.

    1. THE HASH WOULD DEPEND ON THE INGEST ROUTE. `load_markdown` reads with universal
       newlines, so a CRLF file on disk already arrives as LF. `load_pdf` and `load_url`
       take whatever pypdf and trafilatura emit. The same policy text would therefore
       produce two different `raw_sha256` values depending on which loader saw it, and
       DESIGN.md 7.1 ships that hash as the evidence a re-fetch is reproducible.

    2. THE HASH WOULD DEPEND ON THE HOST OS. `Path.write_text` translates `\\n` to the
       platform terminator, so on Windows a cached `\\r\\n` was written as `\\r\\r\\n` and read
       back - through universal newlines - as `\\n\\n`. An EXTRA BLANK LINE. Blank lines are
       what the segmenter splits paragraphs on, so a cache hit could produce different
       clause boundaries, different `clause_id`s and a different `policy_version` than the
       cold fetch of byte-identical upstream text. That is half the key the audit trail is
       built on, moving because of a line-ending convention.

    3. THE CACHE INVARIANT WOULD BE FALSE. This module's docstring promises that deleting
       `policies/.cache/` "must change nothing but runtime". Under (2) it changed the text.

    Canonicalising STRENGTHENS the 7.1 claim rather than weakening it: a reviewer on Linux
    and a build on Windows now agree on the hash for the same page. An upstream server
    switching terminators is not a policy change, and should not read as one - the same
    argument `hashing.normalize` makes one level down, at clause granularity.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def raw_sha256(text: str) -> str:
    """Full sha256 of the extracted text, for provenance. Not the clause hash.

    Callers pass text that has already been through `canonical_text`; see there for why
    the line endings must be settled before this point.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slug_for_path(path: Path) -> str:
    """Derive a doc_slug from a filename stem: `acme-refunds.md` -> `acme-refunds`."""
    return slugify(path.stem)


def slug_for_url(url: str) -> str:
    """Derive a doc_slug from a URL's host and last path segment.

    Host is included because two merchants routinely both serve
    `/returns-policy`, and a clause ID collision between two different companies'
    policies would be a silent, confusing corruption of the audit trail.
    """
    stripped = _URL_RE.sub("", url).split("?", 1)[0].split("#", 1)[0].rstrip("/")
    parts = [p for p in stripped.split("/") if p]
    if not parts:
        return slugify(stripped or url)
    host = parts[0].removeprefix("www.")
    if len(parts) == 1:
        return slugify(host)
    tail = Path(parts[-1]).stem
    return slugify(f"{host}-{tail}")


def _cache_path(url: str) -> Path:
    """Cache filename keyed on the sha256 of the URL.

    Keyed on a hash rather than on the slug so that two URLs which slugify alike
    cannot silently serve each other's cached text.
    """
    return CACHE_DIR / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}.txt"


def load_markdown(path: str | Path) -> LoadedSource:
    """Load a local markdown or plain-text policy file."""
    p = Path(path)
    text = canonical_text(p.read_text(encoding="utf-8"))
    return LoadedSource(
        text=text,
        source=str(path),
        fetched_at=datetime.now(timezone.utc),
        raw_sha256=raw_sha256(text),
        doc_slug=slug_for_path(p),
    )


def load_pdf(path: str | Path) -> LoadedSource:
    """Extract text from a local PDF via pypdf (DESIGN.md Appendix).

    Pages are joined with a blank line so the segmenter sees a paragraph break at
    each page boundary. Joining with a single newline instead would let the last
    sentence of one page fuse with the first heading of the next, producing a
    clause that spans a boundary no human reader would treat as continuous.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised by env, not tests
        raise RuntimeError(
            "PDF ingest needs `pypdf`; install it or pass a markdown source"
        ) from exc

    p = Path(path)
    reader = PdfReader(str(p))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = canonical_text("\n\n".join(page for page in pages if page))
    return LoadedSource(
        text=text,
        source=str(path),
        fetched_at=datetime.now(timezone.utc),
        raw_sha256=raw_sha256(text),
        doc_slug=slug_for_path(p),
    )


def load_url(
    url: str,
    *,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    timeout: float = FETCH_TIMEOUT_SECONDS,
) -> LoadedSource:
    """Fetch a policy page and extract its main text via trafilatura.

    `trafilatura` rather than a raw HTML-to-text pass because navigation chrome,
    cookie banners and footer link farms would otherwise become clauses - and a
    clause that says "Accept all cookies" is one the extractor will happily invent
    an entitlement rule from.

    Cache hits return `from_cache=True` and the cached file's mtime as
    `fetched_at`, so a cached load never claims a fetch timestamp that did not
    happen.
    """
    root = cache_dir if cache_dir is not None else CACHE_DIR
    cached = (root / _cache_path(url).name) if root else None

    if use_cache and cached is not None and cached.exists():
        # newline="" disables translation on the way in, so the cache is a byte-faithful
        # store and `canonical_text` is the single place line endings are allowed to
        # change. `Path.read_text` grew a `newline=` parameter only in 3.12+, so this
        # goes through `open` to stay version-agnostic.
        with cached.open("r", encoding="utf-8", newline="") as handle:
            text = canonical_text(handle.read())
        return LoadedSource(
            text=text,
            source=url,
            fetched_at=datetime.fromtimestamp(cached.stat().st_mtime, tz=timezone.utc),
            raw_sha256=raw_sha256(text),
            doc_slug=slug_for_url(url),
            from_cache=True,
        )

    try:
        import httpx
        import trafilatura
    except ImportError as exc:  # pragma: no cover - exercised by env, not tests
        raise RuntimeError(
            "URL ingest needs `httpx` and `trafilatura`; install them or pass a "
            "local source"
        ) from exc

    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    extracted = trafilatura.extract(
        response.text,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if not extracted or not extracted.strip():
        raise RuntimeError(f"no main text extracted from {url!r}")

    text = canonical_text(extracted).strip()
    if use_cache and cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        # newline="" so nothing is translated on the way out. Without it Windows wrote
        # `\r\n` as `\r\r\n`, which read back as an extra blank line and moved the clause
        # boundaries - see `canonical_text`. Belt and braces: the text is already LF by
        # the line above, and this makes the cache incapable of reintroducing the problem.
        with cached.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    return LoadedSource(
        text=text,
        source=url,
        fetched_at=datetime.now(timezone.utc),
        raw_sha256=raw_sha256(text),
        doc_slug=slug_for_url(url),
    )


def load(source: str | Path, **kwargs) -> LoadedSource:
    """Dispatch on the shape of `source`: URL, `.pdf`, or markdown/text."""
    text_source = str(source)
    if _URL_RE.match(text_source):
        return load_url(text_source, **kwargs)
    if Path(text_source).suffix.lower() == ".pdf":
        return load_pdf(text_source)
    return load_markdown(text_source)
