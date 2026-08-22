"""STEP 2 coverage for harness/ingest/loaders.py - the DESIGN.md 7.1 fetcher.

This module had no tests, which is how the bug these tests were written for survived:
`load_url` cached its fetched text with `Path.write_text` and recomputed `raw_sha256`
from whatever came back on the next cache hit. Two independent failures came out of that,
and neither needs a network to reproduce:

  * on Linux the `\\r\\n` in the fetched text survived the write, and the universal-newline
    read turned it into `\\n` - so the cold fetch and the cache hit hashed differently
  * on Windows `write_text` translated the `\\n` half of each `\\r\\n` again, storing
    `\\r\\r\\n`, which read back as `\\n\\n` - an extra BLANK LINE, and blank lines are what
    the segmenter splits paragraphs on

The second one is the dangerous one. Different paragraph boundaries mean different
`clause_id`s mean a different `policy_version`, which is half the key every audit row is
written against. It would have surfaced as "the same policy hashes differently on my
machine", days later, with the cache as the last place anyone would look.

The fix is `canonical_text`, applied by all three loaders, plus `newline=""` on both ends
of the cache so it cannot reintroduce the problem. `TestLoadUrlCacheRoundTrip` is the
regression test; everything else is the coverage that should have been here already.

NO NETWORK, NO OPTIONAL DEPENDENCIES. `httpx` and `trafilatura` are imported lazily
inside `load_url`, so the two tests that exercise the fetch path inject stand-ins through
`sys.modules`. That is a seam for a *collaborator* - the loader logic under test is the
real thing, and the only behaviour faked is "the network returned this string".
"""

from __future__ import annotations

import hashlib
import re
import sys
import types

import pytest

from harness.ingest.loaders import (
    CACHE_DIR,
    LoadedSource,
    _cache_path,
    canonical_text,
    load_markdown,
    load_url,
    raw_sha256,
    slug_for_path,
    slug_for_url,
)

URL = "https://shop.example.com/policies/refund-policy"


# --------------------------------------------------------------------------------------
# stand-ins for the lazily imported fetch stack
# --------------------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def install_fake_fetch(monkeypatch: pytest.MonkeyPatch, served: str) -> dict[str, int]:
    """Make `load_url` see `served` as the extracted text. Returns a call counter.

    `trafilatura.extract` is the identity here rather than a real extraction: what these
    tests are about is what happens to the text AFTER extraction, and a fake extractor
    that reflows or strips would hide exactly the whitespace behaviour under test.
    """
    calls = {"get": 0}

    def fake_get(url: str, **kwargs) -> _FakeResponse:
        calls["get"] += 1
        return _FakeResponse(served)

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.get = fake_get
    fake_trafilatura = types.ModuleType("trafilatura")
    fake_trafilatura.extract = lambda text, **kwargs: text

    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "trafilatura", fake_trafilatura)
    return calls


class TestCanonicalText:
    def test_crlf_becomes_lf(self):
        assert canonical_text("a\r\nb") == "a\nb"

    def test_a_lone_cr_becomes_lf(self):
        """Classic-Mac terminators turn up in PDFs more often than anyone expects."""
        assert canonical_text("a\rb") == "a\nb"

    def test_lf_is_left_alone(self):
        assert canonical_text("a\nb") == "a\nb"

    def test_it_is_idempotent(self):
        once = canonical_text("a\r\n\r\nb\rc\n")
        assert canonical_text(once) == once

    def test_a_paragraph_break_does_not_gain_a_line(self):
        """The whole reason this matters. `\\r\\n\\r\\n` is ONE blank line, and it has to stay
        one: the segmenter splits paragraphs on blank lines, so inflating it to two would
        move clause boundaries and therefore clause_ids."""
        assert canonical_text("a\r\n\r\nb") == "a\n\nb"
        assert canonical_text("a\r\n\r\nb").count("\n") == 2

    def test_no_carriage_return_survives(self):
        assert "\r" not in canonical_text("a\r\nb\rc\r\n\r\nd")

    def test_empty_string_is_safe(self):
        assert canonical_text("") == ""


class TestRawSha256:
    def test_it_is_sha256_of_the_utf8_bytes(self):
        text = "Refunds within 30 days."
        assert raw_sha256(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_it_is_bare_hex_with_no_prefix(self):
        """`content_hash` carries a `sha256:` prefix and this does not; they are different
        fields answering different questions (see the module docstring)."""
        digest = raw_sha256("x")
        assert not digest.startswith("sha256:")
        assert len(digest) == 64

    def test_line_endings_no_longer_change_it_once_canonicalised(self):
        assert raw_sha256(canonical_text("a\r\nb")) == raw_sha256(canonical_text("a\nb"))


class TestLoadMarkdown:
    def test_a_crlf_file_loads_as_lf(self, tmp_path):
        p = tmp_path / "acme-refunds.md"
        p.write_bytes(b"# Refunds\r\n\r\nWithin 30 days.\r\n")
        assert "\r" not in load_markdown(p).text

    def test_crlf_and_lf_files_hash_identically(self, tmp_path):
        """Same policy, two checkouts, one hash. Without this the 7.1 claim that a
        reviewer can re-run the fetcher and compare hashes is false across platforms."""
        crlf = tmp_path / "crlf.md"
        crlf.write_bytes(b"# Refunds\r\n\r\nWithin 30 days.\r\n")
        lf = tmp_path / "lf.md"
        lf.write_bytes(b"# Refunds\n\nWithin 30 days.\n")
        assert load_markdown(crlf).raw_sha256 == load_markdown(lf).raw_sha256

    def test_the_slug_comes_from_the_stem(self, tmp_path):
        p = tmp_path / "acme-refunds.md"
        p.write_bytes(b"# Refunds\n\nWithin 30 days.\n")
        assert load_markdown(p).doc_slug == "acme-refunds"

    def test_it_is_not_marked_as_coming_from_cache(self, tmp_path):
        p = tmp_path / "a.md"
        p.write_bytes(b"text\n")
        assert load_markdown(p).from_cache is False


class TestSlugForPath:
    def test_extension_is_dropped(self, tmp_path):
        assert slug_for_path(tmp_path / "Acme Refunds.md") == "acme-refunds"


class TestSlugForUrl:
    def test_the_host_is_included(self):
        """Two merchants both serving /returns-policy must not collide - a clause_id
        collision between two companies' policies would silently corrupt the audit
        trail, which is why the host is in the slug at all."""
        a = slug_for_url("https://shop-a.example.com/returns-policy")
        b = slug_for_url("https://shop-b.example.com/returns-policy")
        assert a != b
        assert "shop-a" in a and "shop-b" in b

    def test_www_is_stripped_so_one_site_is_one_slug(self):
        assert slug_for_url("https://www.example.com/refunds") == slug_for_url(
            "https://example.com/refunds"
        )

    def test_a_trailing_slash_does_not_make_a_second_slug(self):
        assert slug_for_url(URL + "/") == slug_for_url(URL)

    def test_query_and_fragment_are_ignored(self):
        assert slug_for_url(URL + "?utm_source=x#section-2") == slug_for_url(URL)

    def test_a_bare_host_still_produces_something(self):
        """Dots survive: `_DOC_SLUG_RE` is `^[a-z0-9][a-z0-9._-]*$`, so `example.com` is a
        valid slug and does not need flattening to `example-com`."""
        assert slug_for_url("https://example.com") == "example.com"

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com",
            "https://example.com:8443/refunds",
            "https://EXAMPLE.com/Returns_Policy",
            "https://example.com/policies/refund%20policy",
            "https://xn--bcher-kva.example/rckgabe",
        ],
    )
    def test_every_derived_slug_is_a_valid_doc_slug(self, url):
        """A slug is the first COLON-DELIMITED component of every clause_id, so a colon
        leaking in from a URL port would silently corrupt clause_id parsing downstream.
        Pattern duplicated from harness/schemas/clause.py rather than imported, so this
        test fails if the two ever drift apart instead of following the schema."""
        slug = slug_for_url(url)
        assert ":" not in slug
        assert re.match(r"^[a-z0-9][a-z0-9._-]*$", slug), slug


class TestCachePath:
    def test_two_urls_that_slugify_alike_get_different_files(self):
        """Keyed on the sha256 of the URL, not the slug, so near-identical URLs cannot
        serve each other's cached text."""
        a = _cache_path("https://example.com/a/refunds")
        b = _cache_path("https://example.com/b/refunds")
        assert a.name != b.name

    def test_the_same_url_is_stable(self):
        assert _cache_path(URL).name == _cache_path(URL).name

    def test_it_lands_under_the_gitignored_cache_root(self):
        """DESIGN.md 7.1: we ship the fetcher, not the corpus. The cache must stay in the
        gitignored directory or the corpus ships by accident."""
        assert _cache_path(URL).parent == CACHE_DIR


class TestLoadUrlCacheRoundTrip:
    """The regression tests. A cache hit must be indistinguishable from a cold fetch."""

    SERVED = "Refund Policy\r\n\r\nRefunds are available within 30 days.\r\n"

    def test_the_cold_fetch_returns_canonical_text(self, tmp_path, monkeypatch):
        install_fake_fetch(monkeypatch, self.SERVED)
        loaded = load_url(URL, cache_dir=tmp_path)
        assert "\r" not in loaded.text
        assert loaded.from_cache is False

    def test_the_cached_bytes_are_lf_on_every_platform(self, tmp_path, monkeypatch):
        install_fake_fetch(monkeypatch, self.SERVED)
        load_url(URL, cache_dir=tmp_path)
        cached = tmp_path / _cache_path(URL).name
        assert cached.exists(), "the fetch should have populated the cache"
        assert b"\r" not in cached.read_bytes()

    def test_a_cache_hit_hashes_the_same_as_the_cold_fetch(self, tmp_path, monkeypatch):
        """THE test. `raw_sha256` is recomputed on the cache-hit path, so if the round trip
        is not byte-faithful the provenance hash changes with no upstream change at all."""
        calls = install_fake_fetch(monkeypatch, self.SERVED)
        cold = load_url(URL, cache_dir=tmp_path)
        warm = load_url(URL, cache_dir=tmp_path)

        assert calls["get"] == 1, "the second call should not have refetched"
        assert warm.from_cache is True
        assert warm.text == cold.text
        assert warm.raw_sha256 == cold.raw_sha256

    def test_a_cache_hit_does_not_gain_a_paragraph_break(self, tmp_path, monkeypatch):
        """The Windows manifestation, asserted on its consequence rather than its cause:
        `\\r\\r\\n` read back as `\\n\\n` and a new blank line moves clause boundaries."""
        install_fake_fetch(monkeypatch, self.SERVED)
        cold = load_url(URL, cache_dir=tmp_path)
        warm = load_url(URL, cache_dir=tmp_path)
        assert warm.text.count("\n") == cold.text.count("\n")

    def test_deleting_the_cache_changes_nothing_but_runtime(self, tmp_path, monkeypatch):
        """The invariant the module docstring promises, asserted directly."""
        install_fake_fetch(monkeypatch, self.SERVED)
        first = load_url(URL, cache_dir=tmp_path)
        (tmp_path / _cache_path(URL).name).unlink()
        second = load_url(URL, cache_dir=tmp_path)
        assert second.raw_sha256 == first.raw_sha256
        assert second.text == first.text

    def test_a_crlf_cache_file_left_by_anything_else_is_still_read_as_lf(self, tmp_path):
        """No fetch stack installed on purpose: a cache hit must short-circuit before the
        lazy imports, so this also proves a cached load needs neither httpx nor
        trafilatura. The CRLF bytes stand in for a file left by an older build."""
        cached = tmp_path / _cache_path(URL).name
        cached.write_bytes(b"Refund Policy\r\n\r\nWithin 30 days.\r\n")
        loaded = load_url(URL, cache_dir=tmp_path)
        assert loaded.from_cache is True
        assert "\r" not in loaded.text
        assert loaded.raw_sha256 == raw_sha256("Refund Policy\n\nWithin 30 days.\n")

    def test_use_cache_false_refetches(self, tmp_path, monkeypatch):
        calls = install_fake_fetch(monkeypatch, self.SERVED)
        load_url(URL, cache_dir=tmp_path)
        load_url(URL, cache_dir=tmp_path, use_cache=False)
        assert calls["get"] == 2

    def test_an_empty_extraction_is_an_error_not_an_empty_policy(
        self, tmp_path, monkeypatch
    ):
        """An empty document would sail through as a policy with zero clauses, and a
        policy with zero clauses trivially passes every conformance check."""
        install_fake_fetch(monkeypatch, "   \n  \n ")
        with pytest.raises(RuntimeError, match="no main text extracted"):
            load_url(URL, cache_dir=tmp_path)

    def test_the_fetched_source_is_the_url(self, tmp_path, monkeypatch):
        install_fake_fetch(monkeypatch, self.SERVED)
        loaded = load_url(URL, cache_dir=tmp_path)
        assert loaded.source == URL
        assert loaded.is_remote is True


class TestLoadedSourceIsFrozen:
    def test_text_cannot_be_reassigned(self, tmp_path):
        """Frozen because raw_sha256 travels beside text; if a later stage could edit the
        text, the hash recorded next to it would describe bytes that no longer exist."""
        p = tmp_path / "a.md"
        p.write_bytes(b"text\n")
        loaded = load_markdown(p)
        with pytest.raises(Exception):
            loaded.text = "something else"  # type: ignore[misc]

    def test_it_is_the_dataclass_the_rest_of_the_pipeline_expects(self, tmp_path):
        p = tmp_path / "a.md"
        p.write_bytes(b"text\n")
        assert isinstance(load_markdown(p), LoadedSource)
