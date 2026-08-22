"""Fetch the policy corpus. Ships the fetcher, not the corpus (DESIGN.md 7.1).

    "Handling: store URLs, fetch timestamps, and content hashes in the repo plus a
    `fetch_policies.py` script. Cache fetched text locally under a gitignored
    path."

This is a thin CLI over `harness.ingest.loaders.load_url`. It deliberately owns no
fetching logic of its own: duplicating the trafilatura call here would mean the
corpus could be fetched two ways and hash two ways, which would make a hash
mismatch ambiguous between "the policy changed" and "you used the other entrypoint".

WHAT IT PERSISTS, AND WHY THAT IS NOT ALREADY COVERED
`loaders.py` computes `raw_sha256` - the sha256 of the exact extracted bytes - and
then drops it, because `ingest_text` takes only text and `fingerprint_document`
stores clause hashes. Clause hashes answer "did any clause's meaning change?";
they cannot answer "did the fetch return the same bytes?", because normalisation
throws away exactly the difference between those two questions. So the
fetch-provenance hash 7.1 asks for had nowhere to live until this file. It is
written to `policies/fetch.lock.json`, COMMITTED, and contains no policy text.

WHERE THE URL LIST LIVES
`--sources policies/sources.json`, a committed JSON array of objects:

    [{"url": "https://...", "corpus_role": "real", "is_holdout": false,
      "doc_slug": "optional override", "note": "licensing/why-this-one"}]

That filename and shape are NOT in DESIGN.md - 7.1 says URLs live in the repo but
does not say in what file. The list stays empty until the 8 real + 2 synthetic
policies are actually chosen, so nothing here pretends to a corpus that does not
exist yet. URLs may also be passed positionally for a one-off fetch.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Importable as a script from the repo root without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.ingest.loaders import CACHE_DIR, LoadedSource, load_url, slug_for_url

#: Committed fetch provenance: URL -> timestamp + raw content hash. No text.
FETCH_LOCK_PATH = Path("policies/fetch.lock.json")

#: Committed input list. Absent until the corpus is chosen; see the docstring.
DEFAULT_SOURCES_PATH = Path("policies/sources.json")

VALID_ROLES = ("real", "synthetic_stress", "worked_example")


def read_sources(path: Path) -> list[dict]:
    """Load the committed URL list, validating the fields we depend on.

    Validated eagerly rather than at fetch time: a typo in `corpus_role` that
    surfaces only after 8 network round-trips would silently label a real policy
    as a stress fixture, and 7.1 forbids pooling those.
    """
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a JSON array of source objects")

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "url" not in entry:
            raise ValueError(f"{path}[{i}] needs at least a 'url' key")
        role = entry.get("corpus_role", "real")
        if role not in VALID_ROLES:
            raise ValueError(
                f"{path}[{i}] has corpus_role {role!r}; expected one of {VALID_ROLES}"
            )
    return entries


def load_lock(path: Path) -> dict:
    if not path.exists():
        return {"fetched": {}}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw or raw == "{}":
        return {"fetched": {}}
    data = json.loads(raw)
    data.setdefault("fetched", {})
    return data


def write_lock(lock: dict, path: Path) -> Path:
    """Deterministic write, so an unchanged corpus produces no git diff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def record(loaded: LoadedSource, entry: dict, prior: dict | None) -> dict:
    """Build one lockfile row, preserving the timestamp when bytes are unchanged.

    Same reasoning as the manifest's `content_fetched_at`: if every fetch rewrote
    the timestamp, a re-fetch of an unchanged page would dirty a committed file
    and "the lockfile changed" would stop meaning "the source changed".
    """
    changed = prior is None or prior.get("raw_sha256") != loaded.raw_sha256
    return {
        "url": loaded.source,
        "doc_slug": loaded.doc_slug,
        "corpus_role": entry.get("corpus_role", "real"),
        "is_holdout": bool(entry.get("is_holdout", False)),
        "note": entry.get("note", ""),
        "raw_sha256": loaded.raw_sha256,
        "chars": len(loaded.text),
        "content_fetched_at": (
            loaded.fetched_at.isoformat()
            if changed
            else prior.get("content_fetched_at", loaded.fetched_at.isoformat())
        ),
        "last_verified_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python scripts/fetch_policies.py",
        description="Fetch policy pages into the gitignored cache and record "
        "URL + timestamp + content hash (DESIGN.md 7.1).",
    )
    p.add_argument("urls", nargs="*", help="Fetch these URLs instead of the list.")
    p.add_argument(
        "--sources",
        type=Path,
        default=DEFAULT_SOURCES_PATH,
        help=f"Committed URL list (default {DEFAULT_SOURCES_PATH}).",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass the cache and re-fetch. Use this to detect a policy change.",
    )
    p.add_argument(
        "--lock", type=Path, default=FETCH_LOCK_PATH, help="Provenance lockfile."
    )
    p.add_argument(
        "--dry-run", action="store_true", help="List what would be fetched, then stop."
    )
    args = p.parse_args(argv)

    if args.urls:
        entries = [{"url": u} for u in args.urls]
    elif args.sources.exists():
        entries = read_sources(args.sources)
    else:
        print(
            f"No URLs given and {args.sources} does not exist yet.\n"
            "The corpus (8 real + 2 synthetic, DESIGN.md 7.1) has not been chosen. "
            "Create that file or pass URLs positionally.",
            file=sys.stderr,
        )
        return 2

    if not entries:
        print(f"{args.sources} is empty; nothing to fetch.", file=sys.stderr)
        return 2

    if args.dry_run:
        for entry in entries:
            slug = entry.get("doc_slug") or slug_for_url(entry["url"])
            print(f"would fetch {slug:<28} {entry['url']}")
        return 0

    lock = load_lock(args.lock)
    changed_count = 0
    failures = 0

    for entry in entries:
        url = entry["url"]
        try:
            loaded = load_url(url, use_cache=not args.refresh)
        except Exception as exc:
            # One dead URL must not abandon the other seven; a partial corpus is
            # recoverable, a half-written lockfile is not.
            print(f"FAIL  {url}\n      {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue

        slug = entry.get("doc_slug") or loaded.doc_slug
        prior = lock["fetched"].get(url)
        row = record(loaded, entry, prior)
        row["doc_slug"] = slug
        lock["fetched"][url] = row

        if prior and prior.get("raw_sha256") == loaded.raw_sha256:
            state = "unchanged"
        elif prior:
            state = f"CHANGED  {prior.get('raw_sha256', '?')[:12]} -> "
            state += loaded.raw_sha256[:12]
            changed_count += 1
        else:
            state = "new"
            changed_count += 1

        cached = " (cache)" if loaded.from_cache else ""
        print(f"{slug:<28} {row['chars']:>7} chars  {state}{cached}")

    written = write_lock(lock, args.lock)
    print(
        f"\n{len(entries) - failures} fetched, {changed_count} new/changed, "
        f"{failures} failed\ntext cached under {CACHE_DIR}/ (gitignored)\n"
        f"provenance written to {written} (committed, no policy text)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
