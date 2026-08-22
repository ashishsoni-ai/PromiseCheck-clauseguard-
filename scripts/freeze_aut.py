"""Freeze an agent under test and record what was frozen. STEP 4.

DESIGN.md 0, C3: "The agent under test is frozen by commit SHA before any probe exists.
Written day 1, never touched again." DESIGN.md 7.3 adds that the freeze must be
"verifiable in git history", and 1.4 requires an `AUT_COMMIT_SHA` in every audit row.

WHY THE RECORD LIVES OUTSIDE THE FROZEN DIRECTORY
The obvious design - write the computed hash into `aut-naive/AUT_COMMIT_SHA` - eats itself.
The identity being recorded is the tree hash of `aut-naive/`, so writing a file into that
directory changes the tree and invalidates the value the file just recorded. So the record
is written here at the repo root, and the hashes reach the container as Docker build args.

WHY TWO HASHES
`git rev-parse HEAD` is the commit a reviewer can look up, but it moves every time the
harness is touched, so on its own it cannot evidence that the agent predates the probes.
`git rev-parse HEAD:aut-naive` is the tree hash of the agent alone: it changes if and only
if the agent changes. Both are recorded and both are reported by the running container.

Usage (from the repo root):

    python scripts/freeze_aut.py aut-naive --tag aut-naive-v1
    python scripts/freeze_aut.py aut-naive --tag aut-naive-v1 --build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FREEZE_PATH = Path("aut-freeze.json")
SCHEMA_VERSION = 1
IMAGE_PREFIX = "clauseguard"


class FreezeError(RuntimeError):
    pass


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        raise FreezeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def require_clean(aut_dir: str) -> None:
    """Refuse to freeze an agent with uncommitted changes.

    The tree hash is read from HEAD, so a dirty working copy would produce a record that
    describes committed code while the image gets built from something else. That is
    precisely the discrepancy C3 exists to rule out.
    """
    dirty = git("status", "--porcelain", "--", aut_dir)
    if dirty:
        raise FreezeError(
            f"{aut_dir} has uncommitted changes:\n{dirty}\n\n"
            "Commit them first - you cannot freeze a dirty agent."
        )


def tree_hash(aut_dir: str) -> str:
    try:
        return git("rev-parse", f"HEAD:{aut_dir}")
    except FreezeError as exc:
        raise FreezeError(
            f"{exc}\n\n{aut_dir} does not exist at HEAD; commit the directory first."
        ) from exc


def corpus_hashes(aut_dir: Path) -> dict[str, str]:
    """sha256 of each baked-in policy file.

    The tree hash already covers these, but recording them separately answers a question
    the tree hash cannot: *which policy text* was frozen in. A stale agent and a wrong
    agent are different findings, and this is what tells them apart.
    """
    corpus = aut_dir / "corpus"
    if not corpus.is_dir():
        return {}
    return {
        path.name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(corpus.glob("*"))
        if path.is_file()
    }


def ensure_tag(tag: str, commit: str) -> None:
    existing = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{tag}"],
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0:
        pointed_at = existing.stdout.strip()
        if pointed_at != commit:
            raise FreezeError(
                f"tag {tag} already exists and points at {pointed_at[:12]}, not "
                f"{commit[:12]}. Pick a new tag rather than moving a freeze marker - "
                "a moved tag makes every audit row that cited it unverifiable."
            )
        print(f"  tag {tag} already present at {commit[:12]}")
        return
    git("tag", "-a", tag, "-m", f"Freeze {tag} (DESIGN.md C3)", commit)
    print(f"  created tag {tag} at {commit[:12]}")


def write_record(aut: str, record: dict, path: Path = FREEZE_PATH) -> Path:
    """Merge into the freeze record. Deterministic bytes, house style."""
    payload: dict = {"schema_version": SCHEMA_VERSION, "auts": {}}
    if path.exists():
        raw = path.read_text(encoding="utf-8").strip()
        if raw and raw != "{}":
            payload = json.loads(raw)
            payload.setdefault("schema_version", SCHEMA_VERSION)
            payload.setdefault("auts", {})

    payload["auts"][aut] = record
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        # LF explicitly. This is the C3 freeze record - the one file a reviewer is most
        # likely to hash - so it must not differ byte-for-byte with the host OS.
        newline="\n",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze an agent under test (C3).")
    parser.add_argument("aut", help="agent directory, e.g. aut-naive")
    parser.add_argument("--tag", required=True, help="git tag, e.g. aut-naive-v1")
    parser.add_argument(
        "--build", action="store_true", help="run docker build with the freeze args"
    )
    args = parser.parse_args(argv)

    aut_dir = Path(args.aut)
    if not aut_dir.is_dir():
        print(f"error: {aut_dir} is not a directory", file=sys.stderr)
        return 2

    try:
        require_clean(args.aut)
        head = git("rev-parse", "HEAD")
        tree = tree_hash(args.aut)
        frozen_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        print(f"freezing {args.aut}")
        print(f"  repo HEAD  {head}")
        print(f"  tree hash  {tree}")
        ensure_tag(args.tag, head)

        image = f"{IMAGE_PREFIX}/{args.aut}:{args.tag}"
        record = {
            "git_tag": args.tag,
            "repo_head": head,
            "tree_hash": tree,
            "frozen_at": frozen_at,
            "image": image,
            "corpus": corpus_hashes(aut_dir),
        }
        path = write_record(args.aut, record)
        print(f"  recorded in {path}")

        build_cmd = [
            "docker",
            "build",
            "-t",
            image,
            "--build-arg",
            f"AUT_COMMIT_SHA={tree}",
            "--build-arg",
            f"AUT_REPO_HEAD={head}",
            "--build-arg",
            f"AUT_GIT_TAG={args.tag}",
            "--build-arg",
            f"AUT_FROZEN_AT={frozen_at}",
            args.aut,
        ]
        print("\n" + " ".join(build_cmd) + "\n")

        if args.build:
            return subprocess.run(build_cmd).returncode
        print("re-run with --build to build the image, or copy the command above.")
        return 0

    except FreezeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
