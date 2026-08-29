#!/usr/bin/env python
"""Did aut-strong's retrieval actually reach the clause that governs each probe?

STEP 2's standalone proof, and the reason it is a script rather than a unit test:
the claim needs both pinned models at their real weights, so it runs inside the
frozen image. This file owns the ANSWER KEY and makes the judgement; the agent
prints evidence and knows nothing about clauses.

    python scripts/check_aut_retrieval.py emit --out .tmp-queries.txt
    docker run --rm -i --entrypoint python aut-strong:local retrieval.py \
        < .tmp-queries.txt > .tmp-retrieval.jsonl
    python scripts/check_aut_retrieval.py check .tmp-retrieval.jsonl

On Windows use `emit --out` as shown - never `emit > file` - and wrap the docker
line in `cmd /c "..."`. PowerShell 5.1 has no `<` operator at all, and its `>`
writes UTF-16LE with a BOM whatever the command printed, so both redirections
have to come from cmd to stay byte-exact. `check` rejects a BOM by name rather
than failing later with a JSON error that blames this file.

WHY THE KEY IS COMPUTED AND NOT WRITTEN DOWN
A hand-listed set of expected chunk ordinals would be a second source of truth that
drifts the moment CHUNK_CHARS moves, and it would let a wrong answer look right.
Instead: a probe names a `target_rule_id`, that rule (or nested exception - see
lockfiles.py) carries `source_span` values that a test already guarantees are
verbatim policy text, and a span is "retrieved" exactly when some returned chunk's
[start, end) contains it. That is offset arithmetic against the policy file, so the
key is derived from the same lockfiles the run is scored against.

WHAT THIS DOES NOT IMPORT
Nothing from aut-strong/ and nothing from harness/. It reads the agent's JSON and
the repo's lockfiles as data. `harness/` is off limits by the DESIGN.md 1.4
separation the tripwire test enforces; aut-strong/ is off limits because importing
the agent to check the agent would make the chunk offsets self-confirming - the
whole point is that the numbers came out of the container.

A FAILURE HERE IS A FINDING, NOT A KNOB
If the reranker does not surface the governing clause, that is reportable as the
measured limit of retrieval engineering on this corpus. Do not tune CANDIDATE_K or
the window until this passes; that would fit the retrieval configuration to the
probe set, which is the thing DESIGN.md 7.3 calls a thumb on the scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POLICY_DIR = REPO / "policies"
PROBES_LOCK = REPO / "probes" / "probes.lock.json"
RULES_LOCK = REPO / "rules" / "rules.lock.json"

#: Three of the eleven over-promises in docs/results.md's failure table, chosen by
#: the SHAPE of the failure rather than by a measured retrieval trace: in each one the
#: governing text is an exclusion or a multi-condition carve-out, which is where cosine
#: similarity is weakest - an exclusion shares little vocabulary with the request it
#: excludes - and therefore the case that retrieval depth plus a cross-encoder is
#: supposed to address.
#:
#: These are deliberately NOT described as "the probes results.md records as retrieval
#: failures". It records no such thing, and nothing else does either: aut-naive/app.py
#: returns `retrieved_chunk_ids` on every reply, no harness module reads it, and the
#: 38-column audit schema has nowhere to put it. So for run 01a032fd a ranking failure
#: and a "had the text and ignored it" reasoning failure are indistinguishable. What
#: this check can establish is that the governing span REACHES aut-strong's top-k; it
#: cannot establish that a diagnosed cause was repaired, and STEP 7 must not say so.
#:
#: Overridable so the check can be widened to all 30 later, but these three are the
#: pre-registered claim.
DEFAULT_PROBES = (
    "P-acme-008-category_smuggling-001",
    "P-acme-008-condition_stripping-003",
    "P-acme-013-condition_stripping-001",
)


def load_probes(probe_ids: tuple[str, ...]) -> list[dict]:
    lock = json.loads(PROBES_LOCK.read_text(encoding="utf-8"))
    by_id = {p["probe_id"]: p for p in lock["probes"]}
    missing = [p for p in probe_ids if p not in by_id]
    if missing:
        raise SystemExit(f"no such probe in {PROBES_LOCK.name}: {missing}")
    return [by_id[p] for p in probe_ids]


def find_rule(rule_id: str) -> list[dict]:
    """The chain from a top-level rule down to `rule_id`, which may be nested.

    `target_rule_id` is allowed to name an exception rather than a rule, so a flat
    scan over top-level `rule_id`s silently reports a valid probe as unknown. The
    chain is returned rather than the node because a carve-out is only reachable
    through the exclusion above it: the ancestors are context the agent also needs,
    and reporting them separately keeps "found the carve-out" from being confused
    with "found the exclusion".
    """
    rules = json.loads(RULES_LOCK.read_text(encoding="utf-8"))["rules"]

    def walk(node: dict, chain: list[dict]) -> list[dict] | None:
        chain = chain + [node]
        if node.get("rule_id") == rule_id:
            return chain
        for child in node.get("exceptions") or []:
            found = walk(child, chain)
            if found:
                return found
        return None

    for rule in rules:
        found = walk(rule, [])
        if found:
            return found
    raise SystemExit(f"target_rule_id {rule_id!r} is in no rule or exception")


def spans_of(node: dict) -> list[str]:
    return [
        c["source_span"] for c in node.get("conditions") or [] if c.get("source_span")
    ]


def locate(text: str, span: str, *, where: str) -> tuple[int, int]:
    """Character offsets of `span` in `text`, insisting it is unique.

    A span occurring twice would make "contained in a returned chunk" ambiguous:
    the agent could retrieve one occurrence while the clause the probe attacks is
    the other, and the check would pass for the wrong reason. A test already asserts
    these spans are verbatim, so zero occurrences means the lockfile and the policy
    have diverged - which is a finding about the lockfile, not about retrieval.
    """
    first = text.find(span)
    if first < 0:
        raise SystemExit(f"{where}: source_span is not verbatim in the policy: {span!r}")
    if text.find(span, first + 1) >= 0:
        raise SystemExit(
            f"{where}: source_span occurs more than once, so containment is "
            f"ambiguous: {span!r}"
        )
    return first, first + len(span)


def cmd_emit(args: argparse.Namespace) -> int:
    """Write the probe texts, one per line, for the container's stdin.

    Emitted from the lockfile rather than retyped so the query the container scores
    is byte-identical to the one the 30-probe run will send. Multi-turn probes emit
    one line per turn, in order.

    PREFER `--out`, DO NOT PIPE THIS THROUGH A SHELL REDIRECT ON WINDOWS.
    Windows PowerShell 5.1's `>` writes UTF-16LE with a BOM whatever the content
    is, and `python x.py > f` on Windows encodes stdout in the locale codepage
    rather than UTF-8. Either one hands the container bytes it will decode as
    UTF-8, which corrupts the query silently as soon as a probe contains a single
    non-ASCII character - and a corrupted query still retrieves *something*, so
    the run looks successful and the ranking evidence is quietly about a different
    string. `--out` makes this file own the encoding, which is the only place that
    can guarantee it.
    """
    lines = [
        " ".join(turn.split())
        for probe in load_probes(tuple(args.probes))
        for turn in probe["turns"]
        if " ".join(turn.split())
    ]
    if args.out:
        # newline="\n" as well as utf-8: the container splits stdin on lines, and
        # CRLF would leave a trailing \r inside every query, which then differs
        # from the lockfile text the checker looks the record up by.
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(f"{line}\n" for line in lines))
        print(f"wrote {len(lines)} quer(ies) to {args.out} as utf-8 with LF endings")
    else:
        for line in lines:
            print(line)
    return 0


def _reencoded(path: Path, what: str) -> str:
    """The one error message for "these are not the bytes the container printed"."""
    return (
        f"{path} contains {what}, so it is not the bytes the container printed.\n"
        f"Windows PowerShell re-encodes redirected output: 5.1's `>` writes UTF-16LE "
        f"whatever the command emitted, and it has no `<` operator at all. Use cmd's "
        f"redirection on both sides, which is byte-exact:\n"
        f'  cmd /c "docker run --rm -i --entrypoint python aut-strong:local '
        f'retrieval.py < .tmp-queries.txt > {path.name}"\n'
        f"and generate the queries with `emit --out`, never `emit > file`."
    )


def read_evidence(path: Path) -> list[dict]:
    """The container's JSON lines, with a named error for the Windows shell trap.

    Read as bytes first and checked for a byte-order mark. `docker run ... > f`
    under Windows PowerShell 5.1 writes UTF-16LE with a BOM regardless of what the
    command printed, and the resulting UnicodeDecodeError or JSONDecodeError points
    at this file rather than at the shell that caused it. The container's own output
    is pure ASCII - json.dumps defaults to ensure_ascii - so anything here that is
    not ASCII-compatible was introduced in transit and the evidence is not what the
    container produced.
    """
    raw = path.read_bytes()
    for bom, label in (
        (b"\xff\xfe", "UTF-16LE"),
        (b"\xfe\xff", "UTF-16BE"),
        (b"\xef\xbb\xbf", "UTF-8"),
    ):
        if raw.startswith(bom):
            raise SystemExit(_reencoded(path, f"a {label} byte-order mark"))
    # A BOM check alone is not enough, and the gap is the one that matters: UTF-16
    # without a BOM DECODES as UTF-8 without error, because NUL is a valid UTF-8
    # codepoint. The result is a JSONDecodeError pointing at column 2 of this
    # file's parser rather than at the shell, which is the confusion this function
    # exists to remove. The container emits ASCII JSON, so a NUL byte is never
    # legitimate here whatever produced it.
    if b"\x00" in raw:
        raise SystemExit(_reencoded(path, "NUL bytes (so it is UTF-16 or UTF-32)"))
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(
            f"{path} is not valid UTF-8 ({exc}). The container emits ASCII, so this "
            f"file was re-encoded in transit; see the note above about `>`."
        ) from exc
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _short(span: str, limit: int = 52) -> str:
    """One-line form of a span for the attribution list."""
    return span if len(span) <= limit else span[: limit - 3] + "..."


def report_attribution(instances: list[dict], churn: list[int], top_k: int) -> None:
    """Of the difference from aut-naive, how much is the cross-encoder actually buying?

    aut-strong changes four things at once - depth, a reranker, the prompt, the
    temperature - so a bare "8 of 10 governing spans reached the top k" cannot be
    attributed to any one of them. Two counterfactuals are free to compute from this
    same evidence, because the container reports the dense ranking alongside the
    reranked one: the dense top 3 is aut-naive's depth, and the dense top-k is
    aut-strong's depth with the cross-encoder switched off. Printed on every run
    rather than worked out by hand afterwards, so that a configuration change which
    quietly makes the reranker load-bearing - or quietly makes it idle - surfaces
    here rather than in a reviewer's question.

    THIS BLOCK IS DIAGNOSTIC AND DOES NOT TOUCH THE EXIT CODE. The pre-registered
    criterion is "every governing span reaches the returned set". Widening it, or
    splitting it into tiers that a partial result can pass, after seeing a result is
    exactly the thumb on the scale DESIGN.md 7.3 warns about.
    """
    if not instances:
        return
    targets = [i for i in instances if i["target"]]
    print()
    print("=" * 78)
    print("ATTRIBUTION (diagnostic; does not affect pass/fail)")
    print(
        f"  governing span-instances: {len(instances)} "
        f"({len(targets)} target, {len(instances) - len(targets)} ancestor)"
    )
    for label, key in (
        ("dense top 3 (aut-naive's depth)", "naive_k3"),
        (f"dense top {top_k} (aut-strong's depth, no rerank)", "dense_k"),
        (f"reranked top {top_k} (as shipped)", "reranked"),
    ):
        n = sum(1 for i in instances if i[key])
        nt = sum(1 for i in targets if i[key])
        print(
            f"    reached under {label:<44} {n:>2}/{len(instances)}"
            f"   target {nt}/{len(targets)}"
        )

    gained = [i for i in instances if i["reranked"] and not i["dense_k"]]
    lost = [i for i in instances if i["dense_k"] and not i["reranked"]]
    print(
        f"  cross-encoder against depth alone: "
        f"+{len(gained)} / -{len(lost)} span-instances"
    )
    for i in gained:
        print(f"    + promoted into the returned set: {i['probe']} {_short(i['span'])!r}")
    for i in lost:
        print(f"    - demoted out of the returned set: {i['probe']} {_short(i['span'])!r}")
    print(f"  slots the rerank changed against dense top {top_k}, per probe: {churn}")
    if not gained and not lost:
        print(
            "  READ THIS BEFORE WRITING UP A RERANKING GAIN: the reranked set holds\n"
            "  exactly the same governing spans as the plain dense top "
            f"{top_k}, so on this\n"
            "  evidence every gain over aut-naive is the DEPTH increase and the\n"
            "  cross-encoder's contribution to span presence is zero. Not 'the\n"
            "  reranker does not work' - the churn line above shows it re-selects\n"
            "  real slots, and ordering may still matter to the model through\n"
            "  position effects, which nothing here measures. The claim it does not\n"
            "  support is 'reranking surfaced the governing clause'."
        )


def cmd_check(args: argparse.Namespace) -> int:
    records = read_evidence(Path(args.evidence))
    headers = [r for r in records if r.get("kind") == "header"]
    queries = [r for r in records if r.get("kind") == "query"]
    if len(headers) != 1:
        raise SystemExit(f"expected exactly 1 header record, got {len(headers)}")
    header = headers[0]

    print("aut-strong retrieval configuration, as reported by the container")
    for key in (
        "chunk_chars",
        "overlap_chars",
        "n_chunks",
        "candidate_k",
        "top_k",
        "embedder",
        "reranker",
        "fingerprint",
    ):
        print(f"  {key:<14} {header[key]}")

    # The offsets below are resolved against this repo's policy text. If the
    # container baked different bytes, every containment answer would be quietly
    # wrong rather than loudly wrong.
    texts: dict[str, str] = {}
    for name, digest in header["corpus"].items():
        path = POLICY_DIR / name
        if not path.is_file():
            raise SystemExit(f"container reported a corpus file this repo lacks: {name}")
        local = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if local != digest:
            raise SystemExit(
                f"corpus mismatch on {name}\n  container {digest}\n  policies/ {local}\n"
                f"The chunk offsets cannot be interpreted against different bytes."
            )
        texts[name] = path.read_text(encoding="utf-8")
        print(f"  corpus         {name} {digest} (matches policies/)")
    if len(texts) != 1:
        raise SystemExit("this check assumes a single-document corpus")
    text = next(iter(texts.values()))

    by_query = {r["query"]: r for r in queries}
    failures: list[str] = []
    instances: list[dict] = []
    churn: list[int] = []
    print()

    for probe in load_probes(tuple(args.probes)):
        chain = find_rule(probe["scenario"]["target_rule_id"])
        target = chain[-1]
        print("=" * 78)
        print(f"{probe['probe_id']}   policy says: {probe['expected_policy_stance']}")
        print(f"  rule chain: {' > '.join(n['rule_id'] for n in chain)}")

        # The last turn is the one carrying the ask, and retrieval runs per message.
        turn = " ".join(probe["turns"][-1].split())
        record = by_query.get(turn)
        if record is None:
            failures.append(f"{probe['probe_id']}: no evidence line for its last turn")
            print("  *** NO EVIDENCE for this probe's last turn ***")
            continue

        dense3 = {h["ordinal"] for h in record["dense"][:3]}
        dense_k = {h["ordinal"] for h in record["dense"][: header["top_k"]]}
        dense_all = {h["ordinal"] for h in record["dense"]}
        returned = {h["ordinal"] for h in record["reranked"]}
        rank_of = {h["ordinal"]: i for i, h in enumerate(record["reranked"], start=1)}
        dense_rank_of = {h["ordinal"]: h["dense_rank"] for h in record["dense"]}
        churn.append(len(dense_k - returned))

        for node, own in [(n, n is target) for n in chain]:
            for span in spans_of(node):
                lo, hi = locate(text, span, where=node["rule_id"])
                # Over the FULL chunk table, so a span held by a chunk that never
                # became a candidate is diagnosed as a ranking miss rather than as
                # a chunk boundary problem.
                holders = {
                    c["ordinal"]
                    for c in header["chunks"]
                    if c["start"] <= lo and hi <= c["end"]
                }
                hit = sorted(holders & returned)
                # Recorded for report_attribution: was this span present at
                # aut-naive's depth, at aut-strong's depth with the cross-encoder
                # off, and as shipped? Appended for every span a window holds whole,
                # including the ones that miss, so the denominator is the full
                # governing set rather than the set that happened to succeed.
                instances.append(
                    {
                        "probe": probe["probe_id"],
                        "target": own,
                        "span": span,
                        "naive_k3": bool(holders & dense3),
                        "dense_k": bool(holders & dense_k),
                        "reranked": bool(hit),
                    }
                )
                label = "TARGET  " if own else "ancestor"
                short = span if len(span) <= 58 else span[:55] + "..."
                if not holders:
                    # Genuinely split by every window: the reranker was never
                    # offered the span whole, which is chunker.py's business and
                    # would contradict its offset sweep.
                    print(f"  [{label}] SPLIT BY EVERY WINDOW  {short!r}")
                    if own:
                        failures.append(
                            f"{probe['probe_id']}: no window holds the target span "
                            f"whole: {span!r}"
                        )
                    continue
                if hit:
                    best = min(hit, key=lambda o: rank_of[o])
                    print(
                        f"  [{label}] reranked #{rank_of[best]:<2} "
                        f"(dense #{dense_rank_of[best]:<2}) chunk {best:<3} {short!r}"
                    )
                    if own and not (holders & dense3):
                        print(
                            "             ^ recovered: no chunk holding this span was "
                            "in the dense top 3, i.e. aut-naive's depth"
                        )
                else:
                    where = (
                        "in candidates but the rerank dropped it"
                        if holders & dense_all
                        else "never entered the candidate set"
                    )
                    print(f"  [{label}] MISSED ({where}), chunks {sorted(holders)}  {short!r}")
                    if own:
                        failures.append(
                            f"{probe['probe_id']}: target span not returned ({where}): "
                            f"{span!r}"
                        )

    report_attribution(instances, churn, header["top_k"])

    print("\n" + "=" * 78)
    if failures:
        print(f"FAIL - {len(failures)} target span(s) did not reach the returned set:")
        for f in failures:
            print(f"  - {f}")
        print(
            "\nThis is a finding to report, not a signal to retune CANDIDATE_K or the "
            "chunk window against these probes."
        )
        return 1
    print(
        f"PASS - every governing span reached aut-strong's top-{header['top_k']} for "
        f"all {len(args.probes)} probe(s)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    emit = sub.add_parser("emit", help="write probe texts for the container's stdin")
    emit.add_argument("--probes", nargs="+", default=list(DEFAULT_PROBES))
    emit.add_argument(
        "--out",
        help="write to this file as utf-8 with LF endings instead of to stdout; "
        "prefer it on Windows, where a shell redirect re-encodes the queries",
    )
    emit.set_defaults(func=cmd_emit)

    check = sub.add_parser("check", help="score the container's JSON lines")
    check.add_argument("evidence", help="path to the JSON lines the container printed")
    check.add_argument("--probes", nargs="+", default=list(DEFAULT_PROBES))
    check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
