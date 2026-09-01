"""clauseguard CLI - extract / generate / run / check (DESIGN.md 1.8).

Step 7 implements `run`. The other three subcommands are registered but refuse to
execute, because `--help` advertising a surface the build does not have is worse
than one that says so - and `check` especially: it is the gate (DESIGN.md 6), and
a `check` that quietly did nothing would exit 0 and read as a pass.

**`run` does not gate.** Its exit status says whether the run completed, not
whether the agent behaved. DESIGN.md 2 step 11 and the `--max-overpromise`
comparison belong to `check`; harness/execution/__init__.py gives the reason -
"the number has to be readable by someone who disagrees with the threshold" - and
putting the comparison here would make the number and the verdict about the
number the same code path. Exit 1 is reserved for the gate and never returned by
`run`, so a future `check` can claim it without redefining what `run` means.

The console summary is the text form of DESIGN.md 5.2's information hierarchy, in
its order: one number, the 2x3 matrix, the run-over-run strip, the failure table,
then the small print. 5.2's HTML page is Step 8. This exists so the slice is
readable now, and so the numbers a dashboard would eventually render can be
checked against a real run by eye before any HTML is written to render them.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, TextIO

from harness.audit import (
    AuditRow,
    AuditStore,
    Reconciliation,
    VerdictClass,
    iter_over_promises,
    new_run,
)
from harness.execution import (
    DEFAULT_AGENT_CONCURRENCY,
    DEFAULT_AGENT_TIMEOUT_S,
    DEFAULT_JUDGE_PACE_S,
    DEFAULT_PROBES_LOCK,
    DEFAULT_RULES_LOCK,
    AgentIdentity,
    HttpxAgentClient,
    LockfileError,
    ProbesLock,
    RulesLock,
    RunError,
    RunResult,
    clause_index,
    execute_run,
    load_probes,
    load_rules,
)
from harness.ingest import MANIFEST_PATH, Clause, PolicyDocument, ingest, load_manifest

from harness.extract.coverage import compute_coverage
from harness.extract.extractor import (
    DEFAULT_EXTRACTOR_MODEL,
    LitellmExtractorClient,
    extract_rules,
    resolve_extractor_model,
    resolve_extractor_temp,
)
from harness.probe_gen.adversary import (
    LitellmAdversaryClient,
    resolve_adversary_model,
    resolve_adversary_temp,
)
from harness.probe_gen.driver import generate_probes as probe_generate_probes

# `harness/judge/__init__.py` is empty, so the judge is imported by module path -
# the same way harness/execution/runner.py reaches it.
from harness.gate.check import check_run, resolve_threshold
from harness.gate.report import write_report, emit_annotations
from harness.judge.judge import resolve_judge_model, resolve_judge_temp

EXIT_OK: Final = 0
#: Reserved for `clauseguard check` exceeding `--max-overpromise`. `run` never
#: returns it: if findings and operational failure shared an exit code, CI could
#: not tell "the agent over-promised" from "the agent was unreachable", and the
#: second silently looks like the first.
EXIT_GATE_FAILED: Final = 1
EXIT_OPERATIONAL: Final = 2

ENV_PATH: Final = Path(".env")

#: DESIGN.md 5.2 item 5 wants judge kappa and oracle pass rate always visible.
#: Neither is computable from this slice, and both print as unavailable-with-a-
#: reason rather than as a number:
#:
#:   kappa  needs the 200-item gold set of DESIGN.md 9 (Days 4-6) plus a second
#:          labeller. Nothing in the repo holds human labels yet. L3 consistency
#:          (harness/judge/consistency.py) now runs, so `judge_agreement` is a
#:          real measurement on the over-promise cell - but self-agreement is not
#:          kappa. Three samples of one judge agreeing measures stability, not
#:          correctness, and DESIGN.md 4.2's kappa is judge-versus-human.
#:   oracle needs the verbatim-oracle agent of DESIGN.md 4.3 - one handed the
#:          single correct clause - which does not exist yet.
#:
#: Printing 0.0 for either would be the worst option on the table: kappa 0.00
#: reads as a broken judge and a 0% oracle pass rate reads as a broken probe set,
#: and both would be claims this build has not measured. The line stays visible
#: because 5.2's point is that reliability numbers are not hidden in an appendix -
#: and "not yet measured" is itself a reliability fact worth showing.
KAPPA_UNAVAILABLE: Final = (
    "not measured (needs the 200-item gold set, DESIGN.md 9 Days 4-6)"
)
ORACLE_UNAVAILABLE: Final = (
    "not measured (needs the verbatim-oracle agent, DESIGN.md 4.3)"
)

#: The failure table marks spans with these instead of ANSI colour. Wrapping is
#: done with textwrap, which counts escape sequences as visible width and would
#: mis-wrap coloured bodies; plain markers survive redirection into a file too,
#: and this output is meant to be pasted into a bug report.
SPAN_OPEN: Final = ">>"
SPAN_CLOSE: Final = "<<"


class CliError(Exception):
    """An operational failure with a message already fit for a terminal."""


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------
def resolve_policy(
    policy_doc: str,
    *,
    source_override: str | None = None,
    manifest_path: Path = MANIFEST_PATH,
) -> PolicyDocument:
    """Rebuild the `PolicyDocument` named by `probes.lock.json`.

    DESIGN.md 1.8 gives `run` only `--probes` and `--agent`, so the policy has to
    be recovered rather than passed. The manifest already records where the
    document came from and what role it plays, so this reuses the recorded values
    instead of taking them as flags.

    That is deliberate for `corpus_role` in particular. `ingest` makes it
    keyword-only and required so that "a caller cannot quietly promote a fixture
    into evidence" (DESIGN.md 7.1 forbids pooling real and synthetic results); a
    `--corpus-role` flag on `run` would hand exactly that power back to whoever
    types the command. The role was decided when the document was ingested, and
    this reads that decision rather than re-making it.

    Re-ingesting is what makes the staleness check real: the clause text is
    re-hashed here, so `assert_matches_policy` compares the locks against the
    markdown as it is now, not against a hash the manifest is asserting about
    itself.
    """
    manifest = load_manifest(manifest_path)
    documents = manifest.get("documents", {})
    entry = documents.get(policy_doc)
    if entry is None:
        known = ", ".join(sorted(documents)) or "(none)"
        raise CliError(
            f"the probe set is for policy {policy_doc!r}, which is not in "
            f"{manifest_path}. Known documents: {known}. Ingest the policy first "
            f"so its clause hashes are on record - a run cannot verify probes "
            f"against a document it has never read."
        )

    source = source_override or entry.get("source")
    if not source:
        raise CliError(
            f"{manifest_path} has no `source` for {policy_doc!r}, so there is no "
            f"file to re-read. Pass --policy explicitly."
        )
    if not Path(source).is_file():
        raise CliError(
            f"{policy_doc!r} was ingested from {source!r}, which is not a file "
            f"now. Pass --policy to point at where it moved to."
        )

    corpus_role = entry.get("corpus_role")
    if not corpus_role:
        raise CliError(
            f"{manifest_path} records no `corpus_role` for {policy_doc!r}. "
            f"DESIGN.md 7.1 forbids pooling real and synthetic results, so this "
            f"is not defaulted here; re-ingest the document to record it."
        )

    return ingest(
        source,
        corpus_role=corpus_role,
        doc_slug=policy_doc,
        is_holdout=bool(entry.get("is_holdout", False)),
    )


# ---------------------------------------------------------------------------
# Judge credentials
# ---------------------------------------------------------------------------
def ensure_judge_credentials(*, stream: TextIO) -> str:
    """Put the provider key in the environment for litellm, and report where from.

    Fails before the agent phase rather than during judging. The judge runs paced
    and last, so a missing key otherwise surfaces after every probe has already
    been sent to the agent - the most expensive possible moment to learn it.

    Only the *provenance* is ever printed. The key itself is never echoed, never
    logged, and is not passed on a command line: it is read out of `.env` into the
    process, because `$env:GROQ_API_KEY=...` in PowerShell writes the secret to
    `ConsoleHost_history.txt` on disk.
    """
    model = resolve_judge_model()
    if not model.startswith("groq/"):
        # A local judge (Ollama) needs no key, and demanding one would make the
        # offline topology unrunnable for no reason.
        return f"judge {model} needs no provider key"

    from_env = (os.getenv("GROQ_API_KEY") or "").strip()
    if from_env:
        print(
            "  note: GROQ_API_KEY came from the process environment, which WINS "
            "over .env. If you have just rotated the key, the old value may still "
            "be shadowing the new one (Remove-Item Env:\\GROQ_API_KEY).",
            file=stream,
        )
        return "process environment"

    if not ENV_PATH.is_file():
        raise CliError(
            f"judge {model} needs GROQ_API_KEY and there is no {ENV_PATH} to read "
            f"it from. Create it from .env.example; do not export the key in the "
            f"shell, because PowerShell writes $env: assignments to "
            f"ConsoleHost_history.txt."
        )

    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        if name.strip() == "GROQ_API_KEY":
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ["GROQ_API_KEY"] = value
                return str(ENV_PATH)

    raise CliError(
        f"judge {model} needs GROQ_API_KEY, and {ENV_PATH} does not set it to a "
        f"non-empty value."
    )


# ---------------------------------------------------------------------------
# Rendering - DESIGN.md 5.2's hierarchy, in text
# ---------------------------------------------------------------------------
def _rule(stream: TextIO, char: str = "-", width: int = 78) -> None:
    print(char * width, file=stream)


def _one_line(text: str, limit: int = 200) -> str:
    """Collapse a probe turn or agent reply to one line for a table cell."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _mark_span(body: str, span: str | None) -> str:
    """Wrap `span` in markers wherever it occurs in `body`.

    Falls back to an explicit annotation when the span is not found verbatim.
    That case is worth seeing rather than hiding: for `quoted_span` it means the
    judge quoted something that is not in the clause, which is the C2 violation
    the whole span check exists to catch.
    """
    if not span:
        return body
    flat_body = " ".join(body.split())
    flat_span = " ".join(span.split())
    if flat_span and flat_span in flat_body:
        return flat_body.replace(flat_span, f"{SPAN_OPEN}{flat_span}{SPAN_CLOSE}", 1)
    return (
        f"{flat_body}\n"
        f"[span not found verbatim in the text above: {flat_span!r}]"
    )


def _wrap(body: str, *, indent: str = "      ", width: int = 78) -> str:
    lines: list[str] = []
    for paragraph in body.split("\n"):
        lines.extend(
            textwrap.wrap(
                paragraph,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
            )
            or [indent.rstrip()]
        )
    return "\n".join(lines)


def render_headline(result: RunResult, *, stream: TextIO) -> None:
    """DESIGN.md 5.2 item 1: one number, huge, with the two-sided cost beneath it.

    Under-serve sits next to over-promise because 5.2 says showing it "immediately
    signals you understand two-sided cost", and DESIGN.md 8 expects it to come out
    *higher* than over-promise on defensive prompts. A summary that reported only
    over-promises would make a uselessly cautious agent look perfect.
    """
    counts = tally(result.rows)
    attempted = len(result.rows)
    evasive = (
        counts[VerdictClass.EVASIVE_ON_GRANT] + counts[VerdictClass.EVASIVE_ON_DENIAL]
    )
    abstained = sum(1 for row in result.rows if row.judge_abstained)

    _rule(stream, "=")
    print(f"  OVER-PROMISES: {result.over_promises} / {attempted}", file=stream)
    print(
        f"  UNDER-SERVE: {result.under_serves}"
        f"  ·  EVASIVE: {evasive}"
        f"  ·  JUDGE ABSTAINED: {abstained}",
        file=stream,
    )
    _rule(stream, "=")


def tally(rows: Sequence[AuditRow]) -> dict[VerdictClass, int]:
    """Count every cell, including the ones no row landed in."""
    counts = {cell: 0 for cell in VerdictClass}
    for row in rows:
        if row.verdict_class is not None:
            counts[row.verdict_class] += 1
    return counts


def render_matrix(rows: Sequence[AuditRow], *, stream: TextIO) -> None:
    """DESIGN.md 5.2 item 2: the 2x3 matrix, over-promise cell marked.

    Marked with `<-- ` rather than coloured, so the shape survives being piped
    into a file. Rows the judge abstained on or errored on have no
    `verdict_class` and so appear in no cell; they are printed underneath rather
    than dropped, because a matrix whose cells do not sum to the probe count
    invites the reader to assume the difference was a pass.
    """
    counts = tally(rows)
    grid = {
        ("grants", "grants"): VerdictClass.CORRECT_GRANT,
        ("grants", "denies"): VerdictClass.UNDER_SERVE,
        ("grants", "evasive"): VerdictClass.EVASIVE_ON_GRANT,
        ("denies", "grants"): VerdictClass.OVER_PROMISE,
        ("denies", "denies"): VerdictClass.CORRECT_DENIAL,
        ("denies", "evasive"): VerdictClass.EVASIVE_ON_DENIAL,
    }

    print("\n  policy \\ agent      grants     denies    evasive", file=stream)
    for policy_stance in ("grants", "denies"):
        cells = []
        for agent_stance in ("grants", "denies", "evasive"):
            cell = grid[(policy_stance, agent_stance)]
            cells.append(f"{counts[cell]:>10}")
        flag = (
            "   <-- over-promise: the cell that matters"
            if policy_stance == "denies"
            else ""
        )
        print(f"  {policy_stance:<16}{''.join(cells)}{flag}", file=stream)

    scored = sum(counts.values())
    unscored = len(rows) - scored
    if unscored:
        abstained = sum(1 for row in rows if row.judge_abstained)
        errored = sum(1 for row in rows if row.judge_error is not None)
        print(
            f"\n  {unscored} row(s) are in no cell "
            f"({abstained} abstained, {errored} errored): the judge returned no "
            f"stance, so there is no pair to classify.",
            file=stream,
        )


def render_history(
    store: AuditStore, current_run_id: str, *, stream: TextIO, limit: int = 10
) -> None:
    """DESIGN.md 5.2 item 3: the run-over-run strip, with new and fixed named.

    The bars are the count per run; "new failures in red, fixed in green" becomes
    a NEW/FIXED probe list against the immediately preceding run, which is more
    useful in text than colour would be - it names the probe to go and read.
    """
    run_ids = store.run_ids()
    recent = run_ids[-limit:]
    print(f"\n  last {len(recent)} run(s) - over-promises per run:", file=stream)
    for run_id in recent:
        count = store.over_promise_count(run_id)
        marker = "  <-- this run" if run_id == current_run_id else ""
        bar = "#" * min(count, 40)
        print(f"    {run_id}  {count:>4}  {bar}{marker}", file=stream)

    if len(run_ids) < 2:
        print(
            "    (no previous run in this database, so there is no regression "
            "story yet - the strip needs at least two)",
            file=stream,
        )
        return

    previous_id = run_ids[-2] if run_ids[-1] == current_run_id else None
    if previous_id is None:
        # The current run is not the newest in the file. Comparing it against
        # "the one before the newest" would print a diff between two runs the
        # reader did not ask about, which is worse than printing nothing.
        print(
            "    (this run is not the newest in the database; skipping the "
            "new/fixed diff rather than comparing two other runs)",
            file=stream,
        )
        return

    now = {
        row.probe_id
        for row in store.latest_rows(current_run_id)
        if row.is_over_promise
    }
    before = {
        row.probe_id for row in store.latest_rows(previous_id) if row.is_over_promise
    }
    new_failures = sorted(now - before)
    fixed = sorted(before - now)
    print(f"    vs {previous_id}:", file=stream)
    print(
        f"      NEW   ({len(new_failures)}): "
        f"{', '.join(new_failures) if new_failures else '-'}",
        file=stream,
    )
    print(
        f"      FIXED ({len(fixed)}): {', '.join(fixed) if fixed else '-'}",
        file=stream,
    )


def render_failures(
    rows: Sequence[AuditRow],
    clauses: dict[str, Clause],
    *,
    stream: TextIO,
    limit: int | None = None,
) -> None:
    """DESIGN.md 5.2 item 4: the failure table, both spans marked.

    "Two highlighted spans side by side is the entire product in one visual" - the
    committing span in the agent's reply, and the contradicting span in the clause
    it was checked against. `iter_over_promises` already orders these so rows that
    can show both come first.
    """
    failures = list(iter_over_promises(rows))
    print(f"\n  OVER-PROMISES ({len(failures)}), worst-evidenced last:", file=stream)
    if not failures:
        print(
            f"    none. On this probe set that is a claim about {len(rows)} "
            f"probe(s), not about the agent in general.",
            file=stream,
        )
        return

    shown = failures if limit is None else failures[:limit]
    for row in shown:
        _rule(stream)
        print(
            f"  {row.probe_id}   [{row.strategy}, tier {row.difficulty_tier}]",
            file=stream,
        )
        print(f"    policy says: {row.expected_policy_stance}"
              f"   (rule {row.rule_id})", file=stream)
        print(f"    agent said : {row.agent_stance}"
              f"   (asserted: {row.entitlement_asserted})", file=stream)

        print("    probe:", file=stream)
        for turn_number, turn in enumerate(row.probe_turns, start=1):
            print(_wrap(f"t{turn_number}: {_one_line(turn)}"), file=stream)

        print("    agent response (committing span marked):", file=stream)
        print(_wrap(_mark_span(row.agent_response, row.response_span)), file=stream)

        clause = clauses.get(row.cited_clause_id or "")
        if clause is None:
            print(
                f"    cited clause: {row.cited_clause_id!r} is not in the loaded "
                f"policy",
                file=stream,
            )
        else:
            verified = {True: "verified", False: "NOT VERIFIED", None: "no span"}[
                row.span_verified
            ]
            print(
                f"    policy clause {clause.clause_id} "
                f"(contradicting span marked, L2: {verified}):",
                file=stream,
            )
            print(_wrap(_mark_span(clause.text, row.quoted_span)), file=stream)

    if limit is not None and len(failures) > limit:
        _rule(stream)
        print(
            f"  ... {len(failures) - limit} more not shown (--max-failures "
            f"{limit}). Every one is in the audit database.",
            file=stream,
        )


def render_small_print(
    result: RunResult,
    reconciliation: Reconciliation,
    *,
    db_path: Path,
    stream: TextIO,
) -> None:
    """DESIGN.md 5.2 item 5: the reliability numbers, always visible.

    Including the two this build cannot compute. See KAPPA_UNAVAILABLE and
    ORACLE_UNAVAILABLE for why they are words rather than numbers.
    """
    rows = result.rows
    # Each from the field it names, rather than the third by subtraction: L2's
    # outcome lives in `span_verified` and whether there was anything to check
    # lives in `quoted_span`, and those are two questions. The audit row
    # permits a False beside a null span, so subtracting one from a total over
    # the other would silently double-count such a row.
    verified = sum(1 for row in rows if row.span_verified is True)
    unverified = sum(1 for row in rows if row.span_verified is False)
    quoted_nothing = sum(1 for row in rows if not row.quoted_span)

    # Three buckets, each counted from what the row actually is, and a total
    # obtained independently of them - so a set that stopped partitioning
    # would show up as three numbers that no longer sum, rather than as one
    # bucket quietly absorbing the difference. That was the bug: `used_llm`
    # is False both for a row L0 settled and for a row whose judge call
    # failed (that one carries `outcome=None`), so "the rest were settled by
    # the L0 pre-filter" credited the pre-filter with the provider's
    # failures. Measured on run 01a032fd - L0 settled 10 rows and the small
    # print reported 12.
    llm_judged = sum(1 for judged in result.judged if judged.used_llm)
    l0_settled = sum(
        1
        for judged in result.judged
        if judged.outcome is not None and not judged.outcome.used_llm
    )
    never_judged = sum(1 for judged in result.judged if judged.outcome is None)

    print("\n  small print", file=stream)
    print(f"    probes attempted   : {reconciliation.attempted}", file=stream)
    print(
        f"    scorable / abstained / errored: {reconciliation.scorable} / "
        f"{reconciliation.abstained} / {reconciliation.errored}",
        file=stream,
    )
    print(
        f"    abstain rate       : {reconciliation.abstain_rate:.1%} "
        f"(errors excluded from the denominator, DESIGN.md 4.2)",
        file=stream,
    )
    print(f"    error rate         : {reconciliation.error_rate:.1%}", file=stream)
    print(
        f"    L2 spans           : {verified} verified, {unverified} not verified, "
        f"{quoted_nothing} quoted nothing (of {len(rows)} rows)",
        file=stream,
    )
    print(
        f"    judge              : {resolve_judge_model()} at temperature "
        f"{resolve_judge_temp()}",
        file=stream,
    )
    print(
        f"    judge routing      : {llm_judged} LLM, {l0_settled} L0 pre-filter, "
        f"{never_judged} never judged (of {len(result.judged)} rows)",
        file=stream,
    )
    print(f"    judge kappa        : {KAPPA_UNAVAILABLE}", file=stream)
    print(f"    oracle pass rate   : {ORACLE_UNAVAILABLE}", file=stream)
    print(f"    policy version     : {result.policy.policy_version}", file=stream)
    print(
        f"    agent              : {result.identity.aut_name} @ "
        f"{result.identity.aut_commit_sha}",
        file=stream,
    )
    print(f"    harness git sha    : {rows[0].git_sha if rows else '(no rows)'}",
          file=stream)
    print(f"    audit database     : {db_path}", file=stream)
    print(
        f"    timing             : {result.agent_seconds:.1f}s agent + "
        f"{result.judge_seconds:.1f}s judge = {result.elapsed_seconds:.1f}s",
        file=stream,
    )
    if reconciliation.superseded:
        print(
            f"    superseded rows    : {reconciliation.superseded}",
            file=stream,
        )


def render_identity(identity: AgentIdentity, *, stream: TextIO) -> None:
    """C3: the agent under test, and whether its freeze is real.

    Printed before the run, not just in the small print, because the whole claim
    of the tool is that the agent could not have been tuned to the probes. An
    unfrozen agent does not stop the run - `--require-frozen` does that - but it
    must never be possible to read a result without seeing which one it was.
    """
    print(f"  agent      : {identity.aut_name}", file=stream)
    print(f"  commit sha : {identity.aut_commit_sha}", file=stream)
    print(f"  repo head  : {identity.aut_repo_head}", file=stream)
    print(f"  git tag    : {identity.aut_git_tag}", file=stream)
    print(f"  frozen at  : {identity.aut_frozen_at}", file=stream)
    if not identity.is_frozen:
        print(
            "  WARNING: this agent was built outside scripts/freeze_aut.py, so "
            "nothing proves it predates the probes. Commitment C3 is not "
            "evidenced for this run. Re-run with --require-frozen to make this "
            "an error.",
            file=stream,
        )


# ---------------------------------------------------------------------------
# `clauseguard run`
# ---------------------------------------------------------------------------
def cmd_run(
    args: argparse.Namespace,
    *,
    stream: TextIO = sys.stdout,
    agent_factory: Callable[..., object] = HttpxAgentClient,
    judge_client: object | None = None,
) -> int:
    """Load the locks, run the probes, print DESIGN.md 5.2's content. No gate.

    `agent_factory` and `judge_client` are the seams the offline suite substitutes
    at, mirroring `AgentClient` in harness/execution/runner.py ("the seam the
    offline test suite substitutes at") and `execute_run`'s own `judge_client`.
    `main` never passes either, so the production path is the default one - but
    without them the console summary could only ever be checked by a live run
    against a container, and formatting bugs would be found by demo.
    """
    try:
        probes_lock: ProbesLock = load_probes(args.probes)
        rules_lock: RulesLock = load_rules(args.rules)
    except LockfileError as exc:
        raise CliError(str(exc)) from exc

    policy = resolve_policy(
        probes_lock.policy_doc,
        source_override=args.policy,
        manifest_path=Path(args.manifest),
    )

    # Three staleness checks, all before the first network call. `execute_run`
    # re-does the rules-vs-policy one; the probes-vs-rules one is the CLI's,
    # because `execute_run` receives a bare probe sequence and so cannot see the
    # digest the labels were computed under.
    try:
        probes_lock.assert_matches_policy(policy)
        rules_lock.assert_matches_policy(policy)
        probes_lock.assert_matches_rules(rules_lock)
    except LockfileError as exc:
        raise CliError(str(exc)) from exc

    # A substituted judge talks to no provider, so demanding a provider key would
    # only make the offline path need a secret it never sends anywhere.
    provenance = (
        "substituted judge (no provider key needed)"
        if judge_client is not None
        else ensure_judge_credentials(stream=stream)
    )

    print(f"  policy     : {policy.doc_slug} ({len(policy.clauses)} clauses)",
          file=stream)
    print(f"  probes     : {len(probes_lock.probes)} from {probes_lock.path}",
          file=stream)
    print(f"  rules      : {rules_lock.digest}", file=stream)
    print(f"  judge key  : {provenance}", file=stream)

    agent = agent_factory(args.agent, timeout_s=args.agent_timeout)
    store, run_id = new_run(args.db)

    total = len(probes_lock.probes)

    def on_agent_done(done: int) -> None:
        print(f"\r  agent: {done}/{total} probes", end="", file=stream, flush=True)

    def on_judge_progress(done: int, of: int, _judged: object) -> None:
        print(f"\r  judge: {done}/{of} exchanges", end="", file=stream, flush=True)

    try:
        result = execute_run(
            probes=probes_lock.probes,
            rules=rules_lock,
            policy=policy,
            agent=agent,
            store=store,
            run_id=run_id,
            agent_concurrency=args.concurrency,
            judge_client=judge_client,
            judge_pace_s=args.judge_pace,
            require_frozen=args.require_frozen,
            gate_run=False,  # `check` sets this; `run` is not the gate.
            on_agent_done=on_agent_done,
            on_judge_progress=on_judge_progress,
        )
        print("", file=stream)  # close the progress line
    except RunError as exc:
        print("", file=stream)
        raise CliError(str(exc)) from exc
    finally:
        store.close()
        # `HttpxAgentClient` owns the AsyncClient it made, and closing it needs a
        # loop, so it happens here rather than at interpreter exit. A failure to
        # close must not replace whatever the run was already raising. A
        # substituted agent need not have an `aclose` at all.
        aclose = getattr(agent, "aclose", None)
        if aclose is not None:
            import anyio

            try:
                anyio.run(aclose)
            except Exception:  # noqa: BLE001 - see above
                pass

    render_identity(result.identity, stream=stream)
    for warning in result.warnings:
        print(f"  warning: {warning}", file=stream)

    render_headline(result, stream=stream)
    render_matrix(result.rows, stream=stream)

    with AuditStore(args.db) as history:
        render_history(history, result.run_id, stream=stream)

    render_failures(
        result.rows,
        clause_index(result.policy),
        stream=stream,
        limit=args.max_failures,
    )
    render_small_print(
        result, result.reconciliation, db_path=Path(args.db), stream=stream
    )

    print(
        f"\n  run {result.run_id} complete. This command reports; it does not "
        f"gate - `clauseguard check` owns the pass/fail comparison against "
        f"--max-overpromise (DESIGN.md 6).",
        file=stream,
    )
    return EXIT_OK


def cmd_check(args: argparse.Namespace, *, stream: TextIO = sys.stdout) -> int:
    """`clauseguard check` — evaluate a run against a threshold."""
    store = AuditStore(Path(args.store)).initialise()
    run_ids = store.run_ids()
    if not run_ids:
        print("No runs in the audit store — nothing to gate on.", file=stream)
        return EXIT_OK

    run_id = args.run_id or run_ids[-1]
    return check_run(
        run_id,
        store,
        max_over_promise=args.max_overpromise,
        baseline="--baseline" if args.baseline else None,
        annotations=args.annotations,
        stream=stream,
    )


def cmd_report(args: argparse.Namespace, *, stream: TextIO = sys.stdout) -> int:
    """`clauseguard report` — generate clauseguard-report.md."""
    store = AuditStore(Path(args.store)).initialise()
    run_ids = store.run_ids()
    if not run_ids:
        print("No runs in the audit store.", file=stream)
        return EXIT_OPERATIONAL

    run_id = args.run_id or run_ids[-1]

    gate_passed: bool | None = None
    if args.gate_passed:
        gate_passed = True
    elif args.gate_failed:
        gate_passed = False

    report_path = write_report(
        run_id,
        store,
        output_dir=args.output,
        gate_passed=gate_passed,
        threshold=args.threshold,
    )
    print(f"Report written to {report_path}", file=stream)

    if args.annotations:
        rows = store.latest_rows(run_id)
        emit_annotations(rows, stream=stream)

    return EXIT_OK


def cmd_extract(args: argparse.Namespace, *, stream: TextIO = sys.stdout) -> int:
    """Extract candidate rules from a policy document (DESIGN.md 1.2)."""
    from pathlib import Path

    policy = ingest(args.policy, corpus_role="worked_example")
    print(f"policy      : {policy.doc_slug}  {policy.policy_version}", file=stream)
    print(f"clauses     : {len(policy.clauses)}", file=stream)

    model = args.model or resolve_extractor_model()
    warm = os.getenv("EXTRACTOR_WARM_MODEL", "")
    if warm:
        model = warm

    kw = dict(model=model)
    if args.timeout:
        kw["timeout_s"] = args.timeout
    if args.max_tokens:
        kw["max_tokens"] = args.max_tokens
    client = LitellmExtractorClient(**kw)

    print(f"extractor   : {client.model}  (temperature {resolve_extractor_temp()})", file=stream)
    if client.model != DEFAULT_EXTRACTOR_MODEL:
        print(
            f"  NOTE      : one-off comparison run — not the pinned "
            f"{DEFAULT_EXTRACTOR_MODEL}",
            file=stream,
        )
    print("extracting  : this may take a minute...", file=stream)

    rules = extract_rules(policy, client=client)
    out = Path(args.out) if args.out else Path("rules/rules.extracted.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    payload = {
        "schema": "clauseguard/rules.lock/1",
        "policy_doc": policy.doc_slug,
        "policy_version": policy.policy_version,
        "authored_by": f"extracted by harness/extract/extractor.py (model {client.model})",
        "rules": [r.model_dump(mode="json") for r in rules],
    }
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"written     : {out}  ({len(rules)} root rule(s))", file=stream)

    coverage = compute_coverage(policy, rules)
    print(coverage.summary(), file=stream)
    print(f"  DESIGN.md 8 band    : {coverage.band}", file=stream)
    print(f"  rules.lock.json     : NOT touched (hand-authored, reviewed)", file=stream)
    return EXIT_OK


def cmd_generate(args: argparse.Namespace, *, stream: TextIO = sys.stdout) -> int:
    """Generate a candidate probe set with the LLM adversary (DESIGN.md 3.1-3.4)."""
    from pathlib import Path
    import json

    from harness.execution.lockfiles import PROBES_LOCK_SCHEMA, rules_digest
    from harness.probe_gen.driver import STRATEGY_ORDER

    policy = ingest("policies/acme-refunds.md", corpus_role="worked_example")
    rules = load_rules(args.rules)
    print(f"policy      : {policy.doc_slug}  {policy.policy_version}", file=stream)
    print(f"rules       : {len(rules.rules)} root rule(s)", file=stream)

    model = args.model or resolve_adversary_model()
    client = LitellmAdversaryClient(model=model, timeout_s=args.timeout or 240)
    print(f"adversary   : {client.model}  (temperature {resolve_adversary_temp()})", file=stream)

    result = probe_generate_probes(
        rules.rules, policy, client=client, limit_per_rule=args.n_per_rule
    )

    out = Path(args.out) if args.out else Path("probes/probes.generated.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": PROBES_LOCK_SCHEMA,
        "policy_doc": policy.doc_slug,
        "policy_version": policy.policy_version,
        "rules_digest": rules_digest(rules.rules),
        "authored_by": f"generated by clauseguard generate (adversary {client.model}); oracle-checked",
        "probes": [p.model_dump(mode="json") for p in result["generated"]],
    }
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"written     : {out}  ({len(result['generated'])} probes)", file=stream)
    print(f"  probes.lock.json untouched (hand-authored source of truth)", file=stream)

    # ---- Report: raw per-strategy attempt / pass counts ----
    print("\n-- ORACLE / VALIDITY REPORT --", file=stream)
    s = result["stats"]
    print(f"  sampled        : {s.get('sampled', 0)}", file=stream)
    print(f"  unlabellable   : {s.get('unlabellable', 0)}", file=stream)
    print(f"  render_error   : {s.get('render_error', 0)}", file=stream)
    print(f"  oracle passed  : {s.get('oracle_passed', 0)}", file=stream)
    print(f"  oracle failed  : {s.get('oracle_failed', 0)}", file=stream)
    total = s.get("oracle_passed", 0) + s.get("oracle_failed", 0)
    if total:
        print(f"  oracle pass rate: {s.get('oracle_passed', 0)/total:.1%}", file=stream)

    print("\n-- STRATEGY DISTRIBUTION (attempted / passed) --", file=stream)
    for strat in STRATEGY_ORDER:
        attempted = result["attempted"].get(strat.value, 0)
        passed = result["passed"].get(strat.value, 0)
        print(f"  {strat.value:<22} attempted {attempted:>3}   passed {passed:>3}", file=stream)
    return EXIT_OK


def cmd_unimplemented(args: argparse.Namespace, *, stream: TextIO = sys.stdout) -> int:
    raise CliError(
        f"`clauseguard {args.command}` is not implemented in this build. "
        f"{args.reason} It is registered so that --help describes the real "
        f"DESIGN.md 1.8 surface rather than a smaller one."
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clauseguard",
        description=(
            "Policy-conformance harness for money-touching agents. "
            "`run` executes a frozen probe set and reports; `check` gates."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract", help="policy document -> candidate rules (DESIGN.md 1.2)"
    )
    extract.add_argument("--policy", required=True, help="Path to policy markdown file")
    extract.add_argument("--out", default=None, help="Output path (default: rules/rules.extracted.json)")
    extract.add_argument("--model", default=None, help="Override the extractor model (env: CLAUSEGUARD_EXTRACTOR_MODEL)")
    extract.add_argument("--timeout", type=int, default=None, help="Per-call timeout in seconds")
    extract.add_argument("--max-tokens", type=int, default=None, help="Output token cap per call")
    extract.set_defaults(func=cmd_extract)

    generate = subparsers.add_parser(
        "generate", help="rules -> oracle-checked candidate probe set (DESIGN.md 3.2)"
    )
    generate.add_argument("--rules", default=DEFAULT_RULES_LOCK)
    generate.add_argument("--n-per-rule", type=int, default=1,
                          help="Max fact vectors per (rule, strategy) (default 1)")
    generate.add_argument("--out", default=None,
                          help="Output path (default: probes/probes.generated.json)")
    generate.add_argument("--model", default=None,
                          help="Override the adversary model (env: CLAUSEGUARD_ADVERSARY_MODEL)")
    generate.add_argument("--timeout", type=int, default=None,
                          help="Per-call timeout in seconds")
    generate.set_defaults(func=cmd_generate)

    run = subparsers.add_parser(
        "run",
        help="run a probe set against an agent and report (no pass/fail)",
        description=(
            "Runs every probe in the lockfile against the agent, judges the "
            "replies, appends one audit row each, and prints DESIGN.md 5.2's "
            "summary. Exit status reflects whether the run completed, not what "
            "it found."
        ),
    )
    run.add_argument(
        "--probes",
        default=str(DEFAULT_PROBES_LOCK),
        help="probe lockfile (default: %(default)s)",
    )
    run.add_argument(
        "--agent",
        required=True,
        help="base URL of the agent under test, e.g. http://localhost:8000",
    )
    run.add_argument(
        "--rules",
        default=str(DEFAULT_RULES_LOCK),
        help="rule lockfile the probe labels must match (default: %(default)s)",
    )
    run.add_argument(
        "--policy",
        default=None,
        help=(
            "override the policy source path; by default it is read from the "
            "clause manifest, along with the corpus_role recorded at ingest"
        ),
    )
    run.add_argument(
        "--manifest", default=str(MANIFEST_PATH), help="clause manifest path"
    )
    run.add_argument("--db", default="runs.db", help="audit database")
    run.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_AGENT_CONCURRENCY,
        help="parallel agent requests (default: %(default)s)",
    )
    run.add_argument(
        "--agent-timeout",
        type=float,
        default=DEFAULT_AGENT_TIMEOUT_S,
        help="per-request agent timeout in seconds (default: %(default)s)",
    )
    run.add_argument(
        "--judge-pace",
        type=float,
        default=DEFAULT_JUDGE_PACE_S,
        help=(
            "seconds between judge calls, for the token ceiling "
            "(default: %(default)s)"
        ),
    )
    run.add_argument(
        "--require-frozen",
        action="store_true",
        help=(
            "refuse to run against an agent not built by scripts/freeze_aut.py "
            "(commitment C3)"
        ),
    )
    run.add_argument(
        "--max-failures",
        type=int,
        default=None,
        help="truncate the failure table to this many rows (default: all)",
    )
    run.set_defaults(func=cmd_run)

    check = subparsers.add_parser(
        "check",
        help="the gate: exit 0/1 against --max-overpromise or --baseline",
        description=(
            "Evaluates a completed run against an over-promise threshold and "
            "exits 0 (pass) or 1 (fail). The threshold can be an absolute number "
            "(--max-overpromise N) or the previous run's count (--baseline). "
            "Prints the failure table and, in CI, emits GitHub Actions annotations."
        ),
    )
    check.add_argument(
        "--run-id",
        default=None,
        help="Run ID to evaluate (default: most recent run in the store)",
    )
    check.add_argument(
        "--max-overpromise",
        type=int,
        default=None,
        help="Absolute over-promise threshold (default: 0)",
    )
    check.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Compare against the previous run's over-promise count",
    )
    check.add_argument(
        "--annotations",
        action="store_true",
        default=False,
        help="Emit GitHub Actions workflow command annotations",
    )
    check.add_argument(
        "--store",
        default="runs.db",
        help="Path to the audit store (default: runs.db)",
    )
    check.add_argument(
        "--probes",
        default=str(DEFAULT_PROBES_LOCK),
        help="Probe lockfile path for annotations (default: %(default)s)",
    )
    check.set_defaults(func=cmd_check)

    report = subparsers.add_parser(
        "report",
        help="generate clauseguard-report.md from a completed run",
        description=(
            "Writes a Markdown report for the run, with the failure table "
            "and optional gate status. Also emits GitHub Actions annotations "
            "when --annotations is set."
        ),
    )
    report.add_argument(
        "--run-id",
        default=None,
        help="Run ID to report on (default: most recent run in the store)",
    )
    report.add_argument(
        "--store",
        default="runs.db",
        help="Path to the audit store (default: runs.db)",
    )
    report.add_argument(
        "--output",
        default=".",
        help="Output directory for the report (default: current directory)",
    )
    report.add_argument(
        "--gate-passed",
        action="store_true",
        default=None,
        help="Mark the gate as passed in the report",
    )
    report.add_argument(
        "--gate-failed",
        action="store_true",
        default=None,
        help="Mark the gate as failed in the report",
    )
    report.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Gate threshold for the report",
    )
    report.add_argument(
        "--annotations",
        action="store_true",
        default=False,
        help="Emit GitHub Actions annotations to stderr",
    )
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CliError as exc:
        print(f"\nclauseguard: {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL
    except KeyboardInterrupt:
        print("\nclauseguard: interrupted. Rows already written are kept.",
              file=sys.stderr)
        return EXIT_OPERATIONAL


if __name__ == "__main__":
    raise SystemExit(main())
