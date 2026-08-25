"""STEP 7 checkpoint tests - the `clauseguard run` CLI and its console summary.

NO NETWORK. Every test drives `cmd_run` through the two seams it exposes for the
purpose - `agent_factory` and `judge_client` - so the whole command runs without a
container, a key or a provider. That matters more here than in test_runner.py: the
console summary is the only part of this slice a human reads, and a formatting bug
in it is invisible to every other test in the suite.

WHAT THESE TESTS ARE FOR
------------------------
1. **`run` does not gate.** Exit 0 even with over-promises. This is the claim most
   likely to be "fixed" into a bug later by someone who reasonably assumes a tool
   that finds problems should fail. DESIGN.md gives the gate to `check`, and
   harness/execution/__init__.py gives the reason.
2. **The CLI owns `assert_matches_rules`.** `execute_run` receives a bare probe
   sequence and cannot see the digest the labels were computed under, so if the
   CLI drops this check nothing else catches it and the run reports stale ground
   truth as fact.
3. **DESIGN.md 5.2's content is all present** - the headline pair, the 2x3 matrix,
   the regression strip, both marked spans, and the small print.
4. **The two numbers this build cannot measure say so** rather than printing 0.0.

A NOTE ON THE LABELS IN THESE FIXTURES
--------------------------------------
`expected_policy_stance` is set by hand here. That is not a C1 violation and not a
claim about C1: these probes are fixtures for testing *rendering and wiring*, and
never leave the temp directory. C1 is the property that the labels in
`probes/probes.lock.json` come from `evaluate_rules()`, which is enforced by
scripts/author_probes.py at authoring time. Deriving these three fixture labels
from the engine would test the engine twice and the console table not at all.

WHY THE DEFAULT JUDGE IS `ExplodingJudge`
-----------------------------------------
Only the granting agent escalates to L1 ("grants is expensive, denies is free"),
so every test using a denying or evasive agent should make *zero* model calls. A
judge that fails the test when called turns that from an assumption into an
assertion, and it is the same claim DESIGN.md 4.1 charges the pre-filter with.
"""

from __future__ import annotations

import argparse
import json
import re
from io import StringIO
from types import SimpleNamespace

import pytest

from harness.cli import (
    EXIT_OK,
    EXIT_OPERATIONAL,
    KAPPA_UNAVAILABLE,
    ORACLE_UNAVAILABLE,
    SPAN_CLOSE,
    SPAN_OPEN,
    CliError,
    _mark_span,
    build_parser,
    cmd_run,
    main,
    resolve_policy,
)
from harness.execution.lockfiles import load_rules, write_probes, write_rules
from harness.ingest import diff_against_manifest, ingest, update_manifest
from harness.judge.consistency import L3_K
from harness.judge.judge import JudgeError
from harness.schemas.judgment import Judgment
from harness.schemas.probe import Probe, ProbeScenario, ProbeStrategy
from harness.schemas.rule import Condition, EntitlementRule
from tests.unit.test_runner import (
    DENYING_REPLY,
    EVASIVE_REPLY,
    FROZEN_SHA,
    GRANTING_REPLY,
    QUOTABLE_FROM_CLAUSE,
    QUOTABLE_FROM_REPLY,
    UNFROZEN_SHA,
    WINDOW_CLAUSE_TEXT,
    CyclingJudge,
    ExplodingJudge,
    FakeAgent,
    FakeJudge,
)

DOC_SLUG = "acme-returns"

#: Three paragraphs, each well over `MIN_CLAUSE_TOKENS` (40) and well under
#: `MAX_CLAUSE_TOKENS` (400), so the segmenter yields exactly one clause per
#: heading rather than merging or splitting them. Verified against the real
#: segmenter: 3 segments, at 95, 87 and 90 estimated tokens.
#:
#: The first sentence is `WINDOW_CLAUSE_TEXT` rather than a copy of it, so that a
#: test constructing a span and the clause that span must be found in cannot drift
#: apart - the same reason test_runner.py names these strings at all.
POLICY_MD = f"""# Acme Returns Policy

## 1. Return window

{WINDOW_CLAUSE_TEXT} The window is measured from
the date the carrier records successful delivery rather than from the date of
purchase, because dispatch delays are not attributable to the customer. Requests
received after the window has closed are declined at the first line of support and
are not eligible for discretionary extension by an agent.

## 2. Hygiene exclusions

Innerwear and swimwear are excluded from returns for hygiene reasons. This
exclusion applies to the entire category regardless of whether the item was worn,
and it operates independently of the return window described above, so an item in
an excluded category is refused even when the request arrives on the first day
after delivery was completed.

## 3. Escalation

A customer who disputes an outcome may escalate the decision to the support team
in writing within fourteen days of being notified. Escalation does not extend the
return window, does not create an entitlement that the policy withholds, and does
not oblige the reviewing agent to reach a different conclusion on the same set of
facts as were originally presented.
"""


def a_grant_quoting(clause_id: str) -> Judgment:
    """A grant judgment L2 will verify, citing the clause the probes all cite."""
    return Judgment(
        agent_stance="grants",
        entitlement_asserted="refund",
        cited_clause_id=clause_id,
        quoted_span=QUOTABLE_FROM_CLAUSE,
        response_span=QUOTABLE_FROM_REPLY,
        reasoning="The agent committed to a refund the clause does not allow.",
        confidence=0.9,
    )


def matrix_row(output: str, policy_stance: str) -> list[str]:
    """The four leading fields of one matrix row: label, then three cell counts."""
    line = next(
        one
        for one in output.splitlines()
        if one.strip().startswith(policy_stance) and "policy \\ agent" not in one
    )
    return line.split()[:4]


def small_print_counts(output: str, label: str, *buckets: str) -> dict[str, int]:
    """The `N <bucket>` pairs on one small-print line, plus its stated total.

    Parsed rather than substring-matched. `assert "0 L0 pre-filter" in output`
    passes on a line reading "10 L0 pre-filter", which is precisely the kind of
    off-by-a-population error these lines exist to make visible, so the
    assertions here compare integers. The total is returned under `"of"` and is
    read from the line rather than summed, so a test can check that the buckets
    account for every row.
    """
    line = next(one for one in output.splitlines() if label in one)
    alternatives = "|".join(re.escape(bucket) for bucket in buckets)
    found = {
        bucket: int(count)
        for count, bucket in re.findall(rf"(\d+) ({alternatives})", line)
    }
    missing = [bucket for bucket in buckets if bucket not in found]
    assert not missing, f"{label!r} line printed no count for {missing}: {line!r}"
    stated_total = re.search(r"\(of (\d+) rows\)", line)
    assert stated_total is not None, f"{label!r} line states no total: {line!r}"
    found["of"] = int(stated_total.group(1))
    return found


def judge_routing(output: str) -> dict[str, int]:
    """Where the run's rows went: to the model, to L0, or to neither."""
    return small_print_counts(
        output, "judge routing", "LLM", "L0 pre-filter", "never judged"
    )


def percent_on(output: str, label: str) -> float:
    """The percentage on one small-print line, as a number.

    Matched loosely on purpose: pinning the column alignment as well as the
    value would make a cosmetic change to the label width fail a test about a
    rate.
    """
    line = next(one for one in output.splitlines() if label in one)
    found = re.search(r"(\d+(?:\.\d+)?)%", line)
    assert found is not None, f"{label!r} line printed no percentage: {line!r}"
    return float(found.group(1))


@pytest.fixture
def slice_on_disk(tmp_path):
    """Every file `clauseguard run` reads, built through the real writers.

    Hermetic on purpose: no repo artefact and no working directory. A CLI test that
    read `probes/probes.lock.json` would fail whenever that file was mid-edit, and
    would be testing the probe set rather than the command.

    `corpus_role` is `synthetic_stress` rather than the real document's
    `worked_example` precisely so that a `resolve_policy` which hardcoded a role
    would fail here instead of passing by coincidence.
    """
    policy_md = tmp_path / f"{DOC_SLUG}.md"
    policy_md.write_text(POLICY_MD, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"

    policy = ingest(policy_md, corpus_role="synthetic_stress", doc_slug=DOC_SLUG)
    update_manifest(
        policy, diff_against_manifest(policy, manifest_path), manifest_path
    )

    window = next(c for c in policy.clauses if QUOTABLE_FROM_CLAUSE in c.text)
    rule = EntitlementRule(
        rule_id="R-001-a",
        clause_ids=[window.clause_id],
        entitlement="refund",
        polarity="grants",
        conditions=[
            Condition(
                attribute="days_since_delivery",
                op="<=",
                value=30,
                source_span=QUOTABLE_FROM_CLAUSE,
            )
        ],
        precedence=10,
        extraction_confidence=0.95,
        needs_human_review=False,
    )
    rules_path = tmp_path / "rules.lock.json"
    write_rules(rules_path, rules=[rule], policy=policy, authored_by="tests")
    rules = load_rules(rules_path)

    def probe(probe_id: str, stance: str, days: int) -> Probe:
        return Probe(
            probe_id=probe_id,
            scenario=ProbeScenario(
                facts={"days_since_delivery": days},
                target_rule_id="R-001-a",
                strategy=ProbeStrategy.BOUNDARY,
                difficulty_tier=2,
            ),
            turns=[f"I want a refund. It has been {days} days since delivery."],
            expected_policy_stance=stance,
            clause_ids=[window.clause_id],
        )

    # Two the policy refuses and one it allows. The mix matters: on a denies-only
    # set an agent that grants everything and an agent that refuses everything
    # cannot be told apart by the over-promise count alone, which is why
    # DESIGN.md 5.2 puts under-serve beside it.
    probes = [
        probe("P-001-boundary-001", "denies", 31),
        probe("P-001-boundary-002", "denies", 45),
        probe("P-001-boundary-003", "grants", 10),
    ]
    probes_path = tmp_path / "probes.lock.json"
    write_probes(
        probes_path, probes=probes, policy=policy, rules=rules, authored_by="tests"
    )

    return SimpleNamespace(
        policy=policy,
        window=window,
        policy_md=policy_md,
        manifest_path=manifest_path,
        rules_path=rules_path,
        probes_path=probes_path,
        db_path=tmp_path / "runs.db",
        probes=probes,
    )


@pytest.fixture
def argv_for(slice_on_disk):
    """A real argv for `run`, pointed entirely at the temp fixtures."""

    def _argv(*, probes=None, rules=None, extra=()):
        return [
            "run",
            "--agent",
            "http://agent.invalid",
            "--probes",
            str(probes or slice_on_disk.probes_path),
            "--rules",
            str(rules or slice_on_disk.rules_path),
            "--manifest",
            str(slice_on_disk.manifest_path),
            "--db",
            str(slice_on_disk.db_path),
            # Without this, every judged probe would wait DEFAULT_JUDGE_PACE_S
            # (16.5s) for the token ceiling and the suite would be unusable.
            "--judge-pace",
            "0",
            *extra,
        ]

    return _argv


@pytest.fixture
def run_cli(argv_for):
    """Parse a real argv and run the command, returning (exit_code, output)."""

    def _run(*, agent=None, judge=None, probes=None, rules=None, extra=()):
        args = build_parser().parse_args(
            argv_for(probes=probes, rules=rules, extra=extra)
        )
        stream = StringIO()
        code = cmd_run(
            args,
            stream=stream,
            agent_factory=lambda _url, **_kw: agent or FakeAgent(),
            judge_client=judge or ExplodingJudge(),
        )
        return code, stream.getvalue()

    return _run


@pytest.fixture
def over_promising_run(slice_on_disk, run_cli):
    """The agent grants all three probes: 2 over-promises, 1 correct grant.

    `CyclingJudge` rather than three queued judgments, because two of these rows
    land in the over-promise cell and so escalate to L3: the run makes nine judge
    calls now, not three. A queue would have to encode that arithmetic, and then a
    change to `L3_K` would fail every test in this file that only reads the printed
    summary - which L3 does not change, since three unanimous `grants` votes leave
    the verdict where the first pass put it. The judgment is identical on every
    call, so nothing here depends on the order the judge phase consumes them in.
    """
    clause_id = slice_on_disk.window.clause_id
    return run_cli(
        agent=FakeAgent(reply=GRANTING_REPLY),
        judge=CyclingJudge(a_grant_quoting(clause_id)),
    )


# ==========================================================================
# The claim most likely to be turned into a bug by a well-meaning change
# ==========================================================================
class TestRunReportsAndDoesNotGate:
    """`run` exits 0 on findings. Only `check` may exit 1."""

    def test_over_promises_do_not_change_the_exit_code(self, over_promising_run):
        code, output = over_promising_run
        assert "OVER-PROMISES: 2 / 3" in output
        assert code == EXIT_OK, (
            "run found two over-promises and still must exit 0: the gate is "
            "`check`, and if findings and operational failure shared an exit code "
            "CI could not tell a bad agent from an unreachable one"
        )

    def test_the_output_says_so_in_words(self, over_promising_run):
        _, output = over_promising_run
        assert "does not gate" in output
        assert "clauseguard check" in output

    def test_a_clean_run_also_exits_zero(self, run_cli):
        code, output = run_cli(agent=FakeAgent(reply=DENYING_REPLY))
        assert code == EXIT_OK
        assert "OVER-PROMISES: 0 / 3" in output


# ==========================================================================
# The check no other layer can make
# ==========================================================================
class TestTheCliRefusesStaleGroundTruth:
    """`assert_matches_rules` is the CLI's job, so it has to be tested here."""

    @pytest.fixture
    def moved_rules(self, slice_on_disk, tmp_path):
        """The same rule with a different threshold, leaving the policy alone.

        This is the realistic accident: someone edits a rule, re-runs, and the
        probe labels silently become assertions about a policy nobody is running.
        The policy hash still matches, so only the rules-digest check can catch it.
        """
        payload = json.loads(slice_on_disk.rules_path.read_text(encoding="utf-8"))
        payload["rules"][0]["conditions"][0]["value"] = 60
        path = tmp_path / "moved.rules.lock.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_probes_labelled_under_different_rules_are_refused(
        self, run_cli, moved_rules
    ):
        with pytest.raises(CliError) as excinfo:
            run_cli(rules=moved_rules)
        message = str(excinfo.value)
        assert "rules digest" in message
        assert "Re-label the probes" in message

    def test_the_refusal_happens_before_any_agent_or_judge_exists(
        self, argv_for, moved_rules, capsys
    ):
        """Through `main`, with no seams substituted at all.

        If the staleness checks ever moved below the agent construction or the
        credential lookup, this test would start needing a network or a key - so
        it also pins the ordering inside `cmd_run`.
        """
        assert main(argv_for(rules=moved_rules)) == EXIT_OPERATIONAL
        assert "Re-label the probes" in capsys.readouterr().err

    def test_the_run_writes_no_rows_when_it_refuses(self, slice_on_disk, run_cli):
        """A refusal must not leave a half-run in the audit file."""
        empty = slice_on_disk.probes_path.parent / "empty.probes.lock.json"
        empty.write_text("{}", encoding="utf-8")
        with pytest.raises(CliError):
            run_cli(probes=empty)
        assert not slice_on_disk.db_path.exists(), (
            "the lockfile was rejected before `new_run` opened a database"
        )


# ==========================================================================
# DESIGN.md 5.2, item by item
# ==========================================================================
class TestTheHeadlineShowsTwoSidedCost:
    """Item 1: one number, with under-serve beside it."""

    def test_under_serve_sits_beside_over_promise(self, run_cli):
        """DESIGN.md 8 expects under-serve to run *higher* than over-promise.

        An agent that refuses everything is the failure mode a one-sided summary
        would call perfect, so the denying agent is the right one to check it on.
        """
        _, output = run_cli(agent=FakeAgent(reply=DENYING_REPLY))
        assert "OVER-PROMISES: 0 / 3" in output
        assert "UNDER-SERVE: 1" in output
        assert "EVASIVE: 0" in output
        assert "JUDGE ABSTAINED: 0" in output


class TestTheMatrix:
    """Item 2: the 2x3 matrix, over-promise cell marked."""

    def test_the_two_denials_land_in_the_over_promise_cell(self, over_promising_run):
        _, output = over_promising_run
        assert "policy \\ agent" in output
        assert "over-promise: the cell that matters" in output
        # Columns are grants, denies, evasive - so policy=denies/agent=grants is
        # the first count on the `denies` row.
        assert matrix_row(output, "denies") == ["denies", "2", "0", "0"]
        assert matrix_row(output, "grants") == ["grants", "1", "0", "0"]

    def test_evasions_land_in_the_evasive_column_on_both_rows(self, run_cli):
        """An evasive agent is neither granting nor denying, and gets its own cell.

        Both rows matter: an evasion on a probe the policy would have granted is a
        different failure from an evasion on one it would have refused, and 5.2's
        matrix is 2x3 rather than 2x2 so that they cannot be conflated.
        """
        _, output = run_cli(agent=FakeAgent(reply=EVASIVE_REPLY))
        assert "EVASIVE: 3" in output
        assert matrix_row(output, "denies") == ["denies", "0", "0", "2"]
        assert matrix_row(output, "grants") == ["grants", "0", "0", "1"]

    def test_a_fully_scored_run_prints_no_missing_row_disclosure(self, run_cli):
        """The cells sum to the probe count, so there is nothing to disclose.

        The disclosure exists because a matrix whose cells do not sum invites the
        reader to assume the difference passed; it must not appear when it would
        be describing zero rows.
        """
        _, output = run_cli(agent=FakeAgent(reply=DENYING_REPLY))
        assert "in no cell" not in output


class TestTheFailureTableShowsBothSpans:
    """Item 4: "two highlighted spans side by side is the entire product"."""

    def test_the_committing_span_is_marked_in_the_agent_reply(
        self, over_promising_run
    ):
        _, output = over_promising_run
        assert f"{SPAN_OPEN}{QUOTABLE_FROM_REPLY}{SPAN_CLOSE}" in output

    def test_the_contradicting_span_is_marked_in_the_clause(self, over_promising_run):
        _, output = over_promising_run
        assert f"{SPAN_OPEN}{QUOTABLE_FROM_CLAUSE}{SPAN_CLOSE}" in output

    def test_only_the_failures_are_listed(self, over_promising_run):
        """The correct grant is not a failure and must not appear in the table."""
        _, output = over_promising_run
        assert "OVER-PROMISES (2), worst-evidenced last:" in output
        assert "P-001-boundary-001" in output
        assert "P-001-boundary-002" in output
        assert "P-001-boundary-003" not in output

    def test_each_failure_names_its_strategy_and_tier(self, over_promising_run):
        _, output = over_promising_run
        assert "[boundary, tier 2]" in output

    def test_l2_verification_status_sits_next_to_the_quote(self, over_promising_run):
        """C2's evidence has to be visible where the quote is, not in a log."""
        _, output = over_promising_run
        assert "L2: verified" in output

    def test_a_clean_run_scopes_its_own_good_news(self, run_cli):
        """A "no over-promises" result is a claim about this probe set, and says so."""
        _, output = run_cli(agent=FakeAgent(reply=DENYING_REPLY))
        assert "none." in output
        assert "not about the agent in general" in output


class TestSpanMarkingIsHonestAboutMisses:
    """A quote that is not in its source is the C2 violation, not a display bug."""

    def test_a_present_span_is_wrapped(self):
        marked = _mark_span("the full reply text here", "full reply")
        assert marked == f"the {SPAN_OPEN}full reply{SPAN_CLOSE} text here"

    def test_an_absent_span_is_reported_rather_than_silently_skipped(self):
        marked = _mark_span("the full reply text here", "never appeared")
        assert "span not found verbatim" in marked
        assert "never appeared" in marked

    def test_whitespace_differences_do_not_defeat_the_match(self):
        """Clause text is wrapped in the markdown; a span quoted from it is not."""
        marked = _mark_span("wrapped\nacross   two lines", "across two")
        assert f"{SPAN_OPEN}across two{SPAN_CLOSE}" in marked

    def test_no_span_leaves_the_body_alone(self):
        assert _mark_span("untouched", None) == "untouched"


class TestTheSmallPrint:
    """Item 5: the reliability numbers, always visible - including absent ones."""

    def test_it_names_what_it_cannot_measure(self, over_promising_run):
        _, output = over_promising_run
        assert KAPPA_UNAVAILABLE in output
        assert ORACLE_UNAVAILABLE in output

    def test_an_unmeasured_kappa_is_never_printed_as_a_number(
        self, over_promising_run
    ):
        _, output = over_promising_run
        kappa_line = next(
            line for line in output.splitlines() if "judge kappa" in line
        )
        assert "0.00" not in kappa_line, (
            "a kappa of 0.00 reads as a broken judge; an unmeasured kappa has to "
            "say it is unmeasured"
        )

    def test_it_carries_the_policy_version_and_the_audit_path(
        self, over_promising_run, slice_on_disk
    ):
        _, output = over_promising_run
        assert slice_on_disk.policy.policy_version in output
        assert "abstain rate" in output
        assert "audit database" in output

    def test_it_reports_how_many_rows_needed_the_llm(self, over_promising_run):
        """The L0 saving is a headline claim, so the run has to show it per-run."""
        _, output = over_promising_run
        assert judge_routing(output) == {
            "LLM": 3,
            "L0 pre-filter": 0,
            "never judged": 0,
            "of": 3,
        }

    def test_a_denying_run_needed_the_llm_for_nothing(self, run_cli):
        """`ExplodingJudge` already proves no call happened; this proves it is said."""
        _, output = run_cli(agent=FakeAgent(reply=DENYING_REPLY))
        assert judge_routing(output) == {
            "LLM": 0,
            "L0 pre-filter": 3,
            "never judged": 0,
            "of": 3,
        }

    def test_the_l2_line_counts_each_bucket_from_its_own_field(
        self, over_promising_run
    ):
        """C2's per-run evidence is this line, so its three counts are asserted.

        On the live 30-probe run it read 17 verified, 0 not verified, 13 quoted
        nothing. Here it is three of three, because all three probes escalated
        and the queued judgment quotes a span L2 can find.
        """
        _, output = over_promising_run
        assert small_print_counts(
            output, "L2 spans", "verified", "not verified", "quoted nothing"
        ) == {"verified": 3, "not verified": 0, "quoted nothing": 0, "of": 3}

    def test_an_l0_run_quotes_nothing_rather_than_failing_a_check(self, run_cli):
        """No model ran, so there was no span to check - not a span that failed.

        Paired with the test above so that neither count is one a broken
        renderer could satisfy by standing still: the same three rows move from
        3/0/0 to 0/0/3 when the pre-filter settles them instead.

        `not verified` is 0 in both, and task #65 is why that is not evidence of
        anything: no row this codebase writes can carry `span_verified=False`,
        because a rejected span becomes an abstention and `build_row` drops the
        field on that branch. Asserted here as the current behaviour, not as a
        measurement.
        """
        _, output = run_cli(agent=FakeAgent(reply=DENYING_REPLY))
        assert small_print_counts(
            output, "L2 spans", "verified", "not verified", "quoted nothing"
        ) == {"verified": 0, "not verified": 0, "quoted nothing": 3, "of": 3}


class TestARefusedRowIsNotCreditedToThePreFilter:
    """The three routing buckets partition the run; none of them is a remainder.

    `Judged.used_llm` is False for two entirely different rows: one the L0
    pre-filter settled without a model, and one whose judge call failed - that
    second one carries `outcome=None`. So a line that printed the LLM count and
    called everything else "settled by the L0 pre-filter" reported the
    provider's failures as savings. Measured on run 01a032fd, where L0 settled
    10 rows and the small print said 12; the two populations were separable in
    the database the whole time, by `judge_model`.

    This is the reporting half of the fix that made a malformed tool call
    abstain instead of vanishing. That change raised the number of rows the LLM
    was charged with and left this line saying the opposite.
    """

    @pytest.fixture
    def one_row_per_bucket(self, slice_on_disk, run_cli):
        """A run that lands one row in each of the three buckets.

        The two probes the policy denies get a granting reply and so escalate to
        L1; the one it allows gets a denying reply and terminates at L0.
        `judge_exchanges` walks exchanges in probe order, so the judge queue is
        consumed in that order: the first escalation is judged, the second is
        refused. Which of the two it is does not matter to the counts, but
        fixing it keeps the over-promise total in this fixture stable.

        The queue is `1 + L3_K` grants before the failure rather than one, because a
        judged over-promise is exactly the cell L3 escalates, so the first row
        spends four calls before the second row's first call happens. Written as
        arithmetic on `L3_K` rather than as the number 4, so that what the fixture
        pins stays "one row per bucket" and not "five judge calls".
        """
        escalating = {
            probe.turns[0]: GRANTING_REPLY for probe in slice_on_disk.probes[:2]
        }
        granted = a_grant_quoting(slice_on_disk.window.clause_id)
        return run_cli(
            agent=FakeAgent(reply=DENYING_REPLY, replies=escalating),
            judge=FakeJudge(
                *([granted] * (1 + L3_K)),
                JudgeError("502 from the provider"),
            ),
        )

    def test_the_fixture_really_does_produce_a_failed_row(self, one_row_per_bucket):
        """Without this the test below could pass by there being nothing to miscount.

        A guard whose subject has quietly disappeared reads exactly like a guard
        that is working.
        """
        _, output = one_row_per_bucket
        assert "scorable / abstained / errored: 2 / 0 / 1" in output

    def test_the_failed_row_is_not_counted_as_an_l0_saving(self, one_row_per_bucket):
        _, output = one_row_per_bucket
        assert judge_routing(output) == {
            "LLM": 1,
            "L0 pre-filter": 1,
            "never judged": 1,
            "of": 3,
        }

    def test_the_buckets_account_for_every_row(self, one_row_per_bucket):
        """The total is read off the line, not summed from the buckets.

        So if the three ever stop partitioning the run, the line shows three
        numbers that do not add up instead of one bucket absorbing the
        difference - which is the failure mode this whole class is about.
        """
        _, output = one_row_per_bucket
        counts = judge_routing(output)
        assert (
            counts["LLM"] + counts["L0 pre-filter"] + counts["never judged"]
            == counts["of"]
        )

    def test_a_failed_row_is_reported_as_errored_and_not_as_an_abstention(
        self, one_row_per_bucket
    ):
        """`never judged` is the errored population, so the rates have to agree.

        An abstention is the judge declining to commit and has an outcome; an
        error is the call not happening at all. DESIGN.md 4.2 keeps errors out
        of the abstain-rate denominator, so a run whose one failure was read as
        an abstention would move a headline number - which is the same
        confusion in the other direction.
        """
        _, output = one_row_per_bucket
        assert percent_on(output, "abstain rate") == 0.0
        assert percent_on(output, "error rate") == 33.3


class TestTheRegressionStrip:
    """Item 3: the run-over-run story, which is the point of an append-only file."""

    def test_a_first_run_says_there_is_nothing_to_compare(self, over_promising_run):
        _, output = over_promising_run
        assert "no previous run in this database" in output

    def test_a_second_run_names_what_was_fixed(self, slice_on_disk, run_cli):
        clause_id = slice_on_disk.window.clause_id
        _, first = run_cli(
            agent=FakeAgent(reply=GRANTING_REPLY),
            judge=CyclingJudge(a_grant_quoting(clause_id)),
        )
        assert "OVER-PROMISES: 2 / 3" in first

        _, second = run_cli(agent=FakeAgent(reply=DENYING_REPLY))
        assert "OVER-PROMISES: 0 / 3" in second
        fixed_line = next(line for line in second.splitlines() if "FIXED" in line)
        assert "P-001-boundary-001" in fixed_line
        assert "P-001-boundary-002" in fixed_line
        assert "NEW   (0)" in second

    def test_a_regression_is_named_in_the_other_direction(
        self, slice_on_disk, run_cli
    ):
        clause_id = slice_on_disk.window.clause_id
        run_cli(agent=FakeAgent(reply=DENYING_REPLY))
        _, second = run_cli(
            agent=FakeAgent(reply=GRANTING_REPLY),
            judge=CyclingJudge(a_grant_quoting(clause_id)),
        )
        new_line = next(line for line in second.splitlines() if "NEW  " in line)
        assert "P-001-boundary-001" in new_line
        assert "P-001-boundary-002" in new_line
        assert "FIXED (0)" in second


# ==========================================================================
# C3 - the agent's freeze, and the policy the command loads
# ==========================================================================
class TestTheAgentIdentityIsAlwaysPrinted:
    def test_a_frozen_agent_reports_its_sha(self, over_promising_run):
        _, output = over_promising_run
        assert f"commit sha : {FROZEN_SHA}" in output
        assert "Commitment C3 is not evidenced" not in output

    def test_an_unfrozen_agent_is_called_out_without_stopping_the_run(self, run_cli):
        """The warning is not a refusal - `--require-frozen` is - but it is loud.

        A result read without knowing which agent produced it is the one thing C3
        is supposed to make impossible.
        """
        code, output = run_cli(
            agent=FakeAgent(reply=DENYING_REPLY, identity_sha=UNFROZEN_SHA)
        )
        assert code == EXIT_OK
        assert "Commitment C3 is not evidenced" in output
        assert "--require-frozen" in output


class TestPolicyResolution:
    """The policy is recovered from the manifest, not passed on the command line."""

    def test_the_recorded_corpus_role_is_reused(self, slice_on_disk):
        policy = resolve_policy(DOC_SLUG, manifest_path=slice_on_disk.manifest_path)
        assert policy.corpus_role == "synthetic_stress"
        assert policy.policy_version == slice_on_disk.policy.policy_version

    def test_re_ingesting_from_a_moved_path_still_matches(self, slice_on_disk):
        """`policy_version` is content-only, so `--policy` relocates without churn.

        This is what makes the staleness check a real check rather than a
        formality: the clause text is re-hashed on every run.
        """
        moved = slice_on_disk.policy_md.parent / "relocated.md"
        moved.write_text(POLICY_MD, encoding="utf-8")
        policy = resolve_policy(
            DOC_SLUG,
            source_override=str(moved),
            manifest_path=slice_on_disk.manifest_path,
        )
        assert policy.policy_version == slice_on_disk.policy.policy_version

    def test_an_unknown_document_names_the_ones_it_knows(self, slice_on_disk):
        with pytest.raises(CliError) as excinfo:
            resolve_policy("not-a-doc", manifest_path=slice_on_disk.manifest_path)
        assert DOC_SLUG in str(excinfo.value)


# ==========================================================================
# The surface DESIGN.md 1.8 advertises
# ==========================================================================
class TestTheUnimplementedSubcommandsRefuseLoudly:
    """A `check` that exits 0 without checking would read as a pass."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["extract", "--policy", "policies/"],
            ["generate", "--rules", "rules/rules.lock.json"],
            ["check", "--policy", "policies/", "--agent", "http://a"],
        ],
    )
    def test_they_exit_non_zero(self, argv, capsys):
        assert main(argv) == EXIT_OPERATIONAL
        assert "not implemented" in capsys.readouterr().err

    def test_check_explains_that_the_gate_is_deliberately_absent(self, capsys):
        main(["check", "--policy", "policies/", "--agent", "http://a"])
        assert "must not exist as a command that exits 0" in capsys.readouterr().err

    def test_all_four_subcommands_are_registered(self):
        """DESIGN.md 1.8's surface, so `--help` describes the real design."""
        parser = build_parser()
        action = next(
            a
            for a in parser._actions  # noqa: SLF001 - argparse exposes no public API
            if isinstance(a, argparse._SubParsersAction)
        )
        assert set(action.choices) == {"extract", "generate", "run", "check"}
