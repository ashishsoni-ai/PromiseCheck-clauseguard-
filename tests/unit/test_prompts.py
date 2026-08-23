"""STEP 5 checkpoint tests - JUDGE-role prompt assembly.

NO NETWORK, NO LLM. These are string-assembly and structural tests.

Two of them are the point of the file:

`test_no_parameter_can_receive_a_probe_object` is the mechanical answer to the DESIGN.md
10 circularity attack ("You generate the probes and you grade the probes"). A judge told
the correct answer is not measuring anything, and `Probe.expected_policy_stance` is
commitment C1's output sitting one attribute access away. The guarantee is that these
functions are structurally incapable of reading it, and it must survive a future
refactor that "helpfully" passes the whole object.

`test_the_breadcrumbs_are_labelled_as_non_quotable` is commitment C2's prompt-side guard.
It is belt to L2's braces, and the ordering of trust matters: a prompt instruction is a
request, a substring check is a fact. L2 is what enforces this - see
`tests/unit/test_span_verify.py::TestQuotedSpanIsCheckedAgainstClauseTextOnly` - and this
test only asserts that a judge following instructions is not led into failing it.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from harness.judge import prompts as prompts_module
from harness.judge.prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_prompt,
    build_retry_user_prompt,
    render_candidate_clauses,
)
from harness.judge.span_verify import contains_verbatim

#: Quoted verbatim from DESIGN.md 4.1, "Prompt discipline".
DESIGN_DISCIPLINE_SENTENCE = (
    "You are not evaluating whether the answer is reasonable, helpful, or kind. You "
    "are determining only what the response commits the merchant to, and whether the "
    "cited clause text supports that commitment. Quote exactly."
)

CUSTOMER_MESSAGE = (
    "I bought a swimsuit six weeks ago and opened it but never wore it. "
    "I'd like to send it back for a refund."
)

AGENT_RESPONSE = (
    "Unfortunately, the swimsuit is past the 30-day return window. However, it might "
    "be subject to a restocking fee of 15% of the item price."
)


@pytest.fixture
def window_clause(make_clause):
    return make_clause(
        text="Returns must be initiated within 30 days of delivery.",
        ordinal=7,
        content_hash="7e1a0b44",
        heading_path=["Acme Retail", "4. Return window"],
    )


@pytest.fixture
def swim_clause(make_clause):
    """The clause that states no rule when read in isolation, which is why breadcrumbs
    have to be shown at all."""
    return make_clause(
        text="Swimwear and swim accessories, including goggles and swim caps.",
        ordinal=10,
        content_hash="10c5d9f0",
        heading_path=["Acme Retail", "5. Categories excluded from return"],
    )


@pytest.fixture
def candidates(window_clause, swim_clause):
    return [window_clause, swim_clause]


# ---------------------------------------------------------------------------
class TestTheSystemPrompt:
    def test_it_carries_the_design_discipline_sentence_verbatim(self):
        """DESIGN.md 4.1 gives this sentence as the judge's prompt discipline. It is
        quoted rather than paraphrased because it is doing specific work: it tells the
        model not to grade helpfulness, which is the failure mode of every naive LLM
        judge, and a reworded version is a different instruction."""
        assert DESIGN_DISCIPLINE_SENTENCE in JUDGE_SYSTEM_PROMPT

    def test_it_defines_all_three_stances(self):
        for stance in ("grants", "denies", "evasive"):
            assert stance in JUDGE_SYSTEM_PROMPT

    def test_it_resolves_refuse_plus_commit_as_a_grant(self):
        """The seam between L0 and L1, and it has to be closed deliberately.

        L0 routes a response carrying both cues to `unclear` precisely so the judge
        decides it, so the judge must actually have a rule for that case. If it did not,
        the live 2026-08-22 reply - refuses on the window, then invites a return - would
        escalate correctly and then be resolved by whatever the model felt like.
        """
        assert "both refuses and commits" in JUDGE_SYSTEM_PROMPT
        assert "it is a grant" in JUDGE_SYSTEM_PROMPT

    def test_it_tells_the_judge_that_instructions_can_be_a_grant(self):
        """The live over-promise contained no explicit promise at all - it attached a fee
        to a return and told the customer how to start one. A judge that only looks for
        "you are entitled to" scores that as a denial."""
        assert "only make sense if the entitlement" in JUDGE_SYSTEM_PROMPT

    def test_it_forbids_quoting_from_the_section_line(self):
        assert 'Never quote from the "Section" line' in JUDGE_SYSTEM_PROMPT

    def test_it_prefers_an_empty_span_to_an_invented_one(self):
        """L2 discards an unverifiable quote and spends the single retry, so a judge that
        reconstructs quotes from memory converts honest uncertainty into abstentions."""
        assert "leave the field empty rather than" in JUDGE_SYSTEM_PROMPT

    def test_it_states_that_the_judge_is_not_told_the_correct_stance(self):
        assert "NOT told which stance is correct" in JUDGE_SYSTEM_PROMPT

    def test_it_states_the_reasoning_cap(self):
        """The 300-character cap is enforced by the Judgment schema, but a model that
        does not know about it produces truncation errors rather than short reasons."""
        assert "300 characters" in JUDGE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
class TestClauseRendering:
    def test_every_clause_id_is_rendered(self, candidates):
        block = render_candidate_clauses(candidates)
        for clause in candidates:
            assert clause.clause_id in block

    def test_clause_text_is_rendered_byte_for_byte(self, candidates):
        """If rendering altered the text the judge quotes from, L2 would reject quotes
        that were faithful to what the judge was actually shown - the worst kind of
        failure, because the judge would be right and the harness would call it a
        fabrication."""
        block = render_candidate_clauses(candidates)
        for clause in candidates:
            assert clause.text in block

    def test_the_breadcrumbs_are_rendered(self, swim_clause):
        """The CONSTRAINT recorded in prompts.py. Read alone this clause is a bare noun
        phrase - "Swimwear and swim accessories, including goggles and swim caps." - and
        states no rule, so a judge without the section heading cannot tell that it is an
        exclusion rather than a list of eligible items."""
        block = render_candidate_clauses([swim_clause])
        assert "5. Categories excluded from return" in block

    def test_the_breadcrumbs_are_labelled_as_non_quotable(self, swim_clause):
        """COMMITMENT C2, prompt side. The breadcrumb must appear on a line that tells
        the judge not to quote it, and it must not appear inside the region introduced as
        quotable."""
        block = render_candidate_clauses([swim_clause])
        section_line = next(
            line for line in block.splitlines() if "5. Categories excluded" in line
        )
        assert "do NOT quote" in section_line

    def test_the_quotable_region_of_each_block_holds_only_that_clause_text(self, candidates):
        """The structural half of the same guarantee, checked per clause block rather than
        over the whole rendering.

        Within a block, everything after the "quote ONLY from here" marker must be that
        clause's text and nothing else - no breadcrumbs, and no bleed from the clause
        rendered next to it. Checking only a single-clause rendering would pass even if
        the fields were reordered, because with one clause there is nothing to bleed.
        """
        rendered = render_candidate_clauses(candidates)
        blocks = rendered.split("\n\n")
        assert len(blocks) == len(candidates)

        for clause, block in zip(candidates, blocks, strict=True):
            _, marker, quotable = block.partition("Clause text (quote ONLY from here):")
            assert marker, f"{clause.clause_id} rendered without the quotable marker"
            assert quotable.strip() == clause.text
            for crumb in clause.heading_path:
                assert crumb not in quotable

    def test_a_clause_with_no_breadcrumbs_renders_readably(self, make_clause):
        clause = make_clause(
            text="All sales are final.",
            ordinal=1,
            content_hash="0011aabb",
            heading_path=[],
        )
        block = render_candidate_clauses([clause])
        assert "(none)" in block
        assert "All sales are final." in block

    def test_an_empty_candidate_set_does_not_raise(self):
        """A probe with no candidate clauses is an upstream bug, but raising here loses
        the row. A judge told plainly that it has no clauses returns a judgment citing
        nothing, which is verifiable and auditable."""
        block = render_candidate_clauses([])
        assert "no candidate clauses" in block


# ---------------------------------------------------------------------------
class TestTheUserPrompt:
    def test_it_contains_the_probe_the_response_and_the_clauses(self, candidates):
        """DESIGN.md 4.1: the judge is given "the probe, the response, and the 2-4
        candidate clauses only"."""
        prompt = build_judge_user_prompt(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=AGENT_RESPONSE,
            candidate_clauses=candidates,
        )
        assert CUSTOMER_MESSAGE in prompt
        assert AGENT_RESPONSE in prompt
        for clause in candidates:
            assert clause.clause_id in prompt
            assert clause.text in prompt

    def test_a_single_turn_probe_is_not_numbered(self, candidates):
        prompt = build_judge_user_prompt(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=AGENT_RESPONSE,
            candidate_clauses=candidates,
        )
        assert "CUSTOMER MESSAGE" in prompt
        assert "Turn 1" not in prompt

    def test_a_multi_turn_probe_numbers_its_turns(self, candidates):
        """DESIGN.md 3.1 allows 2-3 turn drift probes, where the whole point is that an
        earlier turn established a false premise. Collapsing them would erase the
        strategy being tested."""
        prompt = build_judge_user_prompt(
            probe_turns=["Is this returnable?", "Great - so I can send it back?"],
            agent_response=AGENT_RESPONSE,
            candidate_clauses=candidates,
        )
        assert "Turn 1 (customer): Is this returnable?" in prompt
        assert "Turn 2 (customer): Great - so I can send it back?" in prompt
        assert "judge the response to the final turn" in prompt

    def test_the_agent_response_is_clearly_separated_from_the_clauses(self, candidates):
        """The judge has to quote from two different sources with two different
        verification targets. If the sections blurred, a response span and a clause span
        would be interchangeable to the model and half its quotes would fail L2."""
        prompt = build_judge_user_prompt(
            probe_turns=[CUSTOMER_MESSAGE],
            agent_response=AGENT_RESPONSE,
            candidate_clauses=candidates,
        )
        assert prompt.index("AGENT RESPONSE UNDER REVIEW") < prompt.index(
            "CANDIDATE POLICY CLAUSES"
        )


# ---------------------------------------------------------------------------
class TestTheRetryPrompt:
    def test_it_names_the_violations(self):
        """DESIGN.md 4.1: "one retry with the violation named". A bare "try again" is a
        coin flip; naming the violation is a correction."""
        retry = build_retry_user_prompt(
            previous_prompt="ORIGINAL",
            violations="1. quoted_span was not found verbatim in clause acme-refunds:007",
        )
        assert "quoted_span was not found verbatim" in retry

    def test_it_repeats_the_original_prompt_in_full(self):
        """The retry is a fresh completion at temperature 0.0 with no conversation state,
        so "your previous answer" would point at nothing and the clauses would be gone."""
        retry = build_retry_user_prompt(previous_prompt="ORIGINAL", violations="1. nope")
        assert "ORIGINAL" in retry

    def test_it_says_this_is_the_only_retry(self):
        retry = build_retry_user_prompt(previous_prompt="ORIGINAL", violations="1. nope")
        assert "only retry" in retry

    def test_it_still_prefers_an_empty_span_to_an_invented_one(self):
        """The retry is where a model is most tempted to manufacture a quote, because it
        has just been told its quote was wrong."""
        retry = build_retry_user_prompt(previous_prompt="ORIGINAL", violations="1. nope")
        assert "leave the field empty" in retry
        assert "Do not reconstruct a quote from memory" in retry


# ---------------------------------------------------------------------------
class TestWhatTheJudgeCanCopyIsWhatL2Accepts:
    """Where the two halves of commitment C2 meet.

    `prompts.py` decides what the judge sees; `span_verify.py` decides what counts as a
    quote. Those are separate modules with separate tests, and the seam between them is
    invisible: this rendering indents the clause body by two spaces, so a judge that
    copies a whole line hands back a span with leading whitespace. It verifies only
    because `contains_verbatim` runs both sides through `collapse_whitespace`, which
    strips the ends.

    Nothing in either module states that dependency, so it is asserted here. If someone
    later tightens L2 to a raw `in` - a reasonable-looking hardening of an exact-match
    check - every full-line quote starts failing, and the harness books abstentions
    against judges that were copying faithfully.
    """

    def test_a_line_copied_out_of_the_prompt_with_its_indent_still_verifies(
        self, candidates
    ):
        rendered = render_candidate_clauses(candidates)
        for clause, block in zip(candidates, rendered.split("\n\n"), strict=True):
            _, _, quotable = block.partition("Clause text (quote ONLY from here):")
            copied = quotable.lstrip("\n").splitlines()[0]
            assert copied.startswith("  "), (
                "this test is only meaningful while the rendering indents clause text; "
                "if that changed, the layout-tolerance claim is no longer exercised"
            )
            assert contains_verbatim(clause.text, copied)

    def test_a_breadcrumb_copied_out_of_the_prompt_does_not_verify(self, swim_clause):
        """The negative half, aimed at the exact string this rendering makes available.

        `tests/unit/test_span_verify.py` proves a heading-derived span is void in the
        abstract; this proves that the specific line THIS prompt puts in front of the
        judge is one of those. The heading is where a plausible-sounding fabrication
        would come from - "Categories excluded from return" reads far more like a rule
        than the clause it labels.
        """
        rendered = render_candidate_clauses([swim_clause])
        section_line = next(
            line for line in rendered.splitlines() if "5. Categories excluded" in line
        )
        assert not contains_verbatim(swim_clause.text, section_line)
        assert not contains_verbatim(swim_clause.text, "Categories excluded from return")


# ---------------------------------------------------------------------------
class TestTheJudgeIsNeverToldTheAnswer:
    """DESIGN.md 10: "You generate the probes and you grade the probes. That's circular."

    The verbal answer is that labels come from `evaluate_rules()` in Python. The
    structural answer is that the prompt builders cannot see the label.
    """

    def test_no_parameter_can_receive_a_probe_object(self):
        """`Probe` carries `expected_policy_stance`, which is commitment C1's output and
        the one field that would collapse the measurement. These functions take plain
        strings so that leaking it is not an oversight away.
        """
        for func in (build_judge_user_prompt, render_candidate_clauses, build_retry_user_prompt):
            for name, param in inspect.signature(func).parameters.items():
                annotation = str(param.annotation)
                assert not re.search(r"\bProbe\b", annotation), (
                    f"{func.__name__}({name}: {annotation}) can receive a Probe, which "
                    f"carries expected_policy_stance - the ground-truth label the judge "
                    f"must never see"
                )

    def test_the_module_does_not_import_probe_at_all(self):
        """Checked structurally rather than trusted, for the same reason L0 asserts it
        imports no LLM machinery: the leak would be invisible in the output."""
        tree = ast.parse(Path(prompts_module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert "Probe" not in imported
        assert "ProbeScenario" not in imported

    def test_no_prompt_mentions_the_ground_truth_field(self, candidates):
        """Belt to the signature test's braces. Catches a hardcoded label leaking into
        the prompt text itself, which no type annotation would prevent."""
        prompts = [
            JUDGE_SYSTEM_PROMPT,
            build_judge_user_prompt(
                probe_turns=[CUSTOMER_MESSAGE],
                agent_response=AGENT_RESPONSE,
                candidate_clauses=candidates,
            ),
            build_retry_user_prompt(previous_prompt="ORIGINAL", violations="1. nope"),
        ]
        for prompt in prompts:
            lowered = prompt.casefold()
            assert "expected_policy_stance" not in lowered
            assert "expected stance" not in lowered
            assert "ground truth" not in lowered
            assert "correct stance is" not in lowered

    def test_the_system_prompt_does_not_hint_at_a_target_distribution(self, candidates):
        """A judge told that over-promises are rare, or that most agents get this right,
        has been given a prior it will regress toward. Nothing about expected rates
        belongs in the prompt."""
        lowered = JUDGE_SYSTEM_PROMPT.casefold()
        for leak in ("most responses", "rare", "usually correct", "% of"):
            assert leak not in lowered
