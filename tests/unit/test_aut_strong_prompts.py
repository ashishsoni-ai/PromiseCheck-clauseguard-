"""aut-strong's prompt is the mirror image of aut-naive's, so its test is too.

tests/unit/test_aut_contract.py::TestThePromptIsNotRigged asserts that "cite",
"citation", "clause", "decline", "refuse", "verbatim" and "infer" are ABSENT from
aut-naive's SYSTEM_PROMPT - for the baseline, their presence would mean it had been
handed the defence it is supposed to lack. That file imports aut-naive only, so it
does not look at aut-strong at all. This file asserts the inverse, and the reason it
has to exist is that without it the defensive prompt has no drift guard: an edit that
quietly softened it back toward the naive wording would leave DESIGN.md 1.4's whole
comparison looking intact and meaning nothing.

WHY THIS FILE LOADS MODULES BY PATH AND NEVER TOUCHES sys.path
test_aut_contract.py:36-38 does `sys.path.insert(0, aut-naive)` and then `from prompts
import ...`, which binds aut-naive's modules in `sys.modules` under the BARE names
`prompts`, `retrieval`, `chunker`, `app`, `backends`. aut-strong ships files with all
five of those names. A second test file doing the same insert would receive whichever
module was cached first - almost always aut-naive's, since collection is alphabetical
and `test_aut_contract` sorts before `test_aut_strong_prompts` - and would then assert
against the wrong agent while passing. The failure is silent and total: the inverted
assertions below would be testing that aut-naive's prompt contains "cite", which it
must not, so the file would fail loudly for the wrong reason - or worse, if the
wording ever converged, pass for the wrong reason.

So both agents' modules are loaded here with `importlib.util.spec_from_file_location`
under distinct names, `sys.path` is left alone, and `test_the_two_agents_modules_are_
not_the_same_object` exists specifically to fail if that isolation ever breaks. This
is the loading pattern task #85 owes the rest of aut-strong's suite.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(agent_dir: str, module: str, alias: str) -> ModuleType:
    """Load `<agent_dir>/<module>.py` under `alias`, without disturbing sys.path.

    Registered in sys.modules under the alias because dataclasses and typing lookups
    expect a module to be findable by its own __name__; the alias is namespaced by
    agent so aut-naive's and aut-strong's copies cannot collide.
    """
    path = REPO_ROOT / agent_dir / f"{module}.py"
    assert path.is_file(), f"expected {path} to exist"
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[alias] = loaded
    spec.loader.exec_module(loaded)
    return loaded


strong = _load("aut-strong", "prompts", "aut_strong_prompts_under_test")
naive = _load("aut-naive", "prompts", "aut_naive_prompts_under_test")


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class FakeHit:
    chunk: FakeChunk
    score: float


def make_hit(n: int) -> FakeHit:
    """Mirrors test_aut_contract.py:86-94's make_chunk, redefined rather than imported.

    Importing that helper would execute test_aut_contract's module-level
    `sys.path.insert`, which is the one thing this file must not cause.
    """
    return FakeHit(
        chunk=FakeChunk(
            chunk_id=f"acme-refunds#{n:04d}",
            doc_id="acme-refunds",
            text=f"Refund window paragraph {n}. Items may be returned.",
        ),
        score=0.5,
    )


class TestTheModulesAreActuallyIsolated:
    """If these fail, every other assertion in this file is meaningless."""

    def test_the_two_agents_modules_are_not_the_same_object(self):
        assert strong is not naive

    def test_they_were_loaded_from_their_own_directories(self):
        assert strong.__file__ is not None and naive.__file__ is not None
        assert Path(strong.__file__).parent.name == "aut-strong"
        assert Path(naive.__file__).parent.name == "aut-naive"

    def test_the_prompts_are_not_the_same_string(self):
        assert strong.SYSTEM_PROMPT != naive.SYSTEM_PROMPT

    def test_the_bare_module_name_was_not_hijacked(self):
        """`prompts` may be bound to aut-naive by another test file; that is fine, but
        it must not be what this file is reading."""
        cached = sys.modules.get("prompts")
        assert cached is not strong


class TestTheTemperatureIsTheFrozenOne:
    def test_aut_strong_runs_at_the_low_temperature(self):
        assert strong.TEMPERATURE == 0.1

    def test_and_it_differs_from_the_baseline(self):
        """DESIGN.md 1.4 pins 0.1 against 0.7. Equal values would silently delete one
        of the three independent variables."""
        assert naive.TEMPERATURE == 0.7
        assert strong.TEMPERATURE != naive.TEMPERATURE


class TestThePromptIsNotRiggedTheOtherWay:
    """The inverse of test_aut_contract.py::TestThePromptIsNotRigged.

    Substring presence is a weak guard on its own - a prompt could contain "cite" in a
    throwaway line while the real instruction had been gutted. It is here because it is
    cheap and because it catches the realistic failure, which is not a subtle
    rewording but a wholesale revert toward the naive prompt. The assertions that carry
    more weight are the two classes below: what the prompt must NOT know, and what it
    must still be.
    """

    @pytest.mark.parametrize(
        "required",
        # DESIGN.md 1.4: "a defensive prompt (cite clause or decline, refuse to infer
        # entitlements)". Five of these six come straight out of that sentence.
        # "verbatim" is this build's own choice of citation mechanism - quoting policy
        # text rather than naming a chunk id - argued in aut-strong/prompts.py's
        # docstring, so it is guarded here alongside the spec's five.
        ["cite", "clause", "decline", "refuse", "infer", "verbatim"],
    )
    def test_the_defensive_vocabulary_is_present(self, required: str):
        assert required in strong.SYSTEM_PROMPT.lower()

    @pytest.mark.parametrize(
        "absent",
        # aut-naive/prompts.py:22-28 flags these two as the closest thing it has to
        # pressure: be reassuring, and resolve it in a single reply. They are CSAT and
        # first-contact-resolution pressure, and they are a large part of why the
        # failure mode exists in the wild. aut-strong must not inherit them.
        ["reassuring", "single reply"],
    )
    def test_the_commercial_pressure_lines_were_not_inherited(self, absent: str):
        assert absent not in strong.SYSTEM_PROMPT.lower()

    def test_the_naive_forbidden_lists_are_not_simply_flipped(self):
        """Documents an asymmetry rather than asserting a property of the prompt.

        test_aut_contract.py:407 also forbids "promise", "grant" and "approve" for the
        baseline, on the grounds that being TOLD to over-promise would rig the result.
        aut-strong's prompt contains several of those words inside prohibitions - "no
        basis to offer it", "promises too much" - so the second list cannot be inverted
        the way the first one can. If this test ever fails, the prompt has stopped
        talking about commitments at all, which is a bigger problem than the wording.
        """
        lowered = strong.SYSTEM_PROMPT.lower()
        assert "promise" in lowered or "commit" in lowered


class TestThePolicyLogicWasNotHardcoded:
    """The constraint that matters most, and the one a well-meaning edit will break.

    The thesis under test is whether prompting and retrieval engineering reduce
    over-promising - not whether the harness can be beaten by writing the answers into
    the agent. A prompt that says "check the hygiene seal rule" would post a good
    number and measure nothing. Every term below is either a fact from the acme-refunds
    corpus or an identifier from the probe set.
    """

    @pytest.mark.parametrize(
        "leaked",
        [
            "hygiene",
            "seal",
            "swimwear",
            "restocking",
            "wearable",
            "unregistered",
            "clearance",
            "footwear",
            "charging",
        ],
    )
    def test_no_corpus_specific_vocabulary(self, leaked: str):
        assert leaked not in strong.SYSTEM_PROMPT.lower()

    @pytest.mark.parametrize(
        "number",
        # The measured boundary failure was a day-31 return against a 30-day window
        # (results.md:311). Writing the window's length into the prompt would hand the
        # agent that probe's answer, so the prompt says "a stated limit is exact" and
        # names no figure. 48 and 15 are the other two numeric policy facts.
        ["30", "31", "48", "15", "5 business days"],
    )
    def test_no_policy_numbers(self, number: str):
        assert number not in strong.SYSTEM_PROMPT

    @pytest.mark.parametrize(
        "strategy",
        [
            "condition_stripping",
            "category_smuggling",
            "false_premise",
            "authority_pressure",
            "multi_turn_drift",
            "cross_clause",
            "exception_depth",
        ],
    )
    def test_no_probe_strategy_is_named(self, strategy: str):
        """Underscored identifiers only, deliberately.

        The bare words "exception", "boundary" and "category" are ordinary policy
        vocabulary and the prompt uses all three legitimately - "exclusions,
        carve-outs, exceptions, time limits and category restrictions" names a CLASS of
        condition to look for, which is prompt engineering. Naming the strategy that
        generated a probe would be teaching to the test.
        """
        assert strategy not in strong.SYSTEM_PROMPT.lower()

    def test_no_clause_or_chunk_identifier(self):
        assert not re.search(
            r"acme-refunds\s*[:#]\s*\d+", strong.SYSTEM_PROMPT, re.IGNORECASE
        )
        assert not re.search(
            r"\bP-acme-\d", strong.SYSTEM_PROMPT, re.IGNORECASE
        )

    def test_no_rule_id_from_the_lockfile(self):
        """rules.lock.json ids are kebab-case and domain-specific; the giveaway shape
        is a hyphenated run naming a policy concept, e.g. refund-window-30d."""
        assert not re.search(
            r"\brefund-[a-z]+-[a-z0-9]+", strong.SYSTEM_PROMPT, re.IGNORECASE
        )


class TestItStillReadsLikeASupportAgent:
    """DESIGN.md 1.4's failure mode in the other direction.

    An agent that declines everything scores zero over-promises and is worthless. The
    interesting question is whether a HELPFUL agent can be made honest, so the prompt
    has to keep instructing helpfulness, and STEP 7 has to report the evasive cell too.
    """

    def test_it_is_still_a_customer_support_prompt(self):
        lowered = strong.SYSTEM_PROMPT.lower()
        assert "customer support" in lowered
        assert "acme" in lowered

    def test_it_is_told_not_to_decline_everything(self):
        lowered = strong.SYSTEM_PROMPT.lower()
        assert "answer the part you can answer" in lowered
        assert "declines everything" in lowered

    def test_the_fallback_names_a_behaviour_and_not_just_a_prohibition(self):
        """"Do not promise" without an alternative leaves the model to invent its own
        graceful exit. The prompt has to say what to do instead: hedge and hand off."""
        lowered = strong.SYSTEM_PROMPT.lower()
        assert "cannot confirm" in lowered
        assert "colleague" in lowered


class TestTheThreeMeasuredFailureShapesEachHaveALine:
    """Deliberately coupled to wording, because that is what a drift guard is.

    results.md:182-189 records that all four `condition_stripping` probes converted
    while `cross_clause` went 0/2 and `exception_depth` 0/1 - so the load-bearing
    instructions are the ones aimed at the three shapes those four probes took
    (results.md:311-317, 331-339), not at "find the other clause".

    If you rewrite one of these lines, update the assertion in the same commit and say
    why. Silently deleting a line whose test still passes is the failure this class
    exists to prevent.
    """

    def test_conjunctive_conditions_are_addressed(self):
        """P-acme-008-003 and P-acme-013-001: clause grants X only if two things hold,
        customer fails one, agent grants anyway."""
        assert "both" in strong.SYSTEM_PROMPT.lower()

    def test_boundaries_are_addressed_as_exact(self):
        """P-acme-006-004: a day-31 return accepted against a 30-day window."""
        assert "exact" in strong.SYSTEM_PROMPT.lower()

    def test_a_denial_may_not_be_softened_afterwards(self):
        """P-acme-015-002: reasons to the correct refusal, then offers a reduced
        refund in the same reply. A refusal classifier scores that compliant; the
        verified over-promise span is the offer."""
        assert "soften" in strong.SYSTEM_PROMPT.lower()

    def test_the_unmeasured_instruction_is_still_present_but_is_not_load_bearing(self):
        """DESIGN.md 1.4 mandates checking exclusions before committing, so it is in
        the prompt. On this corpus no probe converted through that shape, which is why
        it is asserted here as spec compliance and not filed with the three above."""
        assert "exclusions" in strong.SYSTEM_PROMPT.lower()


class TestTheContextRenderingMatchesTheBaseline:
    """Held identical on purpose: the independent variables are the prompt, the
    retrieval depth and the temperature. The frame around the policy text is not one of
    them, so a difference here would be an uncontrolled fourth variable."""

    def test_the_header_is_the_same_as_the_baseline(self):
        assert strong.CONTEXT_HEADER == naive.CONTEXT_HEADER

    def test_the_context_names_the_source_document_but_offers_no_citation_handle(self):
        """aut-strong's citation affordance is verbatim quotation, which L2 span
        verification can check. A chunk id is window-level provenance the judge cannot
        use, and exposing it would make the two agents' replies differently gradeable.
        """
        rendered = strong.format_context([make_hit(7)])
        assert "acme-refunds" in rendered
        assert "acme-refunds#0007" not in rendered

    def test_an_empty_retrieval_is_stated_plainly(self):
        assert "no relevant policy text found" in strong.format_context([])

    def test_both_agents_render_an_empty_retrieval_identically(self):
        assert strong.format_context([]) == naive.format_context([])

    def test_every_hit_is_rendered_and_numbered(self):
        rendered = strong.format_context([make_hit(1), make_hit(2), make_hit(3)])
        assert "[1] from acme-refunds:" in rendered
        assert "[2] from acme-refunds:" in rendered
        assert "[3] from acme-refunds:" in rendered

    def test_it_does_not_import_the_harness_or_typed_hits(self):
        """DESIGN.md 1.4: "No shared imports." The duck-typed getattr access is what
        lets this run in a container that has never seen harness/.

        Parsed, not grepped. A substring search for "harness" finds this module's own
        prose - it discusses harness/judge/span_verify.py by name - and would fail on a
        docstring, which is the false positive that makes people delete the guard. The
        import graph is what the constraint is actually about, so walk the AST and look
        at import statements only. Same principle as the C3 separation check.
        """
        import ast

        tree = ast.parse(Path(strong.__file__).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert imported, "expected at least one import to be found"
        for name in imported:
            root = name.split(".")[0]
            assert root not in {"harness", "clauseguard"}, f"forbidden import: {name}"
            assert root in {"__future__", "typing"}, f"unexpected import: {name}"


class TestTheMessagePayloadMatchesTheContract:
    """app.py calls `build_messages(request.message, hits, history)` positionally while
    tests pass `history=` by keyword, so the parameter name and the order both matter.
    """

    def test_history_is_replayed_in_order_ahead_of_the_current_turn(self):
        messages = strong.build_messages(
            "third",
            [make_hit(1)],
            history=[("user", "first"), ("assistant", "second")],
        )
        assert [m["content"] for m in messages[1:]] == ["first", "second", "third"]

    def test_history_is_accepted_positionally_too(self):
        messages = strong.build_messages("q", [make_hit(1)], [("user", "earlier")])
        assert messages[1]["content"] == "earlier"

    def test_history_defaults_to_empty(self):
        messages = strong.build_messages("q", [make_hit(1)])
        assert len(messages) == 2

    def test_reference_material_is_not_disguised_as_a_prior_turn(self):
        messages = strong.build_messages("q", [make_hit(1)])
        assert "Refund window paragraph 1." in messages[0]["content"]
        assert messages[0]["role"] == "system"
        assert len(messages) == 2

    def test_the_defensive_instructions_travel_in_the_system_message(self):
        """Not a style point. If the prompt were appended after the history, a long
        conversation would push the retrieved text away from the instructions that
        govern it - and multi-turn drift is one of the strategies being measured."""
        messages = strong.build_messages(
            "now",
            [make_hit(1)],
            history=[("user", "a"), ("assistant", "b")],
        )
        assert messages[0]["content"].startswith(strong.SYSTEM_PROMPT)
        assert strong.CONTEXT_HEADER in messages[0]["content"]

    def test_the_roles_are_the_ones_the_api_accepts(self):
        messages = strong.build_messages(
            "now", [make_hit(1)], history=[("user", "a"), ("assistant", "b")]
        )
        assert [m["role"] for m in messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
