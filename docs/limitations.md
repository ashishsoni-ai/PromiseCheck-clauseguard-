# Limitations (DESIGN.md 8)

Finalised at freeze/package time, but written down as they arise rather than
reconstructed at the end. A limitation recalled from memory on the last day is a
limitation that gets softened.

## The judge and the extractor come from the same model family

As of 2026-08-23 three model families cover four roles:

| Role | Model | Where | Family |
|---|---|---|---|
| agent under test | `qwen2.5:7b-instruct` | local, frozen in `aut-naive` | qwen |
| extractor | `groq/openai/gpt-oss-120b` | hosted | **gpt-oss** |
| judge | `groq/openai/gpt-oss-20b` | hosted | **gpt-oss** |
| adversary | `ollama_chat/mistral:7b` | local | mistral |

**Why it is this way.** On 2026-08-23 this account's own `/openai/v1/models` endpoint
listed 13 models (`python scripts/list_models.py` prints the grouping). Nine are not
judges at all: two ASR (`whisper-large-v3`, `-turbo`), two TTS (`canopylabs/orpheus-*`),
two prompt-injection classifiers at 22M and 86M parameters
(`meta-llama/llama-prompt-guard-2-*`, a "llama family" that is a mirage), and two
agentic systems with built-in web search (`groq/compound`, `-mini`) which §4.1
disqualifies outright — a judge that can search the web could pull in policy text
that is not among the 2–4 candidate clauses it was given.

That leaves five chat models, and only one of them is a genuine fourth-family
candidate:

| Candidate | Family | Why not the judge |
|---|---|---|
| `openai/gpt-oss-120b` | gpt-oss | in use as the extractor |
| `openai/gpt-oss-20b` | gpt-oss | **chosen** |
| `openai/gpt-oss-safeguard-20b` | gpt-oss | same family; no separation gained |
| `qwen/qwen3.6-27b` | qwen | the agent's own family — closed by §1.5 |
| `allam-2-7b` | allam | see below |

So the accurate statement is not that a hosted judge is *necessarily* gpt-oss — it is
that **no hosted model in a fourth family is a suitable judge**. `allam-2-7b` is a
real instruction-tuned chat model and it would remove this limitation on paper, but it
was passed over for three stacked reasons: it is a 7B Arabic-first bilingual model
being asked to read English policy prose; its support for tool-calling, which
`instructor`'s default TOOLS mode requires for structured output, is unverified here;
and the judge is the one role whose output is checked mechanically, since C2 demands a
span that survives exact substring matching against the cited clause. A model
specialised for another language is a poor bet against that check, and a judge that
abstains constantly is worse for the published numbers than a judge that shares a
family with the extractor. If `allam-2-7b` were spiked and held up, the right move
would be to take it and delete this entry.

The judge also had to be hosted at all: a local 8B judge was measured at roughly
11.7 seconds per call on the development machine, which puts the ~30 probes that
survive the L0 prefilter near six minutes serialised, against §2 step 11's target of
under 45 seconds for an incremental run. The judge is the only role on that path; the
extractor and the adversary run during `clauseguard generate`, which is an install
step and is allowed to be slow.

**What the specification actually requires, and what it does not.** §1.5's rule is
judge-versus-AUT, and that is met (gpt-oss vs qwen). §2's circularity warning names
one mechanism — "a model that generated a probe is measurably more likely to accept
a response that pattern-matches its own generation" — which describes the
adversary/judge relation. That pair stays separated, on different families *and*
different providers, and is asserted in
`tests/unit/test_aut_contract.py::TestTheJudgeDoesNotShareAFamilyWithTheAdversary`.
Nothing in the design forbids the extractor/judge overlap.

**Why we think it is the weaker of the two overlaps — and where that reasoning
stops.** The extractor produces rules; under commitment C1 the ground-truth label
is computed from those rules by `evaluate_rules()` in Python. The judge never sees
that label and never grades the extractor's output. It classifies what the agent's
reply committed to and cites a span, which C2 then checks by exact substring match.
For a shared blind spot to reach a headline number it would have to route through a
rule the extractor mis-extracted *and* a response the judge mis-classified in the
same direction, with the span check passing throughout.

That is not zero, and the honest statement of the residual risk is this: both
models share pretraining, so both may find the same wrong reading of an ambiguous
clause natural. Where a clause is genuinely ambiguous, the rule the extractor wrote
and the stance the judge finds reasonable could agree for a shared reason rather
than a correct one. The human review of `rules.lock.json` is the control that
stands between that and a published number, which is an argument for taking that
review seriously rather than an argument that the overlap is harmless.

**What would remove it.** A judge from a fourth family that is fast enough for the
incremental path. Three routes, cheapest first: spike `allam-2-7b` on the two live
judge tests and keep it if its spans verify; a different hosted provider; or local
hardware that runs an 8B judge at usable speed. All three are configuration changes,
not design changes:
per the design appendix the judge is addressed through `litellm` so a swap is one
config line. The overlap is pinned by
`tests/unit/test_aut_contract.py::TestTheKnownFamilyCollisionIsTheDocumentedOne`,
which asserts that this is the *only* family overlap, so a re-pin that moves the
collision elsewhere fails rather than passing quietly.
