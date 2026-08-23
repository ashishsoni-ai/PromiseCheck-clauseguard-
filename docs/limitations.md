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

The judge also had to be hosted at all, though not for the reason first written here.
A local 8B judge was measured at roughly 11.7 seconds per call on the development
machine, which puts the ~30 probes that survive the L0 prefilter near six minutes
serialised. The judge is the only role on the incremental path; the extractor and the
adversary run during `clauseguard generate`, which is an install step and is allowed to
be slow. So the role placement was backwards and moving it was right. What the move
does **not** buy is §2 step 11's 45-second target — see the next entry, which has the
arithmetic.

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

## §2 step 11's 45-second target is not met, and the reason is a token quota

We do not claim "under 45 seconds for an incremental run" anywhere, and this entry
exists so that the absence is a decision rather than an oversight.

**Per-call latency is not the problem, and the number previously published here was
wrong.** `scripts/time_judge.py` measures the hosted judge at **min 0.88s, median 0.92s,
max 1.67s**, with the `litellm` + `instructor` import warmed outside the timed section.
An earlier figure of ~12–16s per call appeared in this repo and is **retracted**: it was
a two-test `pytest` wall clock divided by two, and that wall clock contained a 10.95s
import paid once per *process*, not once per call. The retraction matters more than the
correction, because the bad figure was itself a correction and therefore carried more
authority than the estimate it replaced.

**What binds is throughput.** Groq's `on_demand` tier caps `openai/gpt-oss-20b` at
**8000 tokens per minute**, and one judge call requests **1152–2178 tokens** — the high
end whenever C2's span retry fires, which is the common case for a `grants` judgment,
since `grants` is the one stance the system prompt requires spans for. That is **~5–6
judge calls per minute no matter how fast each one returns.**

| | latency budget | token budget |
|---|---|---|
| ~30 post-L0 probes, k=1 | ~27–36s of judge time | ~66,000 tokens → **~8 min** |
| ~50 calls with L3 k=3 on the consequential class | ~45–60s | ~110,000 tokens → **~14 min** |

So on latency alone an incremental run's judge phase would fit inside 45 seconds with
nothing to spare — and that figure already excludes the AUT fan-out, the extractor, the
report render and the audit write, none of which `time_judge.py` measures. On the token
budget it misses by an order of magnitude, and **no amount of concurrency or prompt
tuning moves a per-minute cap.**

**Concurrency makes it worse, not better.** DESIGN.md §2 step 6 specifies an async
fan-out with a semaphore of 8. Applied to the judge on this tier, **6 of 6 concurrent
calls failed within 0.25 seconds each with `RateLimitError`** — a concurrency cap does
not model a token budget, so a plain semaphore converts the quota into a burst of
`JudgeError` and loses rows. (The semaphore is fine where §2 step 6 aims it, at the AUT
calls, which are local.) A related gap found the same way: the judge path has **no 429
backoff at all** — `SCHEMA_REPAIR_RETRIES` covers schema repair only — even though the
error body states the exact wait to honour, e.g. `"Please try again in 10.559999999s"`.

**What would remove it,** cheapest first: a paid tier, which is a billing change and not
a design change; fewer tokens per call, since the prompt is ~1150 tokens for a *single*
clause and §4.1 allows two to four, so a real run is larger than what was measured; or a
token-budget-aware rate limiter plus 429 backoff, which lets a run take the time the
quota demands instead of failing — the honest version, and the one that keeps the number
publishable. The pacing recipe that works today is **16.5s between calls**, verified
across 16 consecutive live calls with zero rate-limit errors.

Until an end-to-end `clauseguard run` has been timed on the tier the demo actually uses,
the reported wall clock is the measured one and the 45-second figure is quoted as a
design target that this deployment does not reach.
