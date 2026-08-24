# Clauseguard

**A policy-conformance harness and deploy gate for money-touching agents.**

> Every merchant now ships an agent that can promise money. Nobody ships a test
> suite for it. This is that test suite, and it fails your build.

Razorpay Open Track submission. Architecture and build plan live in
[`../DESIGN.md`](../DESIGN.md) — that document is the specification; this README
tracks what is actually built.

## The three commitments

Everything downstream depends on these, and they are visible in the design
rather than bolted on:

**C1 — Ground-truth labels are derived deterministically from rules, never from
an LLM.** The probe generator produces the natural-language *surface* of a
question. The correct answer is computed by evaluating the extracted rule's
conditions in Python. An LLM never decides whether a probe's expected answer is
"grant" or "deny". This is what stops the whole thing being circular.

**C2 — The judge must quote a span that literally exists in the cited clause.**
Exact-substring verification after normalisation. If the quote isn't in the
clause, the judgment is void, retried once, then abstained. This converts "trust
the LLM judge" into a falsifiable mechanical check on every single row.

**C3 — The agent under test is frozen by commit SHA before any probe exists.**
Written once, never touched again. Two agents: one deliberately naive, one
deliberately good. If only the naive one fails, we built a strawman detector.

## Status

Built strictly step by step; each step is tested and demonstrated before the
next begins.

| Step | Component | State |
|---|---|---|
| 0 | Scaffold + dependency setup | done |
| 1 | Pydantic schemas | done |
| 2 | Ingest + clause hashing | done |
| 3 | `evaluate_rules()` correctness core | done |
| 4 | `aut-naive`, frozen by SHA | done, frozen at `aut-naive-v1` |
| 5 | Judge L0 prefilter + L1 classify + L2 span verify | done |
| 6 | Append-only audit store | done |
| 7 | Vertical slice: `clauseguard run` | offline-closed; one live run outstanding |

As of 2026-08-24 the suite is 1129 offline tests, plus 2 `live` tests exercising the
real judge. Step 7's offline proof is complete — the whole slice runs end to end
against a stubbed agent and a stubbed judge — and what remains is a single live
30-probe run against the frozen `aut-naive` container, which is the only evidence
that has to be produced on real hardware against a real provider.

Two things about that run are stated up front rather than discovered by a reader.
It takes roughly **eight minutes**, not the 45 seconds DESIGN.md §2 step 11 targets,
because Groq's free tier caps this model at 8000 tokens per minute and a judge call
requests 1152–2178 of them; the run is paced at 16.5s between judge calls to stay
under that ceiling, and `harness/judge/ratelimit.py` honours the provider's stated
wait if a 429 lands anyway. That figure is published as measured rather than
restated as a target. And L3 self-consistency (the k=3 majority on the consequential
class, DESIGN.md §2 step 8) is a stub: scoped out of Step 5 deliberately, not
missed. The live measurements since then have made it load-bearing rather than
polish — but they also showed it only addresses half the problem, which the third
limitations entry explains.

Everything after Step 7 — rule extraction, probe-generation automation,
`aut-strong`, the CI gate, the dashboard — is deliberately not started yet.

## Layout

```
harness/       the harness itself; the only thing that imports harness/
aut-naive/     agent under test #1: separate container, zero harness imports
aut-strong/    agent under test #2: separate container, zero harness imports
policies/      policy documents + .clauseguard/manifest.json clause hashes
rules/         rules.lock.json - human-reviewed extracted rules
probes/        probes.lock.json - version-controlled probe corpus
tests/         unit / integration / gold labels
```

`rules.lock.json` and `probes.lock.json` are treated exactly like dependency
lockfiles. CI never does bulk generation; `clauseguard generate` is the
`npm install` you run deliberately.

## Setup

Requires Python 3.11+, and Ollama. Docker too, for `aut-naive`.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .
copy .env.example .env          # then fill in GROQ_API_KEY
```

Two local models. The agent's is frozen in its own code; the adversary is pinned in
`.env.example`. Every role must come from a family that differs from the agent's
(see §1.5 of the design), and the adversary must also differ from the judge:

```bash
ollama pull qwen2.5:7b-instruct   # the agent under test
ollama pull mistral:7b            # adversary
```

The extractor and the judge are hosted on Groq, which is what `GROQ_API_KEY` is
for. The judge ran locally for part of 2026-08-23 and moved back, because the judge
is the only role on the incremental path and a local 8B judge measured ~11.7s per
call on the development machine. Hosted, it measures ~0.9s per call — but that does
not deliver §2 step 11's 45-second target, because the free tier caps this model at
8000 tokens per minute and a judge call costs 1152–2178 of them. The binding
constraint is a token quota, not latency; `docs/limitations.md` has the arithmetic
and this project does not claim the 45-second figure.

Check that what the config pins still exists before a run — provider inventory
expires without warning:

```bash
python scripts/list_models.py
```

Verify:

```bash
python -c "import fastapi, sqlmodel, instructor, litellm; print('deps ok')"
pytest
```

The default test suite is offline and deterministic. Tests that hit a real
provider or a local model are marked `live` and deselected unless you ask for
them. Use `--tb=short`: the default long traceback prints each frame's argument
values, and litellm takes `api_key` and `headers` as arguments.

```bash
pytest -m live --tb=short
```

## Corpus honesty

Real policy documents are fetched, not vendored: the repository ships URLs,
fetch timestamps, content hashes, and the fetcher — not the corpus. Synthetic
policies are labelled as stress fixtures, not evidence, and real vs. synthetic
results are always broken out separately rather than pooled.

## Limitations

Kept in [`docs/limitations.md`](docs/limitations.md) and stated up front rather
than discovered by a reviewer. Three real entries so far.

The judge and the extractor come from the same model family, because no hosted model
in a fourth family was a suitable judge and a local judge was far too slow for the
incremental path; the entry names the candidate that was passed over and why.

§2 step 11's 45-second run target is **not met** — the hosted judge is fast per call
(~0.9s) but the free tier's 8000 tokens-per-minute cap allows only ~5–6 judge calls a
minute, so an incremental run's judge phase is minutes rather than seconds. That entry
carries the measurement, the arithmetic, and what would remove it.

**The judge's stance flips on a probe detail that bears on nothing.** Appending an
order reference to the customer's message — which the clause, the reply and the
question all ignore — moved the flagship over-promise fixture from 8/8 `grants` to 1/8
(Fisher exact, two-tailed, p = 0.0014). The flip runs toward `denies`, the direction
that hides an over-promise, and `response_span` shows why: the judge quotes a different
sentence of the same unchanged reply. k=3 majority voting does not fix it. This is a
limitation of the measuring instrument, so it is stated before any headline number is,
and the entry says plainly which rate must not be quoted as the judge's accuracy.
