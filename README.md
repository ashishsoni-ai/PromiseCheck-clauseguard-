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
| 6 | Append-only audit store | not started |
| 7 | Vertical slice: `clauseguard run` | not started |

As of 2026-08-23 the suite is 852 offline tests, plus 2 `live` tests exercising the
real judge. L3 self-consistency (the k=3 majority on the consequential class,
DESIGN.md §2 step 8) is a stub: scoped out of Step 5 deliberately, not missed.

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

Three local models, one per role. The agent's is frozen in its own code; the judge
and adversary are pinned in `.env.example` and must come from families that differ
from the agent's and from each other (see §1.5 of the design):

```bash
ollama pull qwen2.5:7b-instruct   # the agent under test
ollama pull llama3.1:8b           # judge
ollama pull mistral:7b            # adversary
```

Only the extractor is hosted, on Groq, which is the one thing `GROQ_API_KEY` is
for. Check that what the config pins still exists before a run — provider
inventory expires without warning:

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
than discovered by a reviewer.
