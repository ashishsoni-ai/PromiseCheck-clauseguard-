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
| 0 | Scaffold + dependency setup | in review |
| 1 | Pydantic schemas | not started |
| 2 | Ingest + clause hashing | not started |
| 3 | `evaluate_rules()` correctness core | not started |
| 4 | `aut-naive`, frozen by SHA | not started |
| 5 | Judge L0 prefilter + L1 classify + L2 span verify | not started |
| 6 | Append-only audit store | not started |
| 7 | Vertical slice: `clauseguard run` | not started |

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

Requires Python 3.11+, and for `aut-naive`, Docker plus Ollama.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .
copy .env.example .env          # then fill in GROQ_API_KEY
```

Verify:

```bash
python -c "import fastapi, sqlmodel, instructor, litellm; print('deps ok')"
pytest
```

The default test suite is offline and deterministic. Tests that hit a real
provider are marked `live` and deselected unless you ask for them:

```bash
pytest -m live
```

## Corpus honesty

Real policy documents are fetched, not vendored: the repository ships URLs,
fetch timestamps, content hashes, and the fetcher — not the corpus. Synthetic
policies are labelled as stress fixtures, not evidence, and real vs. synthetic
results are always broken out separately rather than pooled.

## Limitations

Kept in [`docs/limitations.md`](docs/limitations.md) and stated up front rather
than discovered by a reviewer.
