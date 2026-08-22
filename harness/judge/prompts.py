"""JUDGE-role prompts, temp 0.0, different model family from AUT (DESIGN.md 2). STEP 5.

CONSTRAINT (decided 2026-08-22, must be honoured when this is implemented):
the cited clause shown to the judge MUST include its `heading_path` breadcrumbs,
for the same reason the extractor needs them - see harness/extract/prompts.py.
DESIGN.md 4.1 has the judge see clauses in isolation, and in isolation
`acme-refunds:010` reads "Swimwear and swim accessories, including goggles and
swim caps.", which states no rule at all.

CAUTION FOR COMMITMENT C2: the heading path is CONTEXT, not quotable text. L2 span
verification substring-matches against `Clause.text` only. If the breadcrumbs were
concatenated into the same field the judge is told to quote from, a judge could
return a span drawn from a heading, and it would verify - which would let a
fabrication pass the check that exists to catch fabrications. Render them as a
separate labelled field, and keep span verification pointed at `Clause.text`.
"""
