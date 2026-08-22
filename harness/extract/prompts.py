"""EXTRACTOR-role prompts - cold, literal, forensic (DESIGN.md 2). Later step.

CONSTRAINT (decided 2026-08-22, must be honoured when this is implemented):
every clause sent to the extractor MUST carry its `heading_path` alongside
`Clause.text`, rendered as breadcrumbs, e.g.

    Section: Acme Retail > 5. Categories excluded from return
    Clause:  Swimwear and swim accessories, including goggles and swim caps.

Why: segmentation splits list stems from their items, so `acme-refunds:008`
("The following categories cannot be returned once the hygiene seal has been
broken:") and its bullets 009-011 are individually non-normative. Clause 010 read
alone is a bare noun phrase carrying no rule; the extractor would either invent an
entitlement or extract nothing. The heading path is what restores the polarity.

The alternative - merging stem and bullets into one clause - was rejected because
it would make editing one excluded category churn its neighbours' content hashes,
costing extraction calls on clauses that did not change.
"""
