"""60-second run dashboard (DESIGN.md 5.2). The minimal summary view is served by
`harness/web/app.py` (one route, one template) — it renders what `clauseguard run`
prints. This module is where any richer dashboard work would land; it is deferred
because the CLI already serves the content and a larger web surface would not add
substance to the submission.
"""