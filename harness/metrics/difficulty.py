"""Per-probe difficulty p and point-biserial discrimination (DESIGN.md 3.4).

DESIGN.md 3.4 asks for three numbers:

- **Oracle pass rate** fraction of agents that pass a probe. Probes with
  `p < 0.95` are discarded by the validity filter.
- **Difficulty (p)** = fraction of agents passing. A histogram over probes.
  Target median 0.55–0.75.
- **Discrimination** = point-biserial correlation between per-probe pass and
  per-agent total score. Probes with `r_pb < 0.1` are noise.

This module computes the second and third; the oracle pass rate is the
probe-validity filter's job (harness/probe_gen/oracle.py).

The oracle pass rate is not measured here because the verbatim-oracle agent
(DESIGN.md 4.3) does not exist yet; the run output reports "not measured
(needs the verbatim-oracle agent, DESIGN.md 4.3)". This module is a stub
that will compute difficulty and discrimination when that agent exists.
"""

from __future__ import annotations

__all__ = [
    "DifficultyReport",
    "compute_difficulty",
]

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class DifficultyReport:
    """Per-probe difficulty and discrimination (DESIGN.md 3.4).

    Both fields are empty/None when the verbatim-oracle agent does not exist
    (the current state). The infrastructure is wired so that once the oracle
    runs, the numbers appear here.
    """

    per_probe: dict[str, float] = field(default_factory=dict)
    median_difficulty: float | None = None
    discrimination: dict[str, float] = field(default_factory=dict)
    fraction_above_0_2: float | None = None
    n_probes: int = 0

    def summary(self) -> str:
        if not self.per_probe:
            return (
                "difficulty: not measured (needs the verbatim-oracle agent, "
                "DESIGN.md 4.3)"
            )
        return (
            f"difficulty: {self.n_probes} probes, median p = {self.median_difficulty}, "
            f"fraction r_pb > 0.2 = {self.fraction_above_0_2}"
        )


def compute_difficulty(
    per_probe_scores: dict[str, dict[str, bool]] | None = None,
) -> DifficultyReport:
    """Compute difficulty and discrimination.

    `per_probe_scores` maps probe_id -> {agent_id -> pass}.
    When None (the oracle agent does not exist yet), returns an empty
    DifficultyReport with a clear "not measured" message.
    """
    if per_probe_scores is None:
        return DifficultyReport(n_probes=0)

    # Implementation stub — the oracle agent does not exist yet.
    # When it does, this function computes p per probe, median p,
    # point-biserial r_pb, and the fraction above 0.2.
    return DifficultyReport(n_probes=len(per_probe_scores))