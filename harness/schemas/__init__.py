"""Clauseguard schemas.

Five models are specified by DESIGN.md and implemented here:

    Condition, EntitlementRule   DESIGN.md 1.2
    ProbeScenario, Probe         DESIGN.md 3.1
    Judgment                     DESIGN.md 4.1

Clause and PolicyDocument are described in prose by DESIGN.md 1.1 rather than
given as code, and are modelled here to match that description plus the audit row
in 5.1.

Re-exported so call sites read `from harness.schemas import Probe` rather than
reaching into module paths that may be reorganised later.
"""

from harness.schemas.clause import Clause, CorpusRole, DocSlug, PolicyDocument
from harness.schemas.judgment import AgentStance, Judgment, PrefilterStance
from harness.schemas.probe import Probe, ProbeScenario, ProbeStrategy
from harness.schemas.rule import (
    MEMBERSHIP_OPS,
    NUMERIC_OPS,
    Condition,
    ConditionOp,
    Entitlement,
    EntitlementRule,
    Polarity,
    PolicyStance,
)

__all__ = [
    # clause.py
    "Clause",
    "CorpusRole",
    "DocSlug",
    "PolicyDocument",
    # rule.py
    "Condition",
    "ConditionOp",
    "Entitlement",
    "EntitlementRule",
    "Polarity",
    "PolicyStance",
    "NUMERIC_OPS",
    "MEMBERSHIP_OPS",
    # probe.py
    "Probe",
    "ProbeScenario",
    "ProbeStrategy",
    # judgment.py
    "AgentStance",
    "Judgment",
    "PrefilterStance",
]
