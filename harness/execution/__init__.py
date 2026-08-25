"""Execution: probes in, audit rows out (DESIGN.md 2 steps 6-10).

Three modules, split along the line between what is committed and what is done:

    lockfiles   read/write `rules.lock.json` and `probes.lock.json`, derive versions
    grounding   every `source_span` must be verbatim in a clause its rule cites
    runner      the two-phase run - agent fan-out, then paced judging, then rows

`grounding` sits below both of the others in the import graph - `lockfiles` calls it
when writing, `runner` calls it before phase 1 - so it imports neither.

Re-exported so call sites read `from harness.execution import execute_run` rather
than reaching into module paths, matching `harness.audit` and `harness.schemas`.

Nothing here decides pass or fail. DESIGN.md 2 step 11 and the `--max-overpromise`
comparison are Step 8's gate; a runner that also gated would make "the number" and
"the verdict about the number" the same code path, and the number has to be
readable by someone who disagrees with the threshold.
"""

from __future__ import annotations

from harness.execution.grounding import (
    GroundingReport,
    UngroundedSpan,
    UngroundedSpanError,
    assert_spans_grounded,
    check_spans_grounded,
)
from harness.execution.lockfiles import (
    DEFAULT_PROBES_LOCK,
    DEFAULT_RULES_LOCK,
    PROBES_LOCK_SCHEMA,
    RULES_LOCK_SCHEMA,
    LockfileError,
    ProbesLock,
    RulesLock,
    StaleLockfileError,
    canonical_json,
    load_probes,
    load_rules,
    rule_version,
    rules_digest,
    write_probes,
    write_rules,
)
from harness.execution.runner import (
    AGENT_ATTEMPTS,
    DEFAULT_AGENT_CONCURRENCY,
    DEFAULT_AGENT_TIMEOUT_S,
    DEFAULT_JUDGE_PACE_S,
    L0_JUDGE_MODEL,
    AgentClient,
    AgentIdentity,
    AgentReply,
    AgentUnavailableError,
    Exchange,
    FrozenAgentMismatchError,
    HttpxAgentClient,
    Judged,
    RowContext,
    RunError,
    RunResult,
    UnresolvedClauseError,
    agent_phase,
    assert_clauses_resolve,
    assert_rules_resolve,
    build_row,
    clause_index,
    collect_exchanges,
    execute_run,
    harness_git_sha,
    judge_exchanges,
    new_session_id,
)

__all__ = [
    "AGENT_ATTEMPTS",
    "DEFAULT_AGENT_CONCURRENCY",
    "DEFAULT_AGENT_TIMEOUT_S",
    "DEFAULT_JUDGE_PACE_S",
    "DEFAULT_PROBES_LOCK",
    "DEFAULT_RULES_LOCK",
    "L0_JUDGE_MODEL",
    "PROBES_LOCK_SCHEMA",
    "RULES_LOCK_SCHEMA",
    "AgentClient",
    "AgentIdentity",
    "AgentReply",
    "AgentUnavailableError",
    "Exchange",
    "FrozenAgentMismatchError",
    "GroundingReport",
    "HttpxAgentClient",
    "Judged",
    "LockfileError",
    "ProbesLock",
    "RowContext",
    "RulesLock",
    "RunError",
    "RunResult",
    "StaleLockfileError",
    "UngroundedSpan",
    "UngroundedSpanError",
    "UnresolvedClauseError",
    "agent_phase",
    "assert_clauses_resolve",
    "assert_rules_resolve",
    "assert_spans_grounded",
    "build_row",
    "canonical_json",
    "check_spans_grounded",
    "clause_index",
    "collect_exchanges",
    "execute_run",
    "harness_git_sha",
    "judge_exchanges",
    "load_probes",
    "load_rules",
    "new_session_id",
    "rule_version",
    "rules_digest",
    "write_probes",
    "write_rules",
]
