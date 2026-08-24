"""`rules.lock.json` and `probes.lock.json` - read, write, and version.

DESIGN.md 6.2 item 3 is the whole reason these are files rather than a cache:

    | `probes.lock.json` committed to the repo - full corpus generated offline and
    | version-controlled. CI never does bulk generation. Treat it exactly like a
    | dependency lockfile; `clauseguard generate` is the `npm install` you run
    | deliberately.

So the format is chosen for review-in-a-pull-request, not for compactness: two
space indent, keys in a fixed order, one probe per object, sorted by id. A diff
that a human can read is the point.

WHY `rule_version` IS COMPUTED ON LOAD AND NOT STORED
-----------------------------------------------------
DESIGN.md 5.1 puts `rule_version` on every audit row, next to `policy_version`,
because together they are what "make the regression story provable". A stored
version can disagree with the rule it claims to describe - an editor changes a
threshold and forgets the hash - and a wrong `rule_version` is worse than none,
because it asserts two incomparable runs are comparable. Deriving it from the
rule's own canonical JSON at load time makes that disagreement unrepresentable.

The hash covers the rule *subtree*, exceptions included. An exception is part of
what the rule means: `grants refund if days <= 30` with and without the clearance
carve-out are different rules, and a probe labelled under one must not silently
report itself as labelled under the other.

WHY A STALE LOCKFILE IS AN ERROR AND NOT A WARNING
--------------------------------------------------
Each lockfile records the `policy_version` it was built against. If the policy on
disk has moved, the labels in `probes.lock.json` were computed from clauses that
no longer exist in that form, and DESIGN.md 2 steps 1-4 specify a re-label pass
before such a run is meaningful. That pass is the gate's job (Step 8). Until it
exists, running anyway would publish an over-promise count derived from stale
ground truth, which is the one number the whole project is read off. So loading
refuses, loudly, and names both versions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from harness.schemas.clause import PolicyDocument
from harness.schemas.probe import Probe
from harness.schemas.rule import EntitlementRule

#: Bumped only when the envelope shape changes in a way an old reader would
#: misparse. Recorded in the file so a lockfile from a future branch fails with
#: "schema 2, expected 1" rather than with a KeyError somewhere downstream.
RULES_LOCK_SCHEMA = "clauseguard/rules.lock/1"
PROBES_LOCK_SCHEMA = "clauseguard/probes.lock/1"

DEFAULT_RULES_LOCK = Path("rules/rules.lock.json")
DEFAULT_PROBES_LOCK = Path("probes/probes.lock.json")


class LockfileError(RuntimeError):
    """Base: a lockfile cannot be used as it stands."""


class StaleLockfileError(LockfileError):
    """The lockfile was built against a different version of the policy."""


def canonical_json(payload: object) -> str:
    """Byte-stable JSON for hashing.

    `sort_keys` and no whitespace, so the hash depends on the content and not on
    how the serialiser happened to lay it out. `ensure_ascii=False` because a
    policy category written in Devanagari should hash the same whether or not the
    intermediate representation escaped it.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def rule_version(rule: EntitlementRule) -> str:
    """`sha256:<64 hex>` over the rule's whole subtree (DESIGN.md 5.1).

    Full length rather than truncated: unlike a clause id, this is not scoped by
    an ordinal, so there is no second component to lean on and no reason to
    shorten it.
    """
    digest = hashlib.sha256(
        canonical_json(rule.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()
    return "sha256:" + digest


def rules_digest(rules: Sequence[EntitlementRule]) -> str:
    """One version for a whole rule set, over the ordered per-rule versions.

    Recorded in `probes.lock.json` so that a probe set can say which rules it was
    labelled under. A rule edit therefore invalidates the probe labels visibly,
    which is the mechanism behind DESIGN.md 6.2 item 2 - re-label without
    regenerate.
    """
    payload = "\n".join(rule_version(rule) for rule in rules)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_envelope(path: Path, *, expected_schema: str, kind: str) -> dict:
    if not path.exists():
        raise LockfileError(
            f"{kind} lockfile {str(path)!r} does not exist. It is committed to the "
            f"repository on purpose (DESIGN.md 6.2 item 3); a run that regenerated "
            f"it on the fly would not be reproducible"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LockfileError(f"{kind} lockfile {str(path)!r} is not valid JSON: {exc}")

    if not isinstance(payload, dict) or not payload:
        raise LockfileError(
            f"{kind} lockfile {str(path)!r} is empty. The Step 0 scaffold commits "
            f"it as {{}} so the path exists; it has to be authored before a run "
            f"can mean anything"
        )
    schema = payload.get("schema")
    if schema != expected_schema:
        raise LockfileError(
            f"{kind} lockfile {str(path)!r} declares schema {schema!r}, but this "
            f"build reads {expected_schema!r}"
        )
    return payload


@dataclass(frozen=True)
class RulesLock:
    """A loaded `rules.lock.json`, with every rule's version already derived."""

    path: Path
    policy_doc: str
    policy_version: str
    rules: tuple[EntitlementRule, ...]
    versions: Mapping[str, str]
    digest: str

    def version_of(self, rule_id: str) -> str:
        """The version of the *root* rule that owns `rule_id`.

        A probe's `target_rule_id` may name a nested exception, and an exception
        has no independent version: it is only meaningful inside the tree that
        reaches it, and that tree is what determines the label. So a nested id
        resolves to its root's version rather than to a hash of the fragment.
        """
        try:
            return self.versions[rule_id]
        except KeyError:
            raise LockfileError(
                f"no rule {rule_id!r} in {str(self.path)!r}; known ids: "
                f"{', '.join(sorted(self.versions))}"
            ) from None

    def rule_ids(self) -> frozenset[str]:
        return frozenset(self.versions)

    def assert_matches_policy(self, policy: PolicyDocument) -> None:
        _assert_policy_match(self.path, "rules", self, policy)


@dataclass(frozen=True)
class ProbesLock:
    """A loaded `probes.lock.json`."""

    path: Path
    policy_doc: str
    policy_version: str
    rules_digest: str
    probes: tuple[Probe, ...]

    def assert_matches_policy(self, policy: PolicyDocument) -> None:
        _assert_policy_match(self.path, "probes", self, policy)

    def assert_matches_rules(self, rules: RulesLock) -> None:
        """Refuse a probe set labelled under a different rule set.

        The labels in this file are `evaluate_rules()` output. If the rules have
        moved, the labels are assertions about a policy nobody is running.
        """
        if self.rules_digest != rules.digest:
            raise StaleLockfileError(
                f"{str(self.path)!r} was labelled under rules digest "
                f"{self.rules_digest} but {str(rules.path)!r} now hashes to "
                f"{rules.digest}. The expected_policy_stance values in this probe "
                f"set were computed from the older rules, so they are not ground "
                f"truth for this run. Re-label the probes (DESIGN.md 6.2 item 2 - "
                f"pure Python, milliseconds) rather than running anyway"
            )


def _assert_policy_match(
    path: Path, kind: str, lock: RulesLock | ProbesLock, policy: PolicyDocument
) -> None:
    if lock.policy_doc != policy.doc_slug:
        raise StaleLockfileError(
            f"{kind} lockfile {str(path)!r} is for policy {lock.policy_doc!r}, "
            f"but the document loaded is {policy.doc_slug!r}"
        )
    if lock.policy_version != policy.policy_version:
        raise StaleLockfileError(
            f"{kind} lockfile {str(path)!r} was built against policy_version "
            f"{lock.policy_version} but {policy.doc_slug!r} on disk is now "
            f"{policy.policy_version}. Clause text has changed, so anything this "
            f"file says about it is out of date"
        )


def load_rules(path: str | Path = DEFAULT_RULES_LOCK) -> RulesLock:
    """Load and validate `rules.lock.json`.

    Every rule is validated by the Step 1 schema and the whole set by the Step 3
    engine's `validate_rule_tree`, so a lockfile whose exceptions could never
    fire is rejected here rather than producing quietly wrong labels later.
    """
    # Imported here rather than at module scope so that `harness.execution` does
    # not drag the rules engine into every import of the runner. The engine is
    # only needed by callers that actually read a rules lockfile.
    from harness.rules_engine import validate_rule_tree

    path = Path(path)
    payload = _read_envelope(path, expected_schema=RULES_LOCK_SCHEMA, kind="rules")

    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise LockfileError(
            f"{str(path)!r} carries no rules; a rule set with nothing in it "
            f"labels every probe 'denies' by default and would look like a "
            f"working oracle"
        )

    rules = tuple(EntitlementRule.model_validate(item) for item in raw_rules)
    validate_rule_tree(rules)

    versions: dict[str, str] = {}
    for root in rules:
        version = rule_version(root)
        for node in root.walk():
            if node.rule_id in versions:
                raise LockfileError(
                    f"{str(path)!r} defines rule_id {node.rule_id!r} more than "
                    f"once. The schema only forbids duplicates within one tree, "
                    f"but a duplicate across trees is worse: an audit row's "
                    f"rule_id would not identify which rule produced the label"
                )
            versions[node.rule_id] = version

    return RulesLock(
        path=path,
        policy_doc=str(payload["policy_doc"]),
        policy_version=str(payload["policy_version"]),
        rules=rules,
        versions=versions,
        digest=rules_digest(rules),
    )


def load_probes(path: str | Path = DEFAULT_PROBES_LOCK) -> ProbesLock:
    """Load and validate `probes.lock.json`."""
    path = Path(path)
    payload = _read_envelope(path, expected_schema=PROBES_LOCK_SCHEMA, kind="probes")

    raw_probes = payload.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise LockfileError(
            f"{str(path)!r} carries no probes; there is nothing to run. A run of "
            f"zero probes reports zero over-promises, which is indistinguishable "
            f"from a passing agent"
        )

    probes = tuple(Probe.model_validate(item) for item in raw_probes)

    seen: set[str] = set()
    for probe in probes:
        if probe.probe_id in seen:
            raise LockfileError(
                f"{str(path)!r} defines probe_id {probe.probe_id!r} twice; "
                f"probe_id keys the audit row, so a duplicate double-counts"
            )
        seen.add(probe.probe_id)

    return ProbesLock(
        path=path,
        policy_doc=str(payload["policy_doc"]),
        policy_version=str(payload["policy_version"]),
        rules_digest=str(payload["rules_digest"]),
        probes=probes,
    )


def write_rules(
    path: str | Path,
    *,
    rules: Sequence[EntitlementRule],
    policy: PolicyDocument,
    authored_by: str,
) -> Path:
    """Write `rules.lock.json`. Used by the authoring script, never by a run.

    `authored_by` records what produced the file. For the Step 7 slice that is a
    hand-authoring script, and saying so on the face of the artefact matters:
    DESIGN.md 9 asks for hand-computed labels at this stage, and a reader should
    not have to guess whether an extractor touched it.
    """
    from harness.rules_engine import validate_rule_tree

    validate_rule_tree(rules)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": RULES_LOCK_SCHEMA,
        "policy_doc": policy.doc_slug,
        "policy_version": policy.policy_version,
        "authored_by": authored_by,
        "rules": [rule.model_dump(mode="json") for rule in rules],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return path


def write_probes(
    path: str | Path,
    *,
    probes: Iterable[Probe],
    policy: PolicyDocument,
    rules: RulesLock,
    authored_by: str,
) -> Path:
    """Write `probes.lock.json`, sorted by `probe_id` for a readable diff."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(probes, key=lambda p: p.probe_id)
    payload = {
        "schema": PROBES_LOCK_SCHEMA,
        "policy_doc": policy.doc_slug,
        "policy_version": policy.policy_version,
        "rules_digest": rules.digest,
        "authored_by": authored_by,
        "probes": [probe.model_dump(mode="json") for probe in ordered],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return path
