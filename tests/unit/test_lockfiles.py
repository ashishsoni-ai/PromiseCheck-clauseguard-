"""Byte-level contract for the two lockfile writers.

These files are committed evidence, so their bytes are part of what a reviewer
audits. That makes two properties worth pinning.

FIRST, LF ENDINGS. `Path.write_text` translates `\\n` to `os.linesep` unless
`newline=` says otherwise, so the same writer emits LF on Linux and CRLF on
Windows. `write_rules` and `write_probes` shipped without `newline="\\n"` and
duly wrote CRLF on Windows: 945 stray CR bytes in `probes.lock.json` and 489 in
`rules.lock.json`, which is how the defect was found.

Be honest about what these assertions are worth on which host. On Linux and
macOS `assert b"\\r" not in ...` passes whether or not the writer passes
`newline=`, because the translation it guards against is a no-op there. The
assertion only bites on Windows. That is not a reason to drop it - a
contributor adding a third writer is exactly who it is for - but a green run on
Linux CI is not evidence the parameter is present. Reading the writer is.

SECOND, AND THE REASON THIS WAS A PAPERCUT AND NOT A BROKEN COMMITMENT: the
rules digest does not depend on line endings. `rules_digest` hashes
`rule_version()` over parsed models, and `_read_envelope` reads through
universal newlines before `json.loads`, so a CRLF lockfile and an LF lockfile
produce the same digest and C1's rules-to-probes binding never moved.
`test_the_digest_ignores_line_endings` pins that. If someone later reworks the
digest to hash raw bytes, that test fails, and it should - it would turn every
Windows-authored lockfile into a spurious staleness refusal.

Contrast `harness/ingest/loaders.py`, where the same omission WAS load-bearing
because it hashed text it had written and read back. Same mistake, different
blast radius, and the difference is worth being able to point at.
"""

import json

import pytest

from harness.execution.lockfiles import (
    PROBES_LOCK_SCHEMA,
    RULES_LOCK_SCHEMA,
    load_probes,
    load_rules,
    write_probes,
    write_rules,
)


@pytest.fixture
def written_rules(tmp_path, sample_policy_document, basic_grant_rule):
    path = write_rules(
        tmp_path / "rules.lock.json",
        rules=[basic_grant_rule],
        policy=sample_policy_document,
        authored_by="tests",
    )
    return path


@pytest.fixture
def written_probes(tmp_path, sample_policy_document, sample_probe, written_rules):
    return write_probes(
        tmp_path / "probes.lock.json",
        probes=[sample_probe],
        policy=sample_policy_document,
        rules=load_rules(written_rules),
        authored_by="tests",
    )


class TestLineEndings:
    """See the module docstring on what these prove per host."""

    def test_the_rules_lock_has_no_carriage_returns(self, written_rules):
        assert b"\r" not in written_rules.read_bytes()

    def test_the_probes_lock_has_no_carriage_returns(self, written_probes):
        assert b"\r" not in written_probes.read_bytes()

    @pytest.mark.parametrize("fixture", ["written_rules", "written_probes"])
    def test_it_ends_with_exactly_one_lf(self, fixture, request):
        raw = request.getfixturevalue(fixture).read_bytes()
        assert raw.endswith(b"\n")
        assert not raw.endswith(b"\n\n")

    def test_read_text_cannot_see_the_defect(self, tmp_path):
        """Why the sibling writers' newline test never caught this.

        `test_it_ends_with_a_newline` in test_manifest.py and test_coverage.py
        both assert through `read_text`, which collapses CRLF on the way in. A
        CRLF file passes them. Pinning that here so the weaker assertion is not
        copied into a third writer in the belief that it covers endings.
        """
        crlf = tmp_path / "crlf.json"
        crlf.write_bytes(b'{"a": 1}\r\n')
        assert crlf.read_text(encoding="utf-8").endswith("\n")  # passes anyway
        assert b"\r" in crlf.read_bytes()  # the defect, visible only in bytes


class TestDeterminism:
    def test_rewriting_unchanged_rules_produces_identical_bytes(
        self, tmp_path, sample_policy_document, basic_grant_rule
    ):
        """A lockfile whose bytes churn between identical writes stops being
        readable evidence: every diff looks like a change to the rules."""
        kwargs = dict(
            rules=[basic_grant_rule],
            policy=sample_policy_document,
            authored_by="tests",
        )
        a = write_rules(tmp_path / "a.json", **kwargs)
        b = write_rules(tmp_path / "b.json", **kwargs)
        assert a.read_bytes() == b.read_bytes()

    def test_probes_are_sorted_by_id_regardless_of_input_order(
        self, tmp_path, sample_policy_document, sample_probe, written_rules
    ):
        second = sample_probe.model_copy(update={"probe_id": "P-acme-001-boundary-001"})
        rules = load_rules(written_rules)
        forward = write_probes(
            tmp_path / "f.json",
            probes=[sample_probe, second],
            policy=sample_policy_document,
            rules=rules,
            authored_by="tests",
        )
        reverse = write_probes(
            tmp_path / "r.json",
            probes=[second, sample_probe],
            policy=sample_policy_document,
            rules=rules,
            authored_by="tests",
        )
        assert forward.read_bytes() == reverse.read_bytes()
        ids = [p["probe_id"] for p in json.loads(forward.read_text())["probes"]]
        assert ids == sorted(ids)


class TestDigestIsNewlineIndependent:
    def test_the_digest_ignores_line_endings(self, written_rules):
        """The property that made the CRLF defect a papercut. If this fails,
        every Windows-authored lockfile becomes a spurious staleness refusal.
        """
        lf_digest = load_rules(written_rules).digest

        crlf = written_rules.read_bytes().replace(b"\n", b"\r\n")
        assert b"\r\n" in crlf, "the fixture must actually be CRLF to prove anything"
        with written_rules.open("wb") as handle:
            handle.write(crlf)

        assert load_rules(written_rules).digest == lf_digest

    def test_a_crlf_probes_lock_still_loads(self, written_probes):
        expected = [p.probe_id for p in load_probes(written_probes).probes]
        with written_probes.open("wb") as handle:
            handle.write(written_probes.read_bytes().replace(b"\n", b"\r\n"))
        assert [p.probe_id for p in load_probes(written_probes).probes] == expected


class TestEnvelope:
    def test_the_rules_lock_stamps_its_schema(self, written_rules):
        assert json.loads(written_rules.read_text())["schema"] == RULES_LOCK_SCHEMA

    def test_the_probes_lock_stamps_its_schema(self, written_probes):
        assert json.loads(written_probes.read_text())["schema"] == PROBES_LOCK_SCHEMA

    def test_the_probes_lock_records_the_rules_digest(
        self, written_probes, written_rules
    ):
        recorded = json.loads(written_probes.read_text())["rules_digest"]
        assert recorded == load_rules(written_rules).digest

    def test_authored_by_reaches_the_file(self, written_rules):
        """DESIGN.md 9 wants hand-authored labels legible as such, so a reader
        does not have to guess whether an extractor produced this."""
        assert json.loads(written_rules.read_text())["authored_by"] == "tests"

    def test_it_creates_the_parent_directory(
        self, tmp_path, sample_policy_document, basic_grant_rule
    ):
        path = write_rules(
            tmp_path / "deep" / "nested" / "rules.lock.json",
            rules=[basic_grant_rule],
            policy=sample_policy_document,
            authored_by="tests",
        )
        assert path.is_file()
