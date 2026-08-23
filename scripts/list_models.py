"""Preflight the provider model IDs the config pins. No DESIGN.md section owns this.

A model ID is provider *inventory*, not design. DESIGN.md 1.5 and 2 step 8 constrain the
model FAMILY - "the judge must come from a different model family than the agent under
test" - and never name an ID, precisely because IDs are decommissioned on the provider's
schedule and a family is a property of the measurement. So this script does not check that
the config is correct; it checks that what the config names still exists.

WHY IT IS WORTH A FILE
On 2026-08-23 `groq/llama-3.3-70b-versatile` - the pinned judge and extractor model -
started returning 404 `model_not_found`, and the first thing to notice was a live judge
test failing. That is the correct behaviour (`JudgeError`, never an abstention - see
"WHAT THE ABSTAIN RATE IS ALLOWED TO MEAN" in harness/judge/judge.py), but it is an
expensive place to learn it. DESIGN.md 2 step 6 fans out with a semaphore of 8, so
discovering a dead ID part-way through a 480-row run costs the whole run. Exit code 1 when
a pinned ID is absent makes this usable as a gate before that happens.

WHY IT DOES NOT PRINT THE KEY, EVER
The same 2026-08-23 failure dumped the raw `Authorization` header into a pytest traceback:
pytest's long traceback format prints each frame's argument values, and litellm passes
`headers` as an argument. The key was in the terminal in plain text and had to be rotated.
This script therefore reports only whether a key was FOUND, never any part of its value.
For the same reason it reads `.env` itself rather than asking you to type
`$env:GROQ_API_KEY="gsk_..."`, which would write the secret to PowerShell's
ConsoleHost_history.txt.

WHY IT PARSES `.env` BY HAND INSTEAD OF CALLING load_dotenv
`tests/conftest.py` deliberately does NOT load `.env`, and that is load-bearing: `.env`
also carries `CLAUSEGUARD_JUDGE_MODEL` and `CLAUSEGUARD_JUDGE_TEMP`, which
`resolve_judge_model()` and `resolve_judge_temp()` read, so auto-loading it would let a
local file change offline test behaviour and cost the suite its determinism. A standalone
operator script has no such duty and may read `.env` - but it must not do so by importing
a helper that the test path could later pick up too.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
ENV_PATH: Final = REPO_ROOT / ".env"

#: Mirrors the Groq-hosted entries in `.env.example`. Kept as literals rather than imported
#: from `harness` so a broken import cannot stop the diagnostic that would explain the
#: breakage.
#:
#: The ADVERSARY is absent because it runs locally on Ollama and this script asks Groq what
#: Groq has - it cannot speak for it. Its equivalent check is the `require_judge_backend`
#: fixture's `/api/tags` lookup in `tests/conftest.py`. Listing a local model here with no way
#: to verify it would be worse than omitting it: this script exits 1 to gate a run, and a gate
#: that reports on things it cannot see is how the gate stops being believed.
#:
#: The JUDGE was briefly absent for that reason and came back on 2026-08-23, when it moved
#: from `ollama_chat/llama3.1:8b` to a hosted model after a local call was measured at ~11.7s
#: against DESIGN.md 2 step 11's 45-second run target. It is the entry this script most needs:
#: the judge runs on every incremental run, step 6 fans out with a semaphore of 8, and a dead
#: judge ID raises `JudgeError` rather than abstaining - so the rows are lost, not merely
#: unverified.
PINNED: Final = {
    "extractor (DESIGN.md 2 step 3)": "groq/openai/gpt-oss-120b",
    "judge (DESIGN.md 4.1)": "groq/openai/gpt-oss-20b",
    "agent's groq fallback, GROQ_MODEL (must stay Qwen)": "qwen/qwen3.6-27b",
}

MODELS_URL: Final = "https://api.groq.com/openai/v1/models"


def read_key() -> tuple[str | None, str]:
    """Return (key, where_it_came_from). The key is never logged by the caller."""
    import os

    from_env = (os.getenv("GROQ_API_KEY") or "").strip()
    if from_env:
        return from_env, "process environment"

    if not ENV_PATH.is_file():
        return None, f"not found (no {ENV_PATH.name} and GROQ_API_KEY unset)"

    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        if name.strip() == "GROQ_API_KEY":
            value = value.strip().strip('"').strip("'")
            if value:
                return value, f"{ENV_PATH.name}"
    return None, f"not found (GROQ_API_KEY empty in {ENV_PATH.name})"


def fetch_model_ids(key: str, provenance: str = "unknown") -> list[str]:
    """The account's own inventory. Authoritative in a way documentation is not.

    `provenance` is carried in only to make the 401 self-diagnosing. It named the exact
    cause on 2026-08-23: a rotated key in `.env` was shadowed by the revoked one still
    sitting in the shell's environment, and the script had that fact in hand but made the
    operator guess. A diagnostic that reports a symptom it could have explained is a
    diagnostic that gets run twice.
    """
    import httpx

    response = httpx.get(
        MODELS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30.0
    )
    if response.status_code == 401:
        raise SystemExit(
            f"401 from Groq: the key was rejected (source: {provenance}).\n"
            "If it came from the process environment and you have just rotated: the "
            "process environment WINS over .env, and $env:GROQ_API_KEY lives as long as "
            "the shell does, so the revoked value is probably still shadowing the new "
            "one. Clear it with `Remove-Item Env:\\GROQ_API_KEY` and re-run - this "
            "script reads .env by itself, so the key never needs to be exported.\n"
            "If it came from .env, that file still holds the revoked key: update it."
        )
    response.raise_for_status()
    return sorted(entry["id"] for entry in response.json()["data"])


def display_family(model_id: str) -> str:
    """Grouping for the printout only.

    NOT the authoritative family test. That is `family_of` in `tests/model_families.py`,
    which is where DESIGN.md 1.5's separation rule is implemented and where both
    `test_aut_contract.py` and `test_judge.py` import it from. Duplicating its logic here
    would create a second source of truth for that rule, so this stays cosmetic.

    (This docstring has now named the wrong file twice: first `aut-naive/backends.py`, where
    `family_of` never lived at all, then `tests/unit/test_aut_contract.py`, which was true
    until the function was extracted on 2026-08-23. A pointer that is only sometimes right
    is the expensive kind, because the next reader trusts it instead of grepping.)

    Trailing version digits are dropped so `qwen3-32b` and `qwen2.5-7b` land in one
    bucket. Without that they head two adjacent lists and the one question this printout
    exists to answer - "which of these is NOT the agent's family?" - gets harder to read,
    not easier.
    """
    head = model_id.split("/")[-1].split("-")[0]
    alpha = re.match(r"[a-zA-Z]+", head)
    return (alpha.group(0) if alpha else head).casefold()


def main() -> int:
    key, provenance = read_key()
    print(f"GROQ_API_KEY: {'found in ' + provenance if key else provenance}")
    if not key:
        return 2

    ids = fetch_model_ids(key, provenance)
    available = set(ids)
    print(f"{len(ids)} models visible to this account\n")

    missing = []
    print("PINNED BY THE CONFIG")
    for role, litellm_id in PINNED.items():
        # litellm addresses models as "groq/<id>"; the provider lists the bare id.
        bare = litellm_id.split("/", 1)[1] if litellm_id.startswith("groq/") else litellm_id
        ok = bare in available
        if not ok:
            missing.append((role, litellm_id))
        print(f"  [{'OK     ' if ok else 'MISSING'}] {litellm_id:<40} {role}")

    print("\nAVAILABLE, GROUPED FOR THE DESIGN.md 1.5 FAMILY CHOICE")
    for family in sorted({display_family(i) for i in ids}):
        members = [i for i in ids if display_family(i) == family]
        print(f"  {family}:")
        for member in members:
            print(f"    {member}")

    if missing:
        print(
            f"\n{len(missing)} pinned model(s) no longer exist. Re-pin before any run: a "
            "404 mid-fan-out costs the whole run, and the judge raises JudgeError rather "
            "than abstaining, so the rows are lost rather than merely unverified."
        )
        return 1
    print("\nAll pinned models are available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
