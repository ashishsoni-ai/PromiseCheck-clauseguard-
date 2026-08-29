"""Find missing probes from the review output."""
import json

with open("_all_pairs.json") as f:
    pairs = json.load(f)

for p in pairs:
    if p["probe_id"] in ("P-acme-006-false_premise-006", "P-acme-006-multi_turn_drift-001"):
        print(f"{p['probe_id']} [{p['agent_id']}]: expected={p['expected_policy_stance']} judge={p['judge_verdict_class']}")
        print(f"  Turn: {p['turns'][0][:200]}")
        print(f"  Response: {p['agent_response'][:300]}")
        print()