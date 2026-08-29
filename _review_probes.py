"""Review all 30 probes with both agent responses for gold labeling."""
import json

with open("_all_pairs.json") as f:
    pairs = json.load(f)

probes = {}
for p in pairs:
    pid = p["probe_id"]
    if pid not in probes:
        probes[pid] = {
            "expected": p["expected_policy_stance"],
            "strategy": p["strategy"],
            "tier": p["difficulty_tier"],
            "turn": p["turns"][0],
            "facts": p["facts"],
            "agents": {}
        }
    probes[pid]["agents"][p["agent_id"]] = {
        "response": p["agent_response"],
        "judge_verdict": p["judge_verdict_class"],
        "judge_stance": p["judge_agent_stance"],
        "judge_model": p["judge_model"],
        "span_verified": p["span_verified"],
        "judge_error": p["judge_error"]
    }

for pid, pdata in sorted(probes.items()):
    print("=" * 80)
    print(f"PROBE: {pid} | strategy={pdata['strategy']} T{pdata['tier']} | expected={pdata['expected']}")
    print(f"CUSTOMER: {pdata['turn'][:200]}")
    print()
    for agent in ["aut-naive", "aut-strong"]:
        a = pdata["agents"].get(agent)
        if not a:
            continue
        print(f"--- [{agent}] judge_verdict={a['judge_verdict']} stance={a['judge_stance']} model={a['judge_model']} span_ok={a['span_verified']} error={a['judge_error']} ---")
        print(f"RESPONSE: {a['response'][:350]}")
        print()