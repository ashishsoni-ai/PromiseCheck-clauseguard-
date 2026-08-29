"""Export all probe/response pairs as JSONL for gold-set labeling."""
import sqlite3, json

conn = sqlite3.connect("runs.db")
rows = conn.execute("""
    SELECT row_id, probe_id, agent_id, expected_policy_stance, 
           agent_response, agent_stance, verdict_class, 
           judge_model, judge_k, span_verified, judge_abstained,
           strategy, difficulty_tier, scenario_facts, probe_turns, judge_error
    FROM audit_rows 
    ORDER BY probe_id, agent_id
""").fetchall()

pairs = []
for r in rows:
    row_id, probe_id, agent_id, expected_stance, agent_response, agent_stance, verdict, judge_model, judge_k, span_verified, judge_abstained, strategy, difficulty, facts_json, turns_json, judge_error = r
    
    pairs.append({
        "row_id": row_id,
        "probe_id": probe_id,
        "agent_id": agent_id,
        "expected_policy_stance": expected_stance,
        "agent_response": agent_response,
        "judge_verdict_class": verdict,
        "judge_agent_stance": agent_stance,
        "judge_model": judge_model,
        "judge_k": judge_k,
        "span_verified": bool(span_verified),
        "judge_abstained": bool(judge_abstained),
        "judge_error": judge_error,
        "strategy": strategy,
        "difficulty_tier": difficulty,
        "facts": json.loads(facts_json),
        "turns": json.loads(turns_json)
    })

with open("_all_pairs.json", "w") as f:
    json.dump(pairs, f, indent=2)

print(f"Exported {len(pairs)} pairs")
print()
print("=== PROBE SUMMARY ===")
probes_seen = set()
for p in pairs:
    pid = p["probe_id"]
    if pid not in probes_seen:
        probes_seen.add(pid)
        print(f"\n{pid} | strategy={p['strategy']} T{p['difficulty_tier']} | expected={p['expected_policy_stance']}")
        print(f"  Turn: {p['turns'][0][:120]}...")
        for p2 in pairs:
            if p2["probe_id"] == pid:
                print(f"  [{p2['agent_id']}] judge={p2['judge_verdict_class']} | {p2['agent_response'][:120]}...")