"""Export all probe/response pairs for gold-set labeling."""
import sqlite3, json, sys

conn = sqlite3.connect("runs.db")
rows = conn.execute("""
    SELECT row_id, probe_id, agent_id, expected_policy_stance, 
           agent_response, agent_stance, verdict_class, 
           judge_model, judge_k, span_verified, judge_abstained,
           strategy, difficulty_tier, scenario_facts, probe_turns
    FROM audit_rows 
    ORDER BY probe_id, agent_id
""").fetchall()

print(f"Total rows: {len(rows)}")
print()

for r in rows:
    row_id, probe_id, agent_id, expected_stance, agent_response, agent_stance, verdict, judge_model, judge_k, span_verified, judge_abstained, strategy, difficulty, facts_json, turns_json = r
    
    facts = json.loads(facts_json)
    turns = json.loads(turns_json)
    
    print("=" * 80)
    print(f"ROW: {row_id}")
    print(f"PROBE: {probe_id}  |  AGENT: {agent_id}  |  STRATEGY: {strategy} (T{str(difficulty)})")
    print(f"EXPECTED STANCE: {expected_stance}")
    print(f"JUDGE VERDICT: agent_stance={agent_stance}  verdict_class={verdict}  judge_k={judge_k}  span_ok={span_verified}  abstained={judge_abstained}")
    print(f"JUDGE MODEL: {judge_model}")
    print()
    print("--- CUSTOMER MESSAGE ---")
    for t in turns:
        print(t)
    print()
    print("--- AGENT RESPONSE ---")
    print(agent_response[:500])
    print()
    print("--- KEY FACTS ---")
    # Show only the decision-relevant facts
    for k, v in sorted(facts.items()):
        print(f"  {k}: {v}")
    print()