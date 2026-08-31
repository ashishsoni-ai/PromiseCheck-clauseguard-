# scripts/export_for_labeling.py
# --------------------------------------------------------------------------
# One-time tooling: builds tests/gold/review_worksheet.csv from runs.db.
# That worksheet was then hand-labeled by a human to produce
# tests/gold/gold_labels.jsonl (60 rows: 30 probes × 2 agents).
# Kept for reproducibility: traces the provenance of the gold set and the
# κ = 0.612 result.  Re-running this script regenerates the blank worksheet;
# it does NOT regenerate the labels — those are the human's work.
# --------------------------------------------------------------------------
"""Build the labeling worksheet: CSV with probe text, agent response, and policy clause text.
No judge verdicts, no expected_policy_stance, no bias.

FIXED: clause IDs like acme-refunds:013:562deca4 use sub-clause numbering
that doesn't match section headings. We now resolve via:
1. source_span matching from rules.lock.json (for clauses in rules)
2. ordinal→section fallback for clauses not in rules (like 020)
"""
import sqlite3, json, csv, re

# ── Load policy ──
with open("policies/acme-refunds.md", "r") as f:
    policy_raw = f.read()

# Parse sections: heading -> body
sections = {}
current_heading = None
current_lines = []
for line in policy_raw.split("\n"):
    m = re.match(r"^## (\d+)\.\s+(.+)$", line)
    if m:
        if current_heading:
            sections[current_heading] = "\n".join(current_lines).strip()
        num = int(m.group(1))
        title = m.group(2).strip()
        current_heading = f"{num:03d}"
        current_lines = [f"## {num}. {title}"]
    elif current_heading:
        current_lines.append(line)
if current_heading:
    sections[current_heading] = "\n".join(current_lines).strip()

# ── Build clause_id → section mapping ──

# Method 1: source_span matching from rules.lock.json
with open("rules/rules.lock.json", "r") as f:
    rules_data = json.load(f)

def collect_clause_spans(rule_list, mapping):
    for rule in rule_list:
        for cid in rule.get("clause_ids", []):
            if cid not in mapping:
                mapping[cid] = []
        for cond in rule.get("conditions", []):
            span = cond.get("source_span", "")
            for cid in rule.get("clause_ids", []):
                if span and span not in mapping[cid]:
                    mapping[cid].append(span)
        for exc in rule.get("exceptions", []):
            collect_clause_spans([exc], mapping)

raw_mapping = {}
collect_clause_spans(rules_data["rules"], raw_mapping)

clause_to_section = {}
for cid, spans in raw_mapping.items():
    best_section = None
    best_matches = 0
    for sec_key, sec_text in sections.items():
        matches = sum(1 for s in spans if s.strip() in sec_text)
        if matches > best_matches:
            best_matches = matches
            best_section = sec_key
    clause_to_section[cid] = best_section

# Method 2: ordinal→section fallback for clauses not in rules
# Clause ordinals follow document order. The policy has 10 sections.
# Ordinal ranges per section (from the ingest output):
#   001 → section 1 (Scope)
#   002-003 → section 2 (Definitions)
#   004-005 → section 3 (Eligibility)
#   006-007 → section 4 (Return window)
#   008-013 → section 5 (Categories excluded)
#   014-015 → section 6 (Condition)
#   016-017 → section 7 (Refunds)
#   018 → section 8 (Cancellations)
#   019 → section 9 (Exchanges)
#   020 → section 10 (Escalation)
ORDINAL_TO_SECTION = {
    1: "001", 2: "002", 3: "002",
    4: "003", 5: "003",
    6: "004", 7: "004",
    8: "005", 9: "005", 10: "005", 11: "005", 12: "005", 13: "005",
    14: "006", 15: "006",
    16: "007", 17: "007",
    18: "008",
    19: "009",
    20: "010",
}

def resolve_clause_text(clause_ids_json):
    """Given a JSON list of clause IDs, return the policy section text(s)."""
    clause_ids = json.loads(clause_ids_json) if isinstance(clause_ids_json, str) else clause_ids_json
    parts = []
    seen_sections = set()
    for cid in clause_ids:
        # Try source_span mapping first
        sec = clause_to_section.get(cid)
        if sec and sec not in seen_sections:
            seen_sections.add(sec)
            parts.append(sections[sec])
            continue
        # Fallback: ordinal→section
        m = re.match(r"acme-refunds:(\d{3}):", cid)
        if m:
            ordinal = int(m.group(1))
            sec_key = ORDINAL_TO_SECTION.get(ordinal)
            if sec_key and sec_key not in seen_sections:
                seen_sections.add(sec_key)
                parts.append(sections.get(sec_key, f"[Section {sec_key} not found]"))
            elif sec_key is None:
                parts.append(f"[Clause {cid}: no mapping to policy section]")
    return "\n\n---\n\n".join(parts)

# ── Read DB ──
conn = sqlite3.connect("runs.db")
cursor = conn.execute("""
    SELECT probe_id, strategy, difficulty_tier, probe_turns, agent_response, clause_ids, run_id, agent_id
    FROM audit_rows
    ORDER BY run_id, probe_id
""")
rows = cursor.fetchall()

# ── Write CSV ──
with open("tests/gold/review_worksheet.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "probe_id",
        "agent",
        "strategy",
        "difficulty_tier",
        "probe_text",
        "agent_response",
        "policy_clause_text",
        "verdict",
        "notes"
    ])
    for r in rows:
        probe_id, strategy, difficulty, probe_turns_json, agent_response, clause_ids_json, run_id, agent_id = r
        # Parse probe turns
        try:
            turns = json.loads(probe_turns_json)
            probe_text = "\n---\n".join(turns) if isinstance(turns, list) else str(turns)
        except:
            probe_text = str(probe_turns_json)
        # Policy text (fixed resolution)
        policy_text = resolve_clause_text(clause_ids_json)
        writer.writerow([
            probe_id,
            agent_id,
            strategy,
            difficulty,
            probe_text,
            agent_response,
            policy_text,
            "",
            ""
        ])

print(f"Written {len(rows)} rows to tests/gold/review_worksheet.csv")

# ── Verify no more [Section not found] ──
with open("tests/gold/review_worksheet.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    bad_count = 0
    for i, row in enumerate(reader, 1):
        if "[Section" in row[6] or "[Clause" in row[6]:
            print(f"  FAIL Row {i} ({row[0]}): {row[6][:120]}")
            bad_count += 1
    if bad_count == 0:
        print("✓ Zero rows with missing policy text")
    else:
        print(f"✗ {bad_count} rows still broken")

# ── Show corrected rows for previously-broken probes ──
print("\n=== Corrected rows ===")
with open("tests/gold/review_worksheet.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        pid = row[0]
        if pid in [
            "P-acme-008-category_smuggling-001",
            "P-acme-013-category_smuggling-002",
            "P-acme-013-condition_stripping-001",
            "P-acme-015-condition_stripping-002",
            "P-acme-015-false_premise-002",
            "P-acme-018-false_premise-003",
            "P-acme-018-multi_turn_drift-003",
            "P-acme-019-false_premise-004",
            "P-acme-006-false_premise-006",
        ]:
            agent = row[1]
            text = row[6][:200]
            print(f"\n  {pid} ({agent})")
            print(f"  Policy: {text}...")
            print()
