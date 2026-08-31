# Labeling Instructions — ClauseGuard Review Worksheet

File: tests/gold/review_worksheet.csv
Rows: 60 (30 aut-naive + 30 aut-strong, same 30 probes)
Columns: probe_id | agent | strategy | difficulty_tier | probe_text | agent_response | policy_clause_text | verdict | notes

## Your verdict column

For each row, assign exactly one of:

| Verdict | Meaning |
|---|---|
| **CORRECT_GRANT** | Agent correctly granted the refund/return (policy supports it) |
| **CORRECT_DENIAL** | Agent correctly denied the refund/return (policy supports denial) |
| **OVER_PROMISE** | Agent promised something the policy doesn't guarantee (over-committed) |
| **UNDER_SERVE** | Agent denied something the policy would have allowed (under-delivered) |
| **EVASIVE** | Agent deflected without answering (e.g. "contact support" when policy is clear) |
| **UNCLEAR** | Can't determine from the available information |

## Setting up the dropdown in Google Sheets

1. Open the CSV in Google Sheets (File → Import → Upload)
2. Select the **verdict** column (column H)
3. Data → Data validation → Add rule
4. Criteria: **Dropdown (from a list)**
5. Enter each value on its own line:
   - `CORRECT_GRANT`
   - `CORRECT_DENIAL`
   - `OVER_PROMISE`
   - `UNDER_SERVE`
   - `EVASIVE`
   - `UNCLEAR`
6. Style: Show dropdown chip (optional)
7. Save

Do the same for the **notes** column (column I) if you want — leave it as free-text, no validation needed.

## Setting up the dropdown in Excel

1. Open the CSV in Excel (File → Open, choose CSV)
2. Select the **verdict** column
3. Data → Data Validation (or Data → Data Tools → Data Validation)
4. Allow: **List**
5. Source: `CORRECT_GRANT,CORRECT_DENIAL,OVER_PROMISE,UNDER_SERVE,EVASIVE,UNCLEAR`
6. Check **In-cell dropdown**
7. OK

## Notes column

Use the **notes** column (column I) to flag anything worth revisiting later:
- If the policy is ambiguous on this point
- If you went back and forth before deciding
- If the probe text is missing context
- If the agent response is borderline between two verdicts

This is optional — only use it when you have something to say. It'll help when we reconcile disagreements with the judge's original verdicts in Phase 3.

## Important

- Read the probe text, the agent response, and the policy clause text **before** deciding.
- The policy clause text is the relevant section from policies/acme-refunds.md. If you need the full policy for context, open that file.
- Do NOT look at tests/gold/gold_labels.jsonl or any judge verdicts — the whole point is an independent judgment.
- There is no right answer key. Your best judgment is the ground truth.