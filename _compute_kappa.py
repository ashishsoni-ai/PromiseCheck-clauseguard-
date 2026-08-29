"""Compute Cohen's kappa and per-class metrics for gold set."""
import json
from collections import Counter

with open("tests/gold/gold_labels.jsonl") as f:
    records = [json.loads(line) for line in f]

# Map verdict classes to a binary: "over_promise" vs "not_over_promise"
# For multi-class kappa, use all classes
all_classes = [
    "CORRECT_DENIAL", "CORRECT_GRANT", "OVER_PROMISE", 
    "UNDER_SERVE", "EVASIVE_ON_DENIAL", "EVASIVE_ON_GRANT"
]

def class_index(c):
    if c is None:
        return -1
    return all_classes.index(c)

# Build confusion matrix
n = len(all_classes)
confusion = [[0]*n for _ in range(n)]

for r in records:
    gold = r["gold_verdict_class"]
    judge = r["judge_verdict_class"]
    if judge is None:
        continue  # skip judge errors
    i = class_index(gold)
    j = class_index(judge)
    confusion[i][j] += 1

print("Confusion matrix (rows=gold, cols=judge):")
print(f"{'':>20}", end="")
for c in all_classes:
    print(f"{c:>20}", end="")
print()
for i, gold_class in enumerate(all_classes):
    print(f"{gold_class:>20}", end="")
    for j in range(n):
        print(f"{confusion[i][j]:>20}", end="")
    print()

# Cohen's kappa
# Observed agreement
total_valid = sum(sum(row) for row in confusion)
agreement = sum(confusion[i][i] for i in range(n))
p_o = agreement / total_valid

# Expected agreement
row_sums = [sum(row) for row in confusion]
col_sums = [sum(confusion[i][j] for i in range(n)) for j in range(n)]
p_e = sum(row_sums[i] * col_sums[i] for i in range(n)) / (total_valid ** 2)

kappa = (p_o - p_e) / (1 - p_e)
print(f"\n=== Cohen's Kappa (multi-class, {total_valid} valid pairs) ===")
print(f"Observed agreement: {p_o:.4f} ({agreement}/{total_valid})")
print(f"Expected agreement: {p_e:.4f}")
print(f"Kappa: {kappa:.4f}")

# Binary kappa: over-promise vs not
print("\n=== Binary (over-promise vs rest) ===")
tp = sum(1 for r in records if r["gold_verdict_class"] == "OVER_PROMISE" and r["judge_verdict_class"] == "OVER_PROMISE")
fp = sum(1 for r in records if r["gold_verdict_class"] != "OVER_PROMISE" and r["judge_verdict_class"] == "OVER_PROMISE")
fn = sum(1 for r in records if r["gold_verdict_class"] == "OVER_PROMISE" and r["judge_verdict_class"] != "OVER_PROMISE" and r["judge_verdict_class"] is not None)
tn = sum(1 for r in records if r["gold_verdict_class"] != "OVER_PROMISE" and r["judge_verdict_class"] != "OVER_PROMISE" and r["judge_verdict_class"] is not None)

# Exclude judge errors from binary
judge_errors = sum(1 for r in records if r["judge_verdict_class"] is None)
print(f"Judge errors excluded: {judge_errors}")

print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f"Over-promise precision: {precision:.4f} ({tp}/{tp+fp})")
print(f"Over-promise recall: {recall:.4f} ({tp}/{tp+fn})")
print(f"Over-promise F1: {f1:.4f}")
print(f"False-alarm rate: {false_alarm_rate:.4f} ({fp}/{fp+tn})")

# Binary kappa
n_bin = tp + fp + fn + tn
p_o_bin = (tp + tn) / n_bin
row_sum_op = tp + fn
row_sum_not = tn + fp
col_sum_op = tp + fp
col_sum_not = tn + fn
p_e_bin = (row_sum_op * col_sum_op + row_sum_not * col_sum_not) / (n_bin ** 2)
kappa_bin = (p_o_bin - p_e_bin) / (1 - p_e_bin)
print(f"Binary kappa: {kappa_bin:.4f}")

# Per-class metrics
print("\n=== Per-class precision/recall ===")
for i, cls in enumerate(all_classes):
    tp_c = confusion[i][i]
    fp_c = sum(confusion[j][i] for j in range(n)) - confusion[i][i]
    fn_c = sum(confusion[i][j] for j in range(n)) - confusion[i][i]
    prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
    rec = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
    print(f"  {cls:>20}: precision={prec:.4f} ({tp_c}/{tp_c+fp_c})  recall={rec:.4f} ({tp_c}/{tp_c+fn_c})")

# Disagreement discussion
print("\n=== Disagreement analysis ===")
disagreements = [r for r in records if r["gold_verdict_class"] != r["judge_verdict_class"] and r["judge_verdict_class"] is not None]
print(f"Total disagreements (excluding judge errors): {len(disagreements)}")
for r in disagreements:
    print(f"\n{r['probe_id']} [{r['agent_id']}]:")
    print(f"  Gold: {r['gold_verdict_class']}  |  Judge: {r['judge_verdict_class']}")
    print(f"  Expected: {r['expected_policy_stance']}  |  Strategy: {r['strategy']} T{r['difficulty_tier']}")