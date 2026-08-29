"""Fix: replace 'gold set' with 'LLM cross-check labels' in architecture.md"""
with open("docs/architecture.md", "r", encoding="utf-8") as f:
    content = f.read()

# The exact text as it appears in the file (confirmed via repr)
old = 'the gold\nset (`tests/gold/gold_labels.jsonl`, LLM-generated labels — κ is inter-LLM agreement, not judge accuracy)'
new = 'the LLM cross-check labels (`tests/gold/gold_labels.jsonl` — κ is inter-LLM agreement, not judge accuracy)'

idx = content.find(old)
if idx >= 0:
    content = content[:idx] + new + content[idx + len(old):]
    with open("docs/architecture.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("REPLACED: architecture.md footnote")
else:
    print("NOT FOUND - checking exact bytes...")
    # Debug
    idx2 = content.find('the gold')
    if idx2 >= 0:
        chunk = content[idx2:idx2+180]
        print(repr(chunk))