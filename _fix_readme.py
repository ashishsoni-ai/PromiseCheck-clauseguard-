"""Fix remaining 'gold set'/'hand labels' in README.md"""
with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

fixes = [
    ("the only control for it is the gold set, which does not exist\nyet.",
     "the only control for it is the LLM cross-check set, which does not exist yet."),
    ("200 hand labels", "200 LLM cross-check labels"),
]

changed = 0
for old, new in fixes:
    if old in content:
        content = content.replace(old, new, 1)
        changed += 1
        print(f"Fixed: {old[:40]}...")
    else:
        print(f"NOT FOUND: {old[:40]}...")

if changed > 0:
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Applied {changed} fix(es)")