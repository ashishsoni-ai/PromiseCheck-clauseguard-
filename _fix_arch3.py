"""Fix architecture.md - L3 annotation and Mermaid."""
with open("docs/architecture.md", "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: L3 table row - the gold set claim
old1 = "over-promise cell + gold set only"
new1 = "over-promise cell only (gold-set path exists but unexercised)"
if old1 in content:
    content = content.replace(old1, new1, 1)
    print("Fix 1 applied")
else:
    print("Fix 1 NOT FOUND")

# Fix 2: Mermaid L3 label
old2 = 'L3["L3 consistency<br/>k=3, temp 0.3<br/>can only lower the count"]'
new2 = 'L3["L3 consistency<br/>k=3, temp 0.3<br/>over-promise cell only<br/>can only lower the count"]'
if old2 in content:
    content = content.replace(old2, new2, 1)
    print("Fix 2 applied")
else:
    print("Fix 2 NOT FOUND")

with open("docs/architecture.md", "w", encoding="utf-8") as f:
    f.write(content)
print("Done")