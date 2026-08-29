"""Fix L3 annotation in architecture.md."""
with open("docs/architecture.md", "rb") as f:
    content = f.read()

# Fix 1: L3 gold-set annotation
old1 = b"over-promise cell + gold set"
new1 = b"over-promise cell only (gold-set path exists but unexercised)"
content = content.replace(old1, new1, 1)

# Fix 2: The Mermaid L3 annotation
old2 = b"L3[\"L3 consistency<br/>k=3, temp 0.3<br/>can only lower the count\"]"
new2 = b"L3[\"L3 consistency<br/>k=3, temp 0.3<br/>over-promise cell only<br/>can only lower the count\"]"
content = content.replace(old2, new2, 1)

with open("docs/architecture.md", "wb") as f:
    f.write(content)
print("Done")