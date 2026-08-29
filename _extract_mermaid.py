"""Extract Mermaid code from architecture.md."""
with open("docs/architecture.md", "r", encoding="utf-8") as f:
    content = f.read()

start = content.index("```mermaid")
end = content.index("```", start + 10)
mermaid_code = content[start+9:end].strip()
print(mermaid_code)