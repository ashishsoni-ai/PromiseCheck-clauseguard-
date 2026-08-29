"""Render Mermaid to PNG using Playwright."""
import sys, os, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

with open("docs/architecture.md", "r", encoding="utf-8") as f:
    content = f.read()

start = content.index("```mermaid")
end = content.index("```", start + 10)
mermaid_code = content[start+9:end].strip()

html = f"""<!DOCTYPE html>
<html><head><script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>body{{margin:0;padding:20px;background:white}}svg{{max-width:100%;height:auto}}</style></head>
<body><div class="mermaid">{mermaid_code}</div>
<script>mermaid.initialize({{theme:'default',startOnLoad:true}});</script></body></html>"""

with open("_temp_mermaid.html", "w", encoding="utf-8") as f:
    f.write(html)

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    page.goto(f"file://{os.path.abspath('_temp_mermaid.html')}")
    page.wait_for_selector(".mermaid svg", timeout=15000)
    page.wait_for_timeout(2000)
    svg = page.locator(".mermaid svg")
    svg.screenshot(path="docs/architecture.png")
    print(f"Wrote screenshot to docs/architecture.png")
    browser.close()

os.remove("_temp_mermaid.html")