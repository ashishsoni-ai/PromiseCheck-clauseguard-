"""Replace 'gold set'/'gold labels'/'hand labels'/'hand-labelled' with 'LLM cross-check labels'
in user-facing docs. Does NOT touch DESIGN.md (historical spec), internal code, or auto-generated files."""

import os, re

REPLACEMENTS = {
    # results.md
    "gold set now exists with 60 hand-labelled pairs": "LLM cross-check labels now exist with 60 labelled pairs",
    "gold set": "LLM cross-check labels",
    "hand-labelled": "LLM cross-check",
    "hand labels": "LLM cross-check labels",
    "gold labels": "LLM cross-check labels",
}

# Specifically targeted replacements for results.md
RESULTS_FIXES = [
    # Line 186
    ("the gold set exists\nand provides an independent accuracy check",
     "the LLM cross-check labels exist\nand provide an independent accuracy check"),
    # Line 204
    ("A gold set now exists with 60 hand-labelled pairs.",
     "LLM cross-check labels now exist for 60 pairs."),
    # Line 272
    ("Cohen's κ vs hand labels", "Cohen's κ vs LLM cross-check labels (not ground truth)"),
    ("the gold set is empty", "the LLM cross-check labels are empty"),
    # Line 409 scoreboard
    ("Judge κ vs hand labels", "Judge κ vs LLM cross-check (not ground truth)"),
    ("60 gold labels", "60 LLM cross-check labels"),
]

LIMITATIONS_FIXES = [
    # Lines 235-237
    ("to the entire gold set", "to the entire LLM cross-check set"),
    ("The gold set is the only control", "The LLM cross-check set is the only control"),
    # Line 258
    ("the gold set measures", "the LLM cross-check set measures"),
    # Line 291
    ("a perturbation panel in the gold set", "a perturbation panel in the LLM cross-check set"),
]

DEMO_SCRIPT_FIXES = [
    ("The gold set is empty", "The LLM cross-check set is empty"),
]

RESULTS_HTML_FIXES = [
    ("No gold set, so no κ.", "No LLM cross-check set, so no κ."),
    ("gold set is empty", "LLM cross-check set is empty"),
    ("hand labels", "LLM cross-check labels"),
    ("needs gold set", "needs LLM cross-check set"),
]

README_FIXES = [
    ("to the gold set", "to the LLM cross-check set"),
    ("the gold set, which does not exist yet", "the LLM cross-check set, which does not exist yet"),
    ("The gold set, and therefore every judge-accuracy number.", "The LLM cross-check set, and therefore every judge-accuracy number."),
    ("hand labels", "LLM cross-check labels"),
    ("gold labels (gold set empty)", "LLM cross-check labels (set empty)"),
]

def apply_fixes(filepath, fixes):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    changed = 0
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new, 1)
            changed += 1
            print(f"  {filepath}: replaced '{old[:40]}...'")
        else:
            # Try case-insensitive
            import re as re2
            pattern = re2.escape(old)
            if re2.search(pattern, content, re2.IGNORECASE):
                content = re2.sub(pattern, new, content, count=1, flags=re2.IGNORECASE)
                changed += 1
                print(f"  {filepath}: case-insensitive replacement for '{old[:40]}...'")
            else:
                print(f"  {filepath}: NOT FOUND '{old[:40]}...'")
    
    if changed > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  => {changed} fix(es) applied to {filepath}")
    else:
        print(f"  => No changes needed for {filepath}")
    return changed

docs_dir = "docs"

print("=== results.md ===")
apply_fixes(os.path.join(docs_dir, "results.md"), RESULTS_FIXES)

print("\n=== limitations.md ===")
apply_fixes(os.path.join(docs_dir, "limitations.md"), LIMITATIONS_FIXES)

print("\n=== demo-script.md ===")
apply_fixes(os.path.join(docs_dir, "demo-script.md"), DEMO_SCRIPT_FIXES)

print("\n=== results.html ===")
apply_fixes(os.path.join(docs_dir, "results.html"), RESULTS_HTML_FIXES)

print("\n=== README.md ===")
apply_fixes("README.md", README_FIXES)

print("\nDone.")