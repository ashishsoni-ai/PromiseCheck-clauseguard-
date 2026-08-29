"""Check architecture.md for stale claims."""
with open("docs/architecture.md", "r", encoding="utf-8") as f:
    content = f.read()

checks = [
    ("empty \u2014 so no", "stale gold-set claim"),
    ("gold set", "gold set reference"),
    ("human ground truth", "overclaim"),
    ("judge accuracy", "overclaim"),
    ("\u03ba =", "kappa claim"),
    ("0.847", "kappa number"),
    ("Cohen", "Cohen reference"),
    ("NOT BUILT", "unbuilt annotation"),
    ("not built", "unbuilt annotation"),
    ("solid", "built annotation"),
    ("built", "built annotation"),
]
for term, label in checks:
    count = content.count(term)
    if count > 0:
        print(f'{label}: "{term}" appears {count} time(s)')
    else:
        print(f'{label}: "{term}" NOT FOUND')