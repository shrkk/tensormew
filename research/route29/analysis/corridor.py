import json, sys
from collections import Counter
from pathlib import Path
X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
for arm in sys.argv[1:]:
    c = Counter(); n = 0; past = 0
    with open(X/arm/"steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line: continue
            o = json.loads(line)["outcome"]; n += 1
            if tuple(o["map"]) == (24, 3):
                if o["x"] == 31 and 10 <= o["y"] <= 14: c[o["y"]] += 1
                if (o["x"] in range(32, 37) and o["y"] == 10) or (o["x"] == 36 and 6 <= o["y"] <= 9): past += 1
    print(f"{arm:30s} decisions={n:6d} corridor(x=31) visits by y {dict(sorted(c.items()))}  beyond-corridor(y=10 east / x=36 north)={past}")
