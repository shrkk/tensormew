"""Reward components per decision in the lower dead-end pocket (24:3, x<=24, y in 14..15) vs the rest of Route 29.  Read-only."""
import json
import sys
from collections import defaultdict
from pathlib import Path

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
for arm in sys.argv[1:]:
    where = {"pocket(x<=24,y14-15)": [0, defaultdict(float)], "other Route 29": [0, defaultdict(float)]}
    with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            e = json.loads(line)
            o = e["outcome"]
            if tuple(o["map"]) != (24, 3) or e["done"]:
                continue
            key = "pocket(x<=24,y14-15)" if (o["x"] <= 24 and o["y"] in (14, 15)) else "other Route 29"
            where[key][0] += 1
            for k, v in (o.get("reward_components") or {}).items():
                where[key][1][k] += v
    print(f"\n{arm}")
    for key, (n, comps) in where.items():
        if not n:
            continue
        tot = sum(comps.values())
        pos = {k: round(v / n, 4) for k, v in sorted(comps.items(), key=lambda kv: -kv[1]) if v / n > 0.0005}
        neg = {k: round(v / n, 4) for k, v in sorted(comps.items(), key=lambda kv: kv[1]) if v / n < -0.0005}
        print(f"  {key:22s} n={n:6d}  net/step={tot / n:+.4f}   +{pos}   -{neg}")
