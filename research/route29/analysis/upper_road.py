"""Has any agent EVER stood on the upper road (Route 29 north branch, x<=34, y<=13)?  Read-only."""
import json
import sys
from collections import Counter
from pathlib import Path

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
arms = sys.argv[1:] or sorted(p.name for p in X.iterdir() if (p / "steps.jsonl").exists())
tot_all = Counter()
for arm in arms:
    p = X / arm / "steps.jsonl"
    if not p.exists():
        continue
    upper = pocket = r29 = 0
    min_y_west = 99
    with open(p, encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            o = json.loads(line)["outcome"]
            if tuple(o["map"]) != (24, 3):
                continue
            r29 += 1
            x, y = o["x"], o["y"]
            if x <= 34:
                if y <= 13:
                    upper += 1
                if y in (14, 15) and x <= 24:
                    pocket += 1
                min_y_west = min(min_y_west, y)
    tot_all["upper"] += upper
    tot_all["pocket"] += pocket
    print(f"{arm:28s} R29={r29:6d} upper_road(x<=34,y<=13)={upper:6d} lower_corridor(x<=24,y14-15)={pocket:6d} min_y_at_x<=34={min_y_west}")
print("TOTAL", dict(tot_all))
