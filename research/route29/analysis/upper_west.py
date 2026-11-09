"""On the upper road (y<=13, x<=34): westmost tiles reached, neighbour tile ids and actions there.  Read-only."""
import json
from collections import Counter
from pathlib import Path

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
ACT = ["up", "down", "left", "right", "A", "B"]
pos = Counter()
nb = {}
acts = Counter()
for arm in sorted(p.name for p in X.iterdir() if (p / "steps.jsonl").exists()):
    with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            e = json.loads(line)
            o = e["outcome"]
            if tuple(o["map"]) != (24, 3) or o["y"] > 13 or o["x"] > 34:
                continue
            r = e["ram_observation"]
            if r[7] > 0.5:
                continue
            key = (o["x"], o["y"])
            pos[key] += 1
            nb[key] = tuple(round(v * 255) for v in r[0:4])   # up, down, left, right collision/tile ids
            if o["x"] <= 16:
                acts[ACT[e["action"]]] += 1
west = sorted(pos, key=lambda k: (k[0], k[1]))[:12]
print("westmost upper-road tiles (x,y): visits, neighbour ids (up,down,left,right)")
for k in west:
    print("  ", k, pos[k], nb[k])
print("most visited upper-road tiles:", [(k, v) for k, v in pos.most_common(8)])
print("actions taken at x<=16 on the upper road:", dict(acts.most_common()))
xs = Counter()
for (x, y), v in pos.items():
    xs[x // 4 * 4] += v
print("upper-road decisions by x-bin:", dict(sorted(xs.items())))
