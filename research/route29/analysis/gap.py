"""The northern frontier (x 31..37, y 3..9 on Route 29): visits, neighbour tile ids and actions tried.  Read-only."""
import json
from collections import Counter, defaultdict
from pathlib import Path

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
ACT = ["up", "down", "left", "right", "A", "B"]
vis = Counter()
nb = {}
acts = defaultdict(Counter)
for arm in sorted(p.name for p in X.iterdir() if (p / "steps.jsonl").exists()):
    with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            e = json.loads(line)
            o = e["outcome"]
            if tuple(o["map"]) != (24, 3) or not (30 <= o["x"] <= 37 and 3 <= o["y"] <= 9):
                continue
            r = e["ram_observation"]
            if r[7] > 0.5:
                continue
            k = (o["x"], o["y"])
            vis[k] += 1
            nb[k] = tuple(round(v * 255) for v in r[0:4])
            acts[k][ACT[e["action"]]] += 1
print("tile (x,y): visits | neighbour ids (up,down,left,right) | top actions   [0=walkable, 7=wall, 18=cut tree, 21=headbutt tree, 24=tall grass]")
for y in range(3, 10):
    for x in range(30, 38):
        k = (x, y)
        if k in vis:
            print(f"  {k}: {vis[k]:5d} | {nb[k]} | {dict(acts[k].most_common(3))}")
