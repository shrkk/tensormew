"""Walk the ROM route map's shortest path across Route 29 tile by tile (OFFLINE DIAGNOSTIC ONLY, never an agent input):
corridor width at each step and how many agent decisions (all arms, all states) ever stood on it.  Read-only."""
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main")
from src import route_graph as rg  # noqa: E402

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")


def rem(x, y):
    return rg.goal_remaining(SimpleNamespace(map_group=24, map_number=3, x=x, y=y))


visits = Counter()
for arm in sorted(p.name for p in X.iterdir() if (p / "steps.jsonl").exists()):
    with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            o = json.loads(line)["outcome"]
            if tuple(o["map"]) == (24, 3):
                visits[(o["x"], o["y"])] += 1

# entrance = on-graph tile with the largest remaining distance on this map
cands = [(rem(x, y), x, y) for x in range(0, 64) for y in range(0, 32) if rem(x, y) is not None]
cands.sort(reverse=True)
r0, x, y = cands[0]
path = [(x, y)]
seen = {(x, y)}
while True:
    cur = rem(x, y)
    nxt = None
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        r = rem(nx, ny)
        if r is not None and r < cur and (nx, ny) not in seen:
            if nxt is None or r < nxt[0]:
                nxt = (r, nx, ny)
    if nxt is None:
        break
    _, x, y = nxt
    path.append((x, y))
    seen.add((x, y))
print(f"route on Route 29: {len(path)} tiles, starts {path[0]} (remaining {r0:.0f}), ends {path[-1]}")
print("idx  tile      remaining  open_neighbours  visits(all arms)")
last_visited = 0
for i, (px, py) in enumerate(path):
    width = sum(1 for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)) if rem(px + dx, py + dy) is not None)
    v = visits[(px, py)]
    if v:
        last_visited = i
    flag = "  <-- 1-wide" if width <= 2 else ""
    if i % 3 == 0 or width <= 2 or i >= len(path) - 12:
        print(f"{i:3d}  {str((px, py)):9s} {rem(px, py):7.0f}   {width}               {v:8d}{flag}")
print(f"\nfurthest path index ever visited: {last_visited} of {len(path) - 1}  -> tile {path[last_visited]}")
print("path tiles never visited (index ranges):", [i for i, p in enumerate(path) if not visits[p]][:5], "...")
