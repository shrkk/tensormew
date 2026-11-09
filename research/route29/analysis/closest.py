"""Did any decision, in any arm's training trace or evaluation, enter Cherrygrove?  And what is the closest any agent got to the goal
(measured with the ROM route map, OFFLINE DIAGNOSTIC ONLY)?  Read-only."""
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main")
from src import route_graph as rg  # noqa: E402

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
maps_seen = Counter()
best = (10 ** 9, None, None)
visits = Counter()
n = 0
for arm in sorted(p.name for p in X.iterdir() if (p / "steps.jsonl").exists()):
    arm_best = (10 ** 9, None)
    with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            o = json.loads(line)["outcome"]
            m = tuple(o["map"])
            maps_seen[m] += 1
            n += 1
            if m != (24, 3):
                continue
            visits[(o["x"], o["y"])] += 1
    print(f"scanned {arm}")
print("\nTOTAL decisions scanned:", n)
print("maps entered (group:number -> decisions):", {f"{k[0]}:{k[1]}": v for k, v in sorted(maps_seen.items())})
print("Cherrygrove is map 26:3 -> decisions there:", maps_seen.get((26, 3), 0))
rows = []
for (x, y), v in visits.items():
    r = rg.goal_remaining(SimpleNamespace(map_group=24, map_number=3, x=x, y=y))
    if r is not None:
        rows.append((r, x, y, v))
rows.sort()
print("\nClosest tiles to the goal ever reached on Route 29 (remaining steps by ROM route map, x, y, visits):")
for r in rows[:10]:
    print("  ", r)
