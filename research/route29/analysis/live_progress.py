import json, sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, "C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main")
from src import route_graph as rg
X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
def rem(x, y): return rg.goal_remaining(SimpleNamespace(map_group=24, map_number=3, x=x, y=y))
for arm in sys.argv[1:]:
    maps = Counter(); best = (10**9, None); n = 0; recent_best = (10**9, None)
    with open(X/arm/"steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line: continue
            e = json.loads(line); o = e["outcome"]; m = tuple(o["map"]); maps[m] += 1; n += 1
            if m == (24, 3):
                r = rem(o["x"], o["y"])
                if r is not None and r < best[0]: best = (r, (o["x"], o["y"]))
    print(f"{arm:28s} decisions={n:6d} cherrygrove(26:3)={maps.get((26,3),0)} closest_remaining={best[0]:.0f} at {best[1]}  top maps={ {f'{k[0]}:{k[1]}':v for k,v in maps.most_common(3)} }")
