"""Route 29 (24:3) occupancy by x-bin: share of decisions, share in battle, and where episodes end.  Read-only."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main")
from src.deep_policy import BATTLE_FLAG_INDEX as BI  # noqa: E402

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
for arm in sys.argv[1:]:
    occ, bat = Counter(), Counter()
    last = {}
    ends = Counter()
    ys = {}
    with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            e = json.loads(line)
            o = e["outcome"]
            if tuple(o["map"]) == (24, 3):
                b = (o["x"] // 5) * 5
                occ[b] += 1
                if float(e["ram_observation"][BI]) > 0.5:
                    bat[b] += 1
            last[e["env"]] = (tuple(o["map"]), o["x"], o["y"])
            if e["done"]:
                ends[(o.get("end_reason"), last[e["env"]][0], (last[e["env"]][1] // 5) * 5)] += 1
    tot = sum(occ.values())
    print(f"\n{arm}: decisions on Route 29 = {tot}")
    print("  x-bin  share  in-battle")
    for b in sorted(occ):
        print(f"  {b:3d}-{b+4:<3d} {occ[b]/tot:5.1%}  {bat[b]/occ[b]:5.0%}")
    print("  episode ends (reason, map, x-bin):", dict(ends))
