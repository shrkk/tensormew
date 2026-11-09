"""How far west along Route 29 (map 24:3) does each arm get?  Min x reached per training window.  Read-only."""
import json
import sys
from pathlib import Path

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
for arm in sys.argv[1:]:
    p = X / arm / "steps.jsonl"
    wins = {}
    ends = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            e = json.loads(line)
            o = e["outcome"]
            k = e["update"] // 100
            w = wins.setdefault(k, {"minx": 999, "maxx": -1, "other_maps": set(), "r29": 0, "n": 0})
            w["n"] += 1
            m = tuple(o["map"])
            if m == (24, 3):
                w["r29"] += 1
                w["minx"] = min(w["minx"], o["x"])
                w["maxx"] = max(w["maxx"], o["x"])
            else:
                w["other_maps"].add(f"{m[0]}:{m[1]}")
    print(f"{arm}: " + " | ".join(f"u{k*100}-{k*100+99}: R29={w['r29']/w['n']:.0%} x[{w['minx'] if w['minx']<999 else '-'}..{w['maxx'] if w['maxx']>=0 else '-'}]"
                                  f" maps={sorted(w['other_maps'])}" for k, w in sorted(wins.items())))
