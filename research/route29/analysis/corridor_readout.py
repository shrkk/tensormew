"""Pre-registered readout for Round C (corridor reward) vs the same-seed plain runs.  Read-only.
Per arm: furthest route idx, decisions beyond (36,6) [idx>=44], dead-end pocket share of Route 29 decisions, corridor reward totals."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main")
exec(open("C:/Users/shrey/AppData/Local/Temp/claude/4ed80d104b8065da/07ed6e13-8a71-4143-bf30-3cec043045d6/scratchpad/route_progress_bfs.py").read().split("X = Path")[0])  # ROUTE, IDX
X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
print(f"{'arm':20s} {'maxidx':>6s} {'beyond(36,6)':>12s} {'idx>=31':>8s} {'pocket share':>12s} {'cg':>3s} {'corr +':>7s} {'corr -':>7s} {'net':>6s} {'episodes':>8s}")
for arm in sys.argv[1:]:
    p = X / arm / "steps.jsonl"
    if not p.exists():
        continue
    mx, b44, c31, r29, pocket, cg, pos, neg, eps = -1, 0, 0, 0, 0, 0, 0.0, 0.0, 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            e = json.loads(line)
            o = e["outcome"]
            if e["done"]:
                eps += 1
            m = tuple(o["map"])
            if m[0] == 26:
                cg += 1
            c = (o.get("reward_components") or {}).get("corridor", 0.0)
            if c > 0:
                pos += c
            elif c < 0:
                neg += c
            if m != (24, 3):
                continue
            r29 += 1
            x, y = o["x"], o["y"]
            if x <= 24 and y in (14, 15):
                pocket += 1
            i = IDX.get((x, y))
            if i is not None:
                mx = max(mx, i)
                c31 += i >= 31
                b44 += i >= 44
    print(f"{arm:20s} {mx:6d} {b44:12d} {c31:8d} {pocket / max(1, r29):12.1%} {cg:3d} {pos:7.1f} {neg:7.1f} {pos + neg:6.1f} {eps:8d}")
