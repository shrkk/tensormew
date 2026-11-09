"""Furthest point along the emulator-verified route (bfs_route.py output) reached by each arm's TRAINING trace.  Read-only diagnostic.
Index 0 = (52,9) on Route 29; (31,14) is index 30; (36,6) is index 43; the exit to Cherrygrove is after index 83."""
import json
import sys
from pathlib import Path

ROUTE = [(52, 9), (52, 10), (52, 11), (52, 12), (52, 13), (51, 13), (50, 13), (49, 13), (48, 13), (47, 13), (47, 14), (47, 15), (46, 15),
         (45, 15), (44, 15), (43, 15), (43, 16), (42, 16), (41, 16), (40, 16), (39, 16), (38, 16), (37, 16), (36, 16), (35, 16), (34, 16),
         (33, 16), (33, 15), (33, 14), (32, 14), (31, 14), (31, 13), (31, 12), (31, 11), (31, 10), (32, 10), (33, 10), (34, 10), (35, 10),
         (36, 10), (36, 9), (36, 8), (36, 7), (36, 6), (35, 6), (34, 6), (33, 6), (32, 6), (31, 6), (30, 6), (30, 5), (29, 5), (28, 5),
         (27, 5), (26, 5), (25, 5), (24, 5), (23, 5), (22, 5), (22, 4), (21, 4), (20, 4), (19, 4), (19, 5), (18, 5), (17, 5), (17, 6),
         (16, 6), (15, 6), (14, 6), (13, 6), (12, 6), (11, 6), (10, 6), (9, 6), (8, 6), (7, 6), (6, 6), (5, 6), (4, 6), (3, 6), (2, 6),
         (1, 6), (0, 6)]
IDX = {t: i for i, t in enumerate(ROUTE)}
X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
print(f"{'arm':22s} {'max idx':>7s} {'at tile':>10s} {'decisions idx>=31 (corridor+)':>30s} {'idx>=44 (beyond (36,6))':>25s} {'cherrygrove':>11s}")
for arm in sys.argv[1:]:
    mx, at, c31, c44, cg = -1, None, 0, 0, 0
    with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            o = json.loads(line)["outcome"]
            m = tuple(o["map"])
            if m[0] == 26:
                cg += 1
            if m != (24, 3):
                continue
            i = IDX.get((o["x"], o["y"]))
            if i is None:
                continue
            if i > mx:
                mx, at = i, (o["x"], o["y"])
            c31 += i >= 31
            c44 += i >= 44
    print(f"{arm:22s} {mx:7d} {str(at):>10s} {c31:30d} {c44:25d} {cg:11d}")
