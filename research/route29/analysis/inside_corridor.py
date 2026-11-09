"""What do agents do INSIDE the 1-tile corridor (x=31, y 10..13)?  Per-decision actions, battle share, and where they go next.  Read-only."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main")
from src.deep_policy import BATTLE_FLAG_INDEX as BI  # noqa: E402

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
ACT = ["up", "down", "left", "right", "A", "B"]
CORR = {(31, y) for y in (10, 11, 12, 13)}
acts, nxt, battles, n = Counter(), Counter(), 0, 0
per_arm = {}
for arm in sys.argv[1:]:
    prev = {}
    a_acts, a_n = Counter(), 0
    with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            e = json.loads(line)
            o = e["outcome"]
            env = e["env"]
            cur = (tuple(o["map"]), o["x"], o["y"])
            p = prev.get(env)
            if p and p[0] == (24, 3) and (p[1], p[2]) in CORR and not e["done"]:
                a_n += 1
                n += 1
                acts[ACT[e["action"]]] += 1
                a_acts[ACT[e["action"]]] += 1
                if float(e["ram_observation"][BI]) > 0.5:
                    battles += 1
                if cur[0] == (24, 3):
                    dx, dy = cur[1] - p[1], cur[2] - p[2]
                    nxt[{(0, -1): "moved up", (0, 1): "moved down", (-1, 0): "moved left", (1, 0): "moved right", (0, 0): "stayed"}.get((dx, dy), "other")] += 1
            prev[env] = cur
    per_arm[arm] = (a_n, dict(a_acts.most_common(4)))
print(f"decisions starting inside the corridor tiles (31,10..13): {n}; of those in battle: {battles}")
print("actions chosen:", {k: f"{v / max(1, n):.0%}" for k, v in acts.most_common()})
print("what happened next:", {k: f"{v / max(1, sum(nxt.values())):.0%}" for k, v in nxt.most_common()})
for arm, (a_n, a) in per_arm.items():
    print(f"  {arm:20s} n={a_n:4d} top actions {a}")
