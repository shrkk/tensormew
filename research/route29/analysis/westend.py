"""What happens at the west end of Route 29 (x <= 16)?  Positions, actions, and what the tiles around look like.  Read-only."""
import json
import sys
from collections import Counter
from pathlib import Path

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
ACT = ["up", "down", "left", "right", "A", "B"]
arm = sys.argv[1]
pos = Counter()
acts = Counter()
episodes = {}
prev = {}
moved_blocked = Counter()
with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
    for line in f:
        if '"event": "step"' not in line:
            continue
        e = json.loads(line)
        o = e["outcome"]
        env = e["env"]
        cur = (tuple(o["map"]), o["x"], o["y"])
        p = prev.get(env)
        if p and p[0] == (24, 3) and p[1] <= 16:
            pos[(p[1], p[2])] += 1
            acts[ACT[e["action"]]] += 1
            moved_blocked["same_tile" if cur == p else "moved"] += 1
            episodes.setdefault((env, e["update"] // 4), 0)
        prev[env] = cur
print(f"{arm}: decisions starting at Route 29 x<=16: {sum(pos.values())}")
print("  actions taken there:", {k: f"{v/sum(acts.values()):.0%}" for k, v in acts.most_common()})
print("  moved vs stayed:", dict(moved_blocked))
print("  westmost tiles (x,y):visits ->", sorted(pos.items())[:14])
print("  most-visited tiles:", pos.most_common(8))
