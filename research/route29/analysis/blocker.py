"""What is at the west end of Route 29?  Neighbor tile ids, dialogue flag and what A does, at x=14.  Read-only."""
import json
from collections import Counter
from pathlib import Path

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
ACT = ["up", "down", "left", "right", "A", "B"]
tiles = Counter()
dialog = Counter()
act_dialog = Counter()
n = 0
with open(X / "warm" / "steps.jsonl", encoding="utf-8") as f:
    for line in f:
        if '"event": "step"' not in line:
            continue
        e = json.loads(line)
        o = e["outcome"]
        if tuple(o["map"]) != (24, 3) or o["x"] != 14:
            continue
        r = e["ram_observation"]
        if r[7] > 0.5:   # in battle
            continue
        n += 1
        tiles[(o["y"], tuple(round(v * 255) for v in r[0:4]))] += 1   # (y, (up, down, left, right) tile ids)
        dialog[bool(r[8] > 0.5)] += 1
        act_dialog[(ACT[e["action"]], bool(r[8] > 0.5))] += 1
print("non-battle decisions at x=14:", n)
print("(y, neighbor tile ids up/down/left/right) -> count:")
for k, v in tiles.most_common(6):
    print("  ", k, v)
print("dialogue open:", dict(dialog))
print("(action, dialogue_open) counts:", dict(act_dialog.most_common(8)))
