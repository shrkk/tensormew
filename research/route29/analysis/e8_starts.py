"""Where do E8's frontier episodes START, and what happens in them?  Read-only (partial trace is fine)."""
import json
import sys
from collections import Counter
from pathlib import Path

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
arm = sys.argv[1]
starts = Counter()
dec_by_start = Counter()
last_state = {}
reward_by_start = Counter()
n = 0
with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
    for line in f:
        if '"event": "step"' not in line:
            continue
        e = json.loads(line)
        o = e["outcome"]
        env = e["env"]
        st = o.get("state", "?")
        n += 1
        dec_by_start[st] += 1
        reward_by_start[st] += e["reward"]
        if last_state.get(env) != st:
            starts[st] += 1
            last_state[env] = st
print(f"{arm}: {n} decisions")
print("episode starts by source (count):")
for k, v in starts.most_common(14):
    d = dec_by_start[k]
    print(f"  {k:30s} starts={v:4d} decisions={d:6d} mean reward/step={reward_by_start[k] / max(1, d):+.4f}")
fr = [k for k in starts if str(k).startswith("frontier:")]
print("distinct frontier start cells:", len(fr))
pocket = [k for k in fr if k.startswith("frontier:24_3_") and int(k.split("_")[2]) <= 24 and int(k.split("_")[3]) in (14, 15)]
print("frontier starts INSIDE the lower dead-end pocket (x<=24,y14-15):", sum(starts[k] for k in pocket), "of", sum(starts[k] for k in fr))
