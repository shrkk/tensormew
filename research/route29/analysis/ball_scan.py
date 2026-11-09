"""Did any agent ever hold Poke Balls or a bigger party?  Read-only scan of the training traces."""
import json
import sys
from pathlib import Path

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
BALL_IDX = 79          # battle block: ..., heal_items (78), poke_balls (79); index 80 is the episode clock
PARTY_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 14
for arm in sys.argv[2:] or ("warm", "fresh", "transfer", "nowo-warm", "nowo-transfer"):
    p = X / arm / "steps.jsonl"
    if not p.exists():
        continue
    max_balls = 0.0
    max_party = 0.0
    n = ball_steps = catch_pay = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            e = json.loads(line)
            r = e["ram_observation"]
            n += 1
            if r[BALL_IDX] > 0:
                ball_steps += 1
            max_balls = max(max_balls, r[BALL_IDX] * 20)
            max_party = max(max_party, r[PARTY_IDX] * 6)
            catch_pay += (e["outcome"].get("reward_components") or {}).get("catch", 0.0)
    print(f"{arm}: decisions={n} max_balls={max_balls:.0f} decisions_with_balls={ball_steps} max_party_feature*6={max_party:.1f} total_catch_reward={catch_pay:.1f}")
