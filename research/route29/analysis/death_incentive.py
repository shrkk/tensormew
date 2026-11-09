"""Is whiteout (-15, episode terminates) cheaper than staying alive for these policies?  Read-only."""
import json
from collections import defaultdict
from pathlib import Path

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
import sys
for arm in (sys.argv[1:] or ("warm", "transfer", "fresh")):
    eps = defaultdict(lambda: {"r": 0.0, "n": 0, "start_u": None})
    done_rows = []
    with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            e = json.loads(line)
            if e["update"] < 284:
                continue
            env = e["env"]
            d = eps[env]
            d["r"] += e["reward"]
            d["n"] += 1
            if e["done"]:
                reason = e["outcome"].get("end_reason")
                done_rows.append((reason, d["n"], d["r"]))
                eps[env] = {"r": 0.0, "n": 0, "start_u": None}
    print(f"\n{arm}: finished episodes in updates 284-383 (reason, decisions, total reward incl. terminal):")
    for reason, n, r in done_rows:
        print(f"   {reason:10s} len={n:5d}  return={r:8.1f}  per-step={r / n:+.4f}")
    wo = [(n, r) for reas, n, r in done_rows if reas == "whiteout"]
    tl = [(n, r) for reas, n, r in done_rows if reas == "time_limit"]
    if wo:
        print(f"   mean whiteout episode: len {sum(n for n, _ in wo) / len(wo):.0f}, return {sum(r for _, r in wo) / len(wo):.1f}")
    if tl:
        print(f"   mean time-limit episode: len {sum(n for n, _ in tl) / len(tl):.0f}, return {sum(r for _, r in tl) / len(tl):.1f}")
