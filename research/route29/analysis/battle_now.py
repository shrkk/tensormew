import json, sys
from pathlib import Path
sys.path.insert(0, "C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main")
from src.deep_policy import BATTLE_FLAG_INDEX as BI
X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
def stats(arm, lo_frac=0.7):
    rows = []
    with open(X/arm/"steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' in line:
                rows.append(json.loads(line))
    if not rows: return
    maxu = max(r["update"] for r in rows)
    lo = int(maxu * lo_frac)
    n=bat=a=0; rew_batt=whiteout=dmg=0.0; wins=0
    for r in rows:
        if r["update"] < lo: continue
        n += 1
        o = r["outcome"]; c = o.get("reward_components") or {}
        rew_batt += c.get("battle", 0); whiteout += c.get("whiteout", 0); dmg += c.get("damage", 0)
        if float(r["ram_observation"][BI]) > 0.5:
            bat += 1; a += 1 if r["action"] == 4 else 0
    print(f"{arm:28s} updates {lo}-{maxu}: decisions={n:6d} in_battle={bat/n:4.0%} A_in_battle={(a/bat if bat else 0):4.0%} battle_reward={rew_batt:6.1f} damage_taken={dmg:7.1f} whiteouts={whiteout/-15:4.1f}")
for arm in sys.argv[1:]:
    stats(arm)
