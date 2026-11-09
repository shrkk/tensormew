"""Read-only game-state analysis of the three arms' training traces (first 100 vs last 100 updates)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
ACT = ["up", "down", "left", "right", "A", "B"]
BI = 38  # placeholder, replaced below
sys.path.insert(0, "C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main")
from src.deep_policy import BATTLE_FLAG_INDEX as BI  # noqa: E402


def analyze(arm, lo, hi):
    n = 0
    maps = Counter()
    maps_battle = Counter()
    comp = defaultdict(float)
    battle_act = Counter()
    lowhp_act = Counter()
    lowhp_n = battle_n = 0
    whiteouts = Counter()
    ends = Counter()
    species = Counter()
    kinds = Counter()
    hp_restored = 0
    prev_lead = {}
    with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            e = json.loads(line)
            u = e["update"]
            if not (lo <= u < hi):
                continue
            o = e["outcome"]
            n += 1
            m = tuple(o["map"])
            maps[m] += 1
            inb = float(e["ram_observation"][BI]) > 0.5
            for k, v in (o.get("reward_components") or {}).items():
                comp[k] += v
            if inb:
                maps_battle[m] += 1
                battle_n += 1
                battle_act[ACT[e["action"]]] += 1
                kinds[o["fun"].get("bm")] += 1
                species[o["fun"].get("es")] += 1
                ph = o["fun"].get("ph") or []
                if ph and ph[0][1] > 0 and ph[0][0] / ph[0][1] <= 0.3:
                    lowhp_n += 1
                    lowhp_act[ACT[e["action"]]] += 1
            if e["done"]:
                r = o.get("end_reason") or "unknown"
                ends[r] += 1
                if r == "whiteout":
                    whiteouts[(m, o.get("ep_steps"))] += 1
    return dict(n=n, maps=maps, maps_battle=maps_battle, comp=dict(comp), battle_act=battle_act, lowhp_n=lowhp_n,
                battle_n=battle_n, lowhp_act=lowhp_act, ends=ends, species=species, kinds=kinds)


def show(arm, tag, r):
    n, bn = r["n"], r["battle_n"]
    print(f"\n--- {arm} [{tag}] decisions={n} battle={bn} ({bn / max(1, n):.0%})")
    tot = sum(v for v in r["comp"].values())
    pos = {k: round(v, 1) for k, v in sorted(r["comp"].items(), key=lambda kv: -kv[1]) if abs(v) > 0.05}
    print("  reward by component (sum over window):", pos, "| net", round(tot, 1))
    print("  top maps (share of decisions):", [(f"{m[0]}:{m[1]}", f"{c / n:.0%}") for m, c in r["maps"].most_common(4)])
    print("  battle actions:", {k: f"{v / max(1, bn):.0%}" for k, v in r["battle_act"].most_common()})
    print("  battle type (1 wild / 2 trainer):", dict(r["kinds"]), "| low-HP(<=30% lead) battle decisions:", r["lowhp_n"],
          "actions:", {k: f"{v / max(1, r['lowhp_n']):.0%}" for k, v in r["lowhp_act"].most_common()})
    print("  episode endings:", dict(r["ends"]))


for arm in (sys.argv[1:] or ("warm", "fresh", "transfer")):
    if not (X / arm / "steps.jsonl").exists():
        continue
    show(arm, "updates 0-99", analyze(arm, 0, 100))
    show(arm, "updates 284-383", analyze(arm, 284, 384))
