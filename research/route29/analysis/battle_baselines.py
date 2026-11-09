"""EVALUATION-ONLY yardstick (never a training signal): win/escape/whiteout rates of two trivial in-battle strategies on Route 29
wild fights, measured with the same battle records the agents are scored with.  Overworld moves are random-walk toward grass."""
import sys
from collections import Counter

sys.path.insert(0, "C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main")
import numpy as np  # noqa: E402

import tools.train_deep_ppo as t  # noqa: E402
from src.battle_record import summarize  # noqa: E402
from src.train_config import from_dict, long_v9  # noqa: E402

cfg = from_dict(dict(supply_injection=False, escape_assist=False, auto_dialogue=False, task_target="falkner_strict",
                     path_progress=False, goal_potential=False, route_ckpt=False, retreat_penalty=False,
                     stagnation_penalty=False, swarm=False, battle_archive=False, battle_lowhp_weight=0,
                     disable_bump_penalty=False, disable_loop_penalty=False, seed=11, terminate_on_whiteout=True), long_v9())


def run(strategy, battles_wanted=25, max_steps=12000, seed=0):
    rng = np.random.default_rng(seed)
    w = t.EnvWorker(0, ["route29_entry"], 4096, t.worker_env_cfg(cfg, None))
    recs, steps, episodes = [], 0, 0
    try:
        w.reset()
        in_battle = False
        while len(recs) < battles_wanted and steps < max_steps:
            st = w.reader.parse(w.emu.frame)
            in_battle = st.battle_mode != 0
            if in_battle:
                act = 4 if strategy == "always_A" else int(rng.integers(0, 6))
            else:
                act = int(rng.choice([2, 2, 3, 0, 1]))   # wander on the grass
            *_, done, _m, _b, info = w.step(act)
            steps += 1
            if info.get("battle_end"):
                recs.append(info["battle_end"])
            if done:
                episodes += 1
    finally:
        w.close()
    return recs, steps, episodes


for strat in ("always_A", "random_buttons"):
    recs, steps, eps = run(strat)
    s = summarize(recs)
    print(f"\n{strat}: {len(recs)} finished wild/trainer battles in {steps} decisions ({eps} episodes ended)")
    print("  outcomes:", s["outcomes"], "| win_rate:", s["win_rate"], "| whiteout_rate:", s["whiteout_rate"])
    print("  mean decisions/battle:", s["mean_decisions"], "| mean decisions/win:", s["mean_decisions_per_win"],
          "| mean HP lost/battle:", s["mean_hp_lost_frac"])
    print("  enemy species seen:", dict(Counter(r["enemy_species"] for r in recs)))
