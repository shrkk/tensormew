"""Battle records on the real emulator: a wild battle from the Route 29 entry state must produce a well-formed record."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HAVE = (ROOT / "rom" / "pokemon_crystal.gbc").exists() and (ROOT / "states" / "route29_entry.state").exists()


@pytest.fixture(autouse=True)
def _restore_global_story_profile():
    from src import story_path
    yield
    story_path.set_profile(False)


@pytest.mark.skipif(not HAVE, reason="needs the ROM and the route29_entry state")
def test_a_real_wild_battle_yields_a_wellformed_record_and_never_changes_the_reward_path():
    import tools.train_deep_ppo as t
    from src.train_config import from_dict, long_v9
    cfg = from_dict(dict(supply_injection=False, escape_assist=False, auto_dialogue=False, task_target="falkner_strict",
                         path_progress=False, goal_potential=False, route_ckpt=False, retreat_penalty=False,
                         stagnation_penalty=False, swarm=False, battle_archive=False, battle_lowhp_weight=0,
                         disable_bump_penalty=False, disable_loop_penalty=False, seed=5), long_v9())
    w = t.EnvWorker(0, ["route29_entry"], 3000, t.worker_env_cfg(cfg, None))
    records, rewards_with, steps = [], [], 0
    try:
        w.reset()
        rng = __import__("numpy").random.default_rng(3)
        while steps < 2500 and len(records) < 2:
            # walk left and mash A / directions: grass encounters happen on their own; battle buttons are random
            act = int(rng.choice([2, 2, 2, 4, 4, 5, 0, 1, 3]))
            *_, rew, done, _m, _b, info = w.step(act)
            rewards_with.append(rew)
            steps += 1
            if info.get("battle_end"):
                records.append(info["battle_end"])
            if done:
                break
    finally:
        w.close()
    assert records, f"no battle finished in {steps} steps (expected wild encounters on Route 29 grass)"
    for r in records:
        assert r["outcome"] in ("win", "escape", "whiteout", "caught")
        assert r["kind"] in ("wild", "trainer") and r["decisions"] >= 1
        assert 0.0 <= r["hp_lost_frac"] <= 1.5 and isinstance(r["enemy_species"], int)
