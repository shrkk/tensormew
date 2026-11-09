"""Cross-episode (global) novelty: default off and unchanged; persists across resets; familiar tiles stop paying."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
needs_rom = pytest.mark.skipif(not ((ROOT / "rom" / "pokemon_crystal.gbc").exists()
                                    and (ROOT / "states" / "new_bark_outside.state").exists()),
                               reason="needs the ROM and start states")

BASE = dict(supply_injection=False, escape_assist=False, auto_dialogue=False, task_target="falkner_strict",
            path_progress=False, goal_potential=False, route_ckpt=False, retreat_penalty=False,
            stagnation_penalty=False, swarm=False, battle_archive=False, battle_lowhp_weight=0,
            disable_bump_penalty=False, disable_loop_penalty=False, dwell_exempt_loop=False)


@pytest.fixture(autouse=True)
def _restore_global_story_profile():
    from src import story_path
    yield
    story_path.set_profile(False)


def _worker(**extra):
    import tools.train_deep_ppo as t
    from src.train_config import from_dict, long_v9
    cfg = from_dict({**BASE, **extra}, long_v9())
    return t.EnvWorker(0, ["new_bark_outside"], 50, t.worker_env_cfg(cfg, None))


def test_config_defaults_are_historical_and_validated():
    from src.train_config import from_dict, long_v9
    c = from_dict(BASE, long_v9())
    assert c.global_novelty is False and c.curiosity_scale == 0.08
    with pytest.raises(Exception):
        from_dict({**BASE, "curiosity_scale": -0.1}, long_v9())


@needs_rom
def test_global_counts_persist_across_episodes_only_when_enabled():
    for enabled in (True, False):
        w = _worker(global_novelty=enabled)
        try:
            w.reset()
            for _ in range(3):
                w.step(5)                      # B presses: stay on the start tile
            n_before = sum(w.global_tile_counts.values())
            w.reset()                          # a new episode
            assert not w.tile_visit_counts or sum(w.tile_visit_counts.values()) == 1   # per-episode counts restart
            if enabled:
                assert n_before >= 3 and sum(w.global_tile_counts.values()) == n_before + 1
            else:
                assert w.global_tile_counts == {}
        finally:
            w.close()


@needs_rom
def test_familiar_tiles_stop_paying_curiosity_and_tile_bonus_when_enabled():
    """Same situation (standing on a tile the worker has visited 10,000 times in earlier episodes): per-episode novelty still
    pays full curiosity; global novelty pays almost nothing.  Default behaviour is identical to the historical one."""
    pay = {}
    for enabled in (False, True):
        w = _worker(global_novelty=enabled)
        try:
            w.reset()
            st = w.reader.parse(w.emu.frame)
            tile = (st.map_group, st.map_number, st.x, st.y)
            w.global_tile_counts[tile] = 10_000 if enabled else 0
            *_, info = w.step(5)               # B press: same tile, first visit THIS episode count is 1
            pay[enabled] = info["reward_components"]["curiosity"]
        finally:
            w.close()
    assert pay[False] > 0.03 and pay[True] < pay[False] / 10


@needs_rom
def test_curiosity_scale_multiplies_the_term():
    out = {}
    for scale in (0.08, 0.4):
        w = _worker(curiosity_scale=scale)
        try:
            w.reset()
            *_, info = w.step(5)
            out[scale] = info["reward_components"]["curiosity"]
        finally:
            w.close()
    assert out[0.4] == pytest.approx(out[0.08] * 5.0, rel=1e-6)


@needs_rom
def test_global_counts_survive_snapshot_restore():
    import tools.train_deep_ppo as t
    from src.train_config import from_dict, long_v9
    cfg = from_dict({**BASE, "global_novelty": True}, long_v9())
    a = t.EnvWorker(0, ["new_bark_outside"], 50, t.worker_env_cfg(cfg, None))
    b = t.EnvWorker(0, ["new_bark_outside"], 50, t.worker_env_cfg(cfg, None))
    try:
        a.reset()
        for act in (5, 2, 2, 5):
            a.step(act)
        b.restore(a.snapshot())
        assert b.global_tile_counts == a.global_tile_counts and b.global_tile_counts
    finally:
        a.close()
        b.close()
