"""Agent-earned frontier archive: pure logic, worker integration, snapshot, and evaluation isolation."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.frontier_archive import FRONTIER_PREFIX, FrontierArchive, parse_key

ROOT = Path(__file__).resolve().parent.parent
needs_rom = pytest.mark.skipif(not ((ROOT / "rom" / "pokemon_crystal.gbc").exists()
                                    and (ROOT / "states" / "new_bark_outside.state").exists()),
                               reason="needs the ROM and start states")

BASE = dict(supply_injection=False, escape_assist=False, auto_dialogue=False, task_target="falkner_strict",
            path_progress=False, goal_potential=False, route_ckpt=False, retreat_penalty=False,
            stagnation_penalty=False, swarm=False, battle_archive=False, battle_lowhp_weight=0,
            disable_bump_penalty=False, disable_loop_penalty=False)


@pytest.fixture(autouse=True)
def _restore_global_story_profile():
    from src import story_path
    yield
    story_path.set_profile(False)


# ------------------------------------------------------------------ pure logic

def test_observe_reports_first_visit_only_and_counts_every_step():
    a = FrontierArchive()
    assert a.observe((24, 3, 5, 5)) is True
    assert a.observe((24, 3, 5, 5)) is False
    assert a.visits[(24, 3, 5, 5)] == 2


def test_cap_evicts_the_most_visited_cell_never_the_newest():
    a = FrontierArchive(cap=3)
    for i, visits in enumerate((50, 5, 9)):
        cell = (24, 3, i, 0)
        for _ in range(visits):
            a.observe(cell)
        a.add(cell, b"x" * 4)
    newest = (24, 3, 9, 9)
    a.observe(newest)
    a.add(newest, b"y")
    assert len(a) == 3 and newest in a.states
    assert (24, 3, 0, 0) not in a.states          # the 50-visit cell went first


def test_pick_prefers_rare_cells_and_respects_probability_and_empty_archive():
    rng = np.random.default_rng(0)
    a = FrontierArchive(cap=8, prob=1.0)
    assert a.maybe_pick(rng) is None               # empty
    rare, common = (24, 3, 1, 1), (24, 3, 2, 2)
    for _ in range(400):
        a.observe(common)
    a.observe(rare)
    a.add(rare, b"r")
    a.add(common, b"c")
    picks = [a.maybe_pick(rng) for _ in range(300)]
    assert all(p.startswith(FRONTIER_PREFIX) for p in picks)
    share_rare = sum(p.endswith("24_3_1_1") for p in picks) / len(picks)
    assert share_rare > 0.7                         # weight 1/sqrt(1) vs 1/sqrt(400), decayed by selection counts
    assert a.load_key("24_3_1_1") == b"r" and parse_key("24_3_1_1") == (24, 3, 1, 1)
    never = FrontierArchive(prob=0.0)
    never.observe((1, 1, 1, 1)); never.add((1, 1, 1, 1), b"z")
    assert never.maybe_pick(rng) is None


def test_validation_rejects_bad_settings():
    with pytest.raises(ValueError):
        FrontierArchive(cap=0)
    with pytest.raises(ValueError):
        FrontierArchive(prob=1.5)


def test_config_defaults_off_and_validated():
    from src.train_config import from_dict, long_v9
    c = from_dict(BASE, long_v9())
    assert c.frontier_archive is False
    from tools.train_deep_ppo import worker_env_cfg
    assert worker_env_cfg(c, None)["frontier_prob"] == 0.0
    on = from_dict({**BASE, "frontier_archive": True, "frontier_prob": 0.4}, long_v9())
    assert worker_env_cfg(on, None)["frontier_prob"] == 0.4
    for bad in ({"frontier_prob": 1.5}, {"frontier_cap": 0}, {"frontier_min_hp": 2.0}):
        with pytest.raises(Exception):
            from_dict({**BASE, **bad}, long_v9())


# ------------------------------------------------------------------ real emulator

def _worker(prob, **extra):
    import tools.train_deep_ppo as t
    from src.train_config import from_dict, long_v9
    cfg = from_dict({**BASE, "frontier_archive": prob > 0, "frontier_prob": prob if prob > 0 else 0.5, **extra}, long_v9())
    return t.EnvWorker(0, ["new_bark_outside"], 30, t.worker_env_cfg(cfg, None))


@needs_rom
def test_worker_archives_new_cells_and_later_resets_start_from_them():
    from src.milestone_stats import is_base_start
    w = _worker(1.0)
    try:
        w.reset()
        assert w.state_name == "new_bark_outside" and len(w.frontier) == 0     # first episode: nothing archived yet
        for act in (2, 2, 2, 1, 1, 3, 3, 0, 0, 2):
            w.step(act)
        assert len(w.frontier) >= 3
        w.reset()
        assert w.state_name.startswith(FRONTIER_PREFIX) and not is_base_start(w.state_name)
        want = parse_key(w.state_name[len(FRONTIER_PREFIX):])
        st = w.reader.parse(w.emu.frame)
        assert (st.map_group, st.map_number, st.x, st.y) == want              # really started on that agent-earned tile
        assert w.steps == 0 and w.falkner_eligible                            # fresh episode ledgers, goal still earnable
    finally:
        w.close()


@needs_rom
def test_default_worker_has_no_archive_and_never_leaves_the_base_start():
    w = _worker(0.0)
    try:
        assert w.frontier is None
        for _ in range(3):
            w.reset()
            assert w.state_name == "new_bark_outside"
            w.step(2)
    finally:
        w.close()


@needs_rom
def test_unhealthy_states_are_not_archived():
    w = _worker(1.0, frontier_min_hp=0.5)
    try:
        w.reset()
        w.emu.write(*w.emu.symbol("wPartyMon1HP"), bytes([0, 1]))    # lead at 1 HP (max HP stays 19): below the 50% rule
        for act in (2, 2, 1):
            w.step(act)
        assert len(w.frontier) == 0
    finally:
        w.close()


@needs_rom
def test_archive_survives_worker_snapshot_and_restore():
    import tools.train_deep_ppo as t
    a, b = _worker(1.0), _worker(1.0)
    try:
        a.reset()
        for act in (2, 2, 1, 3):
            a.step(act)
        b.restore(a.snapshot())
        assert sorted(b.frontier.states) == sorted(a.frontier.states) and len(b.frontier) >= 2
        assert b.frontier.visits == a.frontier.visits
    finally:
        a.close()
        b.close()


def test_evaluation_never_uses_the_frontier_archive(monkeypatch):
    """sampled_eval_main must build its worker with frontier_prob = 0 even when the training config enables the archive."""
    import tools.evaluate_policy as ev
    import tools.train_deep_ppo as t
    seen = {}

    class Stop(Exception):
        pass

    class FakeWorker:
        def __init__(self, index, states, max_steps, cfg):
            seen.update(cfg)
            raise Stop

    monkeypatch.setattr(t, "EnvWorker", FakeWorker)
    cfg_path = ROOT / "falkner-strict-baseline.json"
    args = SimpleNamespace(uniform_random=True, random_init=False, checkpoint=None, config=str(cfg_path), seed=1,
                           start_state="new_bark_outside", max_steps=5, episodes=1, out_dir=None, name=None, device="cpu")
    from src.train_config import load_config
    real_load = load_config

    def enabled(path):
        c = real_load(path)
        c.frontier_archive, c.frontier_prob = True, 0.9
        return c

    import src.train_config as tc
    monkeypatch.setattr(tc, "load_config", enabled)
    with pytest.raises(Stop):
        ev.sampled_eval_main(args)
    assert seen["frontier_prob"] == 0.0
