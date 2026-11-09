"""ASSISTED catching experiments: start-ball supply (balls only) and the party-size-aware catch reward."""
from pathlib import Path

import pytest

from src.rl_env import progress_reward
from tests.test_reward_blueprint import _sn

ROOT = Path(__file__).resolve().parent.parent
needs_rom = pytest.mark.skipif(not ((ROOT / "rom" / "pokemon_crystal.gbc").exists()
                                    and (ROOT / "states" / "new_bark_outside.state").exists()),
                               reason="needs the ROM and start states")

@pytest.fixture(autouse=True)
def _restore_global_story_profile():
    """Building an egg-profile worker sets a process-global story profile; never leak it into other tests."""
    from src import story_path
    yield
    story_path.set_profile(False)


FARM = {"free": 2, "decay": 0.6, "floor": 0.1, "wild_scale": 0.4, "gate_level": False,
        "catch_new_only": True, "catch_reward": 5.0}


def _catch_pay(party_before, party_scale, species=16):
    before = _sn(party_count=party_before, party_levels=(5,) * party_before, party_species=tuple(range(1, party_before + 1)))
    after = _sn(party_count=party_before + 1, party_levels=(5,) * (party_before + 1),
                party_species=tuple(range(1, party_before + 1)) + (species,))
    comp = {}
    progress_reward(before, after, False, set(), {(24, 3)}, caught_counts={}, components=comp,
                    budget={"cap:catch": 100.0}, farm={**FARM, "party_scale": party_scale})
    return comp.get("catch", 0.0)


def test_catch_reward_scales_with_free_party_slots_and_default_is_unchanged():
    # off (historical): flat catch_reward for a new species regardless of party size
    assert [_catch_pay(n, False) for n in (1, 3, 5)] == [5.0, 5.0, 5.0]
    # on: (6 - party size before the catch) / 5
    assert [_catch_pay(n, True) for n in (1, 2, 3, 4, 5)] == pytest.approx([5.0, 4.0, 3.0, 2.0, 1.0])


def test_full_party_pays_nothing_for_a_box_catch():
    """Party already full: the catch goes to a box (party size unchanged, Pokedex count rises) and must pay 0 when scaled."""
    before = _sn(party_count=6, party_levels=(5,) * 6, party_species=(1, 2, 3, 4, 5, 6), pokedex_caught_count=6)
    after = _sn(party_count=6, party_levels=(5,) * 6, party_species=(1, 2, 3, 4, 5, 6), pokedex_caught_count=7)
    for scale, expected in ((True, 0.0), (False, 5.0)):
        comp = {}
        progress_reward(before, after, False, set(), {(24, 3)}, caught_counts={}, components=comp,
                        budget={"cap:catch": 100.0}, farm={**FARM, "party_scale": scale})
        assert comp.get("catch", 0.0) == pytest.approx(expected)


def test_duplicate_species_still_pays_nothing_when_scaled():
    before = _sn(party_count=1, party_levels=(5,), party_species=(155,))
    after = _sn(party_count=2, party_levels=(5, 5), party_species=(155, 16))
    comp = {}
    progress_reward(before, after, False, set(), {(24, 3)}, caught_counts={16: 1}, components=comp,
                    budget={"cap:catch": 100.0}, farm={**FARM, "party_scale": True})
    assert comp.get("catch", 0.0) == 0.0


def test_config_defaults_are_organic_and_validated():
    from src.train_config import from_dict, long_v9
    base = dict(supply_injection=False, escape_assist=False, auto_dialogue=False, task_target="falkner_strict",
                path_progress=False, goal_potential=False, route_ckpt=False, retreat_penalty=False,
                stagnation_penalty=False, swarm=False, battle_archive=False, battle_lowhp_weight=0,
                disable_bump_penalty=False, disable_loop_penalty=False)
    c = from_dict(base, long_v9())
    assert c.assisted_start_balls == 0 and c.catch_party_scale is False and c.anti_farm_params()["party_scale"] is False
    on = from_dict({**base, "assisted_start_balls": 10, "catch_party_scale": True, "catch_reward": 5.0}, long_v9())
    assert on.anti_farm_params()["party_scale"] is True and on.anti_farm_params()["catch_reward"] == 5.0
    for bad in (-1, 100):
        with pytest.raises(Exception):
            from_dict({**base, "assisted_start_balls": bad}, long_v9())


@needs_rom
def test_start_balls_adds_only_balls_and_default_run_still_has_none():
    import tools.train_deep_ppo as t
    from src.battle_obs import ball_count, equip_balls, heal_item_count
    from src.emulator import Emulator
    from src.ram_parser import RamParser
    from src.states import StateRegistry
    from src.train_config import from_dict, long_v9
    emu = Emulator(headless=True, speed=0)
    try:
        emu.load_state(StateRegistry().load_bytes("new_bark_outside"))
        reader = RamParser(emu)
        before = reader.parse(emu.frame)
        assert ball_count(before) == 0 and heal_item_count(before) == 1 and before.money == 0
        equip_balls(emu, 7)
        after = reader.parse(emu.frame)
        assert ball_count(after) == 7
        # nothing else moved: same potion stock, money, party, events, position
        assert (heal_item_count(after), after.money, after.party_species, after.x, after.y) == \
               (heal_item_count(before), before.money, before.party_species, before.x, before.y)
        assert after.event_flags == before.event_flags
        equip_balls(emu, 0)                      # no-op
        assert ball_count(reader.parse(emu.frame)) == 7
        equip_balls(emu, 3)                      # never lowers an existing stock
        assert ball_count(reader.parse(emu.frame)) == 7
    finally:
        emu.close()
    base = dict(supply_injection=False, escape_assist=False, auto_dialogue=False, task_target="falkner_strict",
                path_progress=False, goal_potential=False, route_ckpt=False, retreat_penalty=False,
                stagnation_penalty=False, swarm=False, battle_archive=False, battle_lowhp_weight=0,
                disable_bump_penalty=False, disable_loop_penalty=False)
    counts = {}
    for balls in (0, 7):
        cfg = from_dict({**base, "assisted_start_balls": balls}, long_v9())
        w = t.EnvWorker(0, ["new_bark_outside"], 10, t.worker_env_cfg(cfg, None))
        try:
            w.reset()
            counts[balls] = ball_count(w.reader.parse(w.emu.frame))
        finally:
            w.close()
    assert counts == {0: 0, 7: 7}
