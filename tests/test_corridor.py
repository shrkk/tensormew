"""Corridor-traversal shaping: pure logic on synthetic grids, and the real Route 29 corridor / dead-end pocket on the emulator."""
from pathlib import Path

import pytest

from src.corridor import CorridorTracker

ROOT = Path(__file__).resolve().parent.parent
HAVE = (ROOT / "rom" / "pokemon_crystal.gbc").exists() and (ROOT / "states" / "route29_entry.state").exists()

B = 0.5


def make_grid(rows):
    cells = {(x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch != "#"}

    def n(c):
        x, y = c
        return sum((x + dx, y + dy) in cells for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)))
    return cells, n


def walk(tracker, n, path):
    """Feed a list of cells (consecutive adjacent tiles); return the per-step rewards."""
    out = []
    for prev, cur in zip(path, path[1:]):
        out.append(tracker.update(prev, cur, n(cur), True))
    return out


# room (3x3) - corridor of 4 tiles - room (3x3) on one row each, walls elsewhere
THROUGH = [
    "###########",
    "#...#####...#"[:11],
    "#.........#"[:11],
    "#...#####...#"[:11],
    "###########",
]
CORRIDOR_MAP = [
    "#############",
    "#...#####...#",
    "#...........#",
    "#...#####...#",
    "#############",
]


def row_path(y, x0, x1):
    step = 1 if x1 >= x0 else -1
    return [(x, y) for x in range(x0, x1 + step, step)]


def test_forward_traversal_into_open_space_pays_once_and_reverse_cancels():
    cells, n = make_grid(CORRIDOR_MAP)
    t = CorridorTracker(B, min_len=3)
    fwd = walk(t, n, row_path(2, 2, 10))          # left room -> corridor (x 4..8) -> right room
    assert sum(fwd) == pytest.approx(B) and fwd.count(B) == 1
    back = walk(t, n, row_path(2, 10, 2))
    assert sum(back) == pytest.approx(-B)
    assert t.outstanding() == 0


def test_oscillating_through_a_corridor_cannot_be_farmed():
    cells, n = make_grid(CORRIDOR_MAP)
    t = CorridorTracker(B, min_len=3)
    total = 0.0
    for _ in range(25):
        total += sum(walk(t, n, row_path(2, 2, 10)))
        total += sum(walk(t, n, row_path(2, 10, 2)))
    assert total == pytest.approx(0.0)
    # ending on the far side leaves exactly one bonus outstanding
    total += sum(walk(t, n, row_path(2, 2, 10)))
    assert total == pytest.approx(B)


DEAD_END = [
    "#############",
    "#...#####...#",
    "#.....#####.#",   # corridor row 2 continues x 4..8 then ends at a wall: dead end
    "#...#####...#",
    "#############",
]


def test_a_corridor_ending_in_a_dead_end_tile_is_never_rewarded():
    rows = ["#############",
            "#...#########",
            "#...........#",   # corridor x 4..10, last tile (11,2) has 1 walkable neighbour
            "#...#########",
            "#############"]
    cells, n = make_grid(rows)
    assert n((11, 2)) == 1
    t = CorridorTracker(B, min_len=3)
    r = walk(t, n, row_path(2, 2, 11))
    assert sum(r) == 0.0
    r2 = walk(t, n, row_path(2, 11, 2))
    assert sum(r2) == 0.0


def test_entering_then_backing_out_of_a_corridor_pays_nothing():
    cells, n = make_grid(CORRIDOR_MAP)
    t = CorridorTracker(B, min_len=3)
    path = row_path(2, 2, 6) + row_path(2, 5, 2)   # in 3 tiles, turn back, leave the same way
    assert sum(walk(t, n, path)) == 0.0
    assert t.outstanding() == 0


def test_stepping_back_inside_then_going_through_still_pays_once():
    cells, n = make_grid(CORRIDOR_MAP)
    t = CorridorTracker(B, min_len=3)
    path = row_path(2, 2, 6) + row_path(2, 5, 3) + row_path(2, 4, 10)   # in, back out, in again, through
    assert sum(walk(t, n, path)) == pytest.approx(B)


def test_a_run_shorter_than_min_len_is_ignored():
    rows = ["#########",
            "#...#...#",
            "#.......#",    # corridor of just 1 tile (x=4) between two rooms
            "#...#...#",
            "#########"]
    cells, n = make_grid(rows)
    t = CorridorTracker(B, min_len=3)
    assert sum(walk(t, n, row_path(2, 2, 6))) == 0.0


def test_at_most_max_corridors_are_credited_and_map_change_resets():
    cells, n = make_grid(CORRIDOR_MAP)
    t = CorridorTracker(B, min_len=3, max_corridors=1)
    assert sum(walk(t, n, row_path(2, 2, 10))) == pytest.approx(B)
    # a second, different corridor (vertical copy placed elsewhere) is beyond the cap
    t.first_dir[((0, 0), (0, 1))] = 1
    t2 = CorridorTracker(B, min_len=3, max_corridors=1)
    walk(t2, n, row_path(2, 2, 10))
    assert len(t2.first_dir) == 1
    t2.reset()
    assert t2.first_dir == {} and t2.run is None and t2.outstanding() == 0


def test_validation():
    with pytest.raises(ValueError):
        CorridorTracker(-1.0)
    with pytest.raises(ValueError):
        CorridorTracker(0.3, min_len=1)


def test_config_defaults_off_and_validated():
    from src.train_config import from_dict, long_v9
    base = dict(supply_injection=False, escape_assist=False, auto_dialogue=False, task_target="falkner_strict",
                path_progress=False, goal_potential=False, route_ckpt=False, retreat_penalty=False,
                stagnation_penalty=False, swarm=False, battle_archive=False, battle_lowhp_weight=0,
                disable_bump_penalty=False, disable_loop_penalty=False)
    c = from_dict(base, long_v9())
    assert c.corridor_bonus == 0.0
    from tools.train_deep_ppo import worker_env_cfg
    assert worker_env_cfg(c, None)["corridor_bonus"] == 0.0
    assert worker_env_cfg(from_dict({**base, "corridor_bonus": 0.4}, long_v9()), None)["corridor_bonus"] == 0.4
    with pytest.raises(Exception):
        from_dict({**base, "corridor_bonus": -0.1}, long_v9())


# --------------------------------------------------------------- real emulator: Route 29

@pytest.fixture(autouse=True)
def _restore_global_story_profile():
    from src import story_path
    yield
    story_path.set_profile(False)


def _worker(bonus):
    import tools.train_deep_ppo as t
    from src.train_config import from_dict, long_v9
    cfg = from_dict(dict(supply_injection=False, escape_assist=False, auto_dialogue=False, task_target="falkner_strict",
                         path_progress=False, goal_potential=False, route_ckpt=False, retreat_penalty=False,
                         stagnation_penalty=False, swarm=False, battle_archive=False, battle_lowhp_weight=0,
                         disable_bump_penalty=False, disable_loop_penalty=False, corridor_bonus=bonus, seed=3), long_v9())
    return t.EnvWorker(0, ["route29_entry"], 4000, t.worker_env_cfg(cfg, None))


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
# emulator-verified (diagnostic BFS) moves toward the foot of the 1-tile corridor; the worker starts one tile east of the search start.
# The first press toward a NEW direction sometimes only turns the character, so later legs press "until arrived" (bounded).
APPROACH = [LEFT] + [DOWN] * 4 + [LEFT] * 5 + [DOWN] * 2 + [LEFT] * 4 + [DOWN] + [LEFT] * 10 + [UP] * 2


def _step(w, move):
    """One worker step; Repel is written first in this TEST ONLY so wild encounters cannot interrupt the scripted walk."""
    bank, addr = w.emu.symbol("wRepelEffect")
    w.emu.write(bank, addr, bytes([255]))
    *_, info = w.step(move)
    st = w.reader.parse(w.emu.frame)
    return (st.x, st.y), info["reward_components"].get("corridor", 0.0)


def _run(w, moves):
    return [_step(w, m) for m in moves]


def _until(w, move, done, limit=12):
    """Press ``move`` until ``done((x, y))`` is true (bounded); returns the (position, corridor reward) of every press."""
    out = []
    for _ in range(limit):
        out.append(_step(w, move))
        if done(out[-1][0]):
            return out
    raise AssertionError(f"never arrived: {out[-4:]}")


def _to_foot(w):
    """Walk to (31,14), the foot of the corridor, returning every step."""
    steps = _run(w, APPROACH)
    steps += _until(w, LEFT, lambda p: p[0] == 31)
    assert steps[-1][0] == (31, 14), steps[-3:]
    return steps


@pytest.mark.skipif(not HAVE, reason="needs the ROM and the route29_entry state")
def test_real_one_tile_corridor_pays_on_exit_and_reverse_cancels_and_default_pays_nothing():
    w = _worker(B)
    try:
        w.reset()
        pre = _to_foot(w)
        assert all(c == 0.0 for _, c in pre)                       # the long walk (2-wide corridors, open ground) pays nothing
        up = _until(w, UP, lambda p: p[1] == 11)
        ys = [p[1] for p, _ in up if p[0] == 31]
        assert ys[-3:] == [13, 12, 11] and all(p[0] == 31 for p, _ in up)      # 2 walled tiles, then the junction at (31,11)
        rewards = [c for _, c in up]
        assert rewards[-1] == pytest.approx(B) and all(c == 0.0 for c in rewards[:-1])   # paid once, on leaving into open ground
        down = _until(w, DOWN, lambda p: p[1] == 14)
        assert sum(c for _, c in down) == pytest.approx(-B)         # coming back out cancels it
    finally:
        w.close()
    w0 = _worker(0.0)
    try:
        w0.reset()
        assert w0.corridor is None
        steps = _to_foot(w0)
        steps += _until(w0, UP, lambda p: p[1] == 11)
        assert all(c == 0.0 for _, c in steps)
    finally:
        w0.close()


@pytest.mark.skipif(not HAVE, reason="needs the ROM and the route29_entry state")
def test_real_dead_end_pocket_is_never_paid():
    w = _worker(B)
    try:
        w.reset()
        _to_foot(w)                                                 # at (31,14)
        pocket = _until(w, LEFT, lambda p: p[0] <= 14, limit=40)    # west along the 2-wide corridor into the x=14 dead-end pocket
        assert pocket[-1][0][0] <= 15
        assert all(c == 0.0 for _, c in pocket)
        back = _until(w, RIGHT, lambda p: p[0] >= 31, limit=40)
        assert all(c == 0.0 for _, c in back)
    finally:
        w.close()

def test_a_two_tile_u_turn_end_cap_is_never_paid_but_a_straight_two_tile_corridor_is():
    # 2-wide strip whose west end-cap tiles (1,2),(1,3) each have exactly 2 walkable neighbours: enter at (2,2), loop through the cap, leave at (2,3)
    strip = ["#########",
             "#########",
             "#.......#",
             "#.......#",
             "#########"]
    cells, n = make_grid(strip)
    assert n((1, 2)) == 2 and n((1, 3)) == 2 and n((2, 2)) == 3
    t = CorridorTracker(B, min_len=2)
    assert sum(walk(t, n, [(2, 2), (1, 2), (1, 3), (2, 3)])) == 0.0        # U-turn: entry and exit are adjacent
    straight = ["#######",
                "#.###.#",
                "#.....#",
                "#.###.#",
                "#######"]
    cells, n = make_grid(straight)
    t2 = CorridorTracker(B, min_len=2)
    assert sum(walk(t2, n, [(1, 1), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)])) == pytest.approx(B)   # 3 corridor tiles, junction to junction