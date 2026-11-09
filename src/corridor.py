"""Generic corridor-traversal shaping (potential style).  No map or route knowledge: it only uses how many of a tile's four
neighbours are walkable (the same collision information the observation already carries).

Terms:
* a CORRIDOR TILE has exactly 2 walkable neighbours;
* a TRAVERSAL is a monotone run of >= ``min_len`` corridor tiles, entered from a tile with >= 3 walkable neighbours and left into one (open
  space or a junction), whose entry and exit are at least 3 tiles apart (a U-turn, such as the end-cap of a 2-wide corridor, does not count).  A dead-end tile has 1 walkable neighbour, so a corridor that ends in one is never rewarded; a run shorter than
  ``min_len`` (e.g. a 2-tile pocket) is never rewarded; stepping back inside the corridor abandons the run.

Reward: the first crossing of a corridor pays +bonus.  Crossing the SAME corridor back the other way pays -bonus (and again +bonus if
it is crossed forward again).  So the term is exactly a potential (+bonus while the agent is on the far side): oscillating through a
corridor nets 0 (no farming), and going into a dead-end room and coming back out nets 0.  Honest limit: a corridor that leads into a
dead-end ROOM (many open tiles, no other exit) pays +bonus while the agent stays in that room; locally that cannot be told apart from
a through route.  Bounded: at most ``max_corridors`` distinct corridors are ever credited per episode.
"""
from __future__ import annotations

DIRS = ("up", "down", "left", "right")


def walkable_count(state) -> int:
    """Number of the four neighbours of the tile the agent stands on that it could walk onto (src.mover.walkable semantics)."""
    from src.mover import walkable
    tiles = getattr(state, "tiles", None) or {}
    return sum(1 for d in DIRS if walkable(tiles.get(d, 0xFF), d))


class CorridorTracker:
    def __init__(self, bonus: float = 0.3, min_len: int = 2, max_corridors: int = 8):
        if bonus < 0 or min_len < 2 or max_corridors < 1:
            raise ValueError("bonus >= 0, min_len >= 2 and max_corridors >= 1 are required")
        self.bonus = float(bonus)
        self.min_len = int(min_len)
        self.max_corridors = int(max_corridors)
        self.reset()

    def reset(self) -> None:
        """A new episode (or a map change): forget runs and credits."""
        self.run = None
        self.prev_n = None
        self.first_dir: dict = {}   # corridor id -> direction of its first crossing (+1 / -1)
        self.on_far: dict = {}      # corridor id -> True while the +bonus is outstanding

    def update(self, prev_cell, cur_cell, n_cur: int, moved: bool) -> float:
        """One decision: the agent went ``prev_cell`` -> ``cur_cell`` (``moved``), and ``cur_cell`` has ``n_cur`` walkable neighbours."""
        if not moved or prev_cell == cur_cell:
            return 0.0
        prev_n, self.prev_n = self.prev_n, n_cur
        run = self.run
        if run is None:
            # entering a corridor from OPEN ground (>= 3 walkable neighbours). A run that starts on a dead-end tile (1 neighbour)
            # is the way back OUT of a dead end and must never be paid; an unknown previous tile (episode start) cannot start a run.
            if n_cur == 2 and prev_n is not None and prev_n >= 3:
                self.run = {"entry": prev_cell, "tiles": [cur_cell]}
            return 0.0
        if cur_cell in run["tiles"] or cur_cell == run["entry"]:
            self.run = None                                 # backed up: not a traversal
            return 0.0
        if n_cur == 2:
            run["tiles"].append(cur_cell)
            return 0.0
        self.run = None                                     # the run ends on a non-corridor tile
        tiles = run["tiles"]
        if len(tiles) < self.min_len or n_cur < 3:          # too short, or a dead-end tile
            return 0.0
        entry = run["entry"]
        if abs(cur_cell[0] - entry[0]) + abs(cur_cell[1] - entry[1]) < 3:
            return 0.0                                      # a U-turn (e.g. the end-cap of a 2-wide corridor): not a traversal
        a, b = tiles[0], tiles[-1]
        cid = (a, b) if a <= b else (b, a)
        d = 1 if (a, b) == cid else -1
        if cid not in self.first_dir:
            if len(self.first_dir) >= self.max_corridors:
                return 0.0
            self.first_dir[cid] = d
            self.on_far[cid] = True
            return self.bonus
        if d == self.first_dir[cid]:
            if not self.on_far.get(cid, False):
                self.on_far[cid] = True
                return self.bonus
            return 0.0
        if self.on_far.get(cid, False):                     # crossed back: undo the credit
            self.on_far[cid] = False
            return -self.bonus
        return 0.0

    def outstanding(self) -> int:
        return sum(1 for v in self.on_far.values() if v)
