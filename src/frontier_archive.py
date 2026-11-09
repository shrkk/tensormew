"""Agent-earned frontier archive (Go-Explore style, organic): emulator states at tiles the worker has rarely stood on.

No route knowledge anywhere: a "cell" is just ``(map_group, map_number, x, y)``.  The worker calls ``observe(cell)`` every step; the
first time a cell is ever seen it may ``add`` the current emulator state (healthy, outside battle/dialogue).  A fraction
``prob`` of episode resets then start from an archived state, picked with weight ``1 / sqrt(visits[cell])`` (rarely visited cells
first) times ``1 / sqrt(1 + times_selected[cell])``.  Visiting a cell from an archived start raises its visit count, so the pick
moves on to newer, rarer cells: the frontier advances by the agent's own play.

Bounded: at most ``cap`` states are kept; when full the cell with the most visits (never the newest) is evicted.  Holds raw
emulator bytes (about 100-130 KB each); lives inside the worker snapshot so exact resume restores it.

Evaluation must NEVER use it (``tools.evaluate_policy`` forces ``frontier_prob = 0``).
"""
from __future__ import annotations

import math

FRONTIER_PREFIX = "frontier:"


def cell_key(cell) -> str:
    return "{}_{}_{}_{}".format(*cell)


def parse_key(key: str):
    return tuple(int(v) for v in key.split("_"))


class FrontierArchive:
    def __init__(self, cap: int = 64, prob: float = 0.5):
        if cap < 1:
            raise ValueError("cap must be >= 1")
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be in [0, 1]")
        self.cap = int(cap)
        self.prob = float(prob)
        self.visits: dict = {}      # cell -> times the worker has stood on it (every step, archived or not)
        self.selected: dict = {}    # cell -> times it was chosen as an episode start
        self.states: dict = {}      # cell -> emulator bytes
        self.order: list = []       # insertion order (newest last) for eviction protection

    def observe(self, cell) -> bool:
        """Count one step on ``cell``; True when this is the first time the worker has ever stood there."""
        first = cell not in self.visits
        self.visits[cell] = self.visits.get(cell, 0) + 1
        return first

    def add(self, cell, data: bytes) -> None:
        if cell in self.states:
            return
        self.states[cell] = bytes(data)
        self.order.append(cell)
        while len(self.states) > self.cap:
            # evict the most-visited cell, never the one just added
            victim = max((c for c in self.order[:-1]), key=lambda c: self.visits.get(c, 0), default=None)
            if victim is None:
                break
            self.states.pop(victim, None)
            self.order.remove(victim)

    def __len__(self) -> int:
        return len(self.states)

    def weight(self, cell) -> float:
        return 1.0 / math.sqrt(max(1, self.visits.get(cell, 1))) / math.sqrt(1 + self.selected.get(cell, 0))

    def maybe_pick(self, rng):
        """With probability ``prob`` (and a non-empty archive) return ``'frontier:<cell>'`` for the next episode start, else None."""
        if not self.states or float(rng.random()) >= self.prob:
            return None
        cells = sorted(self.states)
        weights = [self.weight(c) for c in cells]
        u = float(rng.random()) * sum(weights)
        acc = 0.0
        chosen = cells[-1]
        for c, w in zip(cells, weights):
            acc += w
            if u <= acc:
                chosen = c
                break
        self.selected[chosen] = self.selected.get(chosen, 0) + 1
        return FRONTIER_PREFIX + cell_key(chosen)

    def load_key(self, key: str):
        return self.states.get(parse_key(key))
