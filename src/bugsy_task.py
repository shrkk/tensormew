"""Sparse Bugsy objective; rewards facts acquired through transitions, never loaded facts.

This separate ledger does not change observation widths or encode navigation actions.
An imported post-Falkner save remains a segment fixture, not campaign success.
"""
from __future__ import annotations

EVENT_REWARDS = {
    "EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE": 10.0,
    "EVENT_AZALEA_TOWN_SLOWPOKETAIL_ROCKET": 5.0,
    "EVENT_CLEARED_SLOWPOKE_WELL": 15.0,
}
BUGSY_EVENT = "EVENT_BEAT_BUGSY"
HIVE_BADGE = 1 << 1


def _event(state, name):
    return bool(state.has_event(name))


def defeated_bugsy(state):
    return bool(int(state.badge_bits) & HIVE_BADGE) and _event(state, BUGSY_EVENT)


class BugsyLedger:
    """One ledger per episode, initialized only after loading and reset jitter.

    Completion requires BOTH the Bugsy event and Hive Badge to have been absent
    at initialization. Their script writes can occur on different decisions.
    """
    def __init__(self, initial):
        self.seen = {name for name in EVENT_REWARDS if _event(initial, name)}
        self.eligible = not _event(initial, BUGSY_EVENT) and not (int(initial.badge_bits) & HIVE_BADGE)
        self.success = False
        self.earned = set()

    def step(self, before, after):
        components = {}
        for name, reward in EVENT_REWARDS.items():
            if _event(after, name):
                if name not in self.seen and not _event(before, name):
                    components[name] = reward
                self.seen.add(name)
        if self.eligible and not self.success and defeated_bugsy(after):
            self.success = True
            components[BUGSY_EVENT] = 100.0
        self.earned.update(components)
        return components
