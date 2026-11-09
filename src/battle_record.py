"""Per-battle outcome records (pure bookkeeping: never changes a reward, an observation or an action).

``BattleTracker.update(before, after)`` is called once per decision.  It returns a finished-battle dict on the step a battle ends,
else None.  The outcome uses the same facts the reward code already uses (enemy HP at the end, party wipe, party/dex growth):

* ``win``         wild enemy fainted (or trainer battle left without a party wipe)
* ``caught``      party or Pokedex-caught count rose
* ``whiteout``    the whole party fainted
* ``escape``      the battle ended with the wild enemy still alive and no wipe (ran, or it fled)

``decisions`` counts the agent's decisions while the battle was open (menu navigation included), so ``decisions`` is a direct
measure of how efficiently it fights.  ``hp_lost_frac`` is the change in total party HP over the battle, as a fraction of max HP.
A battle still open when the episode ends is never reported (no outcome is invented).
"""
from __future__ import annotations


def _hp_totals(state):
    hp = getattr(state, "party_hp", ()) or ()
    return sum(p[0] for p in hp), sum(p[1] for p in hp)


class BattleTracker:
    def __init__(self):
        self.cur = None

    def reset(self) -> None:
        """A new episode: any open battle is abandoned (not reported)."""
        self.cur = None

    def update(self, before, after):
        mode_b = getattr(before, "battle_mode", 0)
        mode_a = getattr(after, "battle_mode", 0)
        if mode_b == 0 and mode_a != 0:
            cur_hp, max_hp = _hp_totals(after)
            self.cur = {"kind": "trainer" if mode_a == 2 else "wild", "decisions": 0, "hp_start": cur_hp, "max_hp": max_hp,
                        "enemy_species": int(getattr(after, "enemy_mon_species", 0) or 0)}
            return None
        if self.cur is None:
            return None
        if mode_b != 0:
            self.cur["decisions"] += 1
        if mode_b != 0 and mode_a == 0:
            rec = self.cur
            self.cur = None
            cur_hp, max_hp = _hp_totals(after)
            rec["hp_lost_frac"] = round(max(0.0, (rec["hp_start"] - cur_hp) / rec["max_hp"]), 4) if rec["max_hp"] else 0.0
            wiped = bool(getattr(after, "fainted", False))
            caught = (getattr(after, "party_count", 0) > getattr(before, "party_count", 0)
                      or getattr(after, "pokedex_caught_count", 0) > getattr(before, "pokedex_caught_count", 0))
            if wiped:
                rec["outcome"] = "whiteout"
            elif caught:
                rec["outcome"] = "caught"
            elif rec["kind"] == "trainer":
                rec["outcome"] = "win"
            else:
                e_max = max(getattr(after, "enemy_max_hp", 0) or 0, getattr(before, "enemy_max_hp", 0) or 0)
                rec["outcome"] = "win" if (e_max <= 0 or (getattr(after, "enemy_hp", 0) or 0) == 0) else "escape"
            del rec["hp_start"], rec["max_hp"]
            return rec
        return None


def summarize(records) -> dict:
    """Aggregate finished-battle records: counts by outcome, win rate, mean decisions and HP lost per battle."""
    records = list(records)
    n = len(records)
    out = {"battles": n, "outcomes": {}}
    for r in records:
        out["outcomes"][r["outcome"]] = out["outcomes"].get(r["outcome"], 0) + 1
    decided = sum(out["outcomes"].get(k, 0) for k in ("win", "whiteout", "escape", "caught"))
    out["win_rate"] = round(out["outcomes"].get("win", 0) / decided, 4) if decided else None
    out["whiteout_rate"] = round(out["outcomes"].get("whiteout", 0) / decided, 4) if decided else None
    out["mean_decisions"] = round(sum(r["decisions"] for r in records) / n, 2) if n else None
    out["mean_hp_lost_frac"] = round(sum(r["hp_lost_frac"] for r in records) / n, 4) if n else None
    wins = [r for r in records if r["outcome"] == "win"]
    out["mean_decisions_per_win"] = round(sum(r["decisions"] for r in wins) / len(wins), 2) if wins else None
    return out
