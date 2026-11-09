"""Per-battle outcome classification (pure, synthetic states) and the trace report."""
import json
from types import SimpleNamespace

from src.battle_record import BattleTracker, summarize


def S(mode=0, hp=((19, 19),), e_hp=10, e_max=10, fainted=False, party=1, dex=1, species=16):
    return SimpleNamespace(battle_mode=mode, party_hp=hp, enemy_hp=e_hp, enemy_max_hp=e_max, fainted=fainted,
                           party_count=party, pokedex_caught_count=dex, enemy_mon_species=species)


def run(states):
    """Feed consecutive states; return the single record produced (or None)."""
    t, out = BattleTracker(), []
    for before, after in zip(states, states[1:]):
        r = t.update(before, after)
        if r:
            out.append(r)
    assert len(out) <= 1
    return out[0] if out else None


def test_wild_win_counts_decisions_and_hp_lost():
    r = run([S(0), S(1), S(1, hp=((15, 19),)), S(1, hp=((12, 19),), e_hp=0), S(0, hp=((12, 19),), e_hp=0)])
    assert r["outcome"] == "win" and r["kind"] == "wild" and r["enemy_species"] == 16
    assert r["decisions"] == 3 and abs(r["hp_lost_frac"] - 7 / 19) < 1e-3


def test_escape_when_the_wild_enemy_is_still_alive_and_no_wipe():
    r = run([S(0), S(1), S(1), S(0, e_hp=8)])
    assert r["outcome"] == "escape"


def test_whiteout_beats_everything_else():
    r = run([S(0), S(1), S(1, hp=((0, 19),)), S(0, hp=((0, 19),), fainted=True, e_hp=0)])
    assert r["outcome"] == "whiteout"


def test_caught_by_party_or_dex_growth():
    assert run([S(0), S(1), S(1), S(0, party=2)])["outcome"] == "caught"
    assert run([S(0), S(1), S(1), S(0, dex=2)])["outcome"] == "caught"


def test_trainer_battle_left_without_a_wipe_is_a_win():
    r = run([S(0), S(2), S(2), S(0)])
    assert r["kind"] == "trainer" and r["outcome"] == "win"


def test_open_battle_is_not_reported_and_reset_abandons_it():
    t = BattleTracker()
    assert t.update(S(0), S(1)) is None and t.update(S(1), S(1)) is None
    t.reset()
    assert t.update(S(0), S(0)) is None and t.cur is None


def test_state_without_a_battle_never_creates_a_record():
    t = BattleTracker()
    for _ in range(5):
        assert t.update(S(0), S(0)) is None


def test_summary_rates():
    recs = [{"outcome": "win", "decisions": 4, "hp_lost_frac": 0.2}, {"outcome": "win", "decisions": 6, "hp_lost_frac": 0.4},
            {"outcome": "whiteout", "decisions": 20, "hp_lost_frac": 1.0}, {"outcome": "escape", "decisions": 2, "hp_lost_frac": 0.0}]
    s = summarize(recs)
    assert s["battles"] == 4 and s["outcomes"] == {"win": 2, "whiteout": 1, "escape": 1}
    assert s["win_rate"] == 0.5 and s["whiteout_rate"] == 0.25 and s["mean_decisions_per_win"] == 5.0
    assert summarize([])["win_rate"] is None


def test_report_reads_battle_end_and_drops_abandoned_entries_after_resume(tmp_path):
    from tools.battle_report import read_battles, windows
    rows = [{"event": "step", "update": 0, "outcome": {"battle_end": {"outcome": "win", "decisions": 3, "hp_lost_frac": 0.1}}},
            {"event": "step", "update": 1, "outcome": {}},
            {"event": "step", "update": 2, "outcome": {"battle_end": {"outcome": "whiteout", "decisions": 9, "hp_lost_frac": 1.0}}},
            {"event": "resume", "from_update": 2},
            {"event": "step", "update": 2, "outcome": {"battle_end": {"outcome": "win", "decisions": 5, "hp_lost_frac": 0.2}}}]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    recs, n = read_battles(p)
    assert [b["outcome"] for _, b in recs] == ["win", "win"] and n == 3
    assert windows(recs, 40)[0]["win_rate"] == 1.0
    empty = tmp_path / "e.jsonl"
    empty.write_text(json.dumps({"event": "step", "update": 0, "outcome": {}}))
    assert read_battles(empty)[0] == []
