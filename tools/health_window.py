"""Compact health summary of a training trace, per window of updates (default 40), plus the stop-gate verdict.

Read-only: it reads the append-only ``steps.jsonl`` written by ``src.experiment_trace`` and prints/writes JSON.  It never
touches a run, a checkpoint or the STOP marker.  Per window it reports, from facts already in the trace only:

* battle occupancy: share of decisions that BEGAN in battle (RAM battle flag of the observation the action was taken on);
* legal-A / A-press: share of in-battle decisions where A was legal, and how many times A was pressed in battle;
* completed encounters: transitions battle -> not-battle inside one env's decision sequence (an exit, NOT a verified win);
* party-HP changes inside battle (a proxy for damage exchanged; PP is not in the trace, so it is not claimed);
* episode endings by reason, and decisions / emulator frames.

Gate (spec section 3): a window is *stalled* when occupancy > 0.90, fewer encounters completed than there are envs (less than
one resolved encounter per env per window: this is the operationalization of "no completed encounter"; the literal
"zero" would miss the real round-299 pattern, which had a single exit in 4,608 decisions), no party-HP change occurred
and A was legal but (almost) never pressed.  Two consecutive stalled windows => ``stop_arm_for_diagnosis`` (a diagnostic
trigger only: nothing is edited, no game terminal is created).  Trace entries that precede a ``resume`` marker with
``update >= from_update`` were abandoned by an exact resume and are dropped.

    python -m tools.health_window --trace <steps.jsonl> [--window 40] [--out summary.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

A_ACTION = 4   # up, down, left, right, A, B
STALL_OCCUPANCY = 0.90
STALL_MIN_A_PRESS_RATE = 0.005   # A presses per in-battle decision below this count as "effectively unused"


def read_steps(path) -> tuple[list[dict], list[dict]]:
    """(step events in effective order, other events).  Entries made obsolete by a later resume marker are removed."""
    steps: list[dict] = []
    other: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("event") == "step":
                steps.append(e)
            else:
                other.append(e)
                if e.get("event") == "resume":
                    cut = int(e["from_update"])
                    steps = [s for s in steps if int(s["update"]) < cut]
    return steps, other


def _battle(e: dict, battle_index: int) -> bool:
    ram = e.get("ram_observation") or []
    return len(ram) > battle_index and float(ram[battle_index]) > 0.5


def summarize(steps: list[dict], window: int = 40, battle_index: int | None = None) -> list[dict]:
    if battle_index is None:
        from src.deep_policy import BATTLE_FLAG_INDEX
        battle_index = BATTLE_FLAG_INDEX
    by_env: dict[int, list[dict]] = {}
    for s in steps:
        by_env.setdefault(int(s["env"]), []).append(s)
    wins: dict[int, dict] = {}

    def w_of(update: int) -> dict:
        return wins.setdefault(update // window, {
            "first_update": (update // window) * window, "last_update": update, "decisions": 0, "frames": 0,
            "battle_decisions": 0, "battle_legal_a": 0, "battle_a_presses": 0, "encounters_completed": 0,
            "battle_party_hp_changes": 0, "episode_endings": {}, "_envs": set()})

    for env_steps in by_env.values():
        env_steps.sort(key=lambda s: (int(s["update"]), int(s["rollout_step"])))
        prev_battle, prev_hp = False, None
        for s in env_steps:
            w = w_of(int(s["update"]))
            w["last_update"] = max(w["last_update"], int(s["update"]))
            w["_envs"].add(int(s["env"]))
            out = s.get("outcome") or {}
            tr = out.get("transition") or {}
            w["decisions"] += 1
            w["frames"] += max(0, int(tr.get("frame_after", 0)) - int(tr.get("frame_before", 0)))
            in_battle = _battle(s, battle_index)
            if in_battle:
                w["battle_decisions"] += 1
                mask = s.get("mask")
                if mask is None or bool(mask[A_ACTION]):
                    w["battle_legal_a"] += 1
                if int(s["action"]) == A_ACTION:
                    w["battle_a_presses"] += 1
                hp = tuple(tuple(h) for h in ((out.get("fun") or {}).get("ph") or ()))
                if prev_hp is not None and hp and hp != prev_hp and prev_battle:
                    w["battle_party_hp_changes"] += 1
                prev_hp = hp or prev_hp
            if prev_battle and not in_battle:
                w["encounters_completed"] += 1
            if s.get("done"):
                reason = str(out.get("end_reason") or "unknown")
                w["episode_endings"][reason] = w["episode_endings"].get(reason, 0) + 1
            prev_battle = in_battle
    result = []
    for key in sorted(wins):
        w = wins[key]
        n, b = w["decisions"], w["battle_decisions"]
        w["battle_occupancy"] = round(b / n, 4) if n else 0.0
        w["legal_a_share_in_battle"] = round(w["battle_legal_a"] / b, 4) if b else None
        w["a_press_rate_in_battle"] = round(w["battle_a_presses"] / b, 5) if b else None
        w["envs"] = len(w.pop("_envs"))
        w["stalled"] = bool(
            n and w["battle_occupancy"] > STALL_OCCUPANCY and w["encounters_completed"] < w["envs"]
            and w["battle_party_hp_changes"] == 0 and (w["legal_a_share_in_battle"] or 0) > 0.5
            and (w["a_press_rate_in_battle"] or 0) < STALL_MIN_A_PRESS_RATE)
        result.append(w)
    return result


def verdict(windows: list[dict]) -> dict:
    full = [w for w in windows]   # a partial trailing window is reported but can only stall by the same rule
    consecutive = 0
    for w in full:
        consecutive = consecutive + 1 if w["stalled"] else 0
        if consecutive >= 2:
            return {"action": "stop_arm_for_diagnosis", "reason": "two consecutive stalled health windows",
                    "at_update": w["last_update"]}
    return {"action": "continue", "stalled_windows_in_a_row": consecutive}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    steps, _ = read_steps(args.trace)
    windows = summarize(steps, args.window)
    report = {"trace": str(args.trace), "window_updates": args.window, "windows": windows, "gate": verdict(windows)}
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
