"""Phase 1 continuity / reproducibility / telemetry gates (see outputs/route29-experiment-spec.md section 2.4).

Real-emulator tests need the ROM and are skipped without it, exactly like the existing emulator tests.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.continuation import GenerationError, check_embedded, read_generation, write_generation
from src.run_control import (Counters, LockHeld, RunHalted, RunLock, StatusWriter, check_disk, lr_for_update, pid_alive,
                             stop_requested)

ROOT = Path(__file__).resolve().parent.parent
HAVE_ROM = (ROOT / "rom" / "pokemon_crystal.gbc").exists() and (ROOT / "states" / "new_bark_outside.state").exists()
needs_rom = pytest.mark.skipif(not HAVE_ROM, reason="needs the ROM and start states")


@pytest.fixture(autouse=True)
def _restore_global_story_profile():
    """In-process ``train()`` sets the process-global story profile (egg for long_v9); never leak it into other tests."""
    from src import story_path
    yield
    story_path.set_profile(False)


# ---------------------------------------------------------------- schedules / run control (pure)

def test_lr_schedule_is_a_function_of_global_update_not_of_the_reporting_interval():
    horizon = 400
    uninterrupted = [lr_for_update(3e-4, 0.33, u, horizon) for u in range(400)]
    # A runner that restarts every 40 updates and re-derives the lr from its own interval would sawtooth; the global
    # function evaluated chunk by chunk must reproduce the uninterrupted curve exactly.
    chunked = [lr_for_update(3e-4, 0.33, 40 * c + k, horizon) for c in range(10) for k in range(40)]
    assert chunked == uninterrupted
    assert uninterrupted[0] == 3e-4 and all(b <= a for a, b in zip(uninterrupted, uninterrupted[1:]))
    sawtooth = [3e-4 * (1 - 0.67 * (k / 40)) for k in range(40)]
    assert uninterrupted[40] < sawtooth[0]            # the old behaviour restarted at the full lr here
    assert lr_for_update(3e-4, 1.0, 12345, 10) == 3e-4  # end_frac 1 = constant


def test_lock_refuses_live_owner_replaces_stale_and_release_only_removes_own(tmp_path):
    lock_path = tmp_path / "RUNNING.lock"
    mine = RunLock(lock_path).acquire()
    with pytest.raises(LockHeld):
        _foreign_acquire(lock_path)          # a different identity cannot take a lock whose owner (this process) is alive
    mine.release()
    assert not lock_path.exists()
    # stale file: a pid that is not running
    dead = _dead_pid()
    lock_path.write_text(json.dumps({"pid": dead, "started_utc": "x"}))
    taker = RunLock(lock_path).acquire()
    assert taker.taken_over_from["pid"] == dead
    # someone else rewrote the lock: release must not delete a lock this instance no longer owns
    lock_path.write_text(json.dumps({"pid": dead, "started_utc": "y"}))
    taker.release()
    assert lock_path.exists()


def _foreign_acquire(path):
    """Acquire as a different identity whose owner (this very process) is alive."""
    import os
    other = RunLock(path, pid=os.getpid() + 1_000_003)
    return other.acquire()


def _dead_pid() -> int:
    import subprocess
    import sys
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not pid_alive(p.pid)
    return p.pid


def test_status_heartbeat_is_monotonic_and_atomic(tmp_path):
    w = StatusWriter(tmp_path / "status.json")
    seqs, clocks = [], []
    for i in range(3):
        w.beat(phase="training", last_completed_update=i)
        data = json.loads((tmp_path / "status.json").read_text())
        seqs.append(data["heartbeat_seq"])
        clocks.append(data["heartbeat_monotonic_s"])
    assert seqs == [1, 2, 3] and clocks == sorted(clocks)
    assert not list(tmp_path.glob("*.tmp"))


def test_disk_guard_and_stop_marker(tmp_path):
    check_disk(tmp_path, 0)                      # guard off
    with pytest.raises(RunHalted) as exc:
        check_disk(tmp_path, 10 ** 9)            # an impossible requirement
    assert exc.value.reason == "low_disk"
    assert not stop_requested(tmp_path / "STOP")
    (tmp_path / "STOP").write_text("")
    assert stop_requested(tmp_path / "STOP")


def test_counters_separate_decisions_frames_resets_and_restarts():
    c = Counters()
    c.add_step({"transition": {"frame_before": 100, "frame_after": 120}}, False)
    c.add_step({"transition": {"frame_before": 120, "frame_after": 300}, "end_reason": "time_limit", "reset_frames": 7}, True)
    assert (c.train_decisions, c.train_frames, c.reset_frames) == (2, 200, 7)
    assert c.episodes_ended == {"time_limit": 1}
    d = Counters()
    d.load(json.loads(json.dumps(c.state())))
    assert d.state() == c.state()


# ---------------------------------------------------------------- generations (transactional checkpoint set)

def _writer(payload: dict):
    def write(path, gid):
        torch.save({"generation_id": gid, **payload}, path)
    return write


def _make_generation(root, update=3, **payloads):
    comps = {name: _writer({"v": name + str(update)}) for name in ("policy", "optimizer", "world")}
    return write_generation(root, update, comps, meta={"config_sha256": "c"})


def test_generation_roundtrip_and_pointer_follows_latest(tmp_path):
    g1 = _make_generation(tmp_path, 1)
    g2 = _make_generation(tmp_path, 2)
    gdir, manifest = read_generation(tmp_path, expect_components=("policy", "optimizer", "world"))
    assert gdir == g2 and manifest["update"] == 2 and g1 != g2
    obj = torch.load(gdir / manifest["components"]["world"]["file"], weights_only=False)
    check_embedded("world", obj, manifest)


def test_swapped_optimizer_or_world_file_between_generations_is_rejected(tmp_path):
    g1 = _make_generation(tmp_path, 1)
    g2 = _make_generation(tmp_path, 2)
    # copy generation 1's world into generation 2: hash no longer matches generation 2's manifest
    (g2 / "world.pt").write_bytes((g1 / "world.pt").read_bytes())
    with pytest.raises(GenerationError, match="hash mismatch"):
        read_generation(tmp_path)
    # even with a "repaired" manifest the embedded generation id betrays the swap
    _, m1 = read_generation(g1)
    m2 = json.loads((g2 / "manifest.json").read_text())
    m2["components"]["world"]["sha256"] = hashlib.sha256((g2 / "world.pt").read_bytes()).hexdigest()
    (g2 / "manifest.json").write_text(json.dumps(m2))
    with pytest.raises(GenerationError, match="manifest hash differs"):   # the commit pointer pins the manifest too
        read_generation(tmp_path)
    obj = torch.load(g2 / "world.pt", weights_only=False)
    with pytest.raises(GenerationError, match="belongs to generation"):
        check_embedded("world", obj, m2)


def test_incomplete_generation_without_manifest_or_component_is_not_loadable(tmp_path):
    g = _make_generation(tmp_path, 1)
    (g / "optimizer.pt").unlink()
    with pytest.raises(GenerationError, match="missing"):
        read_generation(tmp_path)
    (g / "manifest.json").unlink()
    with pytest.raises(GenerationError, match="no manifest"):
        read_generation(g)


def test_failed_write_leaves_previous_generation_current(tmp_path):
    good = _make_generation(tmp_path, 1)

    def boom(path, gid):
        raise OSError("disk full")

    with pytest.raises(OSError):
        write_generation(tmp_path, 2, {"policy": _writer({}), "optimizer": boom, "world": _writer({})}, {})
    assert read_generation(tmp_path)[0] == good


@needs_rom
def test_loop_penalty_scale_works_in_strict_runs_and_default_is_unchanged():
    """``disable_loop_penalty`` only acts with the route oracle on; ``loop_penalty_scale`` must work in strict runs."""
    import tools.train_deep_ppo as t
    from src.train_config import from_dict, long_v9
    base = dict(supply_injection=False, escape_assist=False, auto_dialogue=False, task_target="falkner_strict",
                path_progress=False, goal_potential=False, route_ckpt=False, retreat_penalty=False,
                stagnation_penalty=False, swarm=False, battle_archive=False, battle_lowhp_weight=0,
                disable_bump_penalty=False, disable_loop_penalty=False,
                dwell_exempt_loop=False)   # the below-level-target exemption would zero the term and hide the scale
    assert from_dict(base, long_v9()).loop_penalty_scale == 1.0
    out = {}
    for scale in (1.0, 0.5, 0.0):
        cfg = from_dict({**base, "loop_penalty_scale": scale}, long_v9())
        w = t.EnvWorker(0, ["new_bark_outside"], 50, t.worker_env_cfg(cfg, None))
        try:
            w.reset()
            st = w.reader.parse(w.emu.frame)
            w.tile_visit_counts[(st.map_group, st.map_number, st.x, st.y)] = 30   # a heavily revisited tile
            *_, info = w.step(5)                                                   # B press: stays on the same tile
            out[scale] = info["reward_components"]["loop"]
        finally:
            w.close()
    assert out[1.0] == pytest.approx(-0.08) and out[0.5] == pytest.approx(-0.04) and out[0.0] == 0.0
    with pytest.raises(Exception):
        from_dict({**base, "loop_penalty_scale": -1.0}, long_v9())


# ---------------------------------------------------------------- health windows / stop gate (pure, synthetic trace)

def _step(update, rs, env, battle, action, hp=((19, 19),), mask_a=True, done=False, reason=None, frames=20):
    ram = [0.0] * 81
    from src.deep_policy import BATTLE_FLAG_INDEX
    ram[BATTLE_FLAG_INDEX] = 1.0 if battle else 0.0
    return {"event": "step", "update": update, "rollout_step": rs, "env": env, "action": action,
            "ram_observation": ram, "mask": [True, True, True, True, mask_a, True], "done": done,
            "outcome": {"fun": {"ph": [list(h) for h in hp]}, "end_reason": reason,
                        "transition": {"frame_before": 0, "frame_after": frames}}}


def _stalled_window(first, n=40):
    return [_step(first + k, 0, 0, True, 5) for k in range(n)]   # always in battle, A legal, never pressed, HP static


def test_health_window_flags_the_overnight_failure_pattern_and_stops_after_two_windows(tmp_path):
    from tools.health_window import summarize, verdict
    steps = _stalled_window(0) + _stalled_window(40)
    wins = summarize(steps, window=40)
    assert [w["stalled"] for w in wins] == [True, True]
    assert wins[0]["battle_occupancy"] == 1.0 and wins[0]["legal_a_share_in_battle"] == 1.0
    assert wins[0]["a_press_rate_in_battle"] == 0.0 and wins[0]["encounters_completed"] == 0
    assert verdict(wins)["action"] == "stop_arm_for_diagnosis"
    assert verdict(wins[:1])["action"] == "continue"          # one bad window is a reassessment, not a stop


def test_health_window_catches_the_round299_shape_one_lone_exit_among_four_envs():
    """Real round-299 trace: 4 envs, 4,373/4,608 decisions in battle, 1 A press, exactly ONE battle exit in the window.
    A literal 'zero completed encounters' rule would miss it; fewer-than-one-exit-per-env must flag it."""
    from tools.health_window import summarize
    steps = []
    for env in range(4):
        for k in range(40):
            steps.append(_step(k, 0, env, True, 5))
    steps.append(_step(39, 1, 0, False, 5))        # env 0 leaves its battle once, then nothing else changes
    w = summarize(steps, window=40)[0]
    assert w["envs"] == 4 and w["encounters_completed"] == 1 and w["battle_party_hp_changes"] == 0
    assert w["battle_occupancy"] > 0.9 and w["stalled"]


def test_health_window_does_not_stall_when_the_policy_progresses():
    from tools.health_window import summarize, verdict
    healthy = (_stalled_window(0, 30)
               + [_step(30, 0, 0, True, 4), _step(31, 0, 0, True, 4, hp=((12, 19),)), _step(32, 0, 0, False, 2)])
    wins = summarize(healthy, window=40)
    w = wins[0]
    assert w["battle_a_presses"] == 2 and w["battle_party_hp_changes"] == 1 and w["encounters_completed"] == 1
    assert not w["stalled"] and verdict(wins)["action"] == "continue"
    # A windows alternating stalled / healthy never reaches two in a row
    mixed = summarize(_stalled_window(0) + healthy[:0] + [_step(40 + k, 0, 0, False, 2) for k in range(40)]
                      + _stalled_window(80), window=40)
    assert [x["stalled"] for x in mixed] == [True, False, True] and verdict(mixed)["action"] == "continue"


def test_health_window_drops_trace_entries_abandoned_by_an_exact_resume(tmp_path):
    from tools.health_window import read_steps
    rows = [_step(0, 0, 0, False, 2), _step(1, 0, 0, False, 2), _step(2, 0, 0, False, 2),   # update 2 will be redone
            {"event": "resume", "from_update": 2},
            _step(2, 0, 0, False, 3), _step(3, 0, 0, False, 3)]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    steps, other = read_steps(p)
    assert [(s["update"], s["action"]) for s in steps] == [(0, 2), (1, 2), (2, 3), (3, 3)]
    assert other[0]["event"] == "resume"


# ---------------------------------------------------------------- real emulator: worker continuation

def _worker_cfg(**extra):
    cfg = dict(supply_injection=False, escape_assist=False, auto_dialogue=False, task_target="falkner_strict",
               action_mask=True, sticky_prob=0.25, jitter_max=3, seed=77, time_feature=True)
    cfg.update(extra)
    return cfg


def _digest(worker, out):
    screen, ram, rew, done, maps, badges, info = out
    t = info["transition"]
    return (hashlib.sha256(screen.tobytes()).hexdigest(), hashlib.sha256(np.asarray(ram).tobytes()).hexdigest(),
            round(float(rew), 9), bool(done), maps, badges, tuple(info["map"]), info["x"], info["y"], info["ep_steps"],
            info["executed_action"], t["frame_before"], t["frame_after"], info.get("amask"),
            tuple(sorted((k, round(v, 9)) for k, v in info["reward_components"].items())))


@needs_rom
def test_worker_snapshot_restore_replays_identically_across_an_episode_boundary():
    import tools.train_deep_ppo as t
    actions = [2, 2, 3, 4, 0, 1, 4, 5, 2, 3, 4, 4, 0, 2, 5, 3, 1, 4, 2, 2, 3, 0, 4, 1]
    a = t.EnvWorker(0, ["new_bark_outside"], max_steps=9, cfg=_worker_cfg())
    b = t.EnvWorker(0, ["new_bark_outside"], max_steps=9, cfg=_worker_cfg())
    try:
        a.reset()
        for act in actions[:7]:
            a.step(act)
        blob = a.snapshot()                              # mid-episode boundary
        reference = [_digest(a, a.step(act)) for act in actions[7:]]
        b.reset()                                        # b starts somewhere else entirely...
        for act in (4, 4, 4):
            b.step(act)
        b.restore(blob)                                  # ...and is put back into a's world
        replay = [_digest(b, b.step(act)) for act in actions[7:]]
        assert replay == reference
        assert any(r[3] for r in reference), "the replayed window must cross an episode boundary (reset with jitter RNG)"
    finally:
        a.close()
        b.close()


@needs_rom
def test_restored_worker_does_not_repay_loaded_milestones_or_events():
    import tools.train_deep_ppo as t
    a = t.EnvWorker(0, ["new_bark_outside"], max_steps=200, cfg=_worker_cfg(jitter_max=0, sticky_prob=0.0))
    b = t.EnvWorker(0, ["new_bark_outside"], max_steps=200, cfg=_worker_cfg(jitter_max=0, sticky_prob=0.0))
    try:
        a.reset()
        for act in (2, 2, 3, 3, 4):
            a.step(act)
        seen = (set(a.seen_events), set(a.seen_maps), dict(a.budget), a.falkner_eligible)
        b.restore(a.snapshot())
        assert (set(b.seen_events), set(b.seen_maps), dict(b.budget), b.falkner_eligible) == seen
        _, _, rew, _, _, _, info = b.step(4)
        assert info["reward_components"].get("event", 0.0) == 0.0 and abs(rew) < 5.0
    finally:
        a.close()
        b.close()


@needs_rom
def test_one_episode_runs_past_2560_decisions_when_horizon_is_4096():
    """The overnight runner discarded every episode at 2560 decisions/env.  A real worker with max_steps=4096 must keep
    one unbroken episode beyond that, including across a snapshot/restore in the middle."""
    import tools.train_deep_ppo as t
    w = t.EnvWorker(0, ["new_bark_outside"], max_steps=4096, cfg=_worker_cfg(jitter_max=0, sticky_prob=0.0, action_mask=False))
    w2 = t.EnvWorker(0, ["new_bark_outside"], max_steps=4096, cfg=_worker_cfg(jitter_max=0, sticky_prob=0.0, action_mask=False))
    try:
        w.reset()
        for i in range(1300):
            w.step(5)                                    # B presses: idle, cannot end the episode
        w2.restore(w.snapshot())
        last = None
        for i in range(1300):
            last = w2.step(5)
        assert w2.steps == 2600 and not last[3]
        assert last[6]["ep_steps"] == 2600 and last[6]["ep_len"] == 4096 and last[6]["end_reason"] is None
    finally:
        w.close()
        w2.close()


@needs_rom
def test_sampled_evaluation_reports_frames_decisions_and_provenance(tmp_path):
    from tools.evaluate_policy import run_sampled_episode
    import tools.train_deep_ppo as t
    w = t.EnvWorker(0, ["new_bark_outside"], max_steps=6, cfg=_worker_cfg(jitter_max=0, sticky_prob=0.0))
    try:
        ep = run_sampled_episode(w, None, seed=20001)   # uniform-random actions: no network needed
    finally:
        w.close()
    assert ep["decisions"] == ep["steps"] == 6 and ep["model_calls"] == 0
    assert ep["frames"] >= 20 * 6, "every strict action consumes at least 20 emulator frames"
    assert ep["task_success"] is False and ep["end_reason"] == "time_limit"
    assert ep["whiteout_events"] == 0   # counted from the -15 reward term, so it also works when whiteout does not end the episode


# ---------------------------------------------------------------- end to end: trainer interrupted vs uninterrupted

def _tiny_cfg(tmp_path, name, updates, **extra):
    from src.train_config import from_dict, long_v9
    d = dict(supply_injection=False, escape_assist=False, auto_dialogue=False, task_target="falkner_strict",
             path_progress=False, goal_potential=False, route_ckpt=False, retreat_penalty=False, stagnation_penalty=False,
             disable_bump_penalty=False, disable_loop_penalty=False, swarm=False, battle_archive=False, jitter_max=2,
             sticky_prob=0.1, stage_balanced=False, stagger_episodes=False, episode_len_jitter=0, battle_a_bias=0,
             periodic_eval=False, envs=2, rollout_steps=8, max_steps=20, updates=updates, seed=4242,
             reward_scale="none", warmup_rollout=0, aux_loss=False, start_states=["new_bark_outside"],
             # non-trivial schedules on purpose: a reporting boundary that reset any of them would change the weights
             lr_end_frac=0.5, lr_schedule_updates=4, entropy_start=0.03, entropy_end=0.01, entropy_anneal_updates=4,
             intrinsic_decay_updates=4, intrinsic_weight_end=0.5, entropy_boost=False, battle_lowhp_weight=0,
             name="continuity_test", run_id=name, save_path=str(tmp_path / f"{name}.pt"),
             trace_path=str(tmp_path / f"{name}-trace.jsonl"), control_dir=str(tmp_path / f"{name}-control"),
             generation_every=2, snapshot_every=2, num_minibatches=2, epochs=1)
    d.update(extra)
    return from_dict(d, long_v9())


def _run(cfg, monkeypatch, cpu=True, **kw):
    import tools.train_deep_ppo as t
    if cpu:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)   # CPU: bitwise-reproducible float math
    t.train(cfg, states=["new_bark_outside"], live=False, runs_dir=None, metrics_path=str(Path(cfg.save_path).with_suffix(".metrics.jsonl")), **kw)


def _trace_steps(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "step":
            rows.append((e["update"], e["rollout_step"], e["env"], e["action"], round(e["reward"], 9), e["done"],
                         tuple(round(p, 7) for p in e["probabilities"])))
    return rows


def _weights_equal(a, b):
    sa, sb = torch.load(a, map_location="cpu", weights_only=False), torch.load(b, map_location="cpu", weights_only=False)
    sa, sb = sa.get("state_dict", sa), sb.get("state_dict", sb)
    assert sa.keys() == sb.keys()
    return all(torch.equal(sa[k], sb[k]) for k in sa if torch.is_tensor(sa[k]))


@needs_rom
def test_interrupted_and_resumed_training_equals_uninterrupted_training(tmp_path, monkeypatch):
    # Arm U: four updates in one process.
    cfg_u = _tiny_cfg(tmp_path, "uninterrupted", 4)
    _run(cfg_u, monkeypatch)
    # Arm R: two updates, process "dies" (train() returns), exact resume from the verified generation for two more.
    cfg_r1 = _tiny_cfg(tmp_path, "resumed", 2)
    _run(cfg_r1, monkeypatch)
    cfg_r2 = _tiny_cfg(tmp_path, "resumed", 2)
    _run(cfg_r2, monkeypatch, resume_generation="auto")

    u = _trace_steps(tmp_path / "uninterrupted-trace.jsonl")
    r = _trace_steps(tmp_path / "resumed-trace.jsonl")
    assert len(u) == 4 * 8 * 2
    assert r == u, "actions, probabilities, rewards and dones must match the uninterrupted run step for step"
    assert _weights_equal(Path(cfg_u.save_path), Path(cfg_r2.save_path))
    # LR / entropy coefficient / novelty weight per update are identical across the interruption (and really do change)
    def schedule(name):
        rows = [json.loads(l) for l in (tmp_path / f"{name}.metrics.jsonl").read_text().splitlines() if l.strip()]
        return [(r["update"], round(r["lr"], 12), r["entropy_coef"], r["intrinsic_weight"]) for r in rows]
    su, sr = schedule("uninterrupted"), schedule("resumed")
    assert [s[0] for s in su] == [0, 1, 2, 3] and sr == su
    # lr and novelty weight genuinely move; the entropy coefficient is route-progress driven (progress stays 0 in 4 updates, so
    # it correctly stays at its start value) but is still compared above (sr == su) rather than assumed.
    assert len({s[1] for s in su}) == 4 and len({s[3] for s in su}) > 1

    # resume marker + counters + statuses are truthful
    events = [json.loads(l) for l in (tmp_path / "resumed-trace.jsonl").read_text().splitlines()]
    assert any(e["event"] == "resume" and e["from_update"] == 2 for e in events)
    status = json.loads((tmp_path / "resumed-control" / "status.json").read_text())
    ustatus = json.loads((tmp_path / "uninterrupted-control" / "status.json").read_text())
    assert status["phase"] == "completed" and status["resume_kind"] == "exact_generation" and status["exact_continuation"]
    assert status["counters"]["train_decisions"] == ustatus["counters"]["train_decisions"] == 4 * 8 * 2
    assert status["counters"]["train_frames"] == ustatus["counters"]["train_frames"] > status["counters"]["train_decisions"]
    assert status["counters"]["external_restarts"] == 1
    assert status["budget"] == {"updates_this_invocation": 2, "completed_this_invocation": 2, "remaining": 0}
    assert status["completed_updates_total"] == 4 and status["owner_pid"] and "child_pids" in status
    assert status["last_snapshot"]["completed_updates"] == 4
    # immutable snapshots exist for 0, 2, 4 and are never rewritten
    snaps = sorted(p.name for p in (tmp_path / "resumed-control" / "snapshots").glob("update-*.pt"))
    assert snaps == ["update-000000.pt", "update-000002.pt", "update-000004.pt"]
    assert not (tmp_path / "resumed-control" / "RUNNING.lock").exists()


@needs_rom
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_exact_resume_on_cuda_restores_gpu_rng_and_sampled_actions(tmp_path, monkeypatch):
    """GPU limit, measured: two independent fresh CUDA runs with the same seed already differ in weights after 2 updates
    (BatchNorm running stats by ~1e-1; cuDNN/atomic nondeterminism, no resume involved).  So bitwise equality is only
    claimed on CPU (test above).  On CUDA we check what resume can promise: world/RNG/recurrent state continue, so the
    sampled actions and (near-)probabilities over the window agree with the uninterrupted run."""
    _run(_tiny_cfg(tmp_path, "gpu-u", 4), monkeypatch, cpu=False)
    _run(_tiny_cfg(tmp_path, "gpu-r", 2), monkeypatch, cpu=False)
    _run(_tiny_cfg(tmp_path, "gpu-r", 2), monkeypatch, cpu=False, resume_generation="auto")
    u, r = _trace_steps(tmp_path / "gpu-u-trace.jsonl"), _trace_steps(tmp_path / "gpu-r-trace.jsonl")
    assert len(u) == len(r) == 64
    assert [row[:4] for row in r] == [row[:4] for row in u]          # (update, step, env, action) identical
    assert max(abs(a - b) for ra, rb in zip(r, u) for a, b in zip(ra[6], rb[6])) < 5e-3


@needs_rom
def test_stop_marker_budget_and_low_disk_leave_terminal_status_and_start_nothing(tmp_path, monkeypatch):
    import tools.train_deep_ppo as t
    # STOP present before launch: no worker is spawned, no checkpoint is written
    cfg = _tiny_cfg(tmp_path, "stopped", 2)
    (Path(cfg.control_dir)).mkdir(parents=True)
    (Path(cfg.control_dir) / "STOP").write_text("")
    spawned = []
    real = t.VectorEnv
    monkeypatch.setattr(t, "VectorEnv", lambda *a, **k: spawned.append(1) or real(*a, **k))
    _run(cfg, monkeypatch)
    status = json.loads((Path(cfg.control_dir) / "status.json").read_text())
    assert status["phase"] == "stopped" and not spawned and not Path(cfg.save_path).exists()
    assert not (Path(cfg.control_dir) / "RUNNING.lock").exists()
    # low disk: halts visibly with a nonzero exception and a terminal status; nothing is spawned
    cfg2 = _tiny_cfg(tmp_path, "lowdisk", 2, min_free_gib=10 ** 9)
    with pytest.raises(RunHalted):
        _run(cfg2, monkeypatch)
    s2 = json.loads((Path(cfg2.control_dir) / "status.json").read_text())
    assert s2["phase"] == "halted" and s2["halt_reason"] == "low_disk" and not spawned


@needs_rom
def test_stop_marker_mid_run_saves_at_an_update_boundary_and_a_generation(tmp_path, monkeypatch):
    import tools.train_deep_ppo as t
    cfg = _tiny_cfg(tmp_path, "midstop", 6, generation_every=0, snapshot_every=0)
    stop = Path(cfg.control_dir) / "STOP"
    original = t.ppo_update

    def stop_after_second_update(*a, **k):
        out = original(*a, **k)
        if not hasattr(stop_after_second_update, "n"):
            stop_after_second_update.n = 0
        stop_after_second_update.n += 1
        if stop_after_second_update.n == 2:
            stop.write_text("")
        return out

    monkeypatch.setattr(t, "ppo_update", stop_after_second_update)
    _run(cfg, monkeypatch)
    status = json.loads((Path(cfg.control_dir) / "status.json").read_text())
    assert status["phase"] == "stopped" and status["stop_reason"] == "stop_marker"
    assert status["last_completed_update"] == 1 and status["last_generation"]["update"] == 1
    _, manifest = read_generation(Path(cfg.control_dir) / "generations", ("policy", "optimizer", "world"))
    assert manifest["update"] == 1


@needs_rom
def test_source_change_halts_at_a_boundary_and_exact_resume_refuses_changed_code(tmp_path, monkeypatch):
    import tools.train_deep_ppo as t
    real = t.code_fingerprint
    calls = {"n": 0}

    def drifting(root):
        calls["n"] += 1
        fp = dict(real(root))
        if calls["n"] >= 3:                       # call 1 = launch, 2 = before update 0, 3 = before update 1
            fp["src/injected.py"] = "0" * 64
        return fp

    monkeypatch.setattr(t, "code_fingerprint", drifting)
    cfg = _tiny_cfg(tmp_path, "drift", 4)
    with pytest.raises(RunHalted) as exc:
        _run(cfg, monkeypatch)
    assert exc.value.reason == "source_changed"
    status = json.loads((Path(cfg.control_dir) / "status.json").read_text())
    assert status["phase"] == "halted" and status["halt_reason"] == "source_changed"
    assert status["last_completed_update"] == 0 and status["last_generation"]["update"] == 0   # saved exactly at the boundary
    assert not (Path(cfg.control_dir) / "RUNNING.lock").exists()
    # a different source tree can no longer claim an exact continuation of that generation
    monkeypatch.setattr(t, "code_fingerprint", lambda root: {**real(root), "src/other.py": "1" * 64})
    with pytest.raises(RuntimeError, match="source files differ"):
        _run(_tiny_cfg(tmp_path, "drift", 4), monkeypatch, resume_generation="auto")


@needs_rom
def test_unexpected_failure_leaves_failed_status_releases_the_lock_and_starts_no_retry(tmp_path, monkeypatch):
    import tools.train_deep_ppo as t
    cfg = _tiny_cfg(tmp_path, "boom", 4, generation_every=0, snapshot_every=0)
    calls = {"n": 0}
    real_update = t.ppo_update

    def explode_on_second(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise MemoryError("simulated CUDA OOM")      # no silent retry / batch-size change is allowed
        return real_update(*a, **k)

    monkeypatch.setattr(t, "ppo_update", explode_on_second)
    with pytest.raises(MemoryError):
        _run(cfg, monkeypatch)
    status = json.loads((Path(cfg.control_dir) / "status.json").read_text())
    assert status["phase"] == "failed" and "MemoryError" in status["error"]
    assert calls["n"] == 2, "the failed update must not be retried"
    assert not (Path(cfg.control_dir) / "RUNNING.lock").exists()
    # no generation was written for the half-finished update: only completed-update boundaries are resumable
    assert not (Path(cfg.control_dir) / "generations").exists()


@needs_rom
def test_exact_resume_rejects_changed_config_and_never_falls_back_to_fresh_optimizer(tmp_path, monkeypatch):
    cfg = _tiny_cfg(tmp_path, "strict", 2)
    _run(cfg, monkeypatch)
    changed = _tiny_cfg(tmp_path, "strict", 2, lr=1e-4)
    with pytest.raises(RuntimeError, match="config differs"):
        _run(changed, monkeypatch, resume_generation="auto")
    # delete the optimizer component: refused, never a fresh optimizer
    gdir, manifest = read_generation(Path(cfg.control_dir) / "generations")
    (gdir / manifest["components"]["optimizer"]["file"]).unlink()
    with pytest.raises(RuntimeError, match="resume-generation"):
        _run(_tiny_cfg(tmp_path, "strict", 2), monkeypatch, resume_generation="auto")
