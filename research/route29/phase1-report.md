# Route 29 Phase 1 report — continuity, reproducibility, telemetry

Status: **implemented and tested; uncommitted working-tree patch for review.** No training was launched, the historical `STOP` marker is intact, no experiment directory exists, and no protected checkpoint changed (hashes re-verified at the end; see §7). Spec: [route29-experiment-spec.md](./route29-experiment-spec.md) §2. Plan for the next step: [route29-phase2/README.md](./route29-phase2/README.md).

Patch: [route29-phase1.patch](./route29-phase1.patch) — sha256 `a58eb2ba1a1756e76f07bde89817873ca46e5b5053845f34eed2fd9a52ce5ad2`, applies cleanly to a pristine `128fae8` (`git apply --check` exit 0). Base commit `128fae8`, branch `development/bugsy-weight-space-2026-11-09`.

## 1. What changed

| File | Change |
|---|---|
| `src/run_control.py` (new) | `StatusWriter` (atomic status + monotonic heartbeat), `RunLock` (live-owner vs stale, never kills anything, release only if still owner), `check_disk`, `stop_requested`, process-scoped `prevent_sleep`, `lr_for_update` (global schedule), `Counters` (decisions ≠ frames), `code_fingerprint` / `config_identity`, `write_run_manifest`. Pure Python. |
| `src/continuation.py` (new) | Transactional generations: component files + `manifest.json` (SHA256 each) + `CURRENT.json` commit pointer written last; verification chain pointer → manifest → components; embedded generation id check. |
| `tools/train_deep_ppo.py` | Worker `snapshot()/restore()`, `VectorEnv.snapshot()/restore()`; `train()` gains four explicit start modes, status/lock/STOP/disk/source-change handling, immutable snapshots, generations, global LR schedule, frame/decision/episode counters, trace `resume` marker; new CLI `--resume-generation DIR\|auto`, `--init-weights PT`; exit code 3 on a visible halt. |
| `src/emulator.py` | `set_frame_counter()` (relabels the wrapper counter after an exact-continuation load; does not touch the machine). Headless render fix from `128fae8` is untouched. |
| `src/train_config.py` | New fields (all default off ⇒ historical behaviour): `control_dir`, `snapshot_every`, `generation_every`, `min_free_gib`, `lr_schedule_updates`; validation. |
| `tools/evaluate_policy.py` | Sampled evaluation now also reports `frames`, `decisions`, `model_calls`, `checkpoint_sha256`, `seeds`, `task_successes`, wall time. Evaluation semantics unchanged. |
| `tools/health_window.py` (new) | Read-only 40-update summaries of a trace + the spec's stop-gate verdict. |
| `tests/test_continuity.py` (new) | 24 tests (below). |

`tools/local_falkner_loop.py` (the old chunk-restart runner) is **deliberately unchanged and must not be used**: it still starts a new trainer process every 40 updates, so it still discards worlds/GRU state. The new path is one long-lived trainer; reporting boundaries are only checkpoint/snapshot events inside it.

## 2. Start modes (what is and is not claimed)

`status.json` records `resume_kind` and `exact_continuation` so a restart is never mislabeled.

| Mode | Flag | Continuation? |
|---|---|---|
| `fresh_init` | (none) | n/a |
| `warm_start_fresh_optimizer` | `--init-weights PT` | **No.** Weights only; fresh Adam; fresh worlds. Refuses `PT == save_path`. |
| `weights_optimizer_only_restart` | legacy `--resume` | **No.** Optimizer sidecar only; worlds/GRU/RNG/counters restart. Unchanged legacy behaviour, now labeled. |
| `exact_generation` | `--resume-generation auto` | **Yes**, from a verified generation (below). |

## 3. Generation schema (`<control_dir>/generations/gen-<update>-<id>/`)

Written at completed-update boundaries (every `generation_every`, and always on STOP / halt / budget end). No outstanding PPO batch is ever checkpointed. Real example sizes: `policy.pt` 50.6 MB, `optimizer.pt` 101.1 MB, `world.pt` 0.8 MB (~150 MB/generation; ~10 per 384-update arm).

* `policy.pt` — `policy.save` format (weights + BatchNorm buffers), unchanged format.
* `optimizer.pt` — AdamW state; update index; trainer state: route-progress tracker, reward scaler, entropy-guard state, last entropy, best/recent scores, 8-update reward window, novelty weight, milestone/source-split stats, **counters**, and **RNG states** (Python, NumPy, Torch CPU, Torch CUDA per device).
* `world.pt` — per worker: emulator save bytes + wrapper frame counter, all per-episode state (seen tiles/maps/events/warps/interactions, visit counts, reward-budget/cap usage, goal-eligibility and Bugsy ledgers, dwell/idle/stagnation/heal trackers, episode step/limit, last action/mask), the worker's NumPy generator state and selector counter; plus the current observation, mask, and GRU hidden state.
* `manifest.json` — generation id, update, SHA256+size of each component, schema version, config identity hash (excludes only `updates`, paths, run id, control/snapshot cadence), source fingerprint, obs/action/GRU schema, envs/rollout shape, counters.
* `CURRENT.json` — commit pointer, written last; includes the manifest hash.

Resume **refuses** (never silently degrades) on: missing/corrupt/swapped component, pointer/manifest mismatch, generation id mismatch, config identity change, source-file change, schema or env-shape change, unloadable optimizer/world. Loading uses trusted local `torch.load(weights_only=False)`; never load a generation from elsewhere.

**Not captured** (by design or limit): LiveFeed/TerminalMonitor display state (rolling windows restart; `MilestoneRecords` is rebuilt from `history.jsonl`); in-memory rotation counters of swarm/battle/route archives (those are **off** in the strict configs; archive files themselves live on disk); half-completed PPO epochs (unsupported by design).

Resume accounting: decisions the interrupted process took *after* its last generation are unrecoverable; they are recorded as `abandoned_decisions` (from the last heartbeat) and `external_restarts += 1`. The append-only trace gets a `resume` marker with `from_update`; entries before it with `update >= from_update` are abandoned and `tools/health_window.py` drops them.

## 4. Telemetry example (real output of the resumed arm of a test run)

```json
{"phase": "completed", "resume_kind": "exact_generation", "exact_continuation": true,
 "owner_pid": 32740, "child_pids": [12944, 23888], "heartbeat_seq": 5, "heartbeat_monotonic_s": 2.594,
 "last_completed_update": 3, "completed_updates_total": 4,
 "budget": {"updates_this_invocation": 2, "completed_this_invocation": 2, "remaining": 0},
 "counters": {"train_decisions": 64, "train_frames": 1520, "reset_frames": 56,
              "episodes_ended": {"time_limit": 2}, "external_restarts": 1, "abandoned_decisions": 0},
 "goal_counts": {"falkner_success_episodes": 0, "bugsy_success_episodes": 0},
 "last_snapshot": {"completed_updates": 4, "sha256": "29710d9c…"}, "last_generation": {"update": 3},
 "last_evaluation": null, "stop_reason": "budget_exhausted"}
```

64 decisions consumed 1,520 emulator frames here (≈23.8 frames/decision): the two quantities are reported separately. `last_evaluation` is `null` unless an evaluation of that exact update exists; it is never carried over from an older model.

## 5. Acceptance matrix (spec §2.4)

| # | Gate | Result | Evidence |
|---|---|---|---|
| 1 | Headless pixel/WRAM/HRAM/exact-frame regression stays green; no extra tick | **Pass** | `test_headless_observations.py` in the affected suite; `Emulator.tick` untouched. |
| 2 | Cross a boundary: position/RAM/episode/mask/GRU continue; episode beyond 2,560 decisions at horizon 4,096 | **Pass** | Real worker run to 2,600 steps of one episode (`ep_len 4096`, no end) with a snapshot/restore at step 1,300; end-to-end trainer test shows identical per-step trace across the boundary. |
| 3 | Uninterrupted vs save/reload matches | **Pass on CPU (bitwise); GPU limited — see §6** | CPU: 4 updates vs 2+exact-resume+2: identical actions, probabilities, rewards, dones **and** final weights. Worker level: identical replay across an episode reset incl. jitter/sticky RNG. |
| 4 | Swapped optimizer/world or mismatched hash/config/schema rejected | **Pass** | Swapped world file, repaired manifest (caught by pointer + embedded id), deleted optimizer, config change, source change, partial generation. No real snapshot corrupted. |
| 5 | LR/entropy/novelty identical across boundaries; evaluation isolation | **Pass (schedules) / by construction (isolation)** | The e2e test uses a *decaying* LR (0.5), novelty decay and entropy anneal; per-update lr/entropy/novelty equal across the interruption. Evaluation is a separate process reading an immutable snapshot, so it cannot touch training RNG/weights/optimizer/world; the old in-process `PeriodicEvaluator` is disabled in the Phase 2 configs and was not modified or newly tested. |
| 6 | STOP, failure, budget exhaustion → saved terminal status, nothing new launched | **Pass** | STOP before launch (no worker spawned, no checkpoint); STOP mid-run (boundary generation saved); low disk; source change mid-run; simulated OOM (`failed`, no retry, lock released, no generation for the half-finished update); budget end (`completed`). Historical STOP never touched. |
| 7 | Goal = Falkner event 1213 **and** Zephyr bit 0, both absent initially | **Pass (pre-existing)** | `test_bugsy_task.py` (eligibility, preloaded/other-badge cases) in the affected suite; unchanged. |
| 8 | Affected suite + report exact final result | **Pass** | §6. |

Mutation check of the key test: removing the torch RNG restore, or zeroing the GRU state on resume, each makes the CPU end-to-end test fail (then restored; no mutation remains).

## 6. Test evidence and limits (exact)

* Baseline before changes: handoff affected command = **126 passed, 2 skipped** (the "128" in the handoff counts the 2 skips).
* After changes, same command + `tests/test_continuity.py`: **149 passed, 2 skipped** at the time of the run; `test_continuity.py` has since gained one test (the failure-status test) and passes **24/24** on its own. Whole suite (last full run, before that one test was added): **872 passed, 5 skipped, 4 failed, 7 collection errors**.
* The **4 failures** (`test_deep_policy` ×2, `test_ppo` MLP, `test_visual_metrics`) are Windows temp-file lock `PermissionError`s that I reproduced on a pristine `128fae8` copy — pre-existing, same class as the one `128fae8` fixed elsewhere. The **7 errors** are `ModuleNotFoundError: pydantic` in legacy LLM-supervisor tests (not installed; I did not install packages).
* One transient `test_long_v6` failure appeared in an earlier full run; it was my own edit landing on disk while worker processes re-imported the module. It passes alone, after my tests, and in the clean full run.
* **GPU determinism limit (measured):** two independent *fresh* CUDA runs with the same seed already differ after 2 updates (BatchNorm running means by ~1e-1; cuDNN/atomic nondeterminism, no resume involved). So bitwise trajectory equality after resume is certified **on CPU only**. On CUDA the test checks what resume can promise (RNG/world/GRU continue ⇒ same sampled actions, probabilities within 5e-3, stable 3/3). A real GPU run resumed from a generation is an exact *state* continuation, but its future floats will diverge from an uninterrupted GPU run exactly as any two GPU runs do.
* Tests ran short training jobs (2–4 updates, 2 envs) against the real emulator in pytest temp directories. These are tests, not the experiment: nothing was written under `outputs/experiment-log`.

## 7. Integrity checks at hand-off

`git status`: 4 modified + 4 new files under `src/ tools/ tests/`; pre-existing untracked `outputs/` preserved. STOP present; old `RUNNING.lock` absent; 0 trainer/dashboard processes. SHA256 unchanged: `round-0299.pt` and `falkner-baseline-latest.pt` = `3c345f02…283645a`; `…latest.best.pt` = `b27ba456…e8e1f62`; `Downloads/deep_policy.pt` = `a0ab5382…d9bdcb`; `Downloads/ppo_policy.npz` = `beb18739…26e79d`. Reserved seeds 50001/50002 not used. No dependency installed; no global/Claude configuration changed.

## 8. Decisions for you

1. **Stop-gate wording.** Literal spec: "no completed encounter". The real round-299 trace has *one* exit in 4,608 decisions (94.9% battle, 1 A press), which the literal rule would miss. `health_window.py` therefore flags a window when occupancy > 90%, **fewer than one completed encounter per env**, no party-HP change, A legal but pressed < 0.5% of battle decisions; two consecutive windows ⇒ `stop_arm_for_diagnosis` (diagnostic only). Confirmed on the real trace (flags the window). PP is not in the trace, so it is not claimed. Approve or tighten.
2. **Declared schedules** (§ plan): fixed LR 3e-4, fixed entropy coefficient 0.02, entropy-boost off. The overnight run ended at coefficient 0.01864 with LR/entropy on per-invocation sawtooths; every effective difference vs the round-299 config is listed in the plan.
3. **Commit.** I did not commit or push. Say if you want it committed on the development branch.
