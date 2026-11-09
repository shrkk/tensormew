# Route 29 Phase 2 — bounded warm-start vs fresh comparison (PREPARED, NOT RUN)

**Nothing here has been executed.** Training starts only when you explicitly ask for it. Configs: [warm.json](./warm.json), [fresh.json](./fresh.json). Infrastructure/evidence: [route29-phase1-report.md](../route29-phase1-report.md). Contract: [route29-experiment-spec.md](../route29-experiment-spec.md) §3.

## Design (one factor changed: initialization)

| | Arm W (warm) | Arm F (fresh) |
|---|---|---|
| Init | audited `round-0299.pt` weights (sha256 `3c345f02…283645a`), **fresh optimizer**, fresh worlds — a *warm start, not a continuation* | random init, same architecture, fresh optimizer |
| Everything else | identical (config identity hash `97ea0027fb85…` is the same for both) | |
| Observation / action | corrected pixels, 81 inputs, 6 buttons; supply_injection, escape_assist, auto_dialogue, path_progress, goal_potential all **false** | |
| Training budget | **384 updates × 4 envs × 64 steps = 98,304 decisions per arm**, 4,096-decision episodes, workers never reset at a reporting boundary | |
| Evaluation | sampled, isolated, from immutable snapshots at updates **0, 192, 384**; dev seeds 10101/10102/20001/20002 → 4 × ≤4,096 = ≤16,384 decisions per snapshot, ≤49,152 per arm (outside the training cap) | |
| Reserved seeds | 50001/50002 **never used** here | |

### Declared factors, vs the overnight round-299 effective config (everything that differs, excluding paths)

`entropy_boost True→False` · `entropy_end 0.005→0.02` (so entropy coefficient is a **constant 0.02**) · `lr_end_frac 0.33→1.0` (**constant LR 3e-4**, no per-chunk sawtooth) · `updates 40→384` · `seed 20261109→20261110` · `name` · plus new run-control fields (`snapshot_every 192`, `generation_every 40`, `min_free_gib 20`, `lr_schedule_updates 384`). Unchanged and shared by both arms: the preset's rewards and masks, including the novelty-scale decay (a persistent function of the global update index, 1.0 → ~0.71 over 384 updates; it restarts relative to the overnight run but identically in both arms). The 4096 horizon is now actually reached (no 2,560-decision chunk truncation).

Initialization seed for F is the run seed 20261110 (recorded in `manifest.json`); W records the parent hash. Same seed *sets*, not identical trajectories (they diverge as policies diverge): compare task metrics, not paths.

## Pre-flight (read-only; all verified at hand-off, repeat right before launching)

```powershell
$W  = 'C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight'
$R  = "$W/work/tensormew-main"; $PY = "$W/work/audit-env/Scripts/python.exe"; $O = "$W/outputs"
$X  = "$O/experiment-log/claude-route29-20261109"
Set-Location -LiteralPath $R
git status --short; git log -1 --oneline            # expect Phase 1 patch applied; decide on commit first (source change mid-run halts the run)
nvidia-smi --query-gpu=memory.used,memory.total --format=csv   # was 999/16303 MiB, Ollama idle
(Get-PSDrive C).Free/1GB                            # was 634 GiB; guard halts below 20 GiB
Get-FileHash "$O/experiment-log/falkner-baseline/round-0299.pt" -Algorithm SHA256
```

> **Commit or freeze the source before launching.** The run fingerprints `src/*.py` + `tools/*.py` at launch and halts at the next update boundary if any file changes. Exact resume also refuses a different source tree.

## Launch (sequential; one arm at a time keeps GPU/CPU contention out of the comparison)

```powershell
# Arm W
& $PY -m tools.train_deep_ppo --config "$O/route29-phase2/warm.json" --init-weights "$O/experiment-log/falkner-baseline/round-0299.pt"
# Arm F
& $PY -m tools.train_deep_ppo --config "$O/route29-phase2/fresh.json"
```

Do **not** pass `--push-weights`. Do not use `tools.local_falkner_loop` (old chunk-restart runner) and do not remove the historical `experiment-log/falkner-baseline/STOP`.

Outputs per arm under `$X/<arm>/`: `status.json` (heartbeat), `manifest.json` (code commit, source/ROM/state hashes, dependencies, config), `steps.jsonl` (per-decision trace), `snapshots/update-000000|192|384.pt` + `snapshots.jsonl` (immutable), `generations/` (exact-continuation checkpoints every 40 updates + at stop/end), `policy.pt` (latest), `STOP` (yours to create).

## Stop / pause / recover

* **Cooperative stop:** `New-Item "$X/<arm>/STOP"`. The trainer finishes the current update, saves weights + an exact generation, and exits with `phase: stopped`. Nothing new launches. Remove that file before a later resume.
* **Visible halts** (exit code 3, `phase: halted`, `halt_reason`): `low_disk` (<20 GiB), `source_changed`. Unexpected exceptions (e.g. OOM): `phase: failed`, **no retry and no batch-size/precision change**. A halt/failure leaves the last 40-update generation resumable (halts at an update boundary also save a fresh one).
* **Exact resume** (remaining budget = `status.json → budget.remaining`):

```powershell
& $PY -m tools.train_deep_ppo --config "$O/route29-phase2/<arm>.json" --resume-generation auto --updates <remaining>
```

  Refuses on any config/source/schema/hash mismatch. Decisions after the last generation are recorded as `abandoned_decisions`; the trace carries a `resume` marker. Note: after a crash the new process takes over a stale `RUNNING.lock` only if its owner PID is dead; a live owner is never displaced.
* Never kill unrelated Hermes/Claude/Ollama processes; check `status.json → owner_pid / child_pids` for this run's own.

## Health gate every 40 updates (compact, read-only)

```powershell
& $PY -m tools.health_window --trace "$X/<arm>/steps.jsonl" --window 40 --out "$X/<arm>/health.json"
Get-Content "$X/<arm>/status.json"
```

Stop an arm immediately for observation/counter/schema/assistance/regression failures (check `status.json → counters`, `manifest.json → assistance`). If `gate.action` is `stop_arm_for_diagnosis` (two consecutive windows with battle occupancy > 90%, fewer than one completed encounter per env, no party-HP change, legal A almost never pressed): create STOP for that arm and diagnose rather than adding identical rounds — a diagnostic trigger only; nothing is edited. One bounded reassessment of a noisy early window is fine. At update 192 review any intermediate improvement; **do not extend past 384 automatically**. Failure is an informative result.

## Evaluation (immutable snapshots; separate process; run after each snapshot exists)

Snapshot files are named by **completed updates**: `update-000000.pt`, `update-000192.pt`, `update-000384.pt`. This loop evaluates all three for arm W (use `fresh` for arm F):

```powershell
foreach ($u in '000000','000192','000384') { foreach ($seed in 10101,20001) {
  & $PY -m tools.evaluate_policy --sampled `
    --checkpoint "$X/warm/snapshots/update-$u.pt" --config "$O/route29-phase2/warm.json" `
    --episodes 2 --max-steps 4096 --seed $seed --device cuda `
    --out-dir "$X/warm/eval" --name "u$u-seed$seed"
} }
```

(`--seed 10101 --episodes 2` runs seeds 10101,10102; `--seed 20001 --episodes 2` runs 20001,20002.) Each result JSON records `checkpoint_sha256`, seeds, `task_successes`, per-episode `frames`/`decisions`/`model_calls`, and total `compute` incl. wall time. Include evaluation compute in total-compute reporting. Add **no** extra favorable-seed evaluations without logging the protocol deviation.

## Reporting

Primary: earned Falkner event 1213 + Zephyr badge in a full base-start rollout — report the actual zeros until then. Intermediate: A-use by recognized battle context, party-HP change, encounter exits (**not** verified wins), NewBark→Route29 retention, unique tiles, Cherrygrove/event acquisition, per-component reward/cap usage, actual frames vs decisions, abandoned/restarted episodes. Best-smoothed-return files are diagnostic, never winners. Compare successes/retention/frames, not the largest shaped return. Final held-out seeds 50001/50002 only after a verified dev success.

## Estimated cost (not measured for this config)

Policy has 13.2 M parameters; each generation ≈ 150 MB (≈ 1.5 GB per arm); `steps.jsonl` roughly 12 MB per 18 updates in the overnight trace ⇒ order of a few hundred MB per arm. Wall time per arm is not estimated here: the earlier smoke runs are too small to extrapolate; read the `[time]` lines after the first updates.
