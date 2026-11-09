# Route 29 experiment records

Pre-registrations (written BEFORE each round launched), the exact configs of every arm, the analysis scripts, and the Phase 1 / Phase 2
documents. Large artifacts (traces, checkpoints, per-episode evaluation JSON) are NOT here; they live in the experiment-log directory of
the local workspace (`outputs/experiment-log/claude-route29-20261109/`).

* `preregistration.md` - hypotheses, single changed factors, predictions and failure modes, in order of execution; also the run-hygiene
  notes and every protocol deviation.
* `configs/*.json` - one file per arm (control arms: warm, fresh, transfer). Arms differ from their matched control in the fields listed
  in the preregistration. ASSISTED arms (`assisted_start_balls`, `path_progress`, `route_ckpt`) are labeled as such there and must never be
  mixed with the organic strict results.
* `analysis/` - read-only diagnostics. `bfs_route.py` and `route_probe.py` use the ROM route map / emulator search for DIAGNOSIS ONLY
  (never an agent input, never a training signal). `battle_baselines.py` is an evaluation-only yardstick.
