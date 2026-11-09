# Queue E1 â€” does terminating the episode on whiteout teach the agent to die?  (pre-registered before running)

Written 2026-11-09 before any E1 training. Controls already exist and are matched (same seed 20261110, budget 384 updates x 4 envs x 64 steps, config identity `97ea0027fb85`, commit `2b0d5ce`): arms `warm` and `transfer` under `outputs/experiment-log/claude-route29-20261109/`.

## Evidence that motivates it (from the three training traces, updates 284-383)
* Every arm's net reward per step is negative (step -0.01 plus loop/bump/idle penalties).
* Warm: whiteout episodes averaged return **-32** (length 1,763) vs **-131** for the one episode that ran to the time limit (length 3,186). Whiteouts rose from 3/6 episodes (updates 0-99) to 11/12 (updates 284-383).
* Transfer: 4/4 evaluation episodes at update 384 end in whiteout (98k frames total vs ~350k for its update-0 checkpoint).
* With `terminate_on_whiteout=True` a whiteout is a TRUE terminal: no bootstrap, the remaining negative per-step reward is never collected. When expected future return is below -15, ending the episode is the better policy.

## The single changed factor
`terminate_on_whiteout: true -> false`. The -15 whiteout penalty stays. In-game, a whiteout teleports the player to the last Pokemon Center with the party restored; the episode then continues. Everything else identical (rewards, masks, LR 3e-4, entropy 0.02, seeds, budget, assists all off). Two arms, each with its matched control:
* `nowo-warm`     (init: audited round-0299 weights)  vs control `warm`
* `nowo-transfer` (init: transfer init-weights)        vs control `transfer`

## Predictions (stated in advance)
* P1: late-training whiteouts per episode fall vs the matched control (warm control: 11/12 episodes).
* P2: evaluation at update 384 has at most 1 whiteout in 4 episodes (controls: warm 2/4, transfer 4/4).
* P3: the arm does NOT simply stop fighting: late battle occupancy stays above 25% (control warm ~57%, transfer ~6%).
* Progress = any evaluation episode reaching a milestone beyond index 0 (Route 29 arrival), or a Falkner win. Anything less is not progress.

## Ways this can fail (and what I will conclude)
* Whiteouts drop but battles also vanish -> the agent merely avoids danger; the incentive problem moved, not solved.
* Whiteouts unchanged -> the death-incentive explanation is not sufficient; look at battle decision quality instead.
* Non-terminal whiteout could be exploited as a free heal/teleport at -15: watch whiteouts per episode and heal events; a rise would be reported as reward hacking.
* n is tiny (4 eval episodes, one training seed per arm). A difference is a lead, not a result.

## Rules for the unattended hour
* No code edits, no new commits; the code under test is commit `2b0d5ce` (clean tree). Configs only.
* Dev seeds 10101/10102/20001/20002 only; reserved 50001/50002 never used. No weight edits, no assists, no scripted actions.
* Each arm: one 384-update training, then 3 isolated evaluations x 2 seed pairs. No retries, no extension past 384, no further experiments launched automatically.
* Old STOP marker untouched. Per-arm STOP file: `<arm>/STOP`.

---

# Round E2 (pre-registered 2026-11-09, before any E2 training) â€” after E1

## E1 outcome (for the record; full eval after the whiteout-event fix pending for nowo-transfer)
* nowo-warm: whiteouts 0 and no damage taken in evaluation (min lead HP 1.0); late training: 0% in battle, 81% of decisions inside Elm's lab. **P3 failed: the agent hides instead of fighting.** No progress beyond Route 29.
* nowo-transfer: evaluation "whiteout" flag reads 0 but that flag only fires when whiteout ends the episode (disabled here); min lead HP reached 0.0 at update 384 and training logged -45 whiteout reward (3 whiteouts) in the late window, similar to its control. **Terminating on whiteout is not the main cause.**
* Evaluator now counts whiteout events from the -15 term (commit 9d50017).

## Diagnosis driving E2
1. Per-step negatives dominate returns in every arm: step -0.01 always, plus the per-tile loop penalty (largest component: -170..-812 per 100 updates; up to -0.08/step) and wall-bump cost.
2. Fighting cannot pay: wild-battle reward is capped at 6/episode (wild_battle_budget) while ONE whiteout costs 15. A risk-neutral optimizer therefore avoids wild battles (E1 warm hid in a building).
3. `disable_loop_penalty` is a no-op in strict runs; commit 1396642 adds `loop_penalty_scale` (default 1.0 = unchanged).

## Arms (each differs from its matched control `warm` / `transfer` in exactly ONE config field; code = commit 9d50017)
* `e2a-loop0-warm`      : loop_penalty_scale 1.0 -> 0.0, warm init
* `e2b-loop0-transfer`  : loop_penalty_scale 1.0 -> 0.0, transfer init
* `e2c-wildbudget-warm` : wild_battle_budget 6.0 -> 20.0, warm init
Same seed 20261110, 384 updates x 4 envs x 64, assists off, dev seeds 10101/10102/20001/20002 only.

## Predictions
* a/b: loop reward component ~0; late battle occupancy stays > 25% for warm; fewer whiteouts per episode than control; any milestone beyond Route 29 would be progress.
* c: more battle reward collected and battle engagement stays (>25%); risk: farming wild grass (watch Route 29 share and per-episode level rewards).
* Failure/null outcomes are reported as-is. A change that merely moves the agent into another degenerate behaviour (hiding, farming, dithering) counts as a failure.

## Autonomous-loop rules (user lifted the round cap; validity rules stay)
Dev seeds only; no hand-edited weights; no assists / scripted actions; one factor per arm with a control; reserved seeds 50001/50002 only after a verified dev-seed Falkner win; 20 GiB disk guard; every code change tested and committed.

---

# Round E3 (pre-registered 2026-11-09 before any E3 training) â€” after E2 training traces

## E2 training-trace outcome (evaluations still running when this was written)
* e2a (loop 0, warm): late battle 1%, 50% of decisions inside Elm's lab, 0 whiteouts: hiding again.
* e2b (loop 0, transfer): 1 whiteout (control 3); still wanders the start maps; no progress.
* e2c (wild budget 20, warm): late battle 97%, **A = 0%**, menu cycling; the original stall returned. Fight reward size is not the limiting factor: A is never sampled, so no reward signal about fighting is ever received.

## Diagnosis driving E3
Exploration collapse in the battle menu: P(A in battle) ~ 3e-4 means A is essentially never tried. The overnight diagnosis found that a global temperature of 4 restored A presses and battle outcomes. The generic PPO lever is a larger entropy bonus. (battle_a_bias is NOT used: it is a hand-written action prior.)

## Arms (one config field vs matched control; code commit 9d50017 unchanged)
* `e3a-ent05-warm`     : entropy_start = entropy_end 0.02 -> 0.05, warm init  (control `warm`)
* `e3b-ent10-warm`     : entropy_start = entropy_end 0.02 -> 0.10, warm init  (control `warm`)
* `e3c-ent05-transfer` : entropy_start = entropy_end 0.02 -> 0.05, transfer init (control `transfer`)
Seed 20261110, 384 updates, dev seeds only, assists off.

## Predictions
* Late-training P(A | battle) rises well above control (warm control ~35%, e2c 0%); battle exits and HP changes continue.
* Risks stated in advance: high entropy makes overworld movement noisy (watch unique tiles and map coverage), could stall navigation; 0.10 may be too noisy. Progress = any evaluation episode beyond Route 29 or a Falkner win.

---

# Round E4 (organic) + A1 (ASSISTED catching) â€” pre-registered 2026-11-09 before any training, code = commit cf8ffbb

## What E2/E3 showed (training traces; evaluations of E3 still running)
* No lever so far (loop penalty off, bigger fight budget, entropy 0.05/0.10, non-terminal whiteout) produces any evaluation episode beyond Route 29. Agents move between three degenerate behaviours: stall in the battle menu, die in fights, hide in buildings.
* Entropy 0.10 fixes A-starvation (A ~25% of battle decisions) but whiteouts 8/10 episodes. The agent still lacks fight competence.
* Common cause hypothesis: the agent gets few battle decisions per update and each failure is punished; it needs far more fight practice than random grass encounters give.

## E4 (organic, no assistance): agent-earned battle curriculum
Config-only: `battle_archive=true`, `battle_archive_prob=0.5`. The worker saves emulator states at the START of battles the agent itself reached (existing mechanism, max 32 files) and half of the episode resets begin there. Evaluation is always from the base start state with no archive. Provenance limit (stated up front): archived states record emulator bytes and key only, not parent checkpoint hash/action trace as the spec's full curriculum protocol asks; results are exploratory and are not claimed as a campaign result. No manual fights, no resource injection, no imported states.
* `e4a-battlearch-warm`     vs control `warm`
* `e4b-battlearch-transfer` vs control `transfer`
Predictions: late-training battle decisions per update rise strongly; A/battle-win/HP-change statistics improve; whiteouts per episode fall; any evaluation episode beyond Route 29 would be progress. Failure modes: overfitting to archived fights with no gain on base-start evaluation; hiding persists.

## A1 (ASSISTED, labeled; never mixed with organic results): catching
The start state has no balls and no money, so no agent has ever caught anything (0 catch reward, party size 1 in 5 x 98k decisions). `assisted_start_balls=10` gives balls only (no potions/money/flags).
* `a1-ctrl-balls-warm`  : assisted_start_balls 10, otherwise the `warm` control (default catch reward 2.0, cap 4)
* `a1-catch-warm`       : same + catch_reward 5.0, catch_budget 12.0, catch_party_scale true (reward x (6 - party size)/5; full party pays 0). The three catch fields are ONE bundled design (party-size-aware catch reward).
Predictions: party size > 1 in some training episodes or evaluation episodes; catch events > 0; a bigger party reduces whiteouts. Failure/null: no agent ever throws a ball (bag-menu discovery is a long action sequence), which would itself be an informative result.
Same seed 20261110, 384 updates, dev seeds only, warm init, reserved seeds untouched.

---

# Findings after E4/A1 training traces (written before E5 launch)

* **A1 catching (ASSISTED, balls only):** with 10 balls agents DID catch: party size reached 3 in both arms; catch reward totals 6.0 (control, flat 2.0) vs 49.0 (party-size-aware, 5.0 x (6-party)/5). Catching is discoverable once balls exist; without them it is impossible (0 balls and 0 money at start; 0 catches in 5 x 98k decisions). Evaluation numbers pending.
* **E4 battle curriculum:** e4a (warm) no change (80% in battle, A 5%); e4b (transfer) engages (A 44%, 20% in battle) but whiteouts 5/10 episodes. No benefit evident.
* **Where agents actually get stuck (new, from the traces, scripts in analysis/):**
  1. Route 29 grass zone x 35-54: 75-87% of decisions there are in battle.
  2. West of x=35 there are no battles (0%). Agents that cross reach x=14 and NEVER go further: at (14,14)/(14,15) left is blocked and up (y=14) / down (y=15) is blocked: a 2-tile dead-end pocket. ~2,800+ decisions per arm there, 88% on the same tile, A pressed 65-70%.
  3. Offline ROM route map (diagnostic only, never an input): remaining distance INCREASES going west along the y=14/15 corridor; the real road leaves it northward around x~24-30 then runs west along y~10-11. So the blocker is a navigation/exploration failure, not fights.
* **Hypothesis for E5:** every novelty term (new tile 0.1, unvisited neighbour 0.02, new map, warp, curiosity) shares ONE per-episode explore budget of 8.0 (explore_budget, from long_v4). A 4,096-step episode exhausts it early; afterwards finding a new tile pays 0, so nothing rewards discovering the upper road.

# Round E5 (organic) â€” pre-registered before training, code = cf8ffbb

Single config field vs matched control `warm` / `transfer`: `explore_budget` 8.0 -> 30.0 (the long-standing default) or 100.0 (effectively uncapped).
* `e5a-explore30-warm`      vs `warm`
* `e5b-explore100-warm`     vs `warm`
* `e5c-explore30-transfer`  vs `transfer`
Predictions: share of Route 29 decisions at x < 14 or off the lower corridor rises; some episodes reach map 26:3 (Cherrygrove, milestone index 1) = progress; time in the dead-end pocket falls. Risks: reward farming in tiny novel regions (dead-end pocket tiles are finite), noisy wandering, hiding; revisit/loop penalties unchanged.

---

# E5 outcome and the Route 29 maze finding (written before E6), code now 01fb26c

* **E5 (explore_budget 30 / 100): no effect.** Evaluations: still nothing beyond Route 29. The per-episode novelty budget was not the limiting factor.
* **Route 29 geometry from the traces + offline ROM map (diagnostic only):** two dead ends and a third, longer route.
  1. Lower corridor (y=14/15) ends in a walled pocket at x=14: ~100,000 decisions across 18 arms.
  2. Upper road (y<=13) reaches x=22, where the tile to the left is a CUT TREE (collision id 0x12=18; neighbours 0x15=21 are headbutt trees): a second dead end (needs an HM the agent does not have).
  3. The goal route detours north (to about y=8) and then west (~40-step detour from (24,10)). Agents reached min y=5 but rarely follow it.
  Only 11.8k decisions were spent on the upper road vs 100k in the lower pocket.
* Per-episode novelty counts reset every episode, so each episode re-pays the same familiar dead-end tiles. Hypothesis: counts that persist across episodes make familiar tiles worthless and rarely-seen tiles valuable (generic count-based exploration, no route knowledge).

# Round E6 (organic) â€” pre-registered before training, code 01fb26c

Fields: `global_novelty` (false -> true), optionally `curiosity_scale` (0.08 -> 0.4). Matched controls `warm` / `transfer`.
* `e6a-gnov-warm`        : global_novelty true, warm init, vs `warm`
* `e6b-gnov-cur04-warm`  : global_novelty true + curiosity_scale 0.4 (one bundled "stronger global curiosity" design), warm init
* `e6c-gnov-transfer`    : global_novelty true, transfer init, vs `transfer`
Predictions: fewer decisions in the lower pocket (x<=24, y 14-15) and more on the upper/north route (y<=13, x<=34); min y < 8 at x<=34 more often; any episode reaching map 26:3 (Cherrygrove, milestone index 1) is progress. Risks: wandering/hiding, reward farming of rarely-seen tiles, fights still kill the agent in the x 35-54 grass zone before it reaches the maze.

---

# E6 outcome and the optimizer diagnosis (written before E7), code 01fb26c

* **E6 (global novelty): no help.** Maze metrics: e6a (warm) 0 decisions in the maze: stuck in the battle menu at the start (97% battle, A 0%). e6b hid (0% battle). e6c (transfer) did reach both the lower corridor and upper road about as much as its control. Evaluation: no episode beyond Route 29 (final eval files pending at the time of writing).
* **Learning-curve diagnosis (training_metrics, arms e2c/e5a/e6a/e6c):** approx_kl ~ 1e-4 and clip_frac 0.000 in almost every update against target_kl 0.05; ppo.epochs = 1, num_minibatches = 2 on 4 envs x 64 = 256 samples per update. The policy changes very little per update (so no large-step collapse) but also learns slowly from tiny, highly correlated batches; entropy/value-loss swings follow batch composition (menu-stall states vs walking), not policy jumps. The degenerate behaviour each arm settles into is whatever the first ~100 noisy updates happen to reinforce.

# Round E7 (organic) â€” pre-registered before training, code 01fb26c

* `e7a-epochs4-warm`     : ppo.epochs 1 -> 4 (more optimization per batch), warm init, vs `warm`
* `e7c-epochs4-transfer` : ppo.epochs 1 -> 4, transfer init, vs `transfer`
* `e7b-envs16-warm`      : envs 4 -> 16 (4x samples per update; 393k decisions total, so NOT compute-matched: reported as a more-data arm, compared per update and per decision), warm init, vs `warm`
Predictions: approx_kl and clip_frac rise; behaviours settle less randomly; with more data per update, battle-menu stall/hiding should be less frequent; any evaluation episode beyond Route 29 is progress. Risks: faster collapse with 4 epochs (KL runaway; early stop at target_kl 0.05 limits it); the larger-batch arm costs ~4x wall time.

---

# Round E8 (organic) â€” agent-earned frontier archive; pre-registered before training; code = worktree commit 03fecbc (on top of 01fb26c)

User-requested organic alternative to route-progress shaping: start some episodes from states the agent itself reached near the exploration frontier so it can practise the final stretch of Route 29 without rediscovering it each time.

Mechanism (src/frontier_archive.py, Go-Explore style, no route knowledge): a "cell" is (map, x, y). The first time a worker ever stands on a cell (healthy lead >= 50% HP, outside battle/dialogue) it stores the emulator state (cap 64 per worker, most-visited evicted). `frontier_prob` of episode resets start from an archived state chosen with weight 1/sqrt(visits) x 1/sqrt(1+times_selected): rarest cells first; visiting them raises their count so the frontier advances. Evaluation forces frontier_prob = 0 (tested): evaluation always starts at New Bark. frontier starts are NOT base starts (separate metrics). Provenance limit: archive states are in-process emulator bytes, not parent-checkpoint/action-trace records, so this is exploratory and not a campaign claim.

Arms (config-only vs matched controls `warm` / `transfer`; frontier_archive true, frontier_prob 0.5):
* `e8a-frontier-warm`
* `e8b-frontier-transfer`
Predictions: share of training decisions on the northern route beyond the narrow connector ((31,13)->(34,12)->(35,11)) rises vs control; visits to route tiles with index >= 54 (x <= 32 along y~7) become non-zero; any episode entering map 26:3 (Cherrygrove) is progress. Evaluation (base start only) shows whether the practised stretch transfers to full runs. Risks: archived states near fights make the agent practise death; the archive fills with dead-end pocket cells; a few strong cells dominate; improvement in frontier-start training episodes that does not transfer to base-start evaluation.

---

# E8 addendum (written after watching E8a/E8b on the dashboard); v2 code = worktree commit above on top of 03fecbc

Observation: the agent can sit in a walled-off area at the far left. Measured on three earlier arms: in the lower dead-end pocket the agent earns NO positive reward and averages -0.08 to -0.095 per step (loop -0.05, bump -0.012, step -0.01, dwell, idle press), ~9x worse than elsewhere on Route 29 (-0.009). So it is a trap with no escape gradient, not reward farming. E8a's first 7 frontier starts: (48,12) (52,11) (57,9) (47,12) (58,8) (47,9) near the east/middle plus (28,17) on the lower-corridor branch (-0.0685/step); none inside the pocket yet, but rarity-only selection can favour dead ends (they are rarely visited by construction).

v2 = `frontier_productivity` true: weight x (1+sqrt(yield)) / (1+2*failures), where failures = episodes started from the cell that discovered no new cell, yield = new cells found. Arms (vs matched controls; v1 arms e8a/e8b keep running as the rarity-only comparison):
* `e8c-frontier-prod-warm`     : frontier_archive + frontier_productivity, warm init, vs `warm` and vs `e8a`
* `e8d-frontier-prod-transfer` : same, transfer init, vs `transfer` and vs `e8b`
Predictions: fewer frontier starts in branches that lead nowhere; more discoveries per frontier episode; route tiles with index >= 54 become non-zero; any entry into map 26:3 is progress.

---

# E8 outcome, calibration result, and the fight track (written before F1); code = main 110d9e0 + battle records (5abcbf1)

* **E8 (agent-earned frontier archive, v1 rarity-only and v2 productivity-aware): negative.** No evaluation episode beyond Route 29; after 3.0M decisions scanned across all arms there are still 0 decisions in Cherrygrove (26:3) and the closest tile to the goal is unchanged (ROM route index 53 of 82, visited once).
* **Calibration (evaluation-only yardstick, never a training signal), Route 29 wild fights from route29_entry:** always pressing A in battle: 18 wins of 18 fights, 0 whiteouts, mean 86 decisions/battle, 13% HP lost per battle. Random buttons in battle: 1 win of 25 (24 escapes). So these fights are trivially winnable by A-mashing; agents lose them because they do not reliably press A. Background research (subagent, partly unverified): Route 29 wilds are level 2-3 only; Cyndaquil has only Tackle/Leer at level 5.
* New measuring tools: per-battle records (win/caught/escape/whiteout, decisions, HP lost) in the trace and the evaluator (`battle_summary`), `tools/battle_report.py`.

# F1 "fight school" (organic) â€” pre-registered before training

Goal: raise real battle win rate. Config-only bundle vs matched controls `warm` / `transfer` (the bundle is ONE design; individual fields are not isolated): `battle_archive` true with `battle_archive_prob` 0.7 (most episode resets start inside a battle the agent itself reached), `max_steps` 512 (short episodes so fights dominate the data), `wild_battle_scale` 1.0 and `wild_battle_budget` 30 (win/KO/damage reward at full size, higher cap). Evaluation is still the standard 4096-decision base-start evaluation; the success measure is the evaluator's `battle_summary` (win rate, whiteout rate, decisions per win), NOT reward.
* `f1a-fightschool-warm`, `f1b-fightschool-transfer`
Predictions: battle win rate (evaluation) well above the controls' and approaching the A-mash yardstick (1.0); whiteout rate down; no collapse into hiding (battle decisions > 0). Risks: wins only inside battle-start training episodes (no transfer to base-start walking), A-mash with no healing (HP attrition across many fights), the policy fights but never walks on.
Also: re-evaluate the final checkpoints of earlier arms with the new evaluator to get their real win rates.


---

# F1 outcome and Round R1 (route runs from the fight-school weights) — pre-registered before training; code = main 110d9e0

F1 results (battle records, standard base-start evaluation, dev seeds; 12-29 battles per checkpoint, one training run per arm): F1a (warm) win rate 38% -> 65% (u192 -> u384) vs 21% for the plain `warm` control, escapes 21 -> 2, whiteouts 4/17; F1b (transfer) 31% -> 42% (5/12) vs 58% for the `transfer` control. Mixed: one arm clearly improved, the other did not; small samples. No new milestone in either.

R1 asks whether a better fighter helps the route run. Warm start from F1a's final weights (`f1a-fightschool-warm/snapshots/update-000384.pt`, sha256 c22e7090c2fc9bca5680d64953cfca56bd4ff9237b76c8d5310bd92c9b32833b), fresh optimizer, standard 4096-step route config (NO battle archive, standard rewards), 384 updates:
* `r1a-f1a-route`           : config identical to `warm`; ONLY the init weights differ. Control `warm`.
* `r1b-f1a-route-frontier`  : same + frontier_archive + frontier_productivity (frontier_prob 0.5). Control `e8c-frontier-prod-warm`.
Predictions: higher evaluation win rate than the matched controls; more time past the grass zone; route tiles with ROM index >= 54 or map 26:3 would be progress. Risks: the fight-school policy may have lost walking skills (short 512-step episodes), may avoid walking into grass at all, or collapse to hiding as other arms did.

---

# R1 outcome and Round S/B — pre-registered before training; code = main 110d9e0 (clean)

R1 (route runs from the F1a fight-school weights): negative. Both arms collapsed into hiding (0 battles in all 8 evaluation episodes; 0-1% in battle late in training); no milestone beyond Route 29. The fight skill did not survive the standard route reward. After 3.44M scanned decisions, 0 decisions in Cherrygrove (26:3); closest ROM-route tile unchanged (index 53 of 82, one visit).

User approved running both an assisted and an organic arm set, and asked that catching also be incentivized when the party is small (party-size-aware reward: 5 x (6 - party size)/5, so full value at party size 1, zero when full).

## ASSISTED track (labeled; route knowledge ON; never mixed with organic results)
Uses `path_progress` (dense reward for new best progress along the ROM-derived route), which encodes route knowledge. task_target `falkner` (non-strict; task_success is NOT meaningful, judge by milestones and best_path). Warm init (round-0299), 384 updates, standard 4096-step config otherwise.
* `s1-assist-route-warm`        : task_target falkner + path_progress. Control for the question "can it cross when told where progress is?".
* `s2-assist-route-catch-warm`  : s1 + assisted_start_balls 10 + catch_reward 5.0 + catch_budget 12.0 + catch_party_scale (catching incentivized most when the party is small).
Predictions: s1 reaches Cherrygrove (26:3) or ROM route index > 53 in training and evaluation; if it still does not, the bottleneck is fighting/reward, not exploration. s2: party size > 1 in training and, if fights are won, fewer whiteouts. Risks: path reward paid in a lethal grass zone makes the agent die more; reward hacking of the catch term; whiteout termination.

## ORGANIC track: make fights worth their cost
Standard strict route config; bundle `wild_battle_scale` 0.4 -> 2.5 (win 3 -> 7.5, KO and damage scaled too), `wild_battle_budget` 6 -> 60, `battle_free_count` 2 -> 6 (full-size payment for the first 6 wild fights per episode before decay). Rationale: a win currently pays ~2.2 and costs ~0.9 in step cost plus HP; hiding is risk-free.
* `b1-fightpays-warm`  : warm init, vs control `warm`
* `b2-fightpays-f1a`   : init from F1a final weights (sha256 c22e7090...), vs control `r1a-f1a-route`
Predictions: late-training battle occupancy stays > 15% (no hiding); evaluation win rate >= 50% with a non-trivial number of battles; any route progress is a bonus. Risks: farming wild fights (cap 60/episode, decay after 6), whiteouts rise, still hides.

---

# Round S3/S4 (ASSISTED; added at ~update 220 of S1/S2/B1/B2, before these arms trained) — code = main 110d9e0, clean

Mid-run check (s1/s2/b1/b2, ~58k decisions each): none entered Cherrygrove; the closest ROM-route tile for the assisted route-progress arm s1 is (31,14), remaining 268, i.e. the mouth of the 1-wide connector (31,13)->(33,12)->(34,12) that the route needs. Even a dense route-progress reward has not produced the single "up" step through it; fights/dead ends absorb the time. b1/b2 change only fight rewards and cannot affect pathfinding.

Adjustment: route checkpoints (existing long_v6 mechanism, requires path_progress, hence ASSISTED): the worker saves emulator states at route-progress buckets (every 12 tiles) and 40% of resets start from them, so the agent practises from states closer to the goal. Limit stated up front: checkpoints only exist up to the furthest tile any worker has reached, so the first crossing of the connector still has to happen by play.
* `s3-assist-route-ckpt-warm`      : s1 config + route_ckpt true (route_ckpt_prob 0.4). Control `s1-assist-route-warm`.
* `s4-assist-route-ckpt-fightpays` : s3 + the b1 bundle (wild_battle_scale 2.5, wild_battle_budget 60, battle_free_count 6). Control s3.
Predictions: more decisions on route tiles past ROM index 40 than s1; any entry into map 26:3 is progress. Failure: the same absorption in grass/fights, hiding, or dying more because checkpoints start inside the grass zone.

---

# Run-hygiene note (2026-11-09, user instruction: stop what is no longer useful, keep doing it)
* Stopped the four stale E8 dashboards (their arms had finished).
* Stopped b1-fightpays-warm's evaluation chain after 3 of 6 evaluation files: b1 ended in the original battle-menu stall (92% in battle, A 0%), so the remaining evaluations could only re-confirm it. PROTOCOL DEVIATION: b1 has 3/6 evaluation files.
* Kept: s1/s2 evaluations (s1 is the control for s3; s2 answers the catching question) and the s3/s4 trainings.
* b2 check (battle records, training): win rate 26-59% across 100-update windows (255 battles total: 82 wins, 144 escapes, 29 whiteouts), whiteout rate 7-16%; the high battle reward (243) is not a clear exploit, but "fights pay" did NOT raise win rate above F1a's 65%. Evaluation u384: 31 battles, 0 wins (27 escapes, 4 whiteouts), small sample and inconsistent with training windows.

---

# Assisted-arm outcome and the verified route (written before S5/sticky arms); code = main 110d9e0, clean

* s1-s4 (ASSISTED route-progress, with/without catching, route checkpoints, fights-pay): none entered Cherrygrove; best route progress in evaluation ~41-42 tiles (s1 u192 41, s2 u384 42), the same place organic arms stall; s1 regressed to 16 by u384. A dense route-progress reward alone does not cross the connector.
* **DIAGNOSTIC (never used in training or as an input): breadth-first search over real emulator tile moves from route29_entry** (scripts: analysis/bfs_route.py; Repel written in the diagnostic copy only so wild encounters do not interrupt). Cherrygrove (26:3) is reachable in **84 moves**; 547 Route 29 tiles reachable. Route: down4 left5 down2 left4 down1 left10 up2 left2 **up4 (x=31, y=14->10)** right5 up4 left6 up1 left8 up1 left3 down1 left2 down1 left9 up1 left8. **Single-tile bottlenecks: (31,14), (31,13), (31,12) and (21,4).** The corridor on y=14/15 continues west past x=31 to the dead-end pocket at x=14; the opening north at x=31 is one tile wide and needs four consecutive Up presses.
* Hypothesis: with ~6 buttons, tending to press Left, an agent walks past the opening, and even a correct Up must repeat 4x. Generic lever: sticky actions (`sticky_prob`: with that probability the previous executed move is repeated), standard exploration practice; strict configs had it at 0. Training-time only? NO: `sticky_prob` also applies in evaluation via the same worker config; the evaluator therefore runs with the arm's sticky setting, stated here so it is not mistaken for a clean greedy test.

# Round K (organic) — pre-registered before training

Config-only, single factor vs matched controls: `sticky_prob` 0 -> 0.5 or 0.75.
* `k1-sticky50-warm`      : sticky_prob 0.5, warm init, vs `warm`
* `k2-sticky75-warm`      : sticky_prob 0.75, warm init, vs `warm`
* `k3-sticky50-transfer`  : sticky_prob 0.5, transfer init, vs `transfer`
Predictions: more decisions at (31,13), (31,12), (31,11) and on the route beyond ROM index 40 than the controls; any entry into map 26:3 is progress. Risks: sticky actions make fights/menus noisier (the battle path excludes sticky: pick_executed_action ignores it in battle), more wall bumps, hiding.

---

# K outcome (training traces) and Round N (noise floor + long run) — pre-registered before training; code = main c1a710f, clean

K (sticky actions, organic): corridor (x=31, y 10-14) visits / decisions beyond it / closest ROM-route remaining: warm control 145 / 96 / 253; k1 (0.5, warm) 9 / 8 / 259; k2 (0.75, warm) 18 / 3 / 262; transfer control 2 / 0 / n.a.; k3 (0.5, transfer) 25 / 16 / 261. Sticky actions let arms through the 1-tile corridor but did not beat the plain warm control on warm init; helped only the transfer init. Still 0 decisions in Cherrygrove everywhere. Evaluations of k1/k2 pending.

**Methodological gap (stated plainly):** every comparison so far is ONE training run per arm. The plain warm control's 96 decisions beyond the corridor may be seed luck. Without the run-to-run spread of the plain configuration, none of the single-factor results above (positive or negative) can be interpreted.

## Round N (organic, no changed factor except the seed)
* `n1-warm-seed1`, `n2-warm-seed2`, `n3-warm-seed3`: the `warm` config with `seed` 1, 2, 3 (control seed was 20261110). Same init (round-0299 weights), budget and code.
Measures (same as everywhere): corridor visits, decisions beyond the corridor, closest route tile, evaluation win rate / milestones. Prediction: large spread. If the plain config's spread covers the differences seen between arms, those differences are not evidence. Also reported: how many of the 4 plain runs (control + n1-n3) ever cross the corridor.

## Long run (organic): does more training alone help?
* `long1-warm-1536`: the `warm` config with updates 1536 (4x the budget; 393k decisions), same seed 20261110, snapshot_every 192 so snapshots exist at 192, 384, 768, 1152, 1536. Standard evaluation at 192/384 via the arm runner; later snapshots evaluated separately.
Prediction/risk: it either keeps drifting into stall/hide (as shorter runs do) or reaches further along the route; the 384-update prefix of this run is itself a replicate of the `warm` control only up to run-to-run GPU nondeterminism.

---

# K evaluation outcome (written after K finished)
No sticky-action arm moved past Route 29 in evaluation (furthest milestone 0 for all). Battle records at the final checkpoint: k1 5 battles / 0 wins; k2 17 battles / 13 wins (76%) but 47 battles / 0 wins at update 192 (volatile); k3 13 battles / 1 win. Sticky actions are dropped as a routing lever. Round N (seed replicates) and the long run are running to establish the run-to-run spread of the plain configuration and whether more training alone helps.

---

# Round N outcome (the noise floor) and Round P (seed population) — written before P trained; code = main c1a710f

**Noise floor, plain `warm` config, 4 seeds (control 20261110 + seeds 1,2,3), training traces:** corridor (x=31,y10-14) visits 145 / 0 / 102 / 24; decisions BEYOND the corridor 96 / 0 / 0 / 42; closest ROM-route remaining 255 / 271 / 265 / 258; late-training battle occupancy / A-in-battle 58%/37%, 75%/1%, 48%/15%, 13%/21%. Two of four identical-config runs crossed the corridor; none went past ROM route index ~53 (second grass patch, x 30-37, y 4-9). **Consequence: run-to-run spread on the main routing metric (0-96) is larger than the differences between almost all earlier single-run arms; those results (positive and negative) are not distinguishable from noise unless they exceed this spread.** Single-seed comparisons in this log must be read that way.

## Round P (organic): sample more seeds of the plain config, keep the furthest
Rationale: with a spread this large the statistically sound approach is a population of seeds, then continue the ones that got furthest (population-based selection), instead of tuning a single run.
* `p4-warm-seed4`, `p5-warm-seed5`, `p6-warm-seed6`, `p7-warm-seed7`: the `warm` config with seeds 4-7 (only the seed differs).
Measures as before. Selection rule fixed in advance: an arm is "promising" if its training trace has >= 20 decisions on the route past ROM index 51 (tile (36,6)) or any decision in map 26:3; promising arms are continued from their final snapshot for another 384 updates (warm start, fresh optimizer, same config).

---

# Round P outcome, the wall at (36,6), and run hygiene (written before the next round)

**Population of 8 plain-config seeds (control + seeds 1-7), furthest point on the emulator-verified route (84 tiles; (31,14)=idx 30, (36,6)=idx 43):** max idx 43 / 27 / 32 / 40 / 31 / 32 / 32 / 33; decisions beyond (36,6): 0 in all eight; 0 decisions in Cherrygrove. By the pre-registered rule ("promising" = >= 20 decisions beyond (36,6) or any decision in 26:3) NO arm is promising; none is continued. (Clarification of my own pre-registration: "route index 51 (tile (36,6))" mixed two indexings; the rule was applied as "route tiles beyond (36,6)".)
**Long run (long1-warm-1536) COMPLETED its full 1536 updates (393,216 decisions; stop_reason budget_exhausted).** It never reached the corridor: max route idx 26 (33,16), 0 decisions at idx >= 31, vs 27-43 for the 384-update runs. More training alone did not help. CORRECTION OF AN EARLIER ENTRY IN THIS LOG: a note written a few minutes earlier said the run was "stopped at ~850/1536 on a cooperative STOP". That was WRONG: I read a stale status (update ~850) and by the time the STOP marker was created the run was at update 1535/1536 and finished on its own budget; the marker was removed and had no effect. The numbers quoted for it before completion (232k decisions, max idx 26) were from a partial trace; the final trace gives the same result.
**Map of the wall (diagnostic, analysis/map_wall.py):** the route does two counter-intuitive things. (1) At x=31 it leaves the westward corridor going UP four tiles. (2) At (31,10) it turns EAST for five tiles (away from Cherrygrove), then north four tiles through tall grass along x=36, and only then west along y=6. Agents instead go WEST at (31,10) into the heavily visited y=10-12, x=23-28 region (100-999 visits per tile), which is the Cut-tree dead end at (22,11); the y=6 west stretch (x=30..34) is reachable but was never visited by any of the four traces mapped.

---

# Corridor behaviour finding and Round Q (organic, replicated) — pre-registered before training; code = main c1a710f + docs, src/tools clean

User question: could we incentivize entering corridors? Measured first (analysis/inside_corridor.py, 7 plain-config traces): decisions that start inside the 1-tile corridor tiles (31,10..13): n=201, none in battle. Actions: A 41%, up 17%, down 15%, left 14%, right 10%, B 1%. Next position: stayed 60%, moved down 15%, moved up 14%, left 5%, right 5%. So agents DO enter it (7 of 8 plain runs reach route idx >= 31) but act near-randomly inside, with heavy no-effect A pressing; "entering" is not the failing step, "committing to traverse" is.
Mechanism hypothesis: the no-effect A/B press penalty (`idle_press_cost` 0.015 per press) has a per-episode budget `idle_press_budget` = 10 (~667 presses); in a 4096-step episode it is exhausted early, after which A-mashing is free (only the -0.01 step cost).

## Round Q: single field, 4 seeds PAIRED with the existing plain-config runs of the same seed (n1 s1, n2 s2, n3 s3, p4 s4)
* `q1-idle100-seed1`, `q2-idle100-seed2`, `q3-idle100-seed3`, `q4-idle100-seed4`: `idle_press_budget` 10 -> 100 (the penalty keeps applying ~6x longer); everything else identical to the `warm` config; seeds 1-4; warm init.
Noise-floor context (8 plain seeds): furthest route idx 43/27/32/40/31/32/32/33; decisions beyond (36,6) = 0 in all 8; corridor-and-beyond decisions (idx>=31) = 178/0/15/53/1/33/14/45.
Pre-registered readout: (a) share of corridor decisions that are A (plain pooled 41%); (b) share of corridor decisions that move up (plain 14%); (c) per-seed max route idx and decisions beyond (36,6) vs the same-seed plain run; (d) any entry into map 26:3. Interpretation rule: a result counts as evidence only if it moves (a)/(b) clearly beyond the plain spread AND at least 2 of 4 seeds reach idx >= 44 (plain: 0 of 8); otherwise it is noise or null. Risks: the agent shifts A-mashing to other idle behaviour (B presses, wall bumps); penalty makes menu/battle text handling worse (A is needed in battle: the penalty applies to plain overworld presses only, per src.idle_press).
