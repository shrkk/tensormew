# TensorMew review and Bugsy experiment plan

Reviewed November 9, 2026. Baseline: `550bbce70b55ac424ec28c33b99d497cb094c828`, fetched from upstream main. [Development branch](https://github.com/shrkk/tensormew/tree/development/bugsy-weight-space-2026-11-09). The branch was initially created at that exact commit. No trained-policy improvement is claimed by this review.

## Assessment

Keep RL as the main learning mechanism. First establish a reproducible baseline, correct evaluation and environment assistance, and extend the story model through Bugsy. Then test narrowly targeted weight edits against PPO-only controls. An endless loop maximizing the present reward would amplify flaws in the reward and evaluation harness.

This is a source and artifact audit, not a behavioral certification of the user's latest policy. Current main has no tracked policy weights or raw training logs. Historical weights exist, but their training semantics differ from the present trainer. A saved game in which Falkner is beaten does not establish that a particular policy learned the journey and battle.

## What is actually available

Git history confirms that [commit e1090a1](https://github.com/AvinashD24/tensormew/commit/e1090a1ec6edbee3a924fadcdbb573677ecc636a), dated October 28, removed model files. The last historical `models/deep_policy.pt` modification is [3f85176](https://github.com/AvinashD24/tensormew/commit/3f851765e02d10bd0dac90360b4bda3772de7327). I extracted its bytes for read-only inspection with `torch.load(weights_only=True)`:

- 52,972,620 bytes; SHA-256 `a0ab5382c51c1a0bf9849fb2b05cc823218aa78ea2a13673a4106ff5f9d9bdcb`.
- 13,221,702 trainable parameters reported by its metadata, 21 RAM inputs, five actions, GRU width 512. All state-dict tensors are finite.
- Its contemporaneous trainer calls `tactical_battle_step` during battles and `observe(..., include_goals=True)`. Therefore this is a learned-navigation/hybrid artifact, not a demonstrated end-to-end learned-battle checkpoint.
- Current `long_v9` uses 81 inputs and six actions, with different feature meanings. Direct resume would be invalid. Do not zero-pad inputs and describe the result as continuity of training. A separate, explicitly labeled transfer experiment could reuse compatible vision weights, but requires fresh validation.

The manifest includes a community `ilex_src` save with eight badges. It must never be used as evidence of agent-earned progress. Several stage saves, including `stage3_falkner_beaten.state`, are present without manifest provenance. The registry permits direct loading of such files; names alone are insufficient provenance.

## Bugsy objectives, verified against game scripts

The route is Violet → Route 32 → Union Cave 1F → Route 33 → Azalea → Kurt → Slowpoke Well B1F → Azalea Gym → Bugsy. There is a crucial prerequisite: Route 32 checks both the Zephyr Badge and receipt of the Togepi Egg. Elm's aide gives the egg in Violet's Pokémon Center only when the party has a free slot. Hatching it is unnecessary. [Route 32 script](https://github.com/pret/pokecrystal/blob/3c813b0b4d82b39bf62185aceffb76fd6026c70b/maps/Route32.asm), [Violet Center script](https://github.com/pret/pokecrystal/blob/3c813b0b4d82b39bf62185aceffb76fd6026c70b/maps/VioletPokecenter1F.asm).

These are proposed starting weights for a post-Falkner experiment, not empirically tuned values. Milestone payments occur once per campaign trajectory on genuine transitions. Loaded states initialize the ledger without payment. Resume restores the ledger. Diagnostic subepisodes do not imply campaign completion.

| Objective | Verifiable condition | Proposed reward |
|---|---|---:|
| Accept the Togepi Egg | `EVENT_GOT_TOGEPI_EGG_FROM_ELMS_AIDE` changes false→true through play; free party slot if needed | +10 |
| Pass the Route 32 guard | First arrival south of the actual gate, with prerequisite verified | +2 |
| Reach Union Cave | First legitimate entry into Union Cave 1F | +2 |
| Traverse the cave | First legitimate cave exit onto Route 33 | +2 |
| Reach Azalea | First legitimate town arrival | +4 |
| Activate Kurt's intervention | Talk to Kurt; the well-entrance Rocket hide event changes through the script | +5 |
| Clear Slowpoke Well | `EVENT_CLEARED_SLOWPOKE_WELL` changes through the final Rocket battle and ensuing script | +15 |
| Reach Bugsy's challenge | Enter the unlocked gym and start the actual Bugsy trainer battle | diagnostic only |
| Defeat Bugsy and receive the Hive Badge | `EVENT_BEAT_BUGSY` plus Johto Hive Badge bit, absent at start, acquired during the recorded run | +100; success |

Kurt removes the entrance blocker. The final well Rocket's script clears the operation and returns the player to Kurt's house; this scripted relocation is legitimate, not navigational progress to reward twice. The gym's script sets subordinate trainer flags after Bugsy, so do not count every newly set trainer flag as a separate battle victory. [Kurt](https://github.com/pret/pokecrystal/blob/3c813b0b4d82b39bf62185aceffb76fd6026c70b/maps/KurtsHouse.asm), [well](https://github.com/pret/pokecrystal/blob/3c813b0b4d82b39bf62185aceffb76fd6026c70b/maps/SlowpokeWellB1F.asm), [gym](https://github.com/pret/pokecrystal/blob/3c813b0b4d82b39bf62185aceffb76fd6026c70b/maps/AzaleaGym.asm).

Bugsy's team is Metapod 14, Kakuna 14, Scyther 16; Scyther has Quick Attack, Leer and Fury Cutter. Fire or Rock attacks can help, but a particular species, move, capture, or level should not be a mandatory rewarded objective. Let the policy discover workable preparation. Healing, purchasing supplies, training and switching are means to success. Ruins of Alph, Union Cave's deeper floors, Flash, Surf, Cut and the Ilex Forest puzzle are unnecessary for this target. The western rival encounter is outside the required gym path. [Trainer data](https://github.com/pret/pokecrystal/blob/3c813b0b4d82b39bf62185aceffb76fd6026c70b/data/trainers/parties.asm).

For shaping, initially compare sparse milestones alone against a small, phase-aware potential term `beta * (gamma * Phi(s_next) - Phi(s))`. Use the learner's gamma, bounded potentials, correct terminal potentials and truncation bootstrapping. Include task phase in the state; returning to heal or talk to Kurt must be legitimate. Do not switch to Bugsy's distance field while the well is still locked. Policy-invariance requires the theorem's assumptions and boundary treatment; clipping individual shaping payments or adding arbitrary bonuses loses that guarantee. [Ng, Harada and Russell](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

Keep repeatable battle, healing and curiosity rewards off in the clean control. If sparse learning fails, add small bounded aids one at a time, with explicit ablations. A cap limits exploitation but does not make damage→heal or fight→reset farming unprofitable. Test those trajectories directly. Compare completion probability first and frames to success second; don't trade task completion for a higher shaped return.

## Current strengths and concrete inefficiencies

Source references below are relative to the [pinned baseline tree](https://github.com/AvinashD24/tensormew/tree/550bbce70b55ac424ec28c33b99d497cb094c828).

1. **Useful foundations already exist.** `deep_policy.py` combines a ResNet-style screen encoder, RAM trunk and optional GRU. The current worker learns battle button choices. The PPO code handles time-limit bootstrapping, recurrent episode boundaries, per-step masks and frozen BatchNorm statistics during updates. There are source-split progress metrics, self-generated archives and an isolated periodic evaluator. Preserve these.

2. **The task still ends at Falkner.** `story_path.py`, `route_graph.py`, milestone reporting and level targets encode that goal. The graph's goals remain the tile before Falkner. Bugsy's flag exists in `story_flags.py`, but the well/Kurt events and full second-gym profile are missing. `beat_falkner()` accepts any positive badge count. Add specific badge-bit checks and a distinct Bugsy profile rather than appending names to a Falkner-specific abstraction. The comment claiming Route 32 needs only walking is incorrect.

3. **Navigation is partly assisted.** `EnvWorker.step()` calls `_escape()` after four blocked moves; it chooses a least-visited walkable neighbor. That controller can provide exploration the policy did not learn. Make assistance explicit and ablate it. Sticky actions are valid environment noise if specified; heuristic escape is additional control. The PPO action then denotes a composite intervention, which complicates attribution.

4. **Resource use is artificial.** `equip_supplies()` writes at least ten Poké Balls and five Potions into RAM after loads, including archive loads. This can change resource distribution across training resets. A strict organic track must preserve earned inventory; supplies must come from game actions. Keep any supplied-resources benchmark separately labeled.

5. **The action interface cannot directly open Start.** The six actions are directions/A/B. Out-of-battle party management, PC deposits, move teaching and ordinary bag use may require additional menu capabilities; in-battle bag use remains possible. A full party blocks the required Togepi Egg. Before adding a Start action, verify reachable interactions and plan checkpoint migration because action count is part of the schema. Do not quietly hand these decisions to dialogue macros.

6. **Offline evaluation omits training masks.** `tools/evaluate_policy.py:run_sampled_episode` calls both forward paths without `worker.last_mask`; `GreedyPolicy` also uses probability helpers that accept no mask. The periodic evaluator does apply masks. Unify evaluation through the worker and pass the current mask everywhere. Greedy and sampled success answer different questions; report both with identical environment semantics.

7. **Reward complexity obscures causality.** `long_v9` pays 695 for completing the egg-route path at scale 1, versus a base badge payment of 60 × 1.5, with a further time-dependent multiplier. Most dense path reward ends before the actual boss victory. Historical notes describe A-spam, grass farming and white-outs, but raw logs are unavailable, so these are repository-reported observations, not newly measured behavior. More shaping does not establish greater competence.

8. **The current path reward changes preferences.** `PathProgress` pays only new episode bests; retreat penalties explicitly prefer monotone travel and are not policy-invariant. Healing exemptions mitigate this but do not cover all useful detours. A second-gym extension introduces mandatory conversations, gated phases and preparation; simply extending the shortest path will repeat earlier design mistakes.

9. **Long-horizon credit remains difficult.** Gamma 0.999 retains about 0.37 of reward 1,000 decisions away, but GAE with lambda 0.95 has much shorter direct credit weighting. A 256-step BPTT window is not equivalent to end-to-end credit through the full journey. Use successful self-generated state curricula plus repeated complete rollouts to verify stitching. Archive progress must be reported separately from base-start progress. [Go-Explore](https://arxiv.org/abs/2004.12919).

10. **Several architectural issues need measurement.** Scalar species/map identifiers impose arbitrary numeric geometry; embeddings may help. The large visual encoder may be costly for low-resolution navigation; benchmark a smaller encoder rather than replacing a trained network speculatively. BatchNorm refresh changes the policy outside the optimizer, and recurrent states carried across parameter updates can become stale. Measure post-refresh action KL and recurrent burn-in sensitivity before diagnosing either as the cause of a failure.

11. **Frames matter.** Direction steps, warp settling, dialogue macros and battle button presses consume different frame counts. A per-decision speed score is not actual game speed. Log emulator frames and decision count separately; use consistent time accounting before claiming navigation efficiency.

12. **Reproducibility needs a run manifest.** Store code commit, exact config, dependency versions, ROM/symbol hashes, observation/action schemas, checkpoint hash, reset provenance and assistance flags together. `requirements.txt` omits torch and pytest, despite their use. Do not claim a successful dependency install proves GPU execution; benchmark CUDA kernels separately.

## How to apply weight-space programming honestly

The blog defines weight-space programming as direct LLM weight edits rather than optimizer updates. It reports targeted repairs and networks constructed from task knowledge, alongside failures to establish improvements where no localized defect was found. Its Go experiment uses a bounded feedback budget and hidden opponents. These are promising experiments, not evidence that arbitrary Pokémon skills can be added by repeated weight edits. [Blog](https://rikiyat.github.io/blog/weight-space-programming/).

The released car repair explicitly wires a small controller into selected neurons; that is an engineered behavior even though stored in weights. Therefore coordinate-based route rules encoded in tensors conflict with an RL-only discovery claim. Keep such edits in a separately labeled hybrid track. [Car experiment logs](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp4_fix_car/README.md).

My proposed experiment is narrower: identify a reproducible failure already within the policy's capabilities, locate a causally relevant circuit using rollout activations and interventions, and propose a sparse or low-rank correction. Candidate examples are an inappropriate action bias or recurrent loop; neither has been established for this project's missing latest checkpoint. Do not infer a semantic neuron from weight magnitude alone. Restrict initial edits to a small actor/fusion subset; avoid vision/GRU surgery until evidence justifies it. Small parameter count does not imply small behavioral change.

## Repeatable improvement loop

1. Freeze the evaluator and a checkpoint baseline. Establish stochastic and greedy success over multiple seeds. Use both agent-earned post-Falkner starts and complete New Bark starts; report them separately. An imported post-Falkner save is a segment-training fixture, not evidence of campaign completion.
2. Gather structured failure traces: phase, map, position, screen references, previous action, executed buttons, controller overrides, masks, action probabilities, value, HP/PP/inventory, event transitions, reward components and emulator frames. Include stuck duration, repeated edges, battles and white-outs.
3. Classify the bottleneck as task-gate, interface, exploration, memory, battle, resources, implementation or localized policy defect. Fix environment/evaluation defects before weight editing.
4. For each experiment, change one factor. Compare unchanged checkpoint, equal-compute PPO continuation, and direct edit. A later edit-plus-PPO arm is useful but must be labeled separately. Log environment frames, model calls, GPU time and wall time so savings can be evaluated.
5. For an edit, record parent hash, changed tensor indices or low-rank factors, before/after values, norm and count, hypothesis, observed intervention evidence and resulting policy KL on a replay set. Save the full candidate checkpoint. Apply the edit outside PPO rollouts, discard old rollout data, and restart affected optimizer moments deliberately if continuing training. Reset or burn in recurrent state under the new policy.
6. Use separate development, validation and sealed test seeds/state families. Reusing validation indefinitely overfits it; rotate validation and reserve final tests. Vary legal HP, PP, inventory, positioning and encounter timing using genuine play or documented fixtures, not hidden advantage. Retain the original starting scenario as a regression suite.
7. Promote by task success with uncertainty, then completion frames among successful runs, while enforcing Falkner retention and assistance constraints. Twenty seeds can screen candidates, but expand evaluation before accepting small differences; episode count should follow observed variance. Three independent training seeds are a practical initial control, not a universal guarantee.
8. Reject failures, keep the parent immutable, write the reason, then choose the next bottleneck. Never promote solely because shaped return rose. Stop work immediately on user request and checkpoint before stopping when possible.

Recommended first implementation order: evaluator parity → explicit assistance/supply controls → honest baseline → second-gym event/phase profile → self-generated curriculum and PPO continuation → localized weight-edit experiments. Direct editing cannot currently substitute for the missing compatible policy.

## Audit execution and limits

173 targeted tests passed on CPU, covering story flags, path rewards, reward blueprint, long_v9, learned-battle/no-oracle constraints, PPO updates, battle masks and periodic evaluation. One PyTorch scalar-conversion warning. These checks do not establish in-game Bugsy success. Initial test collection attempts were corrected for a nonexistent test filename and missing torch. A normal headless PyBoy smoke test subsequently loaded New Bark successfully; an optional SDL image DLL was rejected by Windows security during dependency setup, without disabling protection. GPU training and sustained episode improvements require subsequent runs.

The historical checkpoint metadata is saved separately in `historical-checkpoint-audit.json`. The review does not modify policy weights or claim to have reproduced the blog's experiments.

## Local delegation and logging contract

Hermes is installed and configured for Ollama `twil8b`, served locally. Hardware discovery found an Intel Core Ultra 7 265KF and NVIDIA RTX 5070 Ti with 16 GB VRAM. Delegate small implementation units to Hermes, independently review patches and run tests, and keep promotion decisions explicit. Serialize local LLM inference and PPO GPU updates when VRAM contention would otherwise slow both. Model orchestration and episode execution are separate operations.

Every experiment should have an append-only JSONL record plus a short Markdown decision summary. Store hypotheses, concise evidence-based rationale, patches, weight changes, commands, failures, seeds, dependency/config hashes, episode metrics and acceptance decisions. Record no secrets and no private internal chain-of-thought. Logs should explain what changed and why sufficiently to reproduce the result.

The canonical OpenAI model name is `gpt-6.1-sol`, with reasoning effort `low` for the requested light orchestration; this audit has not changed this chat's model selection. [Official model guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6.1-sol). Hermes local delegation has been tested through its CLI; ongoing execution must expose process status and a reliable stop mechanism rather than implying this chat keeps running after it ends.


## Implemented development status (November 9)

The upstream findings above describe the pinned baseline. Development now has explicit assistance controls, sampled-evaluation action masks, append-only action/transition traces, and an optional `bugsy_sparse` task. This task replaces all legacy reward with first-acquired Togepi Egg (+10), Kurt entrance intervention (+5), well cleared (+15), and Bugsy event plus specific Hive Badge (+100). Loaded milestones pay nothing; a preloaded or partially inconsistent Bugsy goal cannot count as completion. Navigation actions are not scripted.

The separate task retains the current 81-input/6-action schema for an experimental control. Its story observation and stage dashboards remain the legacy Falkner profile; they are not Bugsy progress metrics. This does not complete the proposed phase-aware Bugsy observation/profile redesign. No coordinate-route controller or direct weight edit was inserted. Sparse exploration from New Bark is difficult; this run establishes a clean control, not evidence that indefinite PPO will solve it.

The event-table audit also corrected HM/TM offsets, Master Ball, and swapped Chuck/Jasmine IDs. The verified Bugsy gates are well-cleared=43 and Kurt entrance intervention=1786. [Pinned event constants](https://github.com/pret/pokecrystal/blob/3c813b0b4d82b39bf62185aceffb76fd6026c70b/constants/event_flags.asm).

Both user-supplied files are preserved. The deep policy is the historical 21-input/5-action checkpoint trained with route-goal inputs and scripted battle support. Its training history does not establish organic gym competence. The 17-input MLP also has a semantic incompatibility: its last feature historically used event progress (starter .20, Mystery Egg .35, delivery .55, Mom .65), whereas current observations use milestone fractions. Equal tensor width does not make those schemas interchangeable. Evaluate each with its historical observation semantics before transfer; do not silently zero-pad the deep policy or treat either as an established Bugsy policy.

Validation: 91 focused tests passed after the task implementation, followed by 7 task tests after adding the pinned-ID regression assertion. Counts overlap. Two CUDA sparse-task smoke updates completed. The resulting policy was evaluated for two 256-decision episodes (seeds 10101/10102): zero Bugsy completions, both time limits. Earlier CPU/GPU bootstrap runs and their zero-Falkner evaluation remain separate artifacts. No learned gym competence has been demonstrated.

A local PPO-only sparse Bugsy baseline loop is running with four emulators, 64-step rollouts, 40 updates per round, 2,048-decision episodes, and two development episodes per round. It records parent checkpoint hashes, exact configs, commands, dependencies, source/ROM/start-state hashes, per-step traces and per-episode Bugsy ledgers. Reserved seeds 50001/50002 are evaluated once on cooperative stop. It makes no automatic source changes or direct policy edits, stops on source changes or low disk space, and uses no external paid API. Status and stop instructions are in `experiment-log/bugsy-baseline/README.md`.

Local Hermes task 002 read the constants file but did not produce a patch within the bounded attempt; it was interrupted. Sol completed the correction. The failed local attempt and failed smoke command/config attempts are preserved in the working audit logs.
