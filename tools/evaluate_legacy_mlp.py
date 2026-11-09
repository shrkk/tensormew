"""Evaluate a 17-input/5-button MLP with historical event-based observations.

Observation semantics match historical commit 3f85176; environment assistance is
disabled and battles use only policy button choices. This is a compatibility
diagnostic, not evidence about how the checkpoint was originally trained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.rl_env import observe
from src.train_config import load_config
from tools.train_deep_ppo import EnvWorker, worker_env_cfg


def historical_story_stage(state):
    if state.badges > 0 or state.has_event('EVENT_BEAT_FALKNER'):
        return 1.0
    for name, value in (
        ('EVENT_TALKED_TO_MOM_AFTER_MYSTERY_EGG_QUEST', .65),
        ('EVENT_GAVE_MYSTERY_EGG_TO_ELM', .55),
        ('EVENT_RIVAL_CHERRYGROVE_CITY', .45),
        ('EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON', .35),
        ('EVENT_GOT_A_POKEMON_FROM_ELM', .20),
    ):
        if state.has_event(name):
            return value
    return {(10, 7): .85, (10, 5): .75}.get((state.map_group, state.map_number), 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--max-steps', type=int, default=2048)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(args.checkpoint)
    assert data['w1'].shape == (64, 17) and data['w2'].shape == (5, 64)
    checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    cfg = load_config('bugsy-sparse-smoke.json')
    env = worker_env_cfg(cfg, None)
    env.update(task_target='falkner', terminate_on_whiteout=False, action_mask=False)
    results = []
    with (output / 'steps.jsonl').open('w', encoding='utf-8') as trace:
        for mode in ('sampled', 'greedy'):
            for seed in (10101, 10102):
                rng = np.random.default_rng(seed)
                worker = EnvWorker(seed, ['new_bark_outside'], args.max_steps + 1, env)
                t0 = time.time()
                maps = set()
                events = set()
                actions = [0] * 5
                try:
                    worker.reset()
                    initial = worker.reader.parse(worker.emu.frame)
                    initial_success = bool(initial.badge_bits & 1) or initial.has_event('EVENT_BEAT_FALKNER')
                    success = False
                    for step in range(args.max_steps):
                        before = worker.reader.parse(worker.emu.frame)
                        obs = observe(before, worker.visited)
                        obs[16] = historical_story_stage(before)
                        logits = data['w2'] @ np.tanh(data['w1'] @ obs + data['b1']) + data['b2']
                        probs = np.exp(logits - logits.max())
                        probs /= probs.sum()
                        action = int(np.argmax(probs) if mode == 'greedy' else rng.choice(5, p=probs))
                        actions[action] += 1
                        _, _, _, done, _, _, info = worker.step(action)
                        assert not done, 'Diagnostic horizon must precede worker reset'
                        after = worker.reader.parse(worker.emu.frame)
                        maps.add((after.map_group, after.map_number))
                        for name in ('EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON', 'EVENT_GAVE_MYSTERY_EGG_TO_ELM', 'EVENT_BEAT_FALKNER'):
                            if after.has_event(name):
                                events.add(name)
                        success = not initial_success and bool(after.badge_bits & 1) and after.has_event('EVENT_BEAT_FALKNER')
                        trace.write(json.dumps(dict(mode=mode, seed=seed, step=step, observation=obs.tolist(),
                            probabilities=probs.tolist(), action=action, transition=info['transition'],
                            badge_bits=after.badge_bits, success=success)) + '\n')
                        if success:
                            break
                    results.append(dict(mode=mode, seed=seed, steps=step + 1, success=success,
                        initial_goal_already_present=initial_success, maps=sorted(maps),
                        events=sorted(events), action_counts=actions, seconds=round(time.time()-t0, 2)))
                finally:
                    worker.close()
    root = Path(__file__).resolve().parent.parent
    manifest = dict(checkpoint_sha256=checkpoint_hash, checkpoint=str(Path(args.checkpoint).resolve()),
        observation_schema='historical 3f85176 base17, event-based story scalar, no route goals',
        actions=['up','down','left','right','a'], assistance=False, action_masks=False,
        start='new_bark_outside (starter already obtained)',
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for folder in ('src','tools') for p in (root/folder).glob('*.py')}, episodes=results)
    (output/'results.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
