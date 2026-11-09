"""Local PPO-only development loop. STOP halts between updates; no automatic code/weight edits.

Run with the repository's Python environment: python -m tools.local_bugsy_loop.
Development seeds are inspected each round. Reserved held-out seeds are never
consumed by this loop. STOP saves the current training job and launches no new job.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent.parent / 'outputs' / 'experiment-log' / 'bugsy-baseline'
STOP = ROOT.parent.parent / 'outputs' / 'experiment-log' / 'STOP'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for folder in ('src', 'tools')
            for p in sorted((ROOT / folder).glob('*.py'))}


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'RUNNING.lock'
    with lock.open('x') as f:
        import os
        f.write(str(os.getpid()))
    status = {'phase': 'initializing', 'method': 'PPO-only sparse Bugsy baseline',
              'stop_marker': str(STOP), 'competence_established': False}
    def report(**values):
        status.update(values, timestamp_utc=datetime.now(timezone.utc).isoformat())
        write(OUT / 'status.json', status)
    try:
        from src.train_config import load_config
        config = json.loads((ROOT / 'bugsy-sparse-smoke.json').read_text())
        config.update(envs=4, rollout_steps=64, max_steps=2048, updates=40,
                      seed=20261109, name='bugsy_sparse_baseline', aux_loss=False,
                      periodic_eval=False, reward_scale='none', warmup_rollout=0)
        frozen_code = code_hashes()
        manifest = {'method': status['method'], 'source_hashes': frozen_code,
                    'rom_hashes': {p.name: digest(p) for p in (ROOT / 'rom').glob('*') if p.suffix in ('.gbc', '.sym')},
                    'initial_state_hashes': {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'states').glob('*.state')},
                    'dependencies': {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'pyboy')},
                    'python': sys.version, 'development_seeds': [10101, 10102],
                    'reserved_heldout_seeds': [50001, 50002], 'config_template': config}
        manifest_path = OUT / 'manifest.json'
        if manifest_path.exists():
            shutil.copy2(manifest_path, OUT / ('manifest-before-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json'))
        write(manifest_path, manifest)
        latest = ROOT / 'logs' / 'bugsy-baseline-latest.pt'
        if latest.exists() and not args.resume:
            raise RuntimeError('Existing loop checkpoint: explicit restart/resume audit required')
        round_number = 0
        parent = None
        if args.resume:
            round_number = max(int(p.name.split('-')[1]) for p in OUT.glob('round-*-config.json'))
            parent = digest(latest)
        while not STOP.exists():
            if code_hashes() != frozen_code:
                raise RuntimeError('Source changed during run; stopped to preserve experiment identity')
            if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                raise RuntimeError('Less than 20 GiB free disk space')
            round_number += 1
            config.update(updates=40, save_path=str(latest),
                          run_id=f'bugsy-baseline-{round_number:04d}',
                          trace_path=str(OUT / f'round-{round_number:04d}-steps.jsonl'))
            config_path = OUT / f'round-{round_number:04d}-config.json'
            write(config_path, config)
            load_config(str(config_path))
            command = [sys.executable, '-m', 'tools.train_deep_ppo', '--config', str(config_path),
                       '--save', str(latest), '--run-id', config['run_id']]
            if parent is not None:
                command.append('--resume')
            report(phase='training', round=round_number, parent_sha256=parent,
                   config_sha256=digest(config_path), command=command)
            with (OUT / f'round-{round_number:04d}-training.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            parent = digest(latest)
            snapshot = OUT / f'round-{round_number:04d}.pt'
            shutil.copy2(latest, snapshot)
            report(phase='development_evaluation', checkpoint_sha256=parent)
            if STOP.exists():
                break
            evaluate(latest, config_path, 10101, f'round-{round_number:04d}-development')
            result = json.loads((OUT / f'round-{round_number:04d}-development.json').read_text())
            episodes = result['episodes']
            report(development_successes=sum(bool(e['task_success']) for e in episodes),
                   development_episodes=len(episodes),
                   development_milestones=[e.get('bugsy_ledger') for e in episodes])
        report(phase='stopped', checkpoint_sha256=parent)
    except Exception as exc:
        report(phase='failed', error=str(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)


def evaluate(checkpoint, config_path, seed, name):
    command = [sys.executable, '-m', 'tools.evaluate_policy', '--sampled', '--checkpoint', str(checkpoint),
               '--config', str(config_path), '--episodes', '2', '--max-steps', '2048',
               '--seed', str(seed), '--out-dir', str(OUT), '--name', name]
    with (OUT / (name + '.log')).open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
