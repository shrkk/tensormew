"""Append-only factual policy traces, without generated private reasoning."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


class ExperimentTrace:
    def __init__(self, path, config, code_commit=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a', encoding='utf-8')
        root = Path(__file__).resolve().parent.parent
        if code_commit is None:
            try:
                code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
            except (OSError, subprocess.CalledProcessError):
                code_commit = None
        files = [p for directory in ('src', 'tools') for p in (root / directory).glob('*.py')]
        files += [p for p in (root / 'rom').glob('*') if p.suffix in ('.gbc', '.sym')]
        hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        self.write({'event': 'start', 'config': config, 'code_commit': code_commit, 'file_sha256': hashes})

    def write(self, event):
        self.file.write(json.dumps(event, allow_nan=False, default=self._encode) + '\n')
        self.file.flush()

    @staticmethod
    def _encode(value):
        if isinstance(value, bytes):
            return value.hex()
        if hasattr(value, 'tolist'):
            return value.tolist()
        raise TypeError(type(value).__name__)

    def step(self, update, step, actions, probs, values, logps, rams, masks, rewards, dones, infos):
        for index, info in enumerate(infos):
            # final_obs contains screen tensors; the terminal transition is captured separately in the worker.
            safe_info = {k: v for k, v in (info or {}).items() if k != 'final_obs'}
            self.write({'event': 'step', 'update': update, 'rollout_step': step, 'env': index,
                        'action': int(actions[index]), 'probabilities': probs[index],
                        'value': float(values[index]), 'log_probability': float(logps[index]),
                        'ram_observation': rams[index], 'mask': None if masks is None else masks[index],
                        'reward': float(rewards[index]), 'done': bool(dones[index]), 'outcome': safe_info})

    def checkpoint(self, path, update):
        p = Path(path)
        if p.exists():
            self.write({'event': 'checkpoint', 'update': update, 'path': str(p),
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})

    def close(self):
        self.file.close()
