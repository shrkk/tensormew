import hashlib
import json

import numpy as np

from src.experiment_trace import ExperimentTrace


def test_trace_preserves_terminal_transition_mask_and_checkpoint_hash(tmp_path):
    trace = ExperimentTrace(tmp_path / 'audit.jsonl', {'seed': 17}, code_commit='test')
    outcome = {'transition': {'map': [10, 7], 'frame_after': 250}, 'map': [24, 4],
               'reward_components': {'badge': 60.0}, 'final_obs': np.zeros((144, 160, 3))}
    trace.step(2, 3, [4], np.array([[0, 0, 0, 0, 1, 0]]), [1.0], [0.0],
               np.array([[1.0, 2.0]]), np.array([[False, False, False, False, True, False]]),
               [60.0], [True], [outcome])
    checkpoint = tmp_path / 'candidate.pt'
    checkpoint.write_bytes(b'candidate')
    trace.checkpoint(checkpoint, 2)
    trace.close()
    events = [json.loads(x) for x in (tmp_path / 'audit.jsonl').read_text().splitlines()]
    assert events[1]['outcome']['transition']['map'] == [10, 7]
    assert events[1]['outcome']['map'] == [24, 4]
    assert 'final_obs' not in events[1]['outcome']
    assert events[1]['mask'] == [False, False, False, False, True, False]
    assert events[2]['sha256'] == hashlib.sha256(b'candidate').hexdigest()
