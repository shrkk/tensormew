from unittest.mock import Mock, patch

from src.train_config import from_dict, long_v9
from tools.train_deep_ppo import EnvWorker, worker_env_cfg


def test_strict_flags_reach_training_and_evaluation_worker():
    cfg = from_dict({'supply_injection': False, 'escape_assist': False, 'auto_dialogue': False}, long_v9())
    env = worker_env_cfg(cfg, None)
    assert all(env[k] is False for k in ('supply_injection', 'escape_assist', 'auto_dialogue'))
    from tools.train_deep_ppo import eval_env_cfg
    ev = eval_env_cfg(cfg)
    assert all(ev[k] is False for k in ('supply_injection', 'escape_assist', 'auto_dialogue'))


def test_strict_supply_and_escape_controls_do_not_touch_emulator():
    worker = EnvWorker.__new__(EnvWorker)
    worker.supply_injection = worker.escape_assist = False
    worker.emu = Mock()
    worker.blocked_count = 4
    with patch('tools.train_deep_ppo.equip_supplies') as equip:
        worker._equip_balls()
        equip.assert_not_called()
    worker._escape()
    assert worker.blocked_count == 0
    assert not worker.emu.mock_calls
