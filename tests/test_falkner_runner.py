import json
from pathlib import Path
import pytest
from src.bugsy_task import defeated_falkner, falkner_goal_eligible
from src.train_config import load_config

ROOT = Path(__file__).resolve().parent.parent


def test_falkner_strict_baseline_config_validation():
    cfg_file = ROOT / 'falkner-strict-baseline.json'
    assert cfg_file.exists()
    cfg = load_config(str(cfg_file))
    assert cfg.task_target == 'falkner_strict'
    assert not cfg.supply_injection
    assert not cfg.escape_assist
    assert not cfg.auto_dialogue
    assert not cfg.path_progress
    assert not cfg.goal_potential
    assert not cfg.route_ckpt
    assert not cfg.retreat_penalty
    assert not cfg.stagnation_penalty
    assert cfg.envs >= 1
    assert cfg.updates >= 1
    assert cfg.max_steps >= 2048


def test_falkner_goal_logic():
    from types import SimpleNamespace
    def s(*events, badge_bits=0):
        return SimpleNamespace(has_event=lambda name: name in events, badge_bits=badge_bits)

    assert falkner_goal_eligible(s())
    assert not falkner_goal_eligible(s('EVENT_BEAT_FALKNER'))
    assert not falkner_goal_eligible(s(badge_bits=1))
    assert not defeated_falkner(s())
    assert not defeated_falkner(s('EVENT_BEAT_FALKNER', badge_bits=0))
    assert not defeated_falkner(s(badge_bits=1))
    assert defeated_falkner(s('EVENT_BEAT_FALKNER', badge_bits=1))
