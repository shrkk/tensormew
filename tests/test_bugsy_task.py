from types import SimpleNamespace

import pytest

from src.bugsy_task import BUGSY_EVENT, EVENT_REWARDS, BugsyLedger
from src.train_config import ConfigError, from_dict, long_v9


def test_pinned_bugsy_gate_and_corrected_event_ids():
    from src.story_flags import EVENT_FLAGS
    expected = {'EVENT_CLEARED_SLOWPOKE_WELL': 43,
                'EVENT_AZALEA_TOWN_SLOWPOKETAIL_ROCKET': 1786,
                'EVENT_BEAT_BUGSY': 1214, 'EVENT_GOT_TM31_MUD_SLAP': 8,
                'EVENT_GOT_HM01_CUT': 16, 'EVENT_GOT_HM06_WHIRLPOOL': 21,
                'EVENT_BEAT_CHUCK': 1218, 'EVENT_BEAT_JASMINE': 1217,
                'EVENT_GOT_MASTER_BALL_FROM_ELM': 124}
    assert {key: EVENT_FLAGS[key] for key in expected} == expected


def test_strict_first_gym_requires_specific_badge_and_event_absent_initially():
    from src.bugsy_task import defeated_falkner, falkner_goal_eligible
    assert falkner_goal_eligible(state())
    assert not falkner_goal_eligible(state('EVENT_BEAT_FALKNER'))
    assert not falkner_goal_eligible(state(badge_bits=1))
    assert not defeated_falkner(state('EVENT_BEAT_FALKNER', badge_bits=2))
    assert not defeated_falkner(state(badge_bits=1))
    assert defeated_falkner(state('EVENT_BEAT_FALKNER', badge_bits=1))


def state(*events, badge_bits=0):
    return SimpleNamespace(has_event=lambda name: name in events, badge_bits=badge_bits)


def test_loaded_milestones_and_repeated_transitions_never_pay():
    egg, kurt, well = EVENT_REWARDS
    before = state(egg)
    ledger = BugsyLedger(before)
    after = state(egg, kurt, well)
    assert ledger.step(before, after) == {kurt: 5.0, well: 15.0}
    assert ledger.step(after, after) == {}
    assert ledger.step(state(), after) == {}


def test_success_requires_specific_badge_and_event_can_arrive_separately():
    ledger = BugsyLedger(state())
    assert ledger.step(state(), state(BUGSY_EVENT, badge_bits=1)) == {}
    assert not ledger.success
    assert ledger.step(state(BUGSY_EVENT, badge_bits=1), state(BUGSY_EVENT, badge_bits=3)) == {BUGSY_EVENT: 100.0}
    assert ledger.success
    assert ledger.step(state(), state(BUGSY_EVENT, badge_bits=3)) == {}


@pytest.mark.parametrize('initial', [state(BUGSY_EVENT), state(badge_bits=2), state(BUGSY_EVENT, badge_bits=3)])
def test_preloaded_or_inconsistent_goal_cannot_be_credited(initial):
    ledger = BugsyLedger(initial)
    assert ledger.step(initial, state(BUGSY_EVENT, badge_bits=3)) == {}
    assert not ledger.success


def test_task_configuration_rejects_assistance_and_falkner_shaping():
    with pytest.raises(ConfigError):
        from_dict({'task_target': 'bugsy_sparse'}, long_v9())
    cfg = from_dict(dict(task_target='bugsy_sparse', supply_injection=False,
                         escape_assist=False, auto_dialogue=False,
                         path_progress=False, goal_potential=False, route_ckpt=False,
                         retreat_penalty=False, stagnation_penalty=False,
                         disable_bump_penalty=False, disable_loop_penalty=False), long_v9())
    from tools.train_deep_ppo import worker_env_cfg
    assert worker_env_cfg(cfg, None)['task_target'] == 'bugsy_sparse'
