"""Training-time sampling temperature: consistent logits across rollout and update, schedule, never persisted."""
import numpy as np
import pytest
import torch

from src.deep_policy import SpatialVisionPolicy
from src.rl_env import DEEP_OBS_DIM
from src.run_control import temperature_for_update


def _policy(gru=False):
    torch.manual_seed(1)
    p = SpatialVisionPolicy(obs_dim=DEEP_OBS_DIM, num_actions=6, use_gru=gru, gru_hidden_dim=32)
    p.eval()
    return p


def _inputs():
    screen = (np.random.default_rng(0).random((144, 160, 3)) * 255).astype(np.uint8)
    ram = np.random.default_rng(1).random(DEEP_OBS_DIM).astype(np.float32)
    s_t = torch.from_numpy(screen.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    return screen, ram, s_t, torch.from_numpy(ram).unsqueeze(0)


def test_schedule_is_linear_global_and_off_by_default():
    assert temperature_for_update(0, 1.0, 100) == 1.0 and temperature_for_update(50, 3.0, 0) == 1.0
    assert temperature_for_update(0, 3.0, 100) == 3.0 and temperature_for_update(50, 3.0, 100) == pytest.approx(2.0)
    assert temperature_for_update(100, 3.0, 100) == 1.0 and temperature_for_update(900, 3.0, 100) == 1.0


def test_temperature_flattens_the_distribution_and_t1_is_unchanged():
    p = _policy()
    _, _, s_t, r_t = _inputs()
    with torch.no_grad():
        base_logits, _ = p(s_t, r_t)
        p.set_temperature(1.0)
        same, _ = p(s_t, r_t)
        p.set_temperature(4.0)
        hot, _ = p(s_t, r_t)
    assert torch.equal(base_logits, same)
    assert torch.allclose(hot, base_logits / 4.0, atol=1e-6)
    ent = lambda l: float(torch.distributions.Categorical(logits=l).entropy())
    assert ent(hot) > ent(base_logits)


def test_rollout_logprob_equals_update_logprob_at_every_temperature():
    """The PPO invariant: the log-prob stored at rollout time must equal the one the update recomputes."""
    p = _policy()
    screen, ram, s_t, r_t = _inputs()
    for t in (1.0, 2.0, 3.5):
        p.set_temperature(t)
        for _ in range(10):
            a, logp, _v = p.act(screen, ram)
            lp, ent, _ = p.evaluate_actions(s_t, r_t, torch.tensor([a]))
            assert float(lp) == pytest.approx(logp, abs=1e-5)


def test_recurrent_rollout_and_sequence_evaluation_agree_at_temperature():
    B, T = 2, 3
    p = _policy(True)
    p.set_temperature(2.5)
    g = torch.Generator().manual_seed(3)
    screens = torch.rand((B, T, 3, 144, 160), generator=g)
    rams = torch.rand((B, T, DEEP_OBS_DIM), generator=g)
    hx = torch.zeros(1, B, 32)
    old_lp, actions, h = [], [], hx
    with torch.no_grad():
        for t in range(T):
            logits, _v, h = p.forward_recurrent(screens[:, t], rams[:, t], h)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            actions.append(a)
            old_lp.append(dist.log_prob(a))
        new_lp, _ent, _vals = p.evaluate_actions_sequence(screens, rams, torch.stack(actions, 1), hx)[:3]
    assert torch.allclose(new_lp, torch.stack(old_lp, 1), atol=1e-4)


def test_temperature_is_never_saved(tmp_path):
    p = _policy()
    p.set_temperature(3.0)
    path = tmp_path / "p.pt"
    p.save(path)
    loaded = SpatialVisionPolicy.load(str(path), device="cpu", override_obs_dim=DEEP_OBS_DIM)
    assert loaded.temperature == 1.0
    with pytest.raises(ValueError):
        p.set_temperature(0.0)


def test_config_defaults_off_and_validated():
    from src.train_config import from_dict, long_v9
    base = dict(supply_injection=False, escape_assist=False, auto_dialogue=False, task_target="falkner_strict",
                path_progress=False, goal_potential=False, route_ckpt=False, retreat_penalty=False,
                stagnation_penalty=False, swarm=False, battle_archive=False, battle_lowhp_weight=0,
                disable_bump_penalty=False, disable_loop_penalty=False)
    c = from_dict(base, long_v9())
    assert c.train_temperature_start == 1.0 and c.train_temperature_updates == 0
    with pytest.raises(Exception):
        from_dict({**base, "train_temperature_start": 0.5}, long_v9())
