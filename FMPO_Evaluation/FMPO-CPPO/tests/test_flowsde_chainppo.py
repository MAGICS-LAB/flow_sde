"""Property tests for FlowSDE / FMPO-ChainPPO (no checkpoints, environments, or GPUs needed)."""
import inspect
import logging
import os

import numpy as np
import pytest
import torch

from model.common.critic import CriticObs
from model.flow.ft_ppo import FlowSDE, PPOFlowFMPO
from model.flow.ft_ppo.ppoflow import PPOFlow as ReinFlowPPOFlow
from model.flow.mlp_flow import FlowMLP

logging.disable(logging.CRITICAL)
H, A, OBS, K, B = 4, 3, 11, 4, 64


def kw(**over):
    d = dict(
        device="cpu",
        policy=FlowMLP(horizon_steps=over.get("horizon_steps", H), action_dim=over.get("act_dim", A), cond_dim=OBS,
                       time_dim=16, mlp_dims=[64, 64, 64], activation_type="ReLU", out_activation_type="Identity",
                       use_layernorm=False, residual_style=True),
        critic=CriticObs(cond_dim=OBS, mlp_dims=[64, 64, 64], activation_type="Mish", residual_style=True),
        actor_policy_path=None, act_dim=A, horizon_steps=H, act_min=-1, act_max=1, obs_dim=OBS, cond_steps=1,
        noise_scheduler_type="learn_decay", inference_steps=K, ft_denoising_steps=K, randn_clip_value=3,
        min_sampling_denoising_std=0.08, min_logprob_denoising_std=0.08, logprob_min=-1.0, logprob_max=1.0,
        clip_ploss_coef=0.01, clip_ploss_coef_base=0.01, clip_ploss_coef_rate=3, clip_vloss_coef=None,
        denoised_clip_value=1.0, max_logprob_denoising_std=0.16, time_dim_explore=0,
        learn_explore_time_embedding=False, use_time_independent_noise=False, noise_hidden_dims=[16, 16],
        logprob_debug_sample=False, logprob_debug_recalculate=False, explore_net_activation_type="Tanh",
        noise_level_a=0.5,
    )
    d.update(over)
    return d


def chain_model(alpha=0.25):
    torch.manual_seed(3)
    return FlowSDE(**kw(use_chain_ppo_loss=True, chain_ppo_aux_coef=alpha,
                        chain_ppo_clip_ploss_coef=0.02, chain_ppo_aux_clip_ploss_coef=0.01))


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return {"state": torch.randn(B, 1, OBS)}, torch.randn(B), torch.randn(B)


def test_alias_and_lambda_schedules():
    assert PPOFlowFMPO is FlowSDE
    t = torch.tensor([0.0, 0.25, 0.5, 0.75])
    ratio = FlowSDE(**kw())
    assert ratio.lambda_schedule == "ratio"
    assert torch.allclose(ratio._get_lambda_t(t), 0.5 * (1 - t) / torch.clamp(t, min=1 / K))
    chain = FlowSDE(**kw(use_chain_ppo_loss=True, lambda_schedule="ratio", constant_lambda=True))
    assert chain.lambda_schedule == "linear" and not chain.constant_lambda
    assert torch.allclose(chain._get_lambda_t(t), 0.5 * (1 - t))


@pytest.mark.parametrize("schedule", ["linear", "ratio"])
def test_eval_mode_is_the_original_ode(schedule):
    # u_theta = 0 and x0 = 2: the ODE dY = u dt must keep Y at 2 (the SDE drift would contract it).
    m = FlowSDE(**kw(act_min=-10, act_max=10, lambda_schedule=schedule))
    m.actor_ft.forward = lambda x, t, cond, learn_exploration_noise=False, step=0: (torch.zeros_like(x), None)
    m.sample_first_point = lambda n: (torch.full((n, H, A), 2.0), torch.zeros(n))
    x, _ = m.get_actions({"state": torch.zeros(2, 1, OBS)}, eval_mode=True, clip_intermediate_actions=False)
    assert torch.all(x == 2.0)


def test_chain_ratio_is_one_before_update_and_loss_decomposes(batch):
    cond, ret, adv = batch
    m = chain_model(0.25)
    torch.manual_seed(4)
    _, chains, lp, slp = m.get_actions(cond, eval_mode=False, save_chains=True, normalize_denoising_horizon=True,
                                       normalize_act_space_dimension=True, ret_step_logprobs=True)
    assert slp.shape == (B, K)
    recomputed, _ = m.get_step_logprobs(cond, chains)
    assert torch.allclose(recomputed, slp, atol=1e-4)
    out = m.loss(cond, chains, ret, torch.zeros(B), adv, lp, old_step_logprobs=slp, verbose=False)
    assert len(out) == 15 and abs(out[6] - 1.0) < 1e-6
    stats = m.last_chain_ppo_stats
    assert out[0].item() == pytest.approx(stats["chain pg loss"] + 0.25 * stats["aux pg loss"], abs=1e-6)
    with pytest.raises(ValueError):
        m.loss(cond, chains, ret, torch.zeros(B), adv, lp, verbose=False)


def test_noaux_reduces_to_chain_term_and_alpha_changes_gradient(batch):
    cond, ret, adv = batch
    torch.manual_seed(4)
    _, chains, lp, slp = chain_model().get_actions(cond, eval_mode=False, save_chains=True, ret_step_logprobs=True)

    def grads(alpha):
        m = chain_model(alpha)
        torch.manual_seed(5)
        with torch.no_grad():
            for p in m.actor_ft.policy.parameters():
                p.add_(0.02 * torch.randn_like(p))
        out = m.loss(cond, chains, ret, torch.zeros(B), adv, lp, old_step_logprobs=slp, verbose=False)
        out[0].backward()
        return torch.cat([p.grad.flatten() for p in m.actor_ft.policy.parameters() if p.grad is not None]), m, out

    g_aux, m_aux, out_aux = grads(0.25)
    g_noaux, m_noaux, out_noaux = grads(0.0)
    assert not torch.allclose(g_aux, g_noaux)
    assert out_noaux[0].item() == pytest.approx(m_noaux.last_chain_ppo_stats["chain pg loss"], abs=1e-6)
    # normalized chain ratio = exp(sum_k dlogp / (H * d_a))
    new_step, _ = m_aux.get_step_logprobs(cond, chains, normalize_act_space_dimension=True)
    expected = torch.exp(new_step.sum(-1) - (slp / (H * A)).sum(-1)).mean().item()
    assert out_aux[6] == pytest.approx(expected, abs=1e-5)


def test_unnormalized_ratio_is_exact_and_entropy_unchanged(batch):
    cond, ret, adv = batch
    torch.manual_seed(9)
    m = FlowSDE(**kw())
    torch.manual_seed(10)
    _, chains, lp_raw = m.get_actions(cond, eval_mode=False, save_chains=True)
    _, _, lp_norm = m.get_actions(cond, eval_mode=False, save_chains=True, normalize_denoising_horizon=True,
                                  normalize_act_space_dimension=True)
    exact = m.loss(cond, chains, ret, torch.zeros(B), adv, lp_raw, verbose=False,
                   normalize_entropy_act_space_dimension=True)
    legacy = m.loss(cond, chains, ret, torch.zeros(B), adv, lp_norm, verbose=False, normalize_denoising_horizon=True,
                    normalize_act_space_dimension=True)
    assert abs(exact[6] - 1.0) < 1e-6
    assert torch.allclose(exact[1], legacy[1])
    with torch.no_grad():
        for p in m.actor_ft.policy.parameters():
            p.add_(0.005 * torch.randn_like(p))
    exact = m.loss(cond, chains, ret, torch.zeros(B), adv, lp_raw, verbose=False,
                   normalize_entropy_act_space_dimension=True)
    new_lp = m.get_logprobs(cond, chains, account_for_initial_stochasticity=True)[0]
    assert exact[6] == pytest.approx(torch.exp(new_lp - lp_raw).mean().item(), rel=1e-4)


@pytest.mark.parametrize("kind", ["chainppo", "flowsde", "flowsde_exact_ratio", "reinflow"])
def test_agent_update_runs(kind):
    from agent.finetune.reinflow.buffer import PPOFlowBuffer
    from agent.finetune.reinflow.train_ppo_flow_agent import TrainPPOFlowAgent

    model = {"chainppo": chain_model(), "flowsde": FlowSDE(**kw()), "flowsde_exact_ratio": FlowSDE(**kw()),
             "reinflow": ReinFlowPPOFlow(**kw())}[kind]
    norm = kind != "flowsde_exact_ratio"
    agent = object.__new__(TrainPPOFlowAgent)
    agent.model, agent.eval_mode, agent.device = model, False, "cpu"
    agent.use_chain_ppo_loss = bool(getattr(model, "use_chain_ppo_loss", False))
    agent.request_step_logprobs = agent.use_chain_ppo_loss
    agent.model_loss_supports_entropy_normalization = (
        "normalize_entropy_act_space_dimension" in inspect.signature(model.loss).parameters)
    n_steps, n_envs = 4, 16
    agent.buffer = PPOFlowBuffer(n_steps=n_steps, n_envs=n_envs, n_ft_denoising_steps=K, horizon_steps=H, act_steps=H,
                                 action_dim=A, n_cond_step=1, obs_dim=OBS, save_full_observation=False,
                                 furniture_sparse_reward=False, best_reward_threshold_for_success=3,
                                 reward_scale_running=False, gamma=0.99, gae_lambda=0.95, reward_scale_const=1.0,
                                 device="cpu", store_step_logprobs=agent.request_step_logprobs)
    agent.buffer.reset()
    for step in range(n_steps):
        cond = {"state": torch.randn(n_envs, 1, OBS)}
        out = agent.get_samples_logprobs(cond, normalize_denoising_horizon=norm, normalize_act_space_dimension=norm,
                                         ret_step_logprobs=agent.request_step_logprobs)
        step_lp = out[3] if agent.request_step_logprobs else None
        agent.buffer.add(step, cond["state"].numpy(), out[1], np.random.randn(n_envs), np.zeros(n_envs, bool),
                         np.zeros(n_envs, bool), np.random.randn(n_envs), out[2], step_lp)
    agent.buffer.returns_trajs = np.random.randn(n_steps, n_envs)
    agent.buffer.advantages_trajs = np.random.randn(n_steps, n_envs)
    agent.buffer.get_explained_var = lambda values, returns: 0.0
    agent.update_epochs, agent.batch_size, agent.n_steps, agent.n_envs = 2, 32, n_steps, n_envs
    agent.lr_schedule, agent.target_kl, agent.repeat_samples = "fixed", None, False
    agent.use_bc_loss, agent.bc_loss_type, agent.bc_coeff = False, "W2", 0.0
    agent.normalize_denoising_horizon = agent.normalize_act_space_dim = norm
    agent.clip_intermediate_actions, agent.account_for_initial_stochasticity = True, True
    agent.initial_ratio_error_threshold, agent.ent_coef, agent.vf_coef, agent.max_grad_norm = 1e-6, 0.0, 0.5, 1.0
    agent.actor_optimizer = torch.optim.Adam(model.actor_ft.parameters(), lr=1e-4)
    agent.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=1e-4)
    agent.itr, agent.n_critic_warmup_itr, agent.actor_update_freq, agent.actor_update_epoch = 1, 0, 1, 1
    agent.agent_update(verbose=False)  # raises if the first-minibatch ratio is not 1
    assert ("aux pg loss" in agent.train_ret_dict) == (kind == "chainppo")


@pytest.mark.parametrize("env", ["hopper-v2", "walker2d-v2", "ant-v2", "Humanoid-v3"])
def test_paper_configs_instantiate(env):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    paper_a = {"hopper-v2": 0.5, "walker2d-v2": 0.9, "ant-v2": 0.4, "Humanoid-v3": 0.5}[env]
    common = ["device=cpu", "sim_device=cpu", "model.actor_policy_path=null"]
    methods = {
        "reinflow": (["model._target_=model.flow.ft_ppo.ppoflow.PPOFlow"], ReinFlowPPOFlow),
        "flowsde": (["model._target_=model.flow.ft_ppo.ppoflow_fmppo.FlowSDE"], FlowSDE),
        "chainppo": (["model._target_=model.flow.ft_ppo.ppoflow_fmppo.FlowSDE", "model.use_chain_ppo_loss=true"], FlowSDE),
    }
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg", "gym", "finetune", env)
    for name, (overrides, cls) in methods.items():
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ft_ppo_reflow_mlp", overrides=common + overrides)
            assert cfg.train.n_train_itr == 601 and cfg.model.noise_level_a == paper_a
            model = instantiate(cfg.model)
            assert type(model) is cls
            if name == "flowsde":
                assert model.lambda_schedule == "ratio" and not model.use_chain_ppo_loss
            if name == "chainppo":
                assert model.lambda_schedule == "linear" and model.use_chain_ppo_loss


def test_gaussian_latents_survive_action_bounding(monkeypatch):
    """Environment action limits must not put boundary atoms in the Gaussian chain."""
    m = chain_model()
    m.sample_first_point = lambda n: (torch.zeros(n, H, A), torch.zeros(n))
    m.actor_ft.forward = lambda x, t, cond, **kwargs: (torch.zeros_like(x), None)
    m.actor_ft.policy.forward = lambda x, t, cond: torch.zeros_like(x)
    # Include a Gaussian tail beyond the old 3-sigma cutoff deterministically.
    monkeypatch.setattr(torch.distributions.Normal, "sample", lambda dist: dist.loc + 4 * dist.scale)
    cond = {"state": torch.zeros(2, 1, OBS)}
    actions, chain, lp, step_lp = m.get_actions(
        cond, eval_mode=False, save_chains=True, ret_step_logprobs=True, clip_intermediate_actions=False)
    assert torch.all(actions <= 1) and torch.all(actions >= -1)
    assert torch.all(chain[:, -1] > 1)
    assert torch.allclose(chain[:, 1], torch.full((2, H, A), 4 * (0.5 / K) ** 0.5))
    recalculated, _ = m.get_step_logprobs(cond, chain, clip_intermediate_actions=False)
    assert torch.allclose(step_lp, recalculated, atol=1e-4)
