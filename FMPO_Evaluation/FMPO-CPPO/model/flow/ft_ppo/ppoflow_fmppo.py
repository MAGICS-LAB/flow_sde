# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import math
import torch
from torch import nn
import copy
import torch.nn.functional as F
from torch import Tensor
import logging
log = logging.getLogger(__name__)
from collections import namedtuple
from typing import Tuple
from torch.distributions.normal import Normal
from model.flow.mlp_flow import FlowMLP, NoisyFlowMLP
Sample = namedtuple("Sample", "trajectories chains")

class FlowSDE(nn.Module):
    """FlowSDE policy optimizer.

    Fine-tunes a pretrained flow policy with PPO using stochastic
    denoising-chain transition probabilities. The default FlowSDE noise
    schedule is lambda_t = a * (1 - t) / t_eff, with t_eff=max(t, 1/K).
    FMPO-ChainPPO is enabled with use_chain_ppo_loss=True, which forces the
    linear schedule lambda_t = a * (1 - t).
    """

    def __init__(self,
                 device,
                 policy,
                 critic,
                 actor_policy_path,
                 act_dim,
                 horizon_steps,
                 act_min,
                 act_max,
                 obs_dim,
                 cond_steps,
                 noise_scheduler_type,
                 inference_steps,
                 ft_denoising_steps,
                 randn_clip_value,
                 min_sampling_denoising_std,
                 min_logprob_denoising_std,
                 logprob_min,
                 logprob_max,
                 clip_ploss_coef,
                 clip_ploss_coef_base,
                 clip_ploss_coef_rate,
                 clip_vloss_coef,
                 denoised_clip_value,
                 max_logprob_denoising_std,
                 time_dim_explore,
                 learn_explore_time_embedding,
                 use_time_independent_noise,
                 noise_hidden_dims,
                 logprob_debug_sample,
                 logprob_debug_recalculate,
                 explore_net_activation_type,
                 noise_level_a=0.6,
                 constant_lambda=False,
                 lambda_schedule="ratio",
                 use_chain_ppo_loss=False,
                 chain_ppo_aux_coef=0.25,
                 chain_ppo_clip_ploss_coef=None,
                 chain_ppo_aux_clip_ploss_coef=None,
                 chain_ppo_normalize_act_space_dimension=True,
                 ):

        super().__init__()
        self.device = device
        self.inference_steps = inference_steps          # number of steps for inference.
        self.ft_denoising_steps = ft_denoising_steps    # could be adjusted
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = self.horizon_steps * self.action_dim
        self.act_min = act_min
        self.act_max = act_max

        self.obs_dim = obs_dim
        self.cond_steps = cond_steps

        self.noise_scheduler_type:str = noise_scheduler_type

        # prevent extreme values sampled from gaussian. deviation from mean should stay within `randn_clip_value` times of std.
        self.randn_clip_value:float = randn_clip_value

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std:float = min_sampling_denoising_std

        # Minimum and maximum std used in calculating denoising logprobs - for stability
        self.min_logprob_denoising_std:float = min_logprob_denoising_std
        self.max_logprob_denoising_std:float = max_logprob_denoising_std

        # Minimum and maximum logprobability in each batch, cutoff within this range to prevent policy collapse
        self.logprob_min:float= logprob_min
        self.logprob_max:float= logprob_max

        self.clip_ploss_coef:float = clip_ploss_coef
        self.clip_ploss_coef_base:float = clip_ploss_coef_base
        self.clip_ploss_coef_rate:float = clip_ploss_coef_rate
        self.clip_vloss_coef:float = clip_vloss_coef

        # clip intermediate actions during inference
        self.denoised_clip_value:float = denoised_clip_value
        self.logprob_debug_sample=logprob_debug_sample
        self.logprob_debug_recalculate=logprob_debug_recalculate

        # noise network settings
        self.learn_explore_time_embedding=learn_explore_time_embedding
        self.time_dim_explore=time_dim_explore
        self.use_time_independent_noise=use_time_independent_noise
        self.noise_hidden_dims=noise_hidden_dims
        self.explore_net_activation_type=explore_net_activation_type

        # Noise level coefficient. Supported schedules:
        #   ratio:  λ_t = a * (1 - t) / t_eff, t_eff=max(t, 1/K), used by FlowSDE.
        #   linear: λ_t = a * (1 - t), used by FMPO-ChainPPO.
        # If constant_lambda=True, λ_t = a.
        # For ratio, the t=0 step uses the next-step denominator 1/K.
        self.noise_level_a = noise_level_a
        self.constant_lambda = constant_lambda
        self.use_chain_ppo_loss = bool(use_chain_ppo_loss)
        self.lambda_schedule = str(lambda_schedule)
        if self.use_chain_ppo_loss and self.constant_lambda:
            log.warning(
                "FMPO-ChainPPO uses lambda_t = a * (1 - t); overriding constant_lambda=True."
            )
            self.constant_lambda = False
        if self.use_chain_ppo_loss and self.lambda_schedule != "linear":
            log.warning(
                "FMPO-ChainPPO uses lambda_schedule='linear' (a * (1 - t)); "
                "overriding requested lambda_schedule='%s'.",
                self.lambda_schedule,
            )
            self.lambda_schedule = "linear"
        if self.lambda_schedule not in ("ratio", "linear"):
            raise ValueError(f"Unknown lambda_schedule={self.lambda_schedule}")

        # FMPO-ChainPPO: J_chain + chain_ppo_aux_coef * J_step
        self.chain_ppo_aux_coef = float(chain_ppo_aux_coef)
        self.chain_ppo_clip_ploss_coef = float(
            clip_ploss_coef if chain_ppo_clip_ploss_coef is None else chain_ppo_clip_ploss_coef
        )
        self.chain_ppo_aux_clip_ploss_coef = float(
            clip_ploss_coef if chain_ppo_aux_clip_ploss_coef is None else chain_ppo_aux_clip_ploss_coef
        )
        self.chain_ppo_normalize_act_space_dimension = chain_ppo_normalize_act_space_dimension
        self.last_chain_ppo_stats = {}
        if self.chain_ppo_aux_coef < 0.0:
            raise ValueError(f"chain_ppo_aux_coef must be non-negative, got {self.chain_ppo_aux_coef}.")
        if self.chain_ppo_clip_ploss_coef <= 0.0:
            raise ValueError(f"chain_ppo_clip_ploss_coef must be positive, got {self.chain_ppo_clip_ploss_coef}.")
        if self.chain_ppo_aux_clip_ploss_coef <= 0.0:
            raise ValueError(
                f"chain_ppo_aux_clip_ploss_coef must be positive, got {self.chain_ppo_aux_clip_ploss_coef}."
            )

        self.actor_old: FlowMLP = policy
        self.load_policy(actor_policy_path, use_ema=True)  # it was false for previous experiments for hopper walker halfcheetah
        for param in self.actor_old.parameters():
            param.requires_grad = False             # don't train this copy, just use it to load checkpoint.
        self.actor_old.to(self.device)

        policy_copy = copy.deepcopy(self.actor_old)
        for param in policy_copy.parameters():
            param.requires_grad = True

        self.init_actor_ft(policy_copy)
        logging.info("Cloned policy for fine-tuning")

        self.critic = critic
        self.critic = self.critic.to(self.device)

        self.report_network_params()

    def init_actor_ft(self, policy_copy):
        self.actor_ft = NoisyFlowMLP(policy=policy_copy,
                                    denoising_steps=self.inference_steps,
                                    learn_explore_noise_from = self.inference_steps - self.ft_denoising_steps,
                                    inital_noise_scheduler_type=self.noise_scheduler_type,
                                    min_logprob_denoising_std = self.min_logprob_denoising_std,
                                    max_logprob_denoising_std = self.max_logprob_denoising_std,
                                    learn_explore_time_embedding=self.learn_explore_time_embedding,
                                    time_dim_explore=self.time_dim_explore,
                                    use_time_independent_noise=self.use_time_independent_noise,
                                    device=self.device,
                                    noise_hidden_dims=self.noise_hidden_dims,
                                    activation_type=self.explore_net_activation_type
                                    )

    def check_gradient_flow(self):
        print(f"{next(self.actor_ft.policy.parameters()).requires_grad}") #True
        print(f"{next(self.actor_ft.mlp_logvar.parameters()).requires_grad}")#True
        print(f"{next(self.actor_ft.time_embedding_explore.parameters()).requires_grad}")#True
        print(f"{self.actor_ft.logvar_min.requires_grad}")#False
        print(f"{self.actor_ft.logvar_max.requires_grad}")#False

    def report_network_params(self):
        logging.info(
            f"Number of network parameters: Total: {sum(p.numel() for p in self.parameters())/1e6} M. Actor:{sum(p.numel() for p in self.actor_old.parameters())/1e6} M. Actor (finetune) : {sum(p.numel() for p in self.actor_ft.parameters())/1e6} M. Critic: {sum(p.numel() for p in self.critic.parameters())/1e6} M"
        )

    def _get_lambda_t(self, t, one_minus_t=None):
        if one_minus_t is None:
            one_minus_t = torch.clamp(1 - t, min=1e-6)
        if self.constant_lambda:
            return self.noise_level_a * torch.ones_like(one_minus_t)

        if self.lambda_schedule == "linear":
            return self.noise_level_a * one_minus_t
        if self.lambda_schedule == "ratio":
            # The FMPO denoising grid uses start-of-step times t=i/K. At t=0,
            # use the next grid point 1/K as the denominator to avoid singularity.
            t_eff = torch.clamp(t, min=1.0 / float(self.inference_steps))
            return self.noise_level_a * (one_minus_t / t_eff)
        raise ValueError(f"Unknown lambda_schedule={self.lambda_schedule}")

    def load_policy(self, network_path, use_ema=False):
        log.info(f"loading policy from %s" % network_path)
        if network_path:
            print(f"network_path={network_path}, self.device={self.device}")
            model_data = torch.load(network_path, map_location=self.device, weights_only=True)
            actor_network_data = {k.replace("network.", ""): v for k, v in model_data["model"].items()}
            if use_ema:
                ema_actor_network_data = {k.replace("network.", ""): v for k, v in model_data["ema"].items()}
                self.actor_old.load_state_dict(ema_actor_network_data)
                logging.info("Loaded ema actor policy from %s", network_path)
            else:
                self.actor_old.load_state_dict(actor_network_data)
                logging.info("Loaded actor policy from %s", network_path)
            print(f"actor_network_data={actor_network_data.keys()}")
        else:
            logging.warning("No actor policy path provided. Not loading any actor policy. Start from randomly initialized policy.")

    @torch.no_grad()
    def sample_first_point(self, B:int)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        B: batchsize
        outputs:
            xt: torch.Tensor of shape `[batchsize, self.horizon_steps, self.action_dim]`
            log_prob: torch.Tensor of shape `[batchsize]`
        '''
        dist = Normal(torch.zeros(B, self.horizon_steps* self.action_dim), 1.0)
        xt= dist.sample()
        log_prob = dist.log_prob(xt).sum(-1).to(self.device)                    # mean() or sum()
        xt=xt.reshape(B, self.horizon_steps, self.action_dim).to(self.device)
        return xt, log_prob

    def get_logprobs(self,
                     cond:dict,
                     x_chain:Tensor,
                     get_entropy =False,
                     normalize_denoising_horizon=False,
                     normalize_act_space_dimension=False,
                     clip_intermediate_actions=True,
                     verbose_entropy_stats=True,
                     debug=True,
                     account_for_initial_stochasticity=False,
                     get_chains_stds=True,
                     normalize_entropy_act_space_dimension=None,
                     ):
        '''
        Computes log probability consistent with Theorem 4.1.

        The transition distribution follows the SDE:
        dZ_t = [(1 + λ_t/2 * t/(1-t)) * u_θ(Z_t, t) - (λ_t/2 * 1/(1-t)) * Z_t] dt + √λ_t dW_t

        In discrete form:
        Z_{t+dt} ~ N(Z_t + drift * dt, λ_t * dt)
        where drift = (1 + λ/2 * t/(1-t)) * v - (λ/2 * 1/(1-t)) * Z

        inputs:
            x_chain: torch.Tensor of shape `[batchsize, self.inference_steps+1, self.horizon_steps, self.action_dim]`

        outputs:
            log_prob. tensor of shape `[batchsize]`
            entropy_rate_est: tensor of shape `[batchsize]`
            chains_stds.mean(): scalar tensor
        '''
        logprob = 0.0
        joint_entropy = 0.0
        entropy_rate_est = 0.0
        logprob_steps = 0

        B = x_chain.shape[0]
        dt = 1.0 / self.inference_steps
        sqrt_dt = dt ** 0.5

        chains_prev = x_chain[:, :-1, :, :].flatten(-2, -1)  # [B, K, H*A]
        chains_next = x_chain[:, 1:, :, :].flatten(-2, -1)   # [B, K, H*A]
        chains_stds = torch.zeros_like(chains_prev, device=self.device)  # [B, K, H*A]
        chains_mean = torch.zeros_like(chains_prev, device=self.device)  # [B, K, H*A]

        # initial probability p(x_0) = N(0, I)
        init_dist = Normal(torch.zeros(B, self.horizon_steps * self.action_dim, device=self.device), 1.0)
        logprob_init = init_dist.log_prob(x_chain[:, 0].reshape(B, -1)).sum(-1)  # [B]
        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)  # [B]
        if account_for_initial_stochasticity:
            logprob += logprob_init
            if get_entropy:
                joint_entropy += entropy_init
            logprob_steps += 1

        steps = torch.linspace(0, 1 - dt, self.inference_steps).repeat(B, 1).to(self.device)  # [B, K]

        for i in range(self.inference_steps):
            t = steps[:, i]  # (B,)
            t_bc = t.view(-1, 1)  # (B, 1) for broadcasting with (B, H*A)

            xt = x_chain[:, i]  # (B, H, A)
            xt_flat = chains_prev[:, i]  # (B, H*A)

            # Get velocity from policy (keep nt for compatibility but don't use it for noise)
            vt, nt = self.actor_ft.forward(xt, t, cond, True, i)  # vt: (B, H, A), nt: (B, H*A)
            vt_flat = vt.flatten(-2, -1)  # (B, H*A)

            # Compute λ_t from the configured schedule (see _get_lambda_t)
            one_minus_t = torch.clamp(1 - t_bc, min=1e-6)
            lambda_t = self._get_lambda_t(t_bc, one_minus_t)

            # Compute drift coefficients per Theorem 4.1
            coef_vel = 1.0 + (lambda_t / 2.0) * (t_bc / one_minus_t)
            coef_xt = (lambda_t / 2.0) / one_minus_t

            # Drift: scaled velocity + contraction term
            drift = coef_vel * vt_flat - coef_xt * xt_flat

            # Mean after Euler step
            mean = xt_flat + drift * dt
            if clip_intermediate_actions:
                mean = mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            # Transition std = √(λ_t * dt)
            trans_std = torch.sqrt(lambda_t * dt)

            chains_mean[:, i] = mean
            chains_stds[:, i] = trans_std
            logprob_steps += 1

        # transition distribution
        chains_dist = Normal(chains_mean, chains_stds)

        # logprobability and entropy of the transitions
        logprob_trans = chains_dist.log_prob(chains_next).sum(-1)  # [B, K] sum over H*A
        if get_entropy:
            entropy_trans = chains_dist.entropy().sum(-1)  # [B, K]

        # logprobability of the whole markov chain
        logprob += logprob_trans.sum(-1)  # [B] accumulate over inference steps
        if self.logprob_debug_recalculate:
            log.info(f"logprob_init={logprob_init.mean().item()}, logprob_trans={logprob_trans.mean().item()}")

        # entropy rate estimate of the whole markov chain
        if get_entropy:
            joint_entropy += entropy_trans.sum(-1)

        if get_entropy:
            entropy_rate_est = joint_entropy / logprob_steps
        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps

        if normalize_act_space_dimension:
            logprob = logprob / self.act_dim_total
        # entropy normalization is independent of the likelihood-ratio normalization
        if normalize_entropy_act_space_dimension is None:
            normalize_entropy_act_space_dimension = normalize_act_space_dimension
        if get_entropy and normalize_entropy_act_space_dimension:
            entropy_rate_est = entropy_rate_est / self.act_dim_total

        if verbose_entropy_stats and get_entropy:
            log.info(f"entropy_rate_est={entropy_rate_est.shape} Entropy Percentiles: 10%={entropy_rate_est.quantile(0.1):.2f}, 50%={entropy_rate_est.median():.2f}, 90%={entropy_rate_est.quantile(0.9):.2f}")

        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, chains_stds.mean()
            return logprob, entropy_rate_est,
        else:
            if get_chains_stds:
                return logprob, chains_stds.mean()
            return logprob

    @torch.no_grad()
    def get_actions(self,
                    cond:dict,
                    eval_mode:bool,
                    save_chains=False,
                    normalize_denoising_horizon=False,
                    normalize_act_space_dimension=False,
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True,
                    ret_logprob=True,
                    ret_step_logprobs=False,
                    ):
        '''
        Implements Theorem 4.1: Correctness for Flow RL via Diffusion-Flow Duality.

        The SDE for Z_t^θ is:
        dZ_t = [(1 + λ_t/2 * t/(1-t)) * u_θ(Z_t, t) - (λ_t/2 * 1/(1-t)) * Z_t] dt + √λ_t dW_t

        inputs:
            cond: dict, containing...
                'state': obs. observation in robotics. torch.Tensor(batchsize, cond_steps, obs_dim)
            eval_mode: bool, whether to deploy with the deterministic ODE dY_t = u_θ(Y_t, t) dt
                instead of SDE inference.
            save_chains: whether to return predictions at each step
            normalize_denoising_horizon: bool, whether to normalize time horizon when calculating the log probability.
            normalize_act_space_dimension: bool, whether to normalize action dimension when calculating the log probability.
            clip_intermediate_actions: bool, whether to clip intermediate actions during the flow.
            ret_step_logprobs: bool, whether to also return the unnormalized per-step transition logprobs `[batchsize, K]` (needed by FMPO-ChainPPO).
        outputs:
            1. (xt, x_chain, log_prob) when `save_chains` =True
            2. (xt, log_prob) when `save_chains` = False
            per-step logprobs are appended as the last element when `ret_step_logprobs` = True
            xt. tensor of shape `[batchsize, self.horizon_steps, self.action_dim]`
            x_chains. tensor of shape `[self.inference_steps +1 ,self.data_shape]`: x0, x1, x2, ... xK
            logprob. tensor of shape `[batchsize]` or None
        '''
        B = cond["state"].shape[0]
        dt = 1.0 / self.inference_steps
        sqrt_dt = dt ** 0.5

        steps = torch.linspace(0, 1 - dt, self.inference_steps).repeat(B, 1).to(self.device)  # [batchsize, num_steps]

        if save_chains:
            x_chain = torch.zeros((B, self.inference_steps + 1, self.horizon_steps, self.action_dim), device=self.device)
        if ret_logprob:
            log_prob = 0.0
            log_prob_steps = 0
            if self.logprob_debug_sample:
                log_prob_list = []
        if ret_step_logprobs:
            step_logprobs = torch.zeros((B, self.inference_steps), device=self.device)

        # sample first point from N(0, I)
        xt, log_prob_init = self.sample_first_point(B)
        if ret_logprob and account_for_initial_stochasticity:
            log_prob += log_prob_init
            log_prob_steps += 1
            if self.logprob_debug_sample:
                log_prob_list.append(log_prob_init.mean().item())

        xt: torch.Tensor
        if save_chains:
            x_chain[:, 0] = xt

        for i in range(self.inference_steps):
            t = steps[:, i]  # (B,)
            t_bc = t.view(-1, 1, 1)  # (B, 1, 1) for broadcasting with (B, H, A)

            # Get velocity from policy (keep nt for compatibility but don't use it for noise)
            vt, nt = self.actor_ft.forward(xt, t, cond, learn_exploration_noise=False, step=i)

            # Compute λ_t from the configured schedule (see _get_lambda_t)
            one_minus_t = torch.clamp(1 - t_bc, min=1e-6)
            lambda_t = self._get_lambda_t(t_bc, one_minus_t)

            # Compute drift coefficients per Theorem 4.1:
            # dZ = [(1 + λ/2 * t/(1-t)) * v - (λ/2 * 1/(1-t)) * Z] dt + √λ dW
            coef_vel = 1.0 + (lambda_t / 2.0) * (t_bc / one_minus_t)
            coef_xt = (lambda_t / 2.0) / one_minus_t

            # Compute drift: scaled velocity + contraction term
            drift = coef_vel * vt - coef_xt * xt

            # Euler step: compute mean after drift
            xt_mean = xt + drift * dt

            if clip_intermediate_actions:
                xt_mean = xt_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            # Diffusion std = √(λ_t * dt)
            diffusion_std = torch.sqrt(lambda_t * dt)

            # Sample from the SDE transition distribution, or deploy with the ODE
            if not eval_mode:
                dist = Normal(xt_mean, diffusion_std)
                # PPO likelihoods refer to the unmodified Gaussian latent chain.
                xt = dist.sample()
            else:
                # original flow ODE: dY_t = u_θ(Y_t, t) dt (no SDE drift correction)
                xt = xt + vt * dt
                if clip_intermediate_actions:
                    xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            if ret_logprob or ret_step_logprobs:
                # compute logprobs using the transition distribution
                dist = Normal(xt_mean, diffusion_std)
                logprob_transition = dist.log_prob(xt).sum(dim=(-2, -1)).to(self.device)
                if ret_logprob:
                    if self.logprob_debug_sample:
                        log_prob_list.append(logprob_transition.mean().item())
                    log_prob += logprob_transition
                    log_prob_steps += 1
                if ret_step_logprobs:
                    step_logprobs[:, i] = logprob_transition

            if save_chains:
                x_chain[:, i + 1] = xt

        # The environment receives a bounded copy. Saved latents and their
        # Gaussian log probabilities remain unchanged, including the final step.
        action = xt.clamp(self.act_min, self.act_max)

        if ret_logprob:
            if normalize_denoising_horizon:
                log_prob = log_prob / log_prob_steps
            if normalize_act_space_dimension:
                log_prob = log_prob / self.act_dim_total
            if self.logprob_debug_sample:
                transform_logprob = torch.log(1 - torch.tanh(x_chain[-1])**2 + 1e-7).sum(dim=(-2, -1)).mean().item()
                print(f"log_prob_list={log_prob_list}, transform={transform_logprob}")

        if ret_logprob:
            if save_chains:
                if ret_step_logprobs:
                    return (action, x_chain, log_prob, step_logprobs)
                return (action, x_chain, log_prob)
            if ret_step_logprobs:
                return (action, log_prob, step_logprobs)
            return (action, log_prob)
        else:
            if save_chains:
                if ret_step_logprobs:
                    return (action, x_chain, step_logprobs)
                return (action, x_chain)
            if ret_step_logprobs:
                return (action, step_logprobs)
            return action

    def get_step_logprobs(
        self,
        cond: dict,
        x_chain: Tensor,
        get_entropy: bool = False,
        normalize_act_space_dimension: bool = False,
        clip_intermediate_actions: bool = True,
        normalize_entropy_act_space_dimension=None,
    ):
        """
        Recompute exact per-step FMPO transition logprobs for a saved denoising chain.

        Returns the unaggregated Gaussian transition densities
            pi_theta(x_k | x_{k-1}, s)
        for every denoising step in the chain, shape [B, K].
        """
        B = x_chain.shape[0]
        K = self.inference_steps
        dt = 1.0 / K

        xt_all = x_chain[:, :-1]
        chains_prev = xt_all.flatten(-2, -1)
        chains_next = x_chain[:, 1:, :, :].flatten(-2, -1)

        steps = torch.linspace(0, 1 - dt, K, device=self.device).unsqueeze(0).expand(B, -1)
        t_batched = steps.reshape(B * K)
        cond_batched = {
            "state": cond["state"].unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, *cond["state"].shape[1:])
        }
        xt_batched = xt_all.reshape(B * K, self.horizon_steps, self.action_dim)

        vt_batched = self.actor_ft.policy.forward(xt_batched, t_batched, cond_batched)
        vt_all = vt_batched.reshape(B, K, -1)

        t_bc = steps.unsqueeze(-1)
        one_minus_t = torch.clamp(1 - t_bc, min=1e-6)
        lambda_t = self._get_lambda_t(t_bc, one_minus_t)

        coef_vel = 1.0 + (lambda_t / 2.0) * (t_bc / one_minus_t)
        coef_xt = (lambda_t / 2.0) / one_minus_t
        drift = coef_vel * vt_all - coef_xt * chains_prev
        chains_mean = chains_prev + drift * dt
        if clip_intermediate_actions:
            chains_mean = chains_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

        chains_stds = torch.sqrt(lambda_t * dt).expand_as(chains_mean)
        chains_dist = Normal(chains_mean, chains_stds)
        step_logprobs = chains_dist.log_prob(chains_next).sum(-1)
        if get_entropy:
            step_entropy = chains_dist.entropy().sum(-1)

        if normalize_act_space_dimension:
            step_logprobs = step_logprobs / self.act_dim_total
        if normalize_entropy_act_space_dimension is None:
            normalize_entropy_act_space_dimension = normalize_act_space_dimension
        if get_entropy and normalize_entropy_act_space_dimension:
            step_entropy = step_entropy / self.act_dim_total

        if get_entropy:
            return step_logprobs, step_entropy, chains_stds.mean()
        return step_logprobs, chains_stds.mean()

    def _get_initial_logprob(self, x_chain: Tensor, normalize_act_space_dimension: bool = False):
        x0_flat = x_chain[:, 0].reshape(x_chain.shape[0], -1)
        init_logprob = (-0.5 * (x0_flat ** 2 + math.log(2 * math.pi))).sum(-1)
        if normalize_act_space_dimension:
            init_logprob = init_logprob / self.act_dim_total
        return init_logprob

    def loss(
        self,
        obs,
        chains,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        old_step_logprobs=None,
        use_bc_loss=False,
        bc_loss_type='W2',
        normalize_denoising_horizon=False,
        normalize_act_space_dimension=False,
        verbose=True,
        clip_intermediate_actions=True,
        account_for_initial_stochasticity=True,
        normalize_entropy_act_space_dimension=None,
    ):
        """
        PPO loss
        obs: dict with key state/rgb; more recent obs at the end
            "state": (B, To, Do)
            "rgb": (B, To, C, H, W)
        chains: (B, K+1, Ta, Da)
        returns: (B, )
        values: (B,)
        advantages: (B,)
        oldlogprobs: (B,)
        old_step_logprobs: (B, K), unnormalized per-step rollout logprobs; required when use_chain_ppo_loss=True
        use_bc_loss: whether to add BC regularization loss
        normalize_denoising_horizon, normalize_act_space_dimension: whether to normalize the chain logprob (and hence the
            PPO ratio) by the number of chain steps and by horizon x action dimension. Both False gives the exact
            whole-chain ratio exp(sum_k log p_new - log p_old).
        normalize_entropy_act_space_dimension: whether to normalize the entropy bonus by horizon x action dimension;
            None follows normalize_act_space_dimension.
        reward_horizon: action horizon that backpropagates gradient, omitted for now.
        Here, B = n_steps x n_envs
        """
        self.last_chain_ppo_stats = {}
        if self.use_chain_ppo_loss:
            return self.loss_chain_ppo(
                obs=obs,
                chains=chains,
                returns=returns,
                oldvalues=oldvalues,
                advantages=advantages,
                oldlogprobs=oldlogprobs,
                old_step_logprobs=old_step_logprobs,
                verbose=verbose,
                clip_intermediate_actions=clip_intermediate_actions,
                account_for_initial_stochasticity=account_for_initial_stochasticity,
            )

        newlogprobs, entropy, noise_std = self.get_logprobs(obs,
                                                            chains,
                                                            get_entropy=True,
                                                            normalize_denoising_horizon=normalize_denoising_horizon,
                                                            normalize_act_space_dimension=normalize_act_space_dimension,
                                                            verbose_entropy_stats=verbose,
                                                            clip_intermediate_actions=clip_intermediate_actions,
                                                            account_for_initial_stochasticity=account_for_initial_stochasticity,
                                                            normalize_entropy_act_space_dimension=normalize_entropy_act_space_dimension)
        if verbose:
            log.info(f"oldlogprobs.min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}, std of oldlogprobs={oldlogprobs.std():5.3f}")
            log.info(f"newlogprobs.min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}, std of newlogprobs={newlogprobs.std():5.3f}")


        # logprob_min/max are per-dimension bounds, so they only apply to dimension-normalized logprobs
        if normalize_act_space_dimension:
            newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
            oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        if verbose and normalize_act_space_dimension:
            if oldlogprobs.min() < self.logprob_min: log.info(f"WARNINIG: old logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.min() < self.logprob_min: log.info(f"WARNINIG: new logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.max() > self.logprob_max: log.info(f"WARNINIG: new logprobs too high")
            if oldlogprobs.max() > self.logprob_max: log.info(f"WARNINIG: old logprobs too high")
        # empirically we noticed that when the min of logprobs gets too negative (say, below -3) or when the std gets larger than 0.5 (usually these two events happen simultaneously) t
        # the perfomance drops.
        # batch normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if verbose:
            with torch.no_grad():
                advantage_stats = {
                    "mean":f"{advantages.mean().item():2.3f}",
                    "std": f"{advantages.std().item():2.3f}",
                    "max": f"{advantages.max().item():2.3f}",
                    "min": f"{advantages.min().item():2.3f}"
                }
                log.info(f"Advantage stats: {advantage_stats}")
                corr = torch.corrcoef(torch.stack([advantages, returns]))[0,1].item()
                log.info(f"Advantage-Reward Correlation: {corr:.2f}")

        # Get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # Get kl difference and whether value clipped
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef: # better not use.
            v_clipped = torch.clamp(newvalues, oldvalues -self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 *torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()
        if verbose:
            with torch.no_grad():
                mse = F.mse_loss(newvalues, returns)
                log.info(f"Value/Reward alignment: MSE={mse.item():.3f}")

        # Entropy loss
        entropy_loss = -entropy.mean()
        # Monitor policy entropy distribution
        if verbose:
            with torch.no_grad():
                log.info(f"Entropy Percentiles: 10%={entropy.quantile(0.1):.2f}, 50%={entropy.median():.2f}, 90%={entropy.quantile(0.9):.2f}")

        # bc loss
        bc_loss = 0.0
        if use_bc_loss:
            if bc_loss_type=='W2':
                # add wasserstein divergence loss via action supervision
                z=torch.zeros((obs['state'].shape[0], self.horizon_steps, self.action_dim), device=self.device)
                a_ω = self.actor_old.sample_action(cond=obs, inference_steps=self.inference_steps, clip_intermediate_actions=True, act_range=[self.act_min, self.act_max],z=z)
                a_θ = self.actor_ft.policy.sample_action(cond=obs, inference_steps=self.inference_steps, clip_intermediate_actions=True, act_range=[self.act_min, self.act_max],z=z)
                bc_loss = F.mse_loss(a_ω.detach(), a_θ)
            else:
                raise NotImplementedError
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            bc_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            oldlogprobs.min(),
            oldlogprobs.max(),
            oldlogprobs.std(),
            newlogprobs.min(),
            newlogprobs.max(),
            newlogprobs.std(),
            noise_std.item(),
            newvalues.mean().item(),#Q function
        )

    def loss_chain_ppo(
        self,
        obs,
        chains,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        old_step_logprobs=None,
        verbose=True,
        clip_intermediate_actions=True,
        account_for_initial_stochasticity=True,
    ):
        """
        Chain-level PPO objective for FMPO (FMPO-ChainPPO).

        The full denoising chain at an env step is treated as one stochastic
        action unit through the summed chain log-ratio. A small auxiliary
        per-step PPO term is added only for optimization stability:
            L = L_chain + chain_ppo_aux_coef * L_aux.
        Set chain_ppo_aux_coef=0 for the NoAux ablation.
        """
        if old_step_logprobs is None:
            raise ValueError("ChainPPO requires per-step old logprobs from the rollout buffer.")

        normalize_act_space_dimension = self.chain_ppo_normalize_act_space_dimension
        new_step_logprobs, step_entropy, noise_std = self.get_step_logprobs(
            obs,
            chains,
            get_entropy=True,
            normalize_act_space_dimension=normalize_act_space_dimension,
            clip_intermediate_actions=clip_intermediate_actions,
            normalize_entropy_act_space_dimension=True,
        )
        old_step_logprobs = old_step_logprobs.to(device=new_step_logprobs.device, dtype=new_step_logprobs.dtype)
        if normalize_act_space_dimension:
            old_step_logprobs = old_step_logprobs / self.act_dim_total

        chain_newlogprobs = new_step_logprobs.sum(-1)
        chain_oldlogprobs = old_step_logprobs.sum(-1)
        if account_for_initial_stochasticity:
            init_logprob = self._get_initial_logprob(
                chains,
                normalize_act_space_dimension=normalize_act_space_dimension,
            )
            chain_newlogprobs = chain_newlogprobs + init_logprob
            chain_oldlogprobs = chain_oldlogprobs + init_logprob

        if verbose:
            log.info(
                f"chain_oldlogprobs.min={chain_oldlogprobs.min():5.3f}, max={chain_oldlogprobs.max():5.3f}, std={chain_oldlogprobs.std():5.3f}"
            )
            log.info(
                f"chain_newlogprobs.min={chain_newlogprobs.min():5.3f}, max={chain_newlogprobs.max():5.3f}, std={chain_newlogprobs.std():5.3f}"
            )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if verbose:
            with torch.no_grad():
                advantage_stats = {
                    "mean": f"{advantages.mean().item():2.3f}",
                    "std": f"{advantages.std().item():2.3f}",
                    "max": f"{advantages.max().item():2.3f}",
                    "min": f"{advantages.min().item():2.3f}",
                }
                log.info(f"Chain advantage stats: {advantage_stats}")

        # whole-chain clipped surrogate J_chain
        chain_logratio = chain_newlogprobs - chain_oldlogprobs
        chain_ratio = chain_logratio.exp()
        with torch.no_grad():
            chain_approx_kl = ((chain_ratio - 1) - chain_logratio).mean()
            chain_clipfrac = (
                (chain_ratio - 1.0).abs() > self.chain_ppo_clip_ploss_coef
            ).float().mean().item()

        chain_pg_loss1 = -advantages * chain_ratio
        chain_pg_loss2 = -advantages * torch.clamp(
            chain_ratio,
            1 - self.chain_ppo_clip_ploss_coef,
            1 + self.chain_ppo_clip_ploss_coef,
        )
        chain_pg_loss = torch.max(chain_pg_loss1, chain_pg_loss2).mean()

        # auxiliary per-step clipped surrogate J_step
        step_logratio = new_step_logprobs - old_step_logprobs
        step_ratio = step_logratio.exp()
        aux_advantages = advantages.unsqueeze(1).expand_as(step_ratio)
        with torch.no_grad():
            aux_approx_kl = ((step_ratio - 1) - step_logratio).mean()
            aux_clipfrac = (
                (step_ratio - 1.0).abs() > self.chain_ppo_aux_clip_ploss_coef
            ).float().mean().item()

        aux_pg_loss1 = -aux_advantages * step_ratio
        aux_pg_loss2 = -aux_advantages * torch.clamp(
            step_ratio,
            1 - self.chain_ppo_aux_clip_ploss_coef,
            1 + self.chain_ppo_aux_clip_ploss_coef,
        )
        aux_pg_loss = torch.max(aux_pg_loss1, aux_pg_loss2).mean()

        pg_loss = chain_pg_loss + self.chain_ppo_aux_coef * aux_pg_loss

        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef:
            v_clipped = torch.clamp(newvalues, oldvalues - self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 * torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()

        entropy = step_entropy.mean(-1)
        entropy_loss = -entropy.mean()
        if verbose:
            with torch.no_grad():
                log.info(
                    f"Chain entropy percentiles: 10%={entropy.quantile(0.1):.2f}, 50%={entropy.median():.2f}, 90%={entropy.quantile(0.9):.2f}"
                )

        self.last_chain_ppo_stats = {
            "chain pg loss": chain_pg_loss.item(),
            "chain approx kl": chain_approx_kl.item(),
            "chain ratio": chain_ratio.mean().item(),
            "chain clipfrac": chain_clipfrac,
            "aux pg loss": aux_pg_loss.item(),
            "aux approx kl": aux_approx_kl.item(),
            "aux ratio": step_ratio.mean().item(),
            "aux clipfrac": aux_clipfrac,
            "chain aux coef": self.chain_ppo_aux_coef,
        }

        return (
            pg_loss,
            entropy_loss,
            v_loss,
            0.0,
            chain_clipfrac,
            chain_approx_kl.item(),
            chain_ratio.mean().item(),
            chain_oldlogprobs.min(),
            chain_oldlogprobs.max(),
            chain_oldlogprobs.std(),
            chain_newlogprobs.min(),
            chain_newlogprobs.max(),
            chain_newlogprobs.std(),
            noise_std.item(),
            newvalues.mean().item(),
        )


# Backwards-compatible alias for existing configs and scripts that
# still target model.flow.ft_ppo.ppoflow_fmppo.PPOFlow.
PPOFlow = FlowSDE
