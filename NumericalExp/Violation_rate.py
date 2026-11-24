"""
Theorem 4.2 validation with 10 random seeds: collects checkpoint pairs, saves them to violation_rate_results.json, and plots the binned violation rate.

Based on thm_4_2.py but:
  - Runs 10 different random seeds
  - No plotting
  - Saves all raw data for later analysis
"""

import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# -----------------------------
# Repro helpers
# -----------------------------

def save_records_json(path, payload):
    """Save raw measurements (numpy scalars/arrays converted) so figures can be re-plotted and checked."""
    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Not JSON serializable: {type(obj).__name__}")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_default)
    print(f"Saved {path}")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# Policy model u_theta(x,t)
# -----------------------------
class VelocityNetwork(nn.Module):
    def __init__(self, dim=2, hidden_dim=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


# -----------------------------
# True velocity for Gaussian bridge (ONLY for eps measurement)
# -----------------------------
class TrueVelocityGaussianBridge:
    def __init__(self, mu, Sigma):
        self.mu = torch.tensor(mu, dtype=torch.float32, device=device)
        self.Sigma = torch.tensor(Sigma, dtype=torch.float32, device=device)
        self.d = self.mu.shape[0]
        self.I = torch.eye(self.d, device=device, dtype=torch.float32)

    def u_star(self, x, t):
        B, d = x.shape
        t_ = t.view(B, 1, 1)
        mu = self.mu.view(1, d)
        Ct = ((1 - t_) ** 2) * self.I + (t_ ** 2) * self.Sigma
        At = (t_) * self.Sigma + (t_ - 1.0) * self.I
        diff = (x - (t.view(B, 1) * mu)).unsqueeze(-1)
        Ct_inv = torch.linalg.inv(Ct)
        return mu + (At @ Ct_inv @ diff).squeeze(-1)

    def sample_xt(self, n, T, seed=0):
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        t = torch.rand(n, generator=g, device=device) * T
        x0 = torch.randn(n, self.d, generator=g, device=device)

        L = torch.linalg.cholesky(self.Sigma)
        z = torch.randn(n, self.d, generator=g, device=device)
        x1 = self.mu.view(1, self.d) + z @ L.T

        xt = (1 - t).view(n, 1) * x0 + t.view(n, 1) * x1
        return t, xt


# -----------------------------
# Reward: bounded in (0,1]
# -----------------------------
def get_scaled_mahal_reward(mu_tgt, Sigma_base, temp: float, min_log: float = -40.0):
    mu = torch.tensor(mu_tgt, device=device, dtype=torch.float32).view(1, -1)
    Sigma = torch.tensor(Sigma_base, device=device, dtype=torch.float32)
    Sigma_inv = torch.linalg.inv(Sigma)

    def reward(x):
        diff = x - mu
        mahal = (diff @ Sigma_inv * diff).sum(dim=1)
        log_r = (-0.5 * temp * mahal).clamp(min=min_log, max=0.0)
        return torch.exp(log_r)

    return reward


# -----------------------------
# ODE rollout (eval)
# -----------------------------
@torch.no_grad()
def ode_rollout(net, x0, n_steps=250, T=0.97):
    dt = T / n_steps
    x = x0.clone()
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
        x = x + net(x, t) * dt
    return x


# -----------------------------
# SDE rollout (Euler-Maruyama)
# -----------------------------
def sde_rollout_diff(net, x0, n_steps=80, T=0.97, lambda_noise=3.0, seed=0):
    g = torch.Generator(device=x0.device)
    g.manual_seed(seed)

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    sqrt_lam = math.sqrt(lambda_noise)

    x = x0
    for i in range(n_steps):
        t_val = i * dt
        t = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)

        v = net(x, t)

        one_minus_t = max(1e-6, 1.0 - t_val)
        coef_u = 1.0 + (lambda_noise / 2.0) * (t_val / one_minus_t)
        coef_x = (lambda_noise / 2.0) * (1.0 / one_minus_t)

        drift = (coef_u * v - coef_x * x) * dt
        dW = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * sqrt_dt

        x = x + drift + sqrt_lam * dW

    return x


@torch.no_grad()
def sde_rollout_nograd(net, x0, n_steps=250, T=0.97, lambda_noise=3.0, seed=0):
    g = torch.Generator(device=x0.device)
    g.manual_seed(seed)

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    sqrt_lam = math.sqrt(lambda_noise)

    x = x0.clone()
    for i in range(n_steps):
        t_val = i * dt
        t = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)

        v = net(x, t)

        one_minus_t = max(1e-6, 1.0 - t_val)
        coef_u = 1.0 + (lambda_noise / 2.0) * (t_val / one_minus_t)
        coef_x = (lambda_noise / 2.0) * (1.0 / one_minus_t)

        drift = (coef_u * v - coef_x * x) * dt
        dW = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * sqrt_dt

        x = x + drift + sqrt_lam * dW

    return x


# -----------------------------
# eps(theta) measurement
#   eps_int  := ∫_0^T E[||u* - uθ||^2] dt   (integrated flow-matching error; ε in App. F.2)
#   eps_sqrt := sqrt(eps_int)               (√ε in App. F.2 / Sec. 5.2; the x-axis of the plot)
# -----------------------------
@torch.no_grad()
def estimate_eps_both(net, ustar, n_samples=20000, T=0.97, seed=123):
    t, xt = ustar.sample_xt(n_samples, T=T, seed=seed)
    u_true = ustar.u_star(xt, t)
    u_pred = net(xt, t)
    err2 = ((u_true - u_pred) ** 2).sum(dim=1).mean().item()
    eps_int = float(T) * float(err2)
    eps_sqrt = math.sqrt(max(eps_int, 0.0))
    return eps_int, eps_sqrt


# -----------------------------
# TV estimator (2D histogram plug-in)
# -----------------------------
def estimate_tv_hist(samples_p, samples_q, bins=90):
    p = np.asarray(samples_p)
    q = np.asarray(samples_q)
    assert p.shape[1] == 2 and q.shape[1] == 2, "TV estimator is 2D only."

    allx = np.concatenate([p, q], axis=0)
    mins = allx.min(axis=0)
    maxs = allx.max(axis=0)
    pad = 0.05 * (maxs - mins + 1e-6)
    mins -= pad
    maxs += pad

    Hp, _, _ = np.histogram2d(
        p[:, 0], p[:, 1],
        bins=bins,
        range=[[mins[0], maxs[0]], [mins[1], maxs[1]]],
        density=False
    )
    Hq, _, _ = np.histogram2d(
        q[:, 0], q[:, 1],
        bins=bins,
        range=[[mins[0], maxs[0]], [mins[1], maxs[1]]],
        density=False
    )

    P = Hp / max(Hp.sum(), 1.0)
    Q = Hq / max(Hq.sum(), 1.0)
    return 0.5 * np.abs(P - Q).sum()


# -----------------------------
# Evaluation: JSDE, JODE, TV
# -----------------------------
@torch.no_grad()
def evaluate_js_and_tv(
    net,
    reward_fn,
    n_samples=20000,
    T=0.97,
    lambda_noise=3.0,
    rollout_steps=250,
    tv_bins=90,
    seed=0,
):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x0 = torch.randn(n_samples, 2, generator=g, device=device)

    zT = sde_rollout_nograd(net, x0, n_steps=rollout_steps, T=T, lambda_noise=lambda_noise, seed=seed + 1)
    yT = ode_rollout(net, x0, n_steps=rollout_steps, T=T)

    jsde = reward_fn(zT).mean().item()
    jode = reward_fn(yT).mean().item()

    tv = estimate_tv_hist(
        samples_p=yT.detach().cpu().numpy(),
        samples_q=zT.detach().cpu().numpy(),
        bins=tv_bins,
    )
    return jsde, jode, tv


# -----------------------------
# Stats helpers
# -----------------------------
def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    rad = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2)))
    return max(0.0, center - rad), min(1.0, center + rad)


# ============================================================
# Stage A: Supervised pretraining (flow matching MSE to u*)
# ============================================================
def supervised_pretrain(
    net: nn.Module,
    ustar: TrueVelocityGaussianBridge,
    T: float,
    steps: int,
    batch_size: int,
    lr: float,
    log_every: int,
    eps_eval_n: int,
    eps_eval_seed: int,
):
    net.train()
    opt = optim.Adam(net.parameters(), lr=lr)

    history = []
    pbar = tqdm(range(1, steps + 1), desc="Supervised pretrain (FM)")

    for it in pbar:
        t, xt = ustar.sample_xt(batch_size, T=T, seed=eps_eval_seed + it)
        with torch.no_grad():
            u_true = ustar.u_star(xt, t)
        u_pred = net(xt, t)
        loss = ((u_pred - u_true) ** 2).sum(dim=1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % log_every == 0 or it == 1 or it == steps:
            eps_int, eps_sqrt = estimate_eps_both(net, ustar, n_samples=eps_eval_n, T=T, seed=eps_eval_seed)
            history.append(
                dict(
                    step=it,
                    fm_loss=float(loss.item()),
                    eps_int=float(eps_int),
                    eps_sqrt=float(eps_sqrt),
                    state_dict_cpu={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                )
            )
            pbar.set_postfix(loss=f"{loss.item():.3e}", eps=f"{eps_sqrt:.3f}")

    return history


# ============================================================
# Stage B: RL fine-tuning from a given start
# ============================================================
def rl_finetune_from_start(
    start_name: str,
    start_state_dict_cpu: dict,
    seed_run: int,
    cfg: dict,
):
    set_seed(seed_run)

    net = VelocityNetwork(hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"]).to(device)
    net.load_state_dict({k: v.to(device) for k, v in start_state_dict_cpu.items()})
    net.train()

    opt = optim.Adam(net.parameters(), lr=cfg["rl_lr"])

    checkpoints = []
    pbar = tqdm(range(1, cfg["rl_steps"] + 1), desc=f"RL finetune start={start_name} seed={seed_run}")

    for step in pbar:
        g = torch.Generator(device=device)
        g.manual_seed(seed_run + 10_000 + step)
        x0 = torch.randn(cfg["rl_batch_size"], 2, generator=g, device=device)

        zT = sde_rollout_diff(
            net, x0,
            n_steps=cfg["sde_steps_train"],
            T=cfg["T"],
            lambda_noise=cfg["lambda_noise"],
            seed=seed_run + 20_000 + step,
        )

        r = cfg["reward_fn"](zT)
        loss = -r.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if cfg["grad_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])

        opt.step()

        if step % cfg["rl_log_every"] == 0:
            pbar.set_postfix(loss=f"{loss.item():.3e}", r=f"{r.mean().item():.3e}")

        if step in cfg["rl_checkpoint_steps"]:
            net.eval()
            eval_seed = cfg["eval_seed_base"] + seed_run * 10_000 + cfg["start_id_map"][start_name]
            jsde, jode, tv = evaluate_js_and_tv(
                net,
                cfg["reward_fn"],
                n_samples=cfg["n_eval"],
                T=cfg["T"],
                lambda_noise=cfg["lambda_noise"],
                rollout_steps=cfg["rollout_steps_eval"],
                tv_bins=cfg["tv_bins"],
                seed=eval_seed,
            )
            eps_int, eps_sqrt = estimate_eps_both(
                net,
                cfg["ustar"],
                n_samples=cfg["n_eps"],
                T=cfg["T"],
                seed=cfg["eps_seed_base"] + seed_run * 1000 + cfg["start_id_map"][start_name],
            )

            checkpoints.append(
                dict(
                    start=start_name,
                    seed=seed_run,
                    step=step,
                    JSDE=jsde,
                    JODE=jode,
                    TV=tv,
                    eps_int=eps_int,
                    eps_sqrt=eps_sqrt,
                )
            )
            net.train()

    return checkpoints


# ============================================================
# Stage C: Build transitions
# ============================================================
def build_pairs(all_checkpoints, R=1.0, tol_improve=1e-6):
    by_run = {}
    for cp in all_checkpoints:
        key = (cp["start"], cp["seed"])
        by_run.setdefault(key, []).append(cp)
    for key in by_run:
        by_run[key] = sorted(by_run[key], key=lambda d: d["step"])

    pairs = []
    for (start, seed), cps in by_run.items():
        for k in range(len(cps) - 1):
            a = cps[k]
            b = cps[k + 1]
            dSDE = b["JSDE"] - a["JSDE"]
            dODE = b["JODE"] - a["JODE"]

            if dSDE > tol_improve:
                TV_a, TV_b = a["TV"], b["TV"]
                penalty = 2.0 * R * (TV_a + TV_b)
                lower_bound = dSDE - penalty

                ode_mono_viol = (dODE < 0.0)
                lb_allows_neg = (lower_bound < 0.0)

                avg_eps_sqrt = 0.5 * (a["eps_sqrt"] + b["eps_sqrt"])
                avg_eps_int = 0.5 * (a["eps_int"] + b["eps_int"])

                pairs.append(
                    dict(
                        start=start,
                        seed=seed,
                        step_a=a["step"],
                        step_b=b["step"],
                        ckpt_idx=a["step"] // 5,  # checkpoint index for filtering
                        dSDE=dSDE,
                        dODE=dODE,
                        TV_a=TV_a,
                        TV_b=TV_b,
                        penalty=penalty,
                        lower_bound=lower_bound,
                        ode_mono_viol=ode_mono_viol,
                        lb_allows_neg=lb_allows_neg,
                        eps_sqrt=avg_eps_sqrt,
                        eps_int=avg_eps_int,
                    )
                )
    return pairs


def plot_binned_violation_rate(pairs, n_bins=6, output_file='binned_violation_rate.pdf'):
    """Plot binned violation rate with confidence intervals across seeds."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 11,
        'mathtext.fontset': 'cm',
    })

    # Extract data
    eps_sqrt = np.array([p['eps_sqrt'] for p in pairs])
    viol = np.array([p['ode_mono_viol'] for p in pairs])
    seeds = np.array([p['seed'] for p in pairs])
    unique_seeds = np.unique(seeds)

    # Quantile binning
    edges = np.quantile(eps_sqrt, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_ids = np.digitize(eps_sqrt, edges[1:-1], right=False)

    # Compute statistics per bin
    bin_stats = []
    for b in range(n_bins):
        mask = (bin_ids == b)
        if mask.sum() == 0:
            continue

        eps_vals = eps_sqrt[mask]
        seed_rates = []
        for seed in unique_seeds:
            seed_mask = mask & (seeds == seed)
            if seed_mask.sum() > 0:
                seed_rates.append(viol[seed_mask].sum() / seed_mask.sum())

        seed_rates = np.array(seed_rates)
        if len(seed_rates) > 0:
            mean_rate = np.mean(seed_rates)
            std_rate = np.std(seed_rates, ddof=1) if len(seed_rates) > 1 else 0.0
            n_seeds = len(seed_rates)
            ci_margin = 1.96 * std_rate / np.sqrt(n_seeds) if n_seeds > 1 else 0.0

            bin_stats.append({
                'eps_median': np.median(eps_vals),
                'rate': mean_rate,
                'ci_low': max(0, mean_rate - ci_margin),
                'ci_high': min(1, mean_rate + ci_margin),
            })

    # Plot
    eps_medians = np.array([s['eps_median'] for s in bin_stats])
    rates = np.array([s['rate'] for s in bin_stats])
    ci_lows = np.array([s['ci_low'] for s in bin_stats])
    ci_highs = np.array([s['ci_high'] for s in bin_stats])

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.errorbar(eps_medians, rates, yerr=[rates - ci_lows, ci_highs - rates],
                fmt='o', color='#8B0000', markersize=8, capsize=5,
                capthick=2, elinewidth=2, label='95% CI')
    ax.plot(eps_medians, rates, '-', color='#8B0000', alpha=0.5, linewidth=7)

    ax.set_xlabel(r'$\sqrt{\epsilon}$', fontsize=26, labelpad=2)
    ax.set_ylabel('Violation Rate', fontsize=22, labelpad=2)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(bottom=-0.05)
    ax.legend(loc='upper left', fontsize=22)

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.close('all')


# ============================================================
# Full experiment runner
# ============================================================
def run_theorem_4_2_validation_10seeds():
    # -------------------------
    # Global config
    # -------------------------
    base_seed = 0
    set_seed(base_seed)

    T = 0.99
    lambda_noise = 1.0
    R = 1.0

    mu_tgt = np.array([2.5, 2.5], dtype=np.float32)
    Sigma_base = np.array([[0.25, 0.18],
                           [0.18, 0.80]], dtype=np.float32)

    temp = 3.0
    reward_fn = get_scaled_mahal_reward(mu_tgt, Sigma_base, temp=temp, min_log=-40.0)

    Sigma_reward = Sigma_base / temp
    ustar = TrueVelocityGaussianBridge(mu_tgt, Sigma_reward)

    hidden_dim = 128
    n_layers = 3

    # -------------------------
    # Stage A: supervised pretrain (FM)
    # -------------------------
    pre_net = VelocityNetwork(hidden_dim=hidden_dim, n_layers=n_layers).to(device)

    pre_cfg = dict(
        T=T,
        pre_steps=900,
        pre_batch_size=2048,
        pre_lr=2e-3,
        pre_log_every=45,
        eps_eval_n=15000,
        eps_eval_seed=777,
    )

    pre_hist = supervised_pretrain(
        pre_net,
        ustar=ustar,
        T=T,
        steps=pre_cfg["pre_steps"],
        batch_size=pre_cfg["pre_batch_size"],
        lr=pre_cfg["pre_lr"],
        log_every=pre_cfg["pre_log_every"],
        eps_eval_n=pre_cfg["eps_eval_n"],
        eps_eval_seed=pre_cfg["eps_eval_seed"],
    )

    # Select 20 base models
    n_starts = 20

    n_available = len(pre_hist)
    if n_starts >= n_available:
        chosen_indices = list(range(n_available))
    else:
        chosen_indices = [int(round(1 + (n_available - 2) * i / (n_starts - 1))) for i in range(n_starts)]
        chosen_indices = sorted(set(chosen_indices))

    chosen = [pre_hist[idx] for idx in chosen_indices]
    starts = []
    for i, h in enumerate(chosen):
        starts.append(
            dict(
                name=f"pre_{i:02d}",
                pre_step=h["step"],
                pre_eps_sqrt=h["eps_sqrt"],
                pre_eps_int=h["eps_int"],
                state_dict_cpu=h["state_dict_cpu"],
            )
        )
    starts = sorted(starts, key=lambda d: d["pre_eps_sqrt"], reverse=True)

    print("\nSelected pretrain starts (worst -> best):")
    for s in starts:
        print(f"  {s['name']}: pre_step={s['pre_step']}, eps_sqrt={s['pre_eps_sqrt']:.4f}")

    # -------------------------
    # Stage B: RL fine-tune with 10 seeds
    # -------------------------
    seeds = [base_seed + i for i in range(10)]  # 10 seeds

    rl_steps = 300
    rl_checkpoint_every = 5
    rl_checkpoint_steps = set(range(rl_checkpoint_every, rl_steps + 1, rl_checkpoint_every))

    rl_cfg = dict(
        T=T,
        lambda_noise=lambda_noise,
        R=R,
        reward_fn=reward_fn,
        ustar=ustar,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        rl_steps=rl_steps,
        rl_batch_size=256,
        rl_lr=5e-5,
        grad_clip=1.0,
        sde_steps_train=100,
        rl_log_every=25,
        rl_checkpoint_steps=rl_checkpoint_steps,
        n_eval=20000,
        rollout_steps_eval=400,
        tv_bins=100,
        n_eps=20000,
        eval_seed_base=999_000,
        eps_seed_base=555_000,
        start_id_map={s["name"]: i for i, s in enumerate(starts)},
    )

    all_checkpoints = []
    for st in starts:
        for sd in seeds:
            cps = rl_finetune_from_start(
                start_name=st["name"],
                start_state_dict_cpu=st["state_dict_cpu"],
                seed_run=sd,
                cfg=rl_cfg,
            )
            all_checkpoints.extend(cps)

    print(f"\nCollected {len(all_checkpoints)} RL checkpoints across {len(starts)} starts x {len(seeds)} seeds.")

    # -------------------------
    # Stage C: build pairs
    # -------------------------
    pairs = build_pairs(all_checkpoints, R=R, tol_improve=1e-12)
    print(f"Built {len(pairs)} ΔSDE>0 transitions.")

    save_records_json("violation_rate_results.json", {
        "notation": "eps_int = int_0^T E||u* - u_theta||^2 dt; eps_sqrt = sqrt(eps_int) (sqrt(eps) in App. F.2)",
        "starts": [{k: v for k, v in s.items() if k != "state_dict_cpu"} for s in starts],
        "seeds": seeds,
        "checkpoints": all_checkpoints,
        "pairs": pairs,
    })

    # -------------------------
    # Plot binned violation rate
    # -------------------------
    plot_binned_violation_rate(pairs, n_bins=6, output_file='binned_violation_rate.pdf')


if __name__ == "__main__":
    run_theorem_4_2_validation_10seeds()
