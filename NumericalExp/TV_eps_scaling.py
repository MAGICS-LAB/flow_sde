

import os
import json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Repro / device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Velocity network uθ(x,t)
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
# Analytic u*(x,t) for linear path with independent coupling:
#   X0 ~ N(0, I),  X1 ~ N(mu, Sigma),  X_t = (1-t)X0 + t X1
# u*(x,t) = E[X1 - X0 | X_t = x]  (closed form because jointly Gaussian)
# -----------------------------
class TrueVelocityGaussianBridge:
    def __init__(self, mu, Sigma):
        """
        mu: (d,) numpy or torch
        Sigma: (d,d) numpy or torch SPD
        """
        mu = torch.tensor(mu, dtype=torch.float32, device=device)
        Sigma = torch.tensor(Sigma, dtype=torch.float32, device=device)
        d = mu.shape[0]
        I = torch.eye(d, device=device)

        self.mu = mu
        self.Sigma = Sigma
        self.I = I
        self.d = d

    def u_star(self, x, t):
        """
        x: (B,d)
        t: (B,) in [0,1)
        returns: (B,d)
        Formula:
          U = X1 - X0
          X_t = X0 + tU
          E[U | X_t=x] = mu + Cov(U,Xt) Cov(Xt)^(-1) (x - t mu)
        where:
          Cov(Xt) = (1-t)^2 I + t^2 Sigma
          Cov(U,Xt)= t Sigma + (t-1) I
        """
        B, d = x.shape
        t_ = t.view(B, 1, 1)                        # (B,1,1)
        mu = self.mu.view(1, d)                     # (1,d)
        I = self.I                                  # (d,d)
        Sigma = self.Sigma                          # (d,d)

        # Build batch matrices
        Ct = ((1 - t_) ** 2) * I + (t_ ** 2) * Sigma  # (B,d,d)
        At = (t_) * Sigma + (t_ - 1.0) * I            # (B,d,d)

        # Solve Ct^{-1} (x - t mu)
        diff = (x - (t.view(B, 1) * mu))              # (B,d)
        diff_vec = diff.unsqueeze(-1)                 # (B,d,1)

        Ct_inv = torch.linalg.inv(Ct)                 # (B,d,d)
        corr = (At @ Ct_inv @ diff_vec).squeeze(-1)   # (B,d)

        return mu + corr

    def sample_xt(self, n, T, seed=0):
        """
        Sample (t, X_t) from the true path p*_t induced by linear coupling:
          X0 ~ N(0,I), X1 ~ N(mu,Sigma), Xt=(1-t)X0 + t X1, with t ~ Uniform[0,T]
        """
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        d = self.d

        t = torch.rand(n, generator=g, device=device) * T
        x0 = torch.randn(n, d, generator=g, device=device)
        # x1 ~ N(mu, Sigma)
        L = torch.linalg.cholesky(self.Sigma)
        z = torch.randn(n, d, generator=g, device=device)
        x1 = self.mu.view(1, d) + z @ L.T

        xt = (1 - t).view(n, 1) * x0 + t.view(n, 1) * x1
        return t, xt

# -----------------------------
# ODE rollout: dY = uθ(Y,t) dt
# -----------------------------
@torch.no_grad()

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

def ode_rollout(net, x0, n_steps=4000, T=0.99):
    dt = T / n_steps
    x = x0
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + net(x, t) * dt
    return x

# -----------------------------
# SDE rollout (Euler–Maruyama):
# dZ = [(1 + λ/2 * t/(1-t)) uθ(Z,t) - (λ/2 * 1/(1-t)) Z] dt + √λ dW
# -----------------------------
@torch.no_grad()
def sde_rollout(net, x0, n_steps=4000, T=0.99, lambda_noise=1.0, seed=0):
    g = torch.Generator(device=x0.device)
    g.manual_seed(seed)

    dt = T / n_steps
    x = x0
    for i in range(n_steps):
        t_val = i * dt
        t = torch.full((x.shape[0],), t_val, device=x.device)

        v = net(x, t)

        one_minus_t = max(1e-6, 1.0 - t_val)
        coef_u = 1.0 + (lambda_noise / 2.0) * (t_val / one_minus_t)
        coef_x = (lambda_noise / 2.0) * (1.0 / one_minus_t)

        drift = (coef_u * v - coef_x * x) * dt
        dW = torch.randn_like(x) * np.sqrt(dt)
        x = x + drift + np.sqrt(lambda_noise) * dW
    return x


@torch.no_grad()
def estimate_epsilon(net, ustar_obj, n_samples=20000, n_time_steps=40, T=0.99, seed=0):
    # discretize t grid in [0,T)
    dt = T / n_time_steps
    total = 0.0
    for i in range(n_time_steps):
        t_val = i * dt
        # sample X_t from p*_t at that fixed t
        g = torch.Generator(device=device)
        g.manual_seed(seed + 1000 * i)

        t = torch.full((n_samples,), t_val, device=device)
        # sample x0, x1, then xt=(1-t)x0 + t x1
        x0 = torch.randn(n_samples, ustar_obj.d, generator=g, device=device)
        L = torch.linalg.cholesky(ustar_obj.Sigma)
        z = torch.randn(n_samples, ustar_obj.d, generator=g, device=device)
        x1 = ustar_obj.mu.view(1, ustar_obj.d) + z @ L.T
        xt = (1 - t_val) * x0 + t_val * x1

        u_true = ustar_obj.u_star(xt, t)
        u_hat = net(xt, t)

        err = ((u_true - u_hat) ** 2).sum(dim=-1)  # ||·||^2
        total += err.mean().item() * dt

    eps_sq = total
    eps = np.sqrt(max(eps_sq, 0.0))
    return eps, eps_sq

# -----------------------------
# TV estimator via KDE (training-free, more accurate than histogram)
# TV(P,Q) = 0.5 * integral |p(x) - q(x)| dx
# -----------------------------
from scipy.stats import gaussian_kde

def estimate_tv_kde(samples_p, samples_q, n_fit=5000, n_eval=10000):
    """
    KDE-based TV estimation (training-free).

    Uses importance sampling with mixture distribution:
    TV = 0.5 * integral |p(x) - q(x)| dx
       = 0.5 * E_r[ |p(x) - q(x)| / r(x) ]  where r = (p + q) / 2

    Args:
        n_fit: samples used to fit KDE (small for speed)
        n_eval: samples used to evaluate integral (can be larger)
    """
    p = samples_p.detach().cpu().numpy().T  # (d, N) for scipy
    q = samples_q.detach().cpu().numpy().T

    # Subsample for KDE fitting (this is what makes it fast)
    n_p, n_q = p.shape[1], q.shape[1]
    idx_p = np.random.choice(n_p, size=min(n_fit, n_p), replace=False)
    idx_q = np.random.choice(n_q, size=min(n_fit, n_q), replace=False)

    # Fit KDEs on subsampled data
    kde_p = gaussian_kde(p[:, idx_p], bw_method='silverman')
    kde_q = gaussian_kde(q[:, idx_q], bw_method='silverman')

    # Evaluate on subsampled combined data
    combined = np.hstack([p, q])
    n_total = combined.shape[1]
    idx_eval = np.random.choice(n_total, size=min(n_eval, n_total), replace=False)
    eval_points = combined[:, idx_eval]

    # Evaluate densities
    p_vals = kde_p(eval_points)
    q_vals = kde_q(eval_points)

    # Mixture density r = (p + q) / 2
    r_vals = 0.5 * (p_vals + q_vals)

    # TV via importance sampling: integral |p - q| / 2 = E_r[ |p - q| / (2r) ]
    weights = np.abs(p_vals - q_vals) / (2.0 * r_vals + 1e-12)
    tv = weights.mean()

    return float(np.clip(tv, 0.0, 1.0))


@torch.no_grad()
def estimate_tv(net, n_samples=100000, n_steps=400, T=0.99, lambda_noise=1.0,
                lim=6.0, bins_list=(50, 80, 120), seed=0):
    """
    Estimate TV(p_T, q_T) using KDE-based method.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x0 = torch.randn(n_samples, 2, generator=g, device=device)
    yT = ode_rollout(net, x0.clone(), n_steps=n_steps, T=T)
    zT = sde_rollout(net, x0.clone(), n_steps=n_steps, T=T, lambda_noise=lambda_noise, seed=seed + 999)

    return estimate_tv_kde(yT, zT, n_fit=5000, n_eval=10000)

# -----------------------------
# Training uθ to approximate u*(x,t) DIRECTLY (so ε matches theorem assumption)
# -----------------------------
def train_to_true_velocity(net, ustar_obj, n_iters=3000, batch_size=512, T=0.99, lr=1e-3, seed=0):
    net.train()
    torch.manual_seed(seed)
    np.random.seed(seed)

    opt = optim.Adam(net.parameters(), lr=lr)
    losses = []
    for it in tqdm(range(n_iters), desc=f"Train {n_iters}", leave=False):
        t, xt = ustar_obj.sample_xt(batch_size, T=T, seed=seed + it)
        u_true = ustar_obj.u_star(xt, t)

        u_hat = net(xt, t)
        loss = ((u_hat - u_true) ** 2).mean()  # proportional to theorem integrand
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

# -----------------------------
# Main validation loop
# -----------------------------
def validate_theorem_4_1():
    # Fixed theorem params
    T = 0.99
    lambda_noise = 1.0

    # Choose Gaussian target so u*(x,t) is analytic
    mu = np.array([2.0, -1.0], dtype=np.float32)
    Sigma = np.array([[0.7, 0.25],
                      [0.25, 0.5]], dtype=np.float32)  # SPD

    ustar = TrueVelocityGaussianBridge(mu=mu, Sigma=Sigma)

    # Different approximation qualities via different training iters
    training_iters = np.linspace(100, 4000, num=20, dtype=int)  # ~100 .. ~4000
    seeds = [0,1,2,3,4,5,6,7,8,9]  # repeat per point for error bars (increase if you want tighter CIs)

    results = []  # list of dicts

    # Seed once before the sweep so the first network's initialization is reproducible.
    # Every run reseeds inside train_to_true_velocity, so later networks are initialized
    # from the RNG state left by the previous run, exactly as in the original sweep.
    torch.manual_seed(0)
    np.random.seed(0)

    for n_iter in training_iters:
        for sd in seeds:
            net = VelocityNetwork(dim=2, hidden_dim=96, n_layers=3).to(device)

            train_to_true_velocity(net, ustar, n_iters=n_iter, batch_size=512, T=T, lr=1e-3, seed=10_000*sd)

            eps, eps_sq = estimate_epsilon(net, ustar, n_samples=20000, n_time_steps=40, T=T, seed=20_000*sd)
            tv = estimate_tv(net, n_samples=50000, n_steps=1000, T=T, lambda_noise=lambda_noise,
                             lim=6.0, bins_list=(50, 80, 120), seed=30_000*sd)

            ratio = tv / (np.sqrt(eps) + 1e-12)  # TV / sqrt(ε), since theorem says TV = O(sqrt ε)

            results.append({
                "iters": int(n_iter),
                "seed": int(sd),
                "eps": float(eps),
                "eps_sq": float(eps_sq),
                "tv": float(tv),
                "ratio": float(ratio),
            })
            print(f"iters={n_iter:5d} seed={sd}  eps={eps:.4e}  TV={tv:.4e}  TV/sqrt(eps)={ratio:.4e}")

    # Aggregate by iters
    iters_unique = sorted(set(r["iters"] for r in results))
    agg = []
    for it in iters_unique:
        sub = [r for r in results if r["iters"] == it]
        eps_vals = np.array([r["eps"] for r in sub])
        tv_vals = np.array([r["tv"] for r in sub])
        ratio_vals = np.array([r["ratio"] for r in sub])
        agg.append({
            "iters": it,
            "eps_mean": eps_vals.mean(),
            "eps_std": eps_vals.std(ddof=1) if len(eps_vals) > 1 else 0.0,
            "tv_mean": tv_vals.mean(),
            "tv_std": tv_vals.std(ddof=1) if len(tv_vals) > 1 else 0.0,
            "ratio_mean": ratio_vals.mean(),
            "ratio_std": ratio_vals.std(ddof=1) if len(ratio_vals) > 1 else 0.0,
        })

    # Fit scaling on all points: log TV = a log eps + b  => TV ~ eps^a
    eps_all = np.array([r["eps"] for r in results])
    tv_all = np.array([r["tv"] for r in results])
    a, b = np.polyfit(np.log(eps_all + 1e-18), np.log(tv_all + 1e-18), 1)
    print(f"\nFitted scaling: TV ~ eps^{a:.3f}  (Theorem 4.1 expects ~ eps^{0.5})")

    # Empirical constant for the inequality TV <= C * sqrt(eps)
    # Use a conservative estimate: 95th percentile over runs of TV/sqrt(eps)
    ratios = np.array([r["ratio"] for r in results])
    C95 = np.quantile(ratios, 0.95)
    print(f"Empirical C (95th pct) s.t. TV <= C*sqrt(eps) holds for ~95% runs: C95 = {C95:.4e}")

    # Check satisfaction rate with this empirical C
    satisfied = (tv_all <= C95 * np.sqrt(eps_all) + 1e-12).mean()
    print(f"Fraction of runs satisfying TV <= C95*sqrt(eps): {100*satisfied:.1f}%")

    save_records_json("theorem4_1_results.json", {
        "config": {"T": T, "lambda_noise": lambda_noise, "training_iters": training_iters, "seeds": seeds},
        "notation": "eps = sqrt(int_0^T E||u* - u_theta||^2 dt) (epsilon in Sec. 5.1 / eq. (4.1)); eps_sq = epsilon^2",
        "runs": results,
        "aggregate": agg,
        "fit_all": {"slope": a, "intercept": b},
        "C95": C95,
    })

    # -------- Plots (clearer) --------
    plot_theorem_4_1(results, agg, a, b, C95)

def plot_theorem_4_1(results, agg, a, b, C95):
    from matplotlib.ticker import LogFormatterSciNotation

    # Use LaTeX-style rendering for better math notation
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 11,
        'mathtext.fontset': 'cm',  # Computer Modern for nice math
    })

    class LogExponentFormatter(LogFormatterSciNotation):
        """Custom formatter that shows just the exponent (e.g., -1 instead of 10^-1)."""
        def __call__(self, x, pos=None):
            if x == 0:
                return '0'
            exponent = int(np.floor(np.log10(abs(x))))
            if 10**exponent == x:
                return f'{exponent}'
            return f'{np.log10(x):.1f}'

    eps_all = np.array([r["eps"] for r in results])
    tv_all = np.array([r["tv"] for r in results])

    # Split data by eps threshold (0.14 ≈ 0.1 * sqrt(2))
    threshold = 0.14
    mask_high = eps_all > threshold
    mask_low = eps_all <= threshold

    eps_high, tv_high = eps_all[mask_high], tv_all[mask_high]
    eps_low, tv_low = eps_all[mask_low], tv_all[mask_low]

    # Fit separate lines for each regime
    if len(eps_high) > 1:
        a_high, b_high = np.polyfit(np.log(eps_high), np.log(tv_high), 1)
    else:
        a_high, b_high = a, b

    if len(eps_low) > 1:
        a_low, b_low = np.polyfit(np.log(eps_low), np.log(tv_low), 1)
    else:
        a_low, b_low = a, b

    print(f"\n=== Two-regime fitting ===")
    print(f"eps > {threshold}: TV ~ eps^{a_high:.3f}  (n={len(eps_high)} points)")
    print(f"eps < {threshold}: TV ~ eps^{a_low:.3f}  (n={len(eps_low)} points)")
    print(f"Theorem 4.1 expects slope = 0.5")

    # Log-log TV vs eps with TWO fitted lines (only plot)
    figsize = (7, 5.5)
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data points with different colors
    ax.loglog(eps_high, tv_high, 'o', color='#8B0000', alpha=0.6,
              markersize=6, label=r'$\epsilon > 0.14$')
    ax.loglog(eps_low, tv_low, 's', color='#d62728', alpha=0.6,
              markersize=5, label=r'$\epsilon \leq 0.14$')

    # Fitted line for high eps regime
    if len(eps_high) > 1:
        xx_high = np.logspace(np.log10(eps_high.min()), np.log10(eps_high.max()), 100)
        fit_high = np.exp(b_high) * xx_high**a_high
        ax.loglog(xx_high, fit_high, '--', color='#8B0000', linewidth=7,
                  label=rf'fit: slope$\sim${a_high:.2f}')

    # Fitted line for low eps regime
    if len(eps_low) > 1:
        xx_low = np.logspace(np.log10(eps_low.min()), np.log10(eps_low.max()), 100)
        fit_low = np.exp(b_low) * xx_low**a_low
        ax.loglog(xx_low, fit_low, '--', color='#d62728', linewidth=7,
                  label=rf'fit: slope$\sim${a_low:.2f}')

    ax.set_xlabel(r'$\log_{10}(\epsilon)$', fontsize=30, labelpad=2)
    ax.set_ylabel(r'$\log_{10}(\mathrm{TV}(p_T||q_T))$', fontsize=30, labelpad=2)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=22)

    # Use exponent-only tick labels
    ax.xaxis.set_major_formatter(LogExponentFormatter())
    ax.yaxis.set_major_formatter(LogExponentFormatter())

    fig.tight_layout()
    fig.savefig('theorem4_1_loglog.pdf', bbox_inches='tight')
    print("Saved theorem4_1_loglog.pdf")
    plt.close('all')

if __name__ == "__main__":
    validate_theorem_4_1()
