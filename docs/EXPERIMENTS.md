# Paper-to-code map

Reference: the November 18, 2026 manuscript in [paper.pdf](paper.pdf).

| Result | Entry point | Inputs | Outputs |
| --- | --- | --- | --- |
| Figure 2(a) | `NumericalExp/TV_eps_scaling.py` | Analytic 2-D Gaussian bridge | `theorem4_1_loglog.pdf`, `theorem4_1_results.json` |
| Figure 2(b) | `NumericalExp/Violation_rate.py` | Pretrained synthetic models and repeated RL seeds | `binned_violation_rate.pdf`, `violation_rate_results.json` |
| Figure 2(c) | `NumericalExp/Trajectory.py` | Synthetic pretraining and SDE fine-tuning | `trajectory_data.pdf`, `trajectory_results.json` |
| Figure 3 | `FMPO_Evaluation/plot_gym_tasks.py` | `FMPO_Evaluation/exp_data/*/seed_*_progress.csv` | Four `*_evaluation.pdf` panels |
| Figure 4(a-c) | `CIFAR32Exp/plot_multiseed_trajectories.py` | Nine supplied `history.jsonl` files | `reward_{weak,medium,strong}.pdf` |
| Figure 4(d) | Same CIFAR plotter | Validation conditional FM losses in the histories | `fm_loss.pdf` |
| Section 5.4 violation rates | `CIFAR32Exp/analyze_ode_sde_conditional.py` | Seed-averaged reward trajectories | `ode_given_sde_summary.json`, `p_ode_given_sde.pdf` |
| Figure 5 | Gym plotter with `--ablation` | Full and NoAux CSVs, five seeds each | Four `*_abl.pdf` panels |
| Table 2 | `FMPO-CPPO/script/export_exp_data.py` | New Hydra runs with `result.pkl` | `latency.csv` and per-run provenance |

## Continuous-control settings

The four task launchers use seeds 42-46, 601 iterations (0-600), evaluation every 10 iterations, and four inference steps. Common policy, critic, rollout, and optimizer settings are in `FMPO_Evaluation/FMPO-CPPO/cfg/gym/finetune/<task>/ft_ppo_reflow_mlp.yaml`.

| Method | Noise schedule | Hopper | Walker2d | Ant | Humanoid |
| --- | --- | ---: | ---: | ---: | ---: |
| Flow-SDE | `a * (1-t) / max(t, 1/K)` | 0.5 | 0.9 | 0.4 | 0.5 |
| FMPO-CPPO | `a * (1-t)` | 0.8 | 0.6 | 0.3 | 0.8 |
| ReinFlow | Learned exploration noise | Config | Config | Config | Config |

FMPO-CPPO uses `alpha=0.25`, chain clip `0.02`, and step clip `0.01`. Its log-ratios are normalized by action-chunk dimension `D=H*d_a`; the auxiliary term averages over inference steps. These are tempered PPO ratios, as described in Section 3.3. Flow-SDE keeps the baseline normalization by `(K+1)*D` described in Appendix F.3. NoAux sets only `alpha=0`.

## CIFAR settings and statistics

The common UNet has width 96 and conditioning dimension 192. Pretraining runs for 30 epochs; the configured weak/medium/strong selections are epochs 2/12/30. Fine-tuning uses 80 updates, ten class visits per update, group size 32, ten SDE training steps, twenty monitor steps, learning rate `4e-6`, and KL weight `0.05`. The noise amplitude is `0.7 * sqrt((1-t)/t)` with the finite-grid endpoint convention in `models/samplers.py`.

The paper reports validation conditional FM losses 0.225/0.196/0.184. These are reported measurements, not values guaranteed by checkpoint selection at those epochs. Conditional FM loss is a proxy for the theoretical marginal error, not its exact value.

For the Section 5.4 statistic, first average each reward trajectory over seeds, then take consecutive differences. Count strictly positive SDE changes and strictly negative ODE changes. The supplied records give 21/45 (Weak), 18/43 (Medium), and 19/46 (Strong) violations. Pooling transitions across seeds computes a different statistic.
