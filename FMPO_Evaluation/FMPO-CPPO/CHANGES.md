# Paper implementation and upstream changes

This component is derived from ReinFlow and DPPO; original license notices remain in the source.

| Method | Model target | Training objective | Noise schedule |
| --- | --- | --- | --- |
| ReinFlow | `model.flow.ft_ppo.ppoflow.PPOFlow` | Inherited PPO | Learned |
| Flow-SDE | `model.flow.ft_ppo.ppoflow_fmppo.FlowSDE` | PPO on a normalized chain log-likelihood | `a*(1-t)/max(t,1/K)` |
| FMPO-CPPO | Same `FlowSDE` class, `use_chain_ppo_loss=true` | `J_chain + alpha*J_step` | `a*(1-t)` |

FMPO-CPPO normalizes log-ratios by action-chunk dimension `D=H*d_a`, uses chain clip 0.02 and step clip 0.01, and averages the auxiliary surrogate over inference steps. These ratios are tempered, as described in Section 3.3. Flow-SDE uses the baseline normalization by `(K+1)*D` from Appendix F.3. Entropy normalization is handled separately from likelihood-ratio normalization.

`chain_ppo_aux_coef=0.25` is the main method; zero disables the auxiliary term. ChainPPO forces the linear noise schedule. The agent and buffer store per-transition old log probabilities for the step-wise term.

ODE evaluation uses `x += u_theta(x,t)*dt`, without the SDE drift correction. Flow-SDE and FMPO-CPPO store unbounded Gaussian latent chains; only the environment action copy is bounded. Their launchers disable intermediate mean clipping. This likelihood correction changes new sampling relative to the earlier supplied release. The ReinFlow baseline retains its inherited sampler.

New runs capture launch-time source/configuration and checkpoint metadata. The exporter distinguishes the training revision from the export revision and supports NoAux folders and noise-level searches.

See the [experiment map](../../docs/EXPERIMENTS.md) for settings and [reproduction notes](../../docs/REPRODUCIBILITY.md) for the provenance and coverage of historical result files.
