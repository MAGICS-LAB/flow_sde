# FMPO-CPPO implementation

Paper-specific continuous-control code, derived from ReinFlow and DPPO.

The [continuous-control guide](../README.md) covers installation, all three methods, NoAux, data downloads, and exporting results. The [experiment map](../../docs/EXPERIMENTS.md) lists the paper settings.

| Component | Location |
| --- | --- |
| SDE policy and ChainPPO objective | `model/flow/ft_ppo/ppoflow_fmppo.py` |
| ReinFlow baseline | `model/flow/ft_ppo/ppoflow.py` |
| Rollouts and PPO updates | `agent/finetune/reinflow/train_ppo_flow_agent.py` |
| Buffer, including per-transition log probabilities | `agent/finetune/reinflow/buffer.py` |
| Four paper task presets | `cfg/gym/finetune/` |
| Method launchers and result exporter | `script/` |
| CPU property and integration tests | `tests/` |

Shared upstream modules required by these paths are retained. Unrelated upstream task presets, baselines, rendering demos, and development utilities have been removed. Upstream copyright notices and [LICENSE](LICENSE) are retained.
