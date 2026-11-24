# Reproduction and artifact provenance

This repository is aligned with the supplied manuscript whose cover says **Last Update: November 18, 2026**. A byte-for-byte copy is included as [paper.pdf](paper.pdf). [artifacts.json](artifacts.json) records its SHA-256 hash and the hashes of the distributed result files.

## What can be reproduced from the included artifacts

`python scripts/reproduce_figures.py` replays the recorded curves for Figure 3 and Figure 4 without training. `python scripts/verify_artifacts.py` checks hashes, the 61-point MuJoCo evaluation grids (iterations 0 through 600), the nine 80-update CIFAR histories, finite metrics, and the seed-averaged violation counts.

| Artifact group | Included | Status |
| --- | ---: | --- |
| MuJoCo main comparison | 60 CSVs | Four tasks, three methods, seeds 42-46 |
| CIFAR fine-tuning | Nine histories | Weak/medium/strong, seeds 42-44 |
| Synthetic raw measurements | None | Full scripts generate new measurements |
| MuJoCo NoAux ablation | None | Launcher and plotter available; requires new runs |
| Original latency records | None | Exporter available for new runs |
| Trained checkpoints | None | Gym download mappings and CIFAR training pipeline available |

The root figure command creates panels from the available histories. It does not assert that the current training source produced those histories. The original records have no corresponding training commit, resolved configuration, or checkpoint hash. Full training, downloads, and A100 timing have not been rerun as part of this cleanup.

## Consolidated source and subsequent fixes

The maintained implementation comes from the newer local author release (`Codebase_release`), rather than the older code previously at the repository root. It includes the ChainPPO mixed objective, NoAux switch, original ODE evaluation, explicit noise schedules, and CPU tests. The original root experiment logs were retained without modification. Unrelated task presets, baseline entry points, demos, review environments, duplicate release archives, and website work were removed from the code repository. Shared upstream modules required by the retained paths remain.

The following corrections affect **new training**, and should be taken into account when comparing it with the distributed historical records:

- Flow-SDE and FMPO-CPPO now retain the unbounded Gaussian latent chain and compute its likelihood before any action bounding. A bounded copy of the final latent is sent to the environment. Gaussian noise is no longer clipped at three standard deviations. Their launchers disable intermediate mean clipping, so the latent Euler-Maruyama transitions use the drift in the paper. The ReinFlow baseline keeps its inherited behavior.
- CIFAR validation is reconstructed from the configurations stored in the actual selected checkpoints. Checkpoints with different split seeds or validation fractions are rejected. This also covers a pretraining `--seed` override, which the static JSON alone could not identify.
- New Gym launches save `run_metadata.json` and `resolved_config.yaml`, including source hash, Git revision and dirty state, configured device, package versions, and the loaded base checkpoint hash. The exporter carries this training metadata forward. Its separate `export_git_commit` describes the checkout at export; legacy runs have unknown training revisions.
- Launchers stop on errors, initialize local data/log paths, and resolve their own working directories. The CIFAR end-to-end driver writes new fine-tuning histories under `results/runs/`, preserving the distributed histories.

Small CPU tests cover sampling/likelihood consistency, ratio normalization, the mixed and NoAux losses, ODE evaluation, PPO agent updates, four task configs, checkpoint split checks, classifier guards, and result export. They establish implementation properties; they do not replace full experiment reproduction.

## Notation and interpretation

For the theory, `I = integral_0^T E[||u_theta-u_star||^2] dt` and `epsilon=sqrt(I)`. The historical `eps_sqrt` field in the two synthetic transfer scripts equals `sqrt(I)`, hence **epsilon** under this definition. Figure 2(b) and its original plotter label that quantity `sqrt(epsilon)`; the legacy values and label are retained for traceability. Comparisons using this axis must account for the notation difference.

The CIFAR validation monitor measures conditional flow-matching MSE. This is a proxy for the theoretical marginal velocity error. The manuscript's 0.225/0.196/0.184 values are reported measurements, not constraints enforced by the epoch-based checkpoint selector. Monitoring uses fixed validation images with fresh time/noise draws, which also introduce Monte Carlo variation.

The Section 5.4 violation rate is computed **after averaging rewards over seeds**. The included records yield Weak 21/45 = 0.466667, Medium 18/43 = 0.418605, and Strong 19/46 = 0.413043. Computing event counts per seed and then pooling them is a different statistic. The analysis script's complement of strict ODE improvement agrees with strict ODE decline on these records because the conditioned transitions contain no zero ODE changes.

The correctness and transfer theorems concern continuous-time inference up to `T<1` under the stated assumptions. Finite integration grids, the endpoint convention for the ratio schedule, and bounded environment actions are practical implementation choices, not additional guarantees established by these scripts.

## Artifact inventory

```bash
python scripts/make_manifest.py
python scripts/make_manifest.py --checkpoints 'path/to/checkpoints/*.pt' --out local_manifest.json
```

The manifest explicitly reports absent groups. Generated outputs and downloaded artifacts are ignored by Git, while the paper snapshot, CSVs, histories, and reference inventory remain versioned. Update `docs/artifacts.json` deliberately if the distributed artifacts themselves change.

## Cleanup validation (November 23, 2026)

Validation ran on Windows with Python 3.12.11 and PyTorch 2.11.0+cpu, using the local scientific runtime and available isolated test dependencies. All **27 CPU tests passed** (18 continuous-control, five CIFAR, four synthetic). All nine available PDF plots were regenerated and visually inspected. All 69 distributed logs match their original bytes, and the paper artifact hash and reported CIFAR violation counts were verified.

The five shell scripts passed Bash syntax checks. Sixteen task/method launcher combinations were exercised with a recording interpreter instead of training, from a directory outside the checkout. Hydra resolved a paper task config without simulator imports or personal logging credentials. The provenance writer was exercised with a dummy checkpoint. CI is configured for Python 3.10 with CPU PyTorch 2.4.1; the remote CI job itself has not run in this local session. A fresh GPU/MuJoCo training environment has not been validated here.
