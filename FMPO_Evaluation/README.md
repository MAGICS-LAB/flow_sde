# Continuous-control experiments

FMPO-CPPO, Flow-SDE, and ReinFlow on Hopper-v2, Walker2d-v2, Ant-v2, and Humanoid-v3 (Section 5.3, Figure 3; Appendix G, Figure 5).

## Plot supplied records

```bash
python -m pip install -r requirements-plots.txt  # from the repository root
python scripts/reproduce_figures.py --suite mujoco
```

`exp_data/` includes 60 CSVs: four tasks, three methods, and seeds 42-46. The plotter computes the mean and population standard deviation across seeds. NoAux CSVs and original timing records were not supplied. Plotting an incomplete seed set fails with a clear error.

## Training environment

Use Linux with a CUDA-capable PyTorch installation and the legacy MuJoCo 2.1 / `mujoco-py` stack required by Gym 0.22 and D4RL. This is a separate environment from the CPU tests and CIFAR experiments. The original training hardware was one A100 40 GB and a ten-core CPU (Appendix F.3).

```bash
conda create -n fmpo-cppo python=3.10
conda activate fmpo-cppo
cd FMPO_Evaluation/FMPO-CPPO
python -m pip install -e '.[gym]'
```

MuJoCo 2.1 binaries must be available at `~/.mujoco/mujoco210`, with their `bin` directory in `LD_LIBRARY_PATH`. Linux builds also need a compiler, Python development headers, and OpenGL/OSMesa development libraries. Check the simulator installation before launching a full run:

```bash
python -c "import mujoco_py, gym, d4rl.gym_mujoco; e=gym.make('Hopper-v2'); print(e.reset()); e.close()"
```

The package chooses local `data/` and `log/` directories by default. `REINFLOW_DATA_DIR` and `REINFLOW_LOG_DIR` can override them. The launchers work from any directory and use the active Python interpreter (`PYTHON_BIN` overrides it).

## Train the paper methods

From `FMPO_Evaluation/FMPO-CPPO`, replace `<task>` with `hopper-v2`, `walker2d-v2`, `ant-v2`, or `Humanoid-v3`:

```bash
bash script/train_reinflow.sh <task>
bash script/train_flowsde.sh <task>
bash script/train_fmppo_chainppo.sh <task>
bash script/train_fmppo_chainppo.sh <task> 0.0  # NoAux
```

The default seeds are 42-46. To run one seed, append `42` (for FMPO-CPPO, use `<task> 0.25 42`). Additional Hydra overrides can be passed through `EXTRA_ARGS`, for example `EXTRA_ARGS='device=cuda:1'`.

Base policies and normalization statistics use the ReinFlow download mappings in `script/download_url.py`. They are fetched on first use. No trained policy checkpoints are bundled here. The four paper task presets are in `cfg/gym/finetune/`; the method launchers set the schedule and objective. See [EXPERIMENTS.md](../docs/EXPERIMENTS.md) for the noise coefficients and ratio normalization.

## Export new runs

```bash
python script/export_exp_data.py --runs 'log/gym/finetune/*/*' --out ../new_exp_data
python ../plot_gym_tasks.py --data-dir ../new_exp_data --output-dir ../../figures/new_mujoco
python ../plot_gym_tasks.py --data-dir ../new_exp_data --output-dir ../../figures/noaux --ablation
```

If a custom log root is configured, adjust `--runs` accordingly. The exporter writes evaluation CSVs, resolved configs, checkpoint hashes, and latency summaries. It labels the Git commit observed at export separately from training provenance. Historical logs cannot establish which source revision generated them; see [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).
