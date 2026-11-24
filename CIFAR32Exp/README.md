# CIFAR-10 flow matching and SDE GRPO

Code and recorded histories for Section 5.4, Figure 4, and Appendix F.4.

## Recreate recorded figures

From this directory, after installing the root `requirements-plots.txt`:

```bash
python plot_multiseed_trajectories.py
python analyze_ode_sde_conditional.py
```

The nine supplied histories are under `results/CIFAR_Exp_seed{42,43,44}/{weak,medium,strong}/history.jsonl`. Each contains 80 updates. Plots and the conditional-improvement summary are written to `results/CIFAR_Exp_plots/`. From the repository root, `python scripts/reproduce_figures.py` instead places outputs under `figures/`.

## Train

Use a separate environment with Python 3.11, then run from this directory:

```bash
python -m pip install -r requirements.txt
bash scripts/run_all.sh
```

Alternatively, create the environment described in `environment.yml`. The driver stores new fine-tuning runs under `results/runs/` (override with `OUTPUT_ROOT`) and uses the active Python interpreter; set `PYTHON_BIN` to override it. It prepares the frozen reward classifier, pretrains the conditional flow model, fine-tunes the three base models for seeds 42/43/44, and generates the plots.

Individual stages:

```bash
python -m models.prepare_classifier --config configs/classifier.json
python train_fm_cifar.py --config configs/fm_cifar.json
python finetune_sde_grpo.py --config configs/grpo_finetune.json --seed 43 --output_dir results/new_seed43
```

Entry points accept `--device cpu`, `--device mps`, `--device cuda`, or `--device cuda:N`. The automatic choice is CUDA, then MPS, then CPU. An unavailable requested device raises an error.

CIFAR-10 and the reward model are downloaded on first use. The official training set is split 90/10 for flow pretraining and validation; the monitor reconstructs the split from the loaded base checkpoints and rejects incompatible checkpoint splits; the test set is not used for fine-tuning. A fixed class-balanced validation subset contains 50 images per class. The classifier configuration requires a pretrained CIFAR-10 ResNet-18 and rejects a locally trained fallback; its report records the checkpoint hash. Set `expected_checkpoint_sha256` to pin a particular prepared classifier.

## Configuration and outputs

- `configs/classifier.json`: reward model candidates, preprocessing, and checkpoint checks.
- `configs/fm_cifar.json`: UNet, split, optimizer, and checkpoint selections (epochs 2/12/30).
- `configs/grpo_finetune.json`: SDE schedule, GRPO, monitoring inputs, and fixed subset seed.
- `results/fm_pretrain/checkpoint_manifest.json`: selected base checkpoints.
- New run metadata, monitor subset indices, checkpoints, and histories are saved under the requested output directory.

See the [experiment map](../docs/EXPERIMENTS.md) for the exact seed-averaged statistic and the [reproduction guide](../docs/REPRODUCIBILITY.md) for the provenance limits of the supplied histories. New training can overwrite histories in the default output folders; use `--output_dir` when keeping an additional run.
