#!/usr/bin/env bash
# Prepare the reward model, pretrain, fine-tune three seeds, and plot new runs.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/runs}"
"$PYTHON_BIN" -m models.prepare_classifier --config configs/classifier.json
"$PYTHON_BIN" train_fm_cifar.py --config configs/fm_cifar.json
RUN_DIRS=()
for SEED in 42 43 44; do
  RUN_DIR="$OUTPUT_ROOT/CIFAR_Exp_seed${SEED}"
  "$PYTHON_BIN" finetune_sde_grpo.py --config configs/grpo_finetune.json --seed "$SEED" --output_dir "$RUN_DIR"
  RUN_DIRS+=("$RUN_DIR")
done
"$PYTHON_BIN" plot_multiseed_trajectories.py --run-dirs "${RUN_DIRS[@]}" --output-dir "$OUTPUT_ROOT/plots"
"$PYTHON_BIN" analyze_ode_sde_conditional.py --run-dirs "${RUN_DIRS[@]}" --output-dir "$OUTPUT_ROOT/plots"
