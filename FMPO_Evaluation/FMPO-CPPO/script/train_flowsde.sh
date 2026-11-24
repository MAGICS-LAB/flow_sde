#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a EXTRA_OVERRIDES <<< "${EXTRA_ARGS:-}"
# Flow-SDE baseline on the gym benchmarks (paper Table 1: lambda_t = a * (1 - t) / t).
# Noise level a follows the paper: Hopper 0.5, Walker2d 0.9, Ant 0.4, Humanoid 0.5
# (set as noise_level_a in each cfg/gym/finetune/<env>/ft_ppo_reflow_mlp.yaml).
#
# Usage:
#   bash script/train_flowsde.sh <env> [seed ...]
#     env:   hopper-v2 | walker2d-v2 | ant-v2 | Humanoid-v3
#     seeds: default 42 43 44 45 46
#
# Extra Hydra overrides can be passed through EXTRA_ARGS, e.g.
#   EXTRA_ARGS="device=cuda:1" bash script/train_flowsde.sh hopper-v2

ENV_NAME=${1:?usage: $0 <env> [seed ...]}
shift
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(42 43 44 45 46)
fi

for seed in "${SEEDS[@]}"; do
    echo "========================================="
    echo "Flow-SDE env=$ENV_NAME seed=$seed"
    echo "========================================="
    "$PYTHON_BIN" script/run.py --config-dir="$PROJECT_DIR/cfg/gym/finetune/${ENV_NAME}" --config-name=ft_ppo_reflow_mlp \
        model._target_=model.flow.ft_ppo.ppoflow_fmppo.FlowSDE \
        model.constant_lambda=false \
        model.lambda_schedule=ratio \
        model.use_chain_ppo_loss=false \
        train.clip_intermediate_actions=false \
        train.n_train_itr=601 \
        train.val_freq=10 \
        train.save_model_freq=9999 \
        wandb=null \
        seed=${seed} \
        "${EXTRA_OVERRIDES[@]}"
done
