#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a EXTRA_OVERRIDES <<< "${EXTRA_ARGS:-}"
# FMPO-CPPO (FMPO-ChainPPO) on the Gym benchmarks: J_chain + alpha * J_step,
# with the settings used in the paper (Sec. 5.3, App. F.3): lambda_t = a(1 - t),
# alpha = 0.25, chain clip 0.02, step clip 0.01, 601 iterations, seeds 42-46.
#
# Usage:
#   bash script/train_fmppo_chainppo.sh <env> [aux_coef] [seed ...]
#     env:      hopper-v2 | walker2d-v2 | ant-v2 | Humanoid-v3
#     aux_coef: alpha for the auxiliary per-step loss (default 0.25; 0.0 = NoAux ablation)
#     seeds:    default 42 43 44 45 46
#
# Examples:
#   bash script/train_fmppo_chainppo.sh hopper-v2
#   bash script/train_fmppo_chainppo.sh ant-v2 0.0 42      # NoAux, seed 42
#
# Extra Hydra overrides can be passed through EXTRA_ARGS, e.g. the unnormalized
# chain ratio:
#   EXTRA_ARGS="model.chain_ppo_normalize_act_space_dimension=false" bash script/train_fmppo_chainppo.sh hopper-v2

ENV_NAME=${1:?usage: $0 <env> [aux_coef] [seed ...]}
AUX_COEF=${2:-0.25}
shift $(( $# >= 2 ? 2 : 1 ))
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(42 43 44 45 46)
fi

# Best noise level a per environment (lambda_t = a * (1 - t))
case "$ENV_NAME" in
    hopper-v2)   NOISE_A=0.8 ;;
    walker2d-v2) NOISE_A=0.6 ;;
    ant-v2)      NOISE_A=0.3 ;;
    Humanoid-v3) NOISE_A=0.8 ;;
    *) echo "Unknown env: $ENV_NAME"; exit 1 ;;
esac

for seed in "${SEEDS[@]}"; do
    echo "========================================="
    echo "FMPO-ChainPPO env=$ENV_NAME a=$NOISE_A aux_coef=$AUX_COEF seed=$seed"
    echo "========================================="
    "$PYTHON_BIN" script/run.py --config-dir="$PROJECT_DIR/cfg/gym/finetune/${ENV_NAME}" --config-name=ft_ppo_reflow_mlp \
        model._target_=model.flow.ft_ppo.ppoflow_fmppo.FlowSDE \
        model.noise_level_a=${NOISE_A} \
        model.constant_lambda=false \
        model.lambda_schedule=linear \
        model.use_chain_ppo_loss=true \
        model.chain_ppo_aux_coef=${AUX_COEF} \
        model.chain_ppo_clip_ploss_coef=0.02 \
        model.chain_ppo_aux_clip_ploss_coef=0.01 \
        model.chain_ppo_normalize_act_space_dimension=true \
        train.clip_intermediate_actions=false \
        train.n_train_itr=601 \
        train.val_freq=10 \
        train.save_model_freq=9999 \
        wandb=null \
        seed=${seed} \
        "${EXTRA_OVERRIDES[@]}"
done
