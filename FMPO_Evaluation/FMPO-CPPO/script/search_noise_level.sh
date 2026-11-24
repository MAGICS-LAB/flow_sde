#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
# Noise-level search used to choose `a` for Flow-SDE and FMPO-CPPO (App. F.3):
# runs a = 0.1, 0.2, ..., 0.9 on one task and seed. Compare the runs with
#   python script/export_exp_data.py --split-noise-level --runs "<run dirs>" --out <dir>
# which writes one folder per noise level (<task>_<method>_a<value>).
#
# Usage:
#   bash script/search_noise_level.sh <method> <env> [seed]
#     method: flowsde | chainppo
#     env:    hopper-v2 | walker2d-v2 | ant-v2 | Humanoid-v3
#     seed:   default 42

METHOD=${1:?usage: $0 <flowsde|chainppo> <env> [seed]}
ENV_NAME=${2:?usage: $0 <flowsde|chainppo> <env> [seed]}
SEED=${3:-42}

case "$METHOD" in
    flowsde)  LAUNCHER=script/train_flowsde.sh ;;
    chainppo) LAUNCHER=script/train_fmppo_chainppo.sh ;;
    *) echo "Unknown method: $METHOD (use flowsde or chainppo)"; exit 1 ;;
esac

for A in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    if [ "$METHOD" = chainppo ]; then
        EXTRA_ARGS="model.noise_level_a=${A} ${EXTRA_ARGS:-}" bash "$LAUNCHER" "$ENV_NAME" 0.25 "$SEED"
    else
        EXTRA_ARGS="model.noise_level_a=${A} ${EXTRA_ARGS:-}" bash "$LAUNCHER" "$ENV_NAME" "$SEED"
    fi
done
