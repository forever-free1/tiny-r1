#!/usr/bin/env bash
set -euo pipefail

# 用法: bash scripts/run_grpo.sh [GPU数] [配置路径] [数据目录(可选)]
NUM_GPUS=${1:-2}
CONFIG_PATH=${2:-configs/grpo_math.yaml}
DATA_PATH=${3:-}

echo "[GRPO] NUM_GPUS=${NUM_GPUS}"
echo "[GRPO] CONFIG=${CONFIG_PATH}"
echo "[GRPO] DATA_PATH=${DATA_PATH:-<HF dataset>}"

if [ -n "${DATA_PATH}" ]; then
  accelerate launch --num_processes "${NUM_GPUS}" train/train_grpo.py \
    --config "${CONFIG_PATH}" \
    --dataset_path "${DATA_PATH}"
else
  accelerate launch --num_processes "${NUM_GPUS}" train/train_grpo.py \
    --config "${CONFIG_PATH}"
fi
