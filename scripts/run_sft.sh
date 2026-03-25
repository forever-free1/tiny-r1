#!/usr/bin/env bash
set -euo pipefail

# 用法: bash scripts/run_sft.sh [GPU数] [配置路径] [数据目录(可选)]
NUM_GPUS=${1:-2}
CONFIG_PATH=${2:-configs/sft_baseline.yaml}
DATA_PATH=${3:-}

echo "[SFT] NUM_GPUS=${NUM_GPUS}"
echo "[SFT] CONFIG=${CONFIG_PATH}"
echo "[SFT] DATA_PATH=${DATA_PATH:-<HF dataset>}"

if [ -n "${DATA_PATH}" ]; then
  accelerate launch --num_processes "${NUM_GPUS}" train/train_sft.py \
    --config "${CONFIG_PATH}" \
    --dataset_path "${DATA_PATH}"
else
  accelerate launch --num_processes "${NUM_GPUS}" train/train_sft.py \
    --config "${CONFIG_PATH}"
fi
