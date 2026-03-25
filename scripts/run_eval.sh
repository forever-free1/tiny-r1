#!/usr/bin/env bash
set -euo pipefail

# 用法: bash scripts/run_eval.sh [模型路径]
MODEL_PATH=${1:-outputs/merged/sft}

echo "[EVAL] MODEL_PATH=${MODEL_PATH}"

python eval/eval_math.py --model_path "${MODEL_PATH}" --max_samples 200
python eval/eval_format.py --model_path "${MODEL_PATH}" --max_samples 200
python eval/sample_generations.py --model_path "${MODEL_PATH}"
