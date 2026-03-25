# tiny-r1

`tiny-r1` 是一个可复现的小型 DeepSeek-R1 风格训练项目模板，聚焦「不改模型结构」前提下的两阶段流程：

1. 基于 `Qwen/Qwen2.5-0.5B` 做指令微调（SFT）
2. 基于数学数据做 mini-GRPO（强化学习）
3. 支持量化与导出到 ModelScope

项目默认采用：
- `transformers`
- `trl`
- `accelerate`

并提供 2×3090 / 4×3090 的分布式训练脚本。

## 1. 项目目标

- 复现一个「R1 风格」的小规模训练路径（SFT -> GRPO）
- 使用公开数据集：
  - SFT：`open-r1/Mixture-of-Thoughts`
  - RL：`open-r1/OpenR1-Math-220k`
- 保持工程可扩展：配置文件驱动、脚本化运行、可复用数据处理逻辑
- 最终可进行 4-bit 量化与 ModelScope 发布准备

## 2. 目录结构

```text
tiny-r1/
├── configs/
│   ├── sft_baseline.yaml
│   ├── sft_variant.yaml
│   └── grpo_math.yaml
├── data/
│   ├── prepare_sft.py
│   ├── prepare_math_rl.py
│   └── prompts.py
├── train/
│   ├── train_sft.py
│   ├── train_grpo.py
│   └── merge_lora.py
├── eval/
│   ├── eval_math.py
│   ├── eval_format.py
│   └── sample_generations.py
├── deploy/
│   ├── quantize.py
│   ├── export_modelscope.py
│   └── inference_demo.py
├── scripts/
│   ├── run_sft.sh
│   ├── run_grpo.sh
│   └── run_eval.sh
├── README.md
├── report.md
└── requirements.txt
```

## 3. 环境依赖

### 3.1 Python 与 CUDA

- Python: `>=3.10`
- CUDA: 建议 `12.x`
- GPU: 2×3090 或 4×3090（24GB）

### 3.2 安装依赖

```bash
pip install -r requirements.txt
```

## 4. 训练流程

### 4.1 数据准备

```bash
python data/prepare_sft.py \
  --dataset_name open-r1/Mixture-of-Thoughts \
  --output_dir data/processed/sft

python data/prepare_math_rl.py \
  --dataset_name open-r1/OpenR1-Math-220k \
  --output_dir data/processed/math_rl
```

### 4.2 SFT 训练（2 卡示例）

```bash
bash scripts/run_sft.sh 2 configs/sft_baseline.yaml
```

### 4.3 GRPO 训练（4 卡示例）

```bash
bash scripts/run_grpo.sh 4 configs/grpo_math.yaml
```

### 4.4 LoRA 合并

```bash
python train/merge_lora.py \
  --base_model Qwen/Qwen2.5-0.5B \
  --lora_path outputs/sft/checkpoint-final \
  --output_dir outputs/merged/sft
```

### 4.5 评估与采样

```bash
bash scripts/run_eval.sh outputs/merged/sft
```

### 4.6 量化与部署准备

```bash
python deploy/quantize.py \
  --model_path outputs/merged/sft \
  --output_dir outputs/quantized/sft-4bit

python deploy/export_modelscope.py \
  --model_dir outputs/quantized/sft-4bit \
  --repo_name your-name/tiny-r1-qwen2.5-0.5b
```

## 5. 分布式训练说明

脚本通过 `accelerate launch --num_processes N` 控制卡数：

- `N=2` 对应 2×3090
- `N=4` 对应 4×3090

首次建议运行：

```bash
accelerate config
```

然后再执行 `scripts/run_sft.sh` 和 `scripts/run_grpo.sh`。

## 6. 可复现实验建议

- 固定随机种子（配置中默认 `seed=42`）
- 对每次实验保存配置快照（`configs/*.yaml`）
- 用 `report.md` 记录：
  - 数据版本
  - 关键超参
  - 训练曲线
  - 评估结果

## 7. 注意事项

- 当前仓库是「小型可复现模板」，默认参数偏保守，优先可跑通。
- GRPO 相关 API 与 `trl` 版本强相关，如有接口变化，请优先升级 `trl` 并同步脚本参数。
- 量化脚本默认保存 transformers 兼容格式，部署到 ModelScope 前建议本地先执行 `deploy/inference_demo.py` 验证。

