"""合并 LoRA 适配器到基础模型并导出。"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 LoRA 权重")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.torch_dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.lora_path)
    merged = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[MERGE] 已输出到: {args.output_dir}")


if __name__ == "__main__":
    main()
