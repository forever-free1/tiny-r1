"""4-bit 量化导出（bitsandbytes 动态量化加载后再保存）。"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="量化模型并导出")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--compute_dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    compute_dtype = getattr(torch, args.compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[QUANT] 已导出 4-bit 模型到: {args.output_dir}")


if __name__ == "__main__":
    main()
